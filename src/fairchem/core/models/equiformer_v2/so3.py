"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.


TODO:
    1. Simplify the case when `num_resolutions` == 1.
    2. Remove indexing when the shape is the same.
    3. Move some functions outside classes and to separate files.
"""

from __future__ import annotations

import math

import torch

try:
    from e3nn import o3
    from e3nn.o3 import FromS2Grid, ToS2Grid
except ImportError:
    pass

from torch.nn import Linear

from .wigner import wigner_D


class CoefficientMappingModule(torch.nn.Module):
    """
    Helper module for coefficients used to reshape lval <--> m and to get coefficients of specific degree or order

    Args:
        lmax_list (list:int):   List of maximum degree of the spherical harmonics
        mmax_list (list:int):   List of maximum order of the spherical harmonics
    """

    def __init__(
        self,
        lmax_list: list[int],
        mmax_list: list[int],
    ):
        super().__init__()

        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(lmax_list)

        # Temporarily use `cpu` as device and this will be overwritten.
        self.device = "cpu"

        # Compute the degree (lval) and order (m) for each entry of the embedding
        l_harmonic = torch.tensor([], device=self.device).long()
        m_harmonic = torch.tensor([], device=self.device).long()
        m_complex = torch.tensor([], device=self.device).long()

        res_size = torch.zeros([self.num_resolutions], device=self.device).long()

        offset = 0
        for i in range(self.num_resolutions):
            for lval in range(self.lmax_list[i] + 1):
                mmax = min(self.mmax_list[i], lval)
                m = torch.arange(-mmax, mmax + 1, device=self.device).long()
                m_complex = torch.cat([m_complex, m], dim=0)
                m_harmonic = torch.cat([m_harmonic, torch.abs(m).long()], dim=0)
                l_harmonic = torch.cat([l_harmonic, m.fill_(lval).long()], dim=0)
            res_size[i] = len(l_harmonic) - offset
            offset = len(l_harmonic)

        num_coefficients = len(l_harmonic)
        # `self.to_m` moves m components from different L to contiguous index
        to_m = torch.zeros([num_coefficients, num_coefficients], device=self.device)
        m_size = torch.zeros([max(self.mmax_list) + 1], device=self.device).long()

        # The following is implemented poorly - very slow. It only gets called
        # a few times so haven't optimized.
        offset = 0
        for m in range(max(self.mmax_list) + 1):
            idx_r, idx_i = self.complex_idx(m, -1, m_complex, l_harmonic)

            for idx_out, idx_in in enumerate(idx_r):
                to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_r)

            m_size[m] = int(len(idx_r))

            for idx_out, idx_in in enumerate(idx_i):
                to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_i)

        to_m = to_m.detach()

        # save tensors and they will be moved to GPU
        self.register_buffer("l_harmonic", l_harmonic)
        self.register_buffer("m_harmonic", m_harmonic)
        self.register_buffer("m_complex", m_complex)
        self.register_buffer("res_size", res_size)
        self.register_buffer("to_m", to_m)
        self.register_buffer("m_size", m_size)

        # Pre-compute m_size as a Python list to avoid device sync barriers
        # when used as arguments to .narrow() on accelerator tensors.
        self.m_size_list: list[int] = m_size.tolist()

        # Dictionary cache for coefficient_idx results (supports multiple lmax/mmax pairs)
        self._coeff_idx_cache: dict[tuple[int, int], torch.Tensor] = {}
        self._rotate_inv_rescale_cache: dict[tuple[int, int], torch.Tensor] = {}

        # for caching the output of `coefficient_idx` (legacy single-pair cache)
        self.lmax_cache, self.mmax_cache = None, None
        self.mask_indices_cache = None
        self.rotate_inv_rescale_cache = None

        # Pre-compute indices for all valid (lmax, mmax) combinations
        self._precompute_all_indices()

    def _precompute_all_indices(self):
        """Pre-compute coefficient_idx for all valid (lmax, mmax) combinations.

        This avoids dynamic-shape ops (masked_select) during forward pass,
        which would cause graph breaks in torch.compile.
        Indices are stored as Python lists to avoid device placement issues.
        """
        max_l = max(self.lmax_list)
        max_m = max(self.mmax_list)
        for lmax in range(max_l + 1):
            for mmax in range(max_m + 1):
                mask = torch.bitwise_and(
                    self.l_harmonic.le(lmax), self.m_harmonic.le(mmax)
                )
                indices = torch.arange(len(mask))
                idx_tensor = torch.masked_select(indices, mask)
                # Store as Python list — will be converted to tensor on correct device at use time
                self._coeff_idx_cache[(lmax, mmax)] = idx_tensor.tolist()

    # Return mask containing coefficients of order m (real and imaginary parts)
    def complex_idx(self, m: int, lmax: int, m_complex, l_harmonic):
        """
        Add `m_complex` and `l_harmonic` to the input arguments
        since we cannot use `self.m_complex`.
        """
        if lmax == -1:
            lmax = max(self.lmax_list)

        indices = torch.arange(len(l_harmonic), device=self.device)
        # Real part
        mask_r = torch.bitwise_and(l_harmonic.le(lmax), m_complex.eq(m))
        mask_idx_r = torch.masked_select(indices, mask_r)

        mask_idx_i = torch.tensor([], device=self.device).long()
        # Imaginary part
        if m != 0:
            mask_i = torch.bitwise_and(l_harmonic.le(lmax), m_complex.eq(-m))
            mask_idx_i = torch.masked_select(indices, mask_i)

        return mask_idx_r, mask_idx_i

    # Return mask containing coefficients less than or equal to degree (lval) and order (m)
    def coefficient_idx(self, lmax: int, mmax: int):
        # Fast path: return from pre-computed dict cache (no dynamic ops)
        key = (lmax, mmax)
        if key in self._coeff_idx_cache:
            idx_list = self._coeff_idx_cache[key]
            return torch.tensor(
                idx_list, dtype=torch.long, device=self.l_harmonic.device
            )

        # Fallback: compute and cache (only needed if called with unexpected args)
        mask = torch.bitwise_and(self.l_harmonic.le(lmax), self.m_harmonic.le(mmax))
        self.device = mask.device
        indices = torch.arange(len(mask), device=self.device)
        mask_indices = torch.masked_select(indices, mask)
        self._coeff_idx_cache[key] = mask_indices.tolist()
        return mask_indices

    # Return the re-scaling for rotating back to original frame
    # this is required since we only use a subset of m components for SO(2) convolution
    def get_rotate_inv_rescale(self, lmax: int, mmax: int):
        key = (lmax, mmax)
        if key in self._rotate_inv_rescale_cache:
            return self._rotate_inv_rescale_cache[key]

        coeff_idx = self.coefficient_idx(lmax, mmax)

        rotate_inv_rescale = torch.ones(
            (1, (lmax + 1) ** 2, (lmax + 1) ** 2), device=self.l_harmonic.device
        )
        for lval in range(lmax + 1):
            if lval <= mmax:
                continue
            start_idx = lval**2
            length = 2 * lval + 1
            rescale_factor = math.sqrt(length / (2 * mmax + 1))
            rotate_inv_rescale[
                :,
                start_idx : (start_idx + length),
                start_idx : (start_idx + length),
            ] = rescale_factor
        rotate_inv_rescale = rotate_inv_rescale[:, :, coeff_idx]
        self._rotate_inv_rescale_cache[key] = rotate_inv_rescale
        return rotate_inv_rescale

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lmax_list={self.lmax_list}, mmax_list={self.mmax_list})"


class SO3_Embedding:
    """
    Helper functions for performing operations on irreps embedding

    Args:
        length (int):           Batch size
        lmax_list (list:int):   List of maximum degree of the spherical harmonics
        num_channels (int):     Number of channels
        device:                 Device of the output
        dtype:                  type of the output tensors
    """

    def __init__(
        self,
        length: int,
        lmax_list: list[int],
        num_channels: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.device = device
        self.dtype = dtype
        self.num_resolutions = len(lmax_list)

        self.num_coefficients = 0
        for i in range(self.num_resolutions):
            self.num_coefficients = self.num_coefficients + int((lmax_list[i] + 1) ** 2)

        embedding = torch.zeros(
            length,
            self.num_coefficients,
            self.num_channels,
            device=self.device,
            dtype=self.dtype,
        )

        self.set_embedding(embedding)
        self.set_lmax_mmax(lmax_list, lmax_list.copy())

    # Clone an embedding of irreps
    def clone(self) -> SO3_Embedding:
        clone = SO3_Embedding(
            0,
            self.lmax_list.copy(),
            self.num_channels,
            self.device,
            self.dtype,
        )
        clone.set_embedding(self.embedding.clone())
        return clone

    # Initialize an embedding of irreps
    def set_embedding(self, embedding) -> None:
        self.length = len(embedding)
        self.embedding = embedding

    # Set the maximum order to be the maximum degree
    def set_lmax_mmax(self, lmax_list: list[int], mmax_list: list[int]) -> None:
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list

    # Expand the node embeddings to the number of edges
    def _expand_edge(self, edge_index: torch.Tensor) -> None:
        embedding = self.embedding[edge_index]
        self.set_embedding(embedding)

    # Initialize an embedding of irreps of a neighborhood
    def expand_edge(self, edge_index: torch.Tensor):
        x_expand = SO3_Embedding(
            0,
            self.lmax_list.copy(),
            self.num_channels,
            self.device,
            self.dtype,
        )
        x_expand.set_embedding(self.embedding[edge_index])
        return x_expand

    # Compute the sum of the embeddings of the neighborhood
    def _reduce_edge(self, edge_index: torch.Tensor, num_nodes: int):
        new_embedding = torch.zeros(
            num_nodes,
            self.num_coefficients,
            self.num_channels,
            device=self.embedding.device,
            dtype=self.embedding.dtype,
        )
        new_embedding.index_add_(0, edge_index, self.embedding)
        self.set_embedding(new_embedding)

    # Reshape the embedding lval -> m
    def _m_primary(self, mapping):
        to_m = mapping.to_m.to(self.embedding.dtype)
        self.embedding = torch.einsum("nac, ba -> nbc", self.embedding, to_m)

    # Reshape the embedding m -> lval
    def _l_primary(self, mapping):
        to_m = mapping.to_m.to(self.embedding.dtype)
        self.embedding = torch.einsum("nac, ab -> nbc", self.embedding, to_m)

    # Rotate the embedding
    def _rotate(self, SO3_rotation, lmax_list: list[int], mmax_list: list[int]):
        if self.num_resolutions == 1:
            embedding_rotate = SO3_rotation[0].rotate(
                self.embedding, lmax_list[0], mmax_list[0]
            )
        else:
            offset = 0
            embedding_rotate = torch.tensor([], device=self.device, dtype=self.dtype)
            for i in range(self.num_resolutions):
                num_coefficients = int((self.lmax_list[i] + 1) ** 2)
                embedding_i = self.embedding[:, offset : offset + num_coefficients]
                embedding_rotate = torch.cat(
                    [
                        embedding_rotate,
                        SO3_rotation[i].rotate(embedding_i, lmax_list[i], mmax_list[i]),
                    ],
                    dim=1,
                )
                offset = offset + num_coefficients

        self.embedding = embedding_rotate
        self.set_lmax_mmax(lmax_list.copy(), mmax_list.copy())

    # Rotate the embedding by the inverse of the rotation matrix
    def _rotate_inv(self, SO3_rotation, mappingReduced):
        if self.num_resolutions == 1:
            embedding_rotate = SO3_rotation[0].rotate_inv(
                self.embedding, self.lmax_list[0], self.mmax_list[0]
            )
        else:
            offset = 0
            embedding_rotate = torch.tensor([], device=self.device, dtype=self.dtype)
            for i in range(self.num_resolutions):
                num_coefficients = mappingReduced.res_size[i]
                embedding_i = self.embedding[:, offset : offset + num_coefficients]
                embedding_rotate = torch.cat(
                    [
                        embedding_rotate,
                        SO3_rotation[i].rotate_inv(
                            embedding_i, self.lmax_list[i], self.mmax_list[i]
                        ),
                    ],
                    dim=1,
                )
                offset = offset + num_coefficients
        self.embedding = embedding_rotate

        # Assume mmax = lmax when rotating back
        for i in range(self.num_resolutions):
            self.mmax_list[i] = int(self.lmax_list[i])
        self.set_lmax_mmax(self.lmax_list, self.mmax_list)

    # Compute point-wise spherical non-linearity
    def _grid_act(self, SO3_grid, act, mappingReduced):
        offset = 0
        for i in range(self.num_resolutions):
            num_coefficients = mappingReduced.res_size[i]

            if self.num_resolutions == 1:
                x_res = self.embedding
            else:
                x_res = self.embedding[
                    :, offset : offset + num_coefficients
                ].contiguous()
            to_grid_mat = SO3_grid[self.lmax_list[i]][
                self.mmax_list[i]
            ].get_to_grid_mat(self.device)
            from_grid_mat = SO3_grid[self.lmax_list[i]][
                self.mmax_list[i]
            ].get_from_grid_mat(self.device)

            x_grid = torch.einsum("bai, zic -> zbac", to_grid_mat, x_res)
            x_grid = act(x_grid)
            x_res = torch.einsum("bai, zbac -> zic", from_grid_mat, x_grid)
            if self.num_resolutions == 1:
                self.embedding = x_res
            else:
                self.embedding[:, offset : offset + num_coefficients] = x_res
            offset = offset + num_coefficients

    # Compute a sample of the grid
    def to_grid(self, SO3_grid, lmax=-1):
        if lmax == -1:
            lmax = max(self.lmax_list)

        to_grid_mat_lmax = SO3_grid[lmax][lmax].get_to_grid_mat(self.device)
        grid_mapping = SO3_grid[lmax][lmax].mapping

        offset = 0
        x_grid_parts = []

        for i in range(self.num_resolutions):
            num_coefficients = int((self.lmax_list[i] + 1) ** 2)
            if self.num_resolutions == 1:
                x_res = self.embedding
            else:
                x_res = self.embedding[
                    :, offset : offset + num_coefficients
                ].contiguous()
            to_grid_mat = to_grid_mat_lmax[
                :,
                :,
                grid_mapping.coefficient_idx(self.lmax_list[i], self.lmax_list[i]),
            ]
            x_grid_parts.append(
                torch.einsum("bai, zic -> zbac", to_grid_mat.to(x_res.dtype), x_res)
            )
            offset = offset + num_coefficients

        return (
            torch.cat(x_grid_parts, dim=3) if len(x_grid_parts) > 1 else x_grid_parts[0]
        )

    # Compute irreps from grid representation
    def _from_grid(self, x_grid, SO3_grid, lmax: int = -1):
        if lmax == -1:
            lmax = max(self.lmax_list)

        from_grid_mat_lmax = SO3_grid[lmax][lmax].get_from_grid_mat(self.device)
        grid_mapping = SO3_grid[lmax][lmax].mapping

        offset = 0
        offset_channel = 0
        for i in range(self.num_resolutions):
            from_grid_mat = from_grid_mat_lmax[
                :,
                :,
                grid_mapping.coefficient_idx(self.lmax_list[i], self.lmax_list[i]),
            ]
            if self.num_resolutions == 1:
                temp = x_grid
            else:
                temp = x_grid[
                    :,
                    :,
                    :,
                    offset_channel : offset_channel + self.num_channels,
                ]
            x_res = torch.einsum("bai, zbac -> zic", from_grid_mat.to(temp.dtype), temp)
            num_coefficients = int((self.lmax_list[i] + 1) ** 2)

            if self.num_resolutions == 1:
                self.embedding = x_res
            else:
                self.embedding[:, offset : offset + num_coefficients] = x_res

            offset = offset + num_coefficients
            offset_channel = offset_channel + self.num_channels


class SO3_Rotation(torch.nn.Module):
    """
    Helper functions for Wigner-D rotations

    Args:
        lmax_list (list:int):   List of maximum degree of the spherical harmonics
    """

    def __init__(
        self,
        lmax: int,
    ):
        super().__init__()
        self.lmax = lmax
        self.mapping = CoefficientMappingModule([self.lmax], [self.lmax])

    def set_wigner(self, rot_mat3x3):
        self.device, self.dtype = rot_mat3x3.device, rot_mat3x3.dtype
        self.wigner = self.RotationToWignerDMatrix(rot_mat3x3, 0, self.lmax)
        self.wigner_inv = torch.transpose(self.wigner, 1, 2).contiguous()
        self.wigner = self.wigner.detach()
        self.wigner_inv = self.wigner_inv.detach()

    # Rotate the embedding
    def rotate(self, embedding, out_lmax: int, out_mmax: int):
        out_mask = self.mapping.coefficient_idx(out_lmax, out_mmax)
        wigner = self.wigner[:, out_mask, :].to(embedding.dtype)
        return torch.bmm(wigner, embedding)

    # Rotate the embedding by the inverse of the rotation matrix
    def rotate_inv(self, embedding, in_lmax: int, in_mmax: int):
        in_mask = self.mapping.coefficient_idx(in_lmax, in_mmax)
        wigner_inv = self.wigner_inv[:, :, in_mask].to(embedding.dtype)
        wigner_inv_rescale = self.mapping.get_rotate_inv_rescale(in_lmax, in_mmax).to(
            embedding.dtype
        )
        wigner_inv = wigner_inv * wigner_inv_rescale
        return torch.bmm(wigner_inv, embedding)

    # Compute Wigner matrices from rotation matrix
    def RotationToWignerDMatrix(
        self, edge_rot_mat, start_lmax: int, end_lmax: int
    ) -> torch.Tensor:
        x = edge_rot_mat @ edge_rot_mat.new_tensor([0.0, 1.0, 0.0])
        alpha, beta = o3.xyz_to_angles(x)
        R = (
            o3.angles_to_matrix(alpha, beta, torch.zeros_like(alpha)).transpose(-1, -2)
            @ edge_rot_mat
        )
        gamma = torch.atan2(R[..., 0, 2], R[..., 0, 0])

        size = (end_lmax + 1) ** 2 - (start_lmax) ** 2
        wigner = torch.zeros(
            len(alpha), size, size, device=self.device, dtype=self.dtype
        )
        start = 0
        for lmax in range(start_lmax, end_lmax + 1):
            block = wigner_D(lmax, alpha, beta, gamma)
            end = start + block.size()[1]
            wigner[:, start:end, start:end] = block
            start = end

        return wigner.detach()


class SO3_Grid(torch.nn.Module):
    """
    Helper functions for grid representation of the irreps

    Args:
        lmax (int):   Maximum degree of the spherical harmonics
        mmax (int):   Maximum order of the spherical harmonics
    """

    def __init__(
        self,
        lmax: int,
        mmax: int,
        normalization: str = "integral",
        resolution: int | None = None,
    ):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.lat_resolution = 2 * (self.lmax + 1)
        if lmax == mmax:
            self.long_resolution = 2 * (self.mmax + 1) + 1
        else:
            self.long_resolution = 2 * (self.mmax) + 1
        if resolution is not None:
            self.lat_resolution = resolution
            self.long_resolution = resolution

        self.mapping = CoefficientMappingModule([self.lmax], [self.lmax])

        device = "cpu"

        to_grid = ToS2Grid(
            self.lmax,
            (self.lat_resolution, self.long_resolution),
            normalization=normalization,  # normalization="integral",
            device=device,
        )
        to_grid_mat = torch.einsum("mbi, am -> bai", to_grid.shb, to_grid.sha).detach()
        # rescale based on mmax
        if lmax != mmax:
            for lval in range(lmax + 1):
                if lval <= mmax:
                    continue
                start_idx = lval**2
                length = 2 * lval + 1
                rescale_factor = math.sqrt(length / (2 * mmax + 1))
                to_grid_mat[:, :, start_idx : (start_idx + length)] = (
                    to_grid_mat[:, :, start_idx : (start_idx + length)] * rescale_factor
                )
        to_grid_mat = to_grid_mat[
            :, :, self.mapping.coefficient_idx(self.lmax, self.mmax)
        ]

        from_grid = FromS2Grid(
            (self.lat_resolution, self.long_resolution),
            self.lmax,
            normalization=normalization,  # normalization="integral",
            device=device,
        )
        from_grid_mat = torch.einsum(
            "am, mbi -> bai", from_grid.sha, from_grid.shb
        ).detach()
        # rescale based on mmax
        if lmax != mmax:
            for lval in range(lmax + 1):
                if lval <= mmax:
                    continue
                start_idx = lval**2
                length = 2 * lval + 1
                rescale_factor = math.sqrt(length / (2 * mmax + 1))
                from_grid_mat[:, :, start_idx : (start_idx + length)] = (
                    from_grid_mat[:, :, start_idx : (start_idx + length)]
                    * rescale_factor
                )
        from_grid_mat = from_grid_mat[
            :, :, self.mapping.coefficient_idx(self.lmax, self.mmax)
        ]

        # save tensors and they will be moved to GPU
        self.register_buffer("to_grid_mat", to_grid_mat)
        self.register_buffer("from_grid_mat", from_grid_mat)

    # Compute matrices to transform irreps to grid
    def get_to_grid_mat(self, device):
        return self.to_grid_mat

    # Compute matrices to transform grid to irreps
    def get_from_grid_mat(self, device):
        return self.from_grid_mat

    # Compute grid from irreps representation
    def to_grid(self, embedding, lmax: int, mmax: int):
        to_grid_mat = self.to_grid_mat[:, :, self.mapping.coefficient_idx(lmax, mmax)]
        return torch.einsum(
            "bai, zic -> zbac", to_grid_mat.to(embedding.dtype), embedding
        )

    # Compute irreps from grid representation
    def from_grid(self, grid, lmax: int, mmax: int):
        from_grid_mat = self.from_grid_mat[
            :, :, self.mapping.coefficient_idx(lmax, mmax)
        ]
        return torch.einsum("bai, zbac -> zic", from_grid_mat.to(grid.dtype), grid)


class SO3_Linear(torch.nn.Module):
    def __init__(
        self, in_features: int, out_features: int, lmax: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lmax = lmax
        self.linear_list = torch.nn.ModuleList()
        for lval in range(lmax + 1):
            if lval == 0:
                self.linear_list.append(Linear(in_features, out_features, bias=bias))
            else:
                self.linear_list.append(Linear(in_features, out_features, bias=False))

    def forward(self, input_embedding, output_scale=None):
        out = []
        for lval in range(self.lmax + 1):
            start_idx = lval**2
            length = 2 * lval + 1
            features = input_embedding.embedding.narrow(1, start_idx, length)
            features = self.linear_list[lval](features)
            if output_scale is not None:
                scale = output_scale.narrow(1, lval, 1)
                features = features * scale
            out.append(features)
        out = torch.cat(out, dim=1)

        out_embedding = SO3_Embedding(
            0,
            input_embedding.lmax_list.copy(),
            self.out_features,
            device=input_embedding.device,
            dtype=input_embedding.dtype,
        )
        out_embedding.set_embedding(out)
        out_embedding.set_lmax_mmax(
            input_embedding.lmax_list.copy(), input_embedding.lmax_list.copy()
        )

        return out_embedding

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, lmax={self.lmax})"


class SO3_LinearV2(torch.nn.Module):
    def __init__(
        self, in_features: int, out_features: int, lmax: int, bias: bool = True
    ) -> None:
        """
        1. Use `torch.einsum` to prevent slicing and concatenation
        2. Need to specify some behaviors in `no_weight_decay` and weight initialization.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lmax = lmax

        self.weight = torch.nn.Parameter(
            torch.randn((self.lmax + 1), out_features, in_features)
        )
        bound = 1 / math.sqrt(self.in_features)
        torch.nn.init.uniform_(self.weight, -bound, bound)
        self.bias = torch.nn.Parameter(torch.zeros(out_features))

        expand_index = torch.zeros([(lmax + 1) ** 2]).long()
        for lval in range(lmax + 1):
            start_idx = lval**2
            length = 2 * lval + 1
            expand_index[start_idx : (start_idx + length)] = lval
        self.register_buffer("expand_index", expand_index)

    def forward(self, input_embedding):
        weight = torch.index_select(
            self.weight, dim=0, index=self.expand_index
        )  # [(L_max + 1) ** 2, C_out, C_in]
        out = torch.einsum(
            "bmi, moi -> bmo", input_embedding.embedding, weight
        )  # [N, (L_max + 1) ** 2, C_out]
        bias = self.bias.view(1, 1, self.out_features)
        out[:, 0:1, :] = out.narrow(1, 0, 1) + bias

        out_embedding = SO3_Embedding(
            0,
            input_embedding.lmax_list.copy(),
            self.out_features,
            device=input_embedding.device,
            dtype=input_embedding.dtype,
        )
        out_embedding.set_embedding(out)
        out_embedding.set_lmax_mmax(
            input_embedding.lmax_list.copy(), input_embedding.lmax_list.copy()
        )

        return out_embedding

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, lmax={self.lmax})"
