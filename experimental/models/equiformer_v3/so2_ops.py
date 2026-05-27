import torch
import math


class SO2MLinear(torch.nn.Module):
    """
    Perform an SO(2) linear operation to features corresponding to +- m

    Args:
        m (int):                    Order of the spherical harmonic coefficients
        num_in_channels (int):      Number of input channels
        num_out_channels (int):     Number of output channels
        lmax (int):                 Maximum degrees (l)
        mmax (int):                 Maximum order (m)
    """

    def __init__(self, m, num_in_channels, num_out_channels, lmax, mmax):
        super(SO2MLinear, self).__init__()

        self.m = m
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.lmax = lmax
        self.mmax = mmax

        num_m_components = self.lmax - self.m + 1
        assert num_m_components > 0

        self.in_features = num_m_components * self.num_in_channels
        self.out_features = num_m_components * self.num_out_channels

        self.fc = torch.nn.Linear(self.in_features, (2 * self.out_features), bias=False)
        self.fc.weight.data.mul_(1 / math.sqrt(2))

        # Fused linear for dense path: absorbs negation into weight matrix
        # to avoid subtraction (works around NCC_ILSA902 compiler bug).
        # Initialized to None; call build_fused_linear() after loading weights.
        self._fc_fused = None

    def build_fused_linear(self):
        """
        Build a fused linear layer that computes complex multiply without subtract.

        Original: fc(x) → chunk → x_r_0 - x_i_1, x_r_1 + x_i_0
        Fused: Reshape input [B, 2, in] → [B, 2*in], apply fused weight → [B, 2*out]
               then reshape to [B, 2, out] = (real, imag) with NO subtract.

        The fused weight absorbs the sign flip: W_fused = [[W_r, -W_i], [W_i, W_r]]
        where W_r and W_i are the real and imaginary parts of the original weight.

        Call this AFTER loading pre-trained weights (it reads from self.fc.weight).
        """
        with torch.no_grad():
            W = self.fc.weight  # [2*out_features, in_features]
            out_feat = self.out_features
            # The original fc produces [2*out_features] per input row.
            # First out_features = "real part output", second = "imaginary part output"
            W_r = W[:out_feat, :]  # real part weights [out, in]
            W_i = W[out_feat:, :]  # imag part weights [out, in]

            # Fused weight: input is [x_real; x_imag] (concatenated, size 2*in)
            # output_real = W_r @ x_real + (-W_i) @ x_imag   (absorbs subtraction!)
            # output_imag = W_i @ x_real + W_r @ x_imag
            # So fused weight matrix is:
            # [[W_r, -W_i],   shape [2*out, 2*in]
            #  [W_i,  W_r]]
            in_feat = self.in_features
            W_fused = torch.zeros(
                2 * out_feat, 2 * in_feat, dtype=W.dtype, device=W.device
            )
            W_fused[:out_feat, :in_feat] = W_r
            W_fused[:out_feat, in_feat:] = -W_i
            W_fused[out_feat:, :in_feat] = W_i
            W_fused[out_feat:, in_feat:] = W_r

        self._fc_fused = torch.nn.Linear(2 * in_feat, 2 * out_feat, bias=False)
        self._fc_fused.weight = torch.nn.Parameter(W_fused)
        self._fc_fused.weight.requires_grad_(self.fc.weight.requires_grad)

    def forward(self, x_m, concat_outputs=True):
        x_m = self.fc(x_m)
        # Use chunk instead of narrow: backward of chunk is cat (no slice_scatter)
        x_r, x_i = torch.chunk(x_m, chunks=2, dim=2)
        x_r_0, x_r_1 = torch.chunk(x_r, chunks=2, dim=1)
        x_i_0, x_i_1 = torch.chunk(x_i, chunks=2, dim=1)
        x_m_r = x_r_0 - x_i_1
        x_m_i = x_r_1 + x_i_0
        x_out = (x_m_r, x_m_i)
        if concat_outputs:
            x_out = torch.cat(x_out, dim=1)
        return x_out

    def forward_fused(self, x_m, concat_outputs=True):
        """
        Fused forward that avoids subtraction (workaround for NCC_ILSA902).

        Input: x_m [B, 2, in_features] where dim=1 is (real_input, imag_input)
        Output: (x_m_r, x_m_i) or concatenated [B, 2, out_features]

        Uses a single linear with pre-negated weights instead of
        fc → chunk → subtract/add.
        """
        assert self._fc_fused is not None, (
            "Call build_fused_linear() before using forward_fused()"
        )
        B = x_m.shape[0]
        # Concatenate real and imag inputs along feature dim: [B, 2, in] → [B, 2*in]
        # dim=1 has size 2: position 0 = real, position 1 = imag
        x_flat = x_m.reshape(B, -1)  # [B, 2*in_features]
        # Apply fused linear: directly produces [output_real; output_imag]
        out = self._fc_fused(x_flat)  # [B, 2*out_features]
        # Split into real and imaginary parts
        x_m_r, x_m_i = torch.chunk(out, 2, dim=1)  # each [B, out_features]
        # Reshape to [B, 1, out_features] to match original output shape
        x_m_r = x_m_r.unsqueeze(1)
        x_m_i = x_m_i.unsqueeze(1)
        x_out = (x_m_r, x_m_i)
        if concat_outputs:
            x_out = torch.cat(x_out, dim=1)
        return x_out


class SO2Linear(torch.nn.Module):
    """
    Perform SO(2) linear operations to all m (orders) components

    Args:
        num_in_channels (int):      Number of input channels
        num_out_channels (int):     Number of output channels
        lmax (int):                 Maximum degrees (l)
        mmax (int):                 Maximum order (m)
        extra_m0_out_channels (int):    If not None, return `outputs` (torch.Tensor) and `extra_m0_features` (torch.Tensor).
    """

    def __init__(
        self, num_in_channels, num_out_channels, lmax, mmax, extra_m0_out_channels=None
    ):
        super(SO2Linear, self).__init__()
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.lmax = lmax
        self.mmax = mmax
        self.extra_m0_out_channels = extra_m0_out_channels

        # for m = 0
        num_in_channels_m0 = (self.lmax + 1) * self.num_in_channels
        num_out_channels_m0 = (self.lmax + 1) * self.num_out_channels
        if self.extra_m0_out_channels is not None:
            self.num_channels_m0_list = [
                self.extra_m0_out_channels,
                num_out_channels_m0,
            ]
            num_out_channels_m0 = num_out_channels_m0 + self.extra_m0_out_channels
        self.fc_m0 = torch.nn.Linear(num_in_channels_m0, num_out_channels_m0)

        # SO(2) linear for non-zero m
        self.so2_m_linear = torch.nn.ModuleList()
        for m in range(1, self.mmax + 1):
            self.so2_m_linear.append(
                SO2MLinear(
                    m,
                    self.num_in_channels,
                    self.num_out_channels,
                    self.lmax,
                    self.mmax,
                )
            )

    def build_fused_linear(self):
        """Build fused linear layers for all SO2MLinear submodules."""
        for m_linear in self.so2_m_linear:
            m_linear.build_fused_linear()

    def forward(self, x):
        """
        1.  `x` shape: [num_edges, num_m_components, num_channels]
        2.  We assume the layout of m components is (0, 0, ...), (1, 1, ...), ...
        """
        num_edges = x.shape[0]
        outputs = []

        # Split x into m-components using torch.split (backward = cat, no slice_scatter)
        split_sizes = [self.lmax + 1] + [
            2 * (self.lmax + 1 - m) for m in range(1, self.mmax + 1)
        ]
        x_splits = torch.split(x, split_sizes, dim=1)

        # Compute m=0 coefficients separately since they only have real values (no imaginary)
        x_m0 = x_splits[0].reshape(num_edges, -1)
        x_m0 = self.fc_m0(x_m0)

        x_m0_extra = None
        # extract extra m0 features
        if self.extra_m0_out_channels is not None:
            x_m0_extra, x_m0 = torch.split(x_m0, self.num_channels_m0_list, dim=1)

        x_m0 = x_m0.view(num_edges, -1, self.num_out_channels)
        outputs.append(x_m0)

        # Compute the values for the m > 0 coefficients
        for m in range(1, self.mmax + 1):
            x_m = x_splits[m].reshape(num_edges, 2, -1)
            # Replace the original one with the followings to prevent one `torch.cat()` for each m > 0
            x_m = self.so2_m_linear[m - 1](x_m, concat_outputs=False)
            x_m_pos, x_m_neg = x_m[0], x_m[1]
            x_m_pos = x_m_pos.view(num_edges, -1, self.num_out_channels)
            x_m_neg = x_m_neg.view(num_edges, -1, self.num_out_channels)
            outputs.append(x_m_pos)
            outputs.append(x_m_neg)

        outputs = torch.cat(outputs, dim=1)

        if self.extra_m0_out_channels is not None:
            return outputs, x_m0_extra
        else:
            return outputs

    def forward_fused(self, x):
        """
        Forward pass using fused linear (no subtraction) for NCC_ILSA902 workaround.

        Same interface as forward(). Requires build_fused_linear() called first.
        """
        num_edges = x.shape[0]
        outputs = []

        # Split x into m-components using torch.split (backward = cat, no slice_scatter)
        split_sizes = [self.lmax + 1] + [
            2 * (self.lmax + 1 - m) for m in range(1, self.mmax + 1)
        ]
        x_splits = torch.split(x, split_sizes, dim=1)

        # Compute m=0 coefficients separately since they only have real values (no imaginary)
        x_m0 = x_splits[0].reshape(num_edges, -1)
        x_m0 = self.fc_m0(x_m0)

        x_m0_extra = None
        # extract extra m0 features
        if self.extra_m0_out_channels is not None:
            x_m0_extra, x_m0 = torch.split(x_m0, self.num_channels_m0_list, dim=1)

        x_m0 = x_m0.view(num_edges, -1, self.num_out_channels)
        outputs.append(x_m0)

        # Compute the values for the m > 0 coefficients using FUSED forward
        for m in range(1, self.mmax + 1):
            x_m = x_splits[m].reshape(num_edges, 2, -1)
            x_m = self.so2_m_linear[m - 1].forward_fused(x_m, concat_outputs=False)
            x_m_pos, x_m_neg = x_m[0], x_m[1]
            x_m_pos = x_m_pos.view(num_edges, -1, self.num_out_channels)
            x_m_neg = x_m_neg.view(num_edges, -1, self.num_out_channels)
            outputs.append(x_m_pos)
            outputs.append(x_m_neg)

        outputs = torch.cat(outputs, dim=1)

        if self.extra_m0_out_channels is not None:
            return outputs, x_m0_extra
        else:
            return outputs
