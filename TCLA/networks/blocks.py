import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from TCLA.networks.s4 import FFTConv


def gaussian_smooth_1d(x, sigma_bins):
    """Apply depthwise Gaussian smoothing along the time axis of [B, C, T]."""
    sigma_bins = float(sigma_bins)
    if sigma_bins <= 0 or x.shape[-1] <= 1:
        return x

    radius = min(max(1, int(math.ceil(3.0 * sigma_bins))), x.shape[-1] - 1)
    grid = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-0.5 * (grid / sigma_bins) ** 2)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1).repeat(x.shape[1], 1, 1)

    padded = F.pad(x, (radius, radius), mode="reflect")
    return F.conv1d(padded, kernel, groups=x.shape[1])


##############################################################
# from https://github.com/zhuyu-cs/PNBA/tree/main/models
##############################################################


class PNBAReadIn(nn.Module):
    """
    Session-agnostic read-in:
    [B, N, T] -> [B, C_proj, D, T] -> [B, C_proj*D, T]
    """

    def __init__(self, out_channels=32, pooled_neurons=8, kernel_size=(3, 3)):
        super().__init__()
        pad_h = kernel_size[0] // 2
        pad_w = kernel_size[1] // 2
        self.out_channels = out_channels
        self.pooled_neurons = pooled_neurons
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(pad_h, pad_w),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((pooled_neurons, None))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.proj(x)
        x = self.pool(x)
        b, c, d, t = x.shape
        x_3d = x.reshape(b, c * d, t)
        return x, x_3d


class PNBAReadOut(nn.Module):
    """
    Session-agnostic read-out:
    [B, C_proj, D, T] -> interpolate to [B, C_proj, N, T] -> [B, N, T]
    """

    def __init__(self, in_channels=32, kernel_size=(3, 3)):
        super().__init__()
        pad_h = kernel_size[0] // 2
        pad_w = kernel_size[1] // 2
        self.out_proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=(pad_h, pad_w),
            bias=True,
        )

    def forward(self, h_4d, target_neurons):
        _, _, _, t = h_4d.shape
        h = F.interpolate(
            h_4d,
            size=(target_neurons, t),
            mode="bilinear",
            align_corners=False,
        )
        h = self.out_proj(h)
        return h.squeeze(1)


class MultiSessionAutoEncoder(nn.Module):
    """
    A wrapper for the AutoEncoder that handles multiple sessions with varying neuron counts.
    It uses session-specific read-in and read-out layers to map to/from a shared latent space.
    """
    def __init__(self, session_configs, shared_ae_config):
        """
        Args:
            session_configs (dict): A dictionary where keys are session_ids and values are
                                    dicts with 'neuron_count'. e.g., {'sessionA': {'neuron_count': 70}, ...}
            shared_ae_config (dict): A dictionary of arguments for the shared AutoEncoder core.
                                    'C_in' in this config will be the shared dimension.
        """
        super().__init__()
        self.session_configs = session_configs
        self.use_pnba_read_adapter = shared_ae_config.pop("use_pnba_read_adapter", False)
        self.pnba_proj_channels = shared_ae_config.pop("pnba_proj_channels", 32)
        self.pnba_pooled_neurons = shared_ae_config.pop("pnba_pooled_neurons", 8)

        if self.use_pnba_read_adapter:
            self.shared_dim = self.pnba_proj_channels * self.pnba_pooled_neurons
            shared_ae_config['C_in'] = self.shared_dim
            self.pnba_readin = PNBAReadIn(
                out_channels=self.pnba_proj_channels,
                pooled_neurons=self.pnba_pooled_neurons,
                kernel_size=(3, 3),
            )
            self.pnba_readout = PNBAReadOut(
                in_channels=self.pnba_proj_channels,
                kernel_size=(3, 3),
            )
        else:
            self.shared_dim = shared_ae_config['C_in']

        # The shared AutoEncoder body
        self.shared_ae = AutoEncoder(**shared_ae_config)

        # Session-specific read-in and read-out layers
        self.read_in_layers = nn.ModuleDict()
        self.read_out_layers = nn.ModuleDict()

        if not self.use_pnba_read_adapter:
            for session_id, config in self.session_configs.items():
                neuron_count = config['neuron_count']
                # Read-in: (B, N_i, L) -> (B, D, L)
                self.read_in_layers[session_id] = nn.Conv1d(neuron_count, self.shared_dim, 1)
                # Read-out: (B, D, L) -> (B, N_i, L)
                self.read_out_layers[session_id] = nn.Conv1d(self.shared_dim, neuron_count, 1)

    def encode(self, x, session_id):
        """
        Encodes data from a specific session into the shared latent space.
        Args:
            x (Tensor): Input tensor of shape [B, N_i, L].
            session_id (str): The identifier for the session.
        Returns:
            Tensor: The latent representation z of shape [B, C_latent, L].
        """
        if self.use_pnba_read_adapter:
            _, x_shared = self.pnba_readin(x)
        else:
            # Map from session-specific dimension to shared dimension
            x_shared = self.read_in_layers[session_id](x)
        # Encode using the shared AE body
        z = self.shared_ae.encode(x_shared)
        return z

    def decode(self, z, session_id, target_neurons=None):
        """
        Decodes a latent representation back to a specific session's rate space.
        Args:
            z (Tensor): The latent representation tensor of shape [B, C_latent, L].
            session_id (str): The identifier for the session.
        Returns:
            Tensor: The reconstructed rates of shape [B, N_i, L].
        """
        # Decode using the shared AE body
        x_shared_hat = self.shared_ae.decode(z)
        if self.use_pnba_read_adapter:
            if target_neurons is None:
                if session_id not in self.session_configs:
                    raise KeyError(
                        f"Session '{session_id}' is not in session_configs and target_neurons was not provided."
                    )
                target_neurons = self.session_configs[session_id]['neuron_count']
            b, _, t = x_shared_hat.shape
            x_4d = x_shared_hat.view(b, self.pnba_proj_channels, self.pnba_pooled_neurons, t)
            x_hat = self.pnba_readout(x_4d, target_neurons=target_neurons)
        else:
            # Map from shared dimension back to session-specific dimension
            x_hat = self.read_out_layers[session_id](x_shared_hat)
        return x_hat

    def forward(self, x, session_id):
        """
        Full forward pass for reconstruction.
        Args:
            x (Tensor): Input tensor of shape [B, N_i, L].
            session_id (str): The identifier for the session.
        Returns:
            Tuple[Tensor, Tensor]: Reconstructed rates x_hat and latent representation z.
        """
        z = self.encode(x, session_id)
        target_neurons = x.shape[1]
        x_hat = self.decode(z, session_id, target_neurons=target_neurons)
        return x_hat, z

    def freeze_shared_ae(self):
        """Freezes the parameters of the shared AE body for fine-tuning."""
        for param in self.shared_ae.parameters():
            param.requires_grad = False

    def unfreeze_shared_ae(self):
        """Unfreezes the parameters of the shared AE body."""
        for param in self.shared_ae.parameters():
            param.requires_grad = True

    def get_finetune_parameters(self, session_id):
        """Returns the parameters of the read-in/out layers for a specific session."""
        if self.use_pnba_read_adapter:
            return list(self.pnba_readin.parameters()) + list(self.pnba_readout.parameters())
        return list(self.read_in_layers[session_id].parameters()) + \
               list(self.read_out_layers[session_id].parameters())
    

##############################################################
# from https://github.com/mackelab/LDNS/tree/main/ldns
##############################################################


class AutoEncoderBlock(nn.Module):
    def __init__(
        self,
        C,
        L,
        kernel="s4",
        bidirectional=False,
        kernel_params=None,
        num_lin_per_mlp=2,
        use_act2=False,
    ):
        super().__init__()
        self.C = C
        self.L = L
        self.bidirectional = bidirectional
        self.kernel_params = kernel_params
        self.time_mixer = self.get_time_mixer()
        self.post_tm_scale = nn.Conv1d(
            C, C, 1, bias=True, groups=C, padding="same"
        )  # channel-wise scale for post-act
        self.channel_mixer = self.get_channel_mixer(num_lin_per_mlp=num_lin_per_mlp)
        self.norm1 = nn.InstanceNorm1d(C, affine=False)  # make sure input is [B, C, L]!
        self.norm2 = nn.InstanceNorm1d(C, affine=False)  # we will use adaLN
        self.act1 = nn.GELU()
        self.act2 = nn.GELU() if use_act2 else nn.Identity()

        self.ada_ln = nn.Parameter(
            torch.zeros(1, C * 6, 1), requires_grad=True
        )  # 3 for each mixer, shift, scale, gate. gate remains unused for now

    @staticmethod
    def affine_op(x_, shift, scale):
        # x is [B, C, L], shift and scale are [B, C, 1]
        assert len(x_.shape) == len(shift.shape), f"{x_.shape} != {shift.shape}"
        return x_ * (1 + scale) + shift

    def get_time_mixer(self):
        time_mixer = FFTConv(
            self.C,
            bidirectional=self.bidirectional,
            activation=None,
        )

        return time_mixer

    def get_channel_mixer(self, num_lin_per_mlp=2):
        layers = [
            Rearrange("b c l -> b l c"),
            nn.Linear(self.C, self.C * 2, bias=False),  # required for zero-init block
        ]
        # extra linear layers prepended by GELU
        for _ in range(max(num_lin_per_mlp - 2, 0)):
            layers.extend(
                [
                    nn.GELU(),
                    nn.Linear(self.C * 2, self.C * 2, bias=False),
                ]
            )
        layers.extend(
            [
                nn.GELU(),
                nn.Linear(self.C * 2, self.C, bias=False),
            ]
        )
        # finally rearrange back to [B, C, L]
        layers.append(Rearrange("b l c -> b c l"))
        return nn.Sequential(*layers)

    def forward(self, x):
        y = x  # x is residual stream
        y = self.norm1(y)
        ada_ln = repeat(self.ada_ln, "1 d c -> b d c", b=x.shape[0])
        shift_tm, scale_tm, gate_tm, shift_cm, scale_cm, gate_cm = ada_ln.chunk(
            6, dim=1
        )
        y = self.affine_op(y, shift_tm, scale_tm)
        y = self.time_mixer(y)
        y = y[0]  # get output not state for gconv and fftconv
        # y = x + gate_tm.unsqueeze(-1) * self.act1(y)
        y = x + self.post_tm_scale(self.act1(y))

        x = y  # x is again residual stream from last layer
        y = self.norm2(y)
        y = self.affine_op(y, shift_cm, scale_cm)
        # y = x + gate_cm.unsqueeze(-1) * self.act2(self.channel_mixer(y))
        y = x + self.act2(self.channel_mixer(y))
        return y


class AutoEncoder(nn.Module):
    def __init__(
        self,
        C_in,
        C,
        C_latent,
        L,
        kernel="s4",
        bidirectional=True,
        kernel_params=None,
        in_groups=None,
        bottleneck_groups=None,
        num_blocks=4,
        num_blocks_decoder=None,
        num_lin_per_mlp=2,
        use_act_bottleneck=False,
        latent_smooth_sigma_bins=0.0,
    ):
        super().__init__()
        self.C_in = C_in
        self.C = C
        self.C_latent = C_latent
        self.L = L
        self.bidirectional = bidirectional
        self.kernel_params = kernel_params
        self.latent_smooth_sigma_bins = float(latent_smooth_sigma_bins)
        if in_groups is None:
            if C % C_in == 0 and C > C_in:
                in_groups = C_in
            else:
                in_groups = 1

        if bottleneck_groups is None:
            if C % C_latent == 0 and C > C_latent:
                bottleneck_groups = C_latent
            else:
                bottleneck_groups = 1

        self.encoder_in = nn.Conv1d(
            C_in,
            C,
            1,
            # groups=in_groups,
        )  # in_groups matter for encoding count data
        self.encoder = nn.ModuleList(
            [
                AutoEncoderBlock(
                    C,
                    L,
                    kernel,
                    bidirectional,
                    kernel_params,
                    num_lin_per_mlp=num_lin_per_mlp,
                )
                for _ in range(num_blocks)
            ]
        )

        self.bottleneck = nn.Conv1d(C, C_latent, 1, groups=bottleneck_groups)
        self.act_bottleneck = nn.GELU() if use_act_bottleneck else nn.Identity()
        self.unbottleneck = nn.Conv1d(C_latent, C, 1, groups=bottleneck_groups)

        if num_blocks_decoder == 0:
            self.decoder = nn.ModuleList([nn.GELU()])  # jsut the activation

        else:
            self.decoder = nn.ModuleList(
                [
                    AutoEncoderBlock(
                        C,
                        L,
                        kernel,
                        bidirectional,
                        kernel_params,
                        num_lin_per_mlp=num_lin_per_mlp,
                    )
                    for _ in range(
                        (
                            num_blocks
                            if num_blocks_decoder is None
                            else num_blocks_decoder
                        )
                    )
                ]
            )

        # self.decoder_out = nn.Conv1d(C, C_in, 1, groups=in_groups)
        self.decoder_out = nn.Conv1d(C, C_in, 1)

    def encode(self, x):
        z = self.encoder_in(x)
        for block in self.encoder:
            z = block(z)
        z = self.bottleneck(z)
        z = gaussian_smooth_1d(z, self.latent_smooth_sigma_bins)
        return z

    def decode(self, z):
        xhat = self.act_bottleneck(z)
        xhat = self.unbottleneck(xhat)
        for block in self.decoder:
            xhat = block(xhat)
        xhat = self.decoder_out(xhat)
        return xhat

    def forward(self, x):
        z = self.encode(x)
        xhat = self.decode(z)
        return xhat, z
