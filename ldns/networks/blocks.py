import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from ldns.networks.s4 import FFTConv



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
        self.shared_dim = shared_ae_config['C_in']

        # The shared AutoEncoder body
        self.shared_ae = AutoEncoder(**shared_ae_config)

        # Session-specific read-in and read-out layers
        self.read_in_layers = nn.ModuleDict()
        self.read_out_layers = nn.ModuleDict()

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
        # Map from session-specific dimension to shared dimension
        x_shared = self.read_in_layers[session_id](x)
        # Encode using the shared AE body
        z = self.shared_ae.encode(x_shared)
        return z

    def decode(self, z, session_id):
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
        x_hat = self.decode(z, session_id)
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
        return list(self.read_in_layers[session_id].parameters()) + \
               list(self.read_out_layers[session_id].parameters())
    

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


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


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
    ):
        super().__init__()
        self.C_in = C_in
        self.C = C
        self.C_latent = C_latent
        self.L = L
        self.bidirectional = bidirectional
        self.kernel_params = kernel_params
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


class DenoiserBlock(nn.Module):
    def __init__(
        self, C, L, bidirectional=True, kernel_params=None, use_act2=False
    ):
        super().__init__()
        self.C = C
        self.L = L
        self.bidirectional = bidirectional
        self.kernel_params = kernel_params
        self.time_mixer = self.get_time_mixer()
        self.channel_mixer = self.get_channel_mixer()
        self.norm1 = nn.InstanceNorm1d(
            C, affine=False
        )  # affine=False because we will use adaLN
        self.norm2 = nn.InstanceNorm1d(C, affine=False)
        self.act1 = nn.GELU()
        self.act2 = nn.GELU() if use_act2 else nn.Identity()
        self.ada_ln = nn.Sequential(  # gets as input [B, C]
            nn.GELU(),
            nn.Linear(
                C // 4,
                C * 6,
                bias=True,
            ),
        )

        # zero-init all weights and biases of ada_ln linear layer
        self.ada_ln[-1].weight.data.zero_()
        self.ada_ln[-1].bias.data.zero_()

    def get_time_mixer(self):
        return FFTConv(
            self.C, bidirectional=self.bidirectional, activation=None
        )

    def get_channel_mixer(self):
        return nn.Sequential(
            Rearrange("b c l -> b l c"),
            nn.Linear(self.C, self.C * 2),
            nn.GELU(),
            nn.Linear(self.C * 2, self.C),
            Rearrange("b l c -> b c l"),
        )

    @staticmethod
    def affine_op(x, shift, scale):
        # x is [B, C, L], shift and scale are [B, C]
        return x * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)

    def forward(self, x, t_cond):
        y = x  # x is residual stream
        y = self.norm1(y)
        shift_tm, scale_tm, gate_tm, shift_cm, scale_cm, gate_cm = self.ada_ln(
            t_cond
        ).chunk(6, dim=1)
        y = self.affine_op(y, shift_tm, scale_tm)
        y = self.time_mixer(y)
        y = y[0]  # get output not state for gconv and fftconv
        y = x + gate_tm.unsqueeze(-1) * self.act1(y)

        x = y  # x is again residual stream from last layer
        y = self.norm2(y)
        y = self.affine_op(y, shift_cm, scale_cm)
        y = x + gate_cm.unsqueeze(-1) * self.act2(self.channel_mixer(y))
        return y
