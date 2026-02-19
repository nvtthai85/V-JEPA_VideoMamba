import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """Selective State Space Model (S6) core.

    Continuous system:  h'(t) = A·h(t) + B·x(t),  y(t) = C·h(t) + D·x(t)
    Discretized (ZOH):  h_k = Ā·h_{k-1} + B̄·x_k,  y_k = C·h_k + D·x_k

    Args:
        d_model:  Input/output dimension (D).
        d_state:  SSM state expansion factor (N). Default 16.
        d_conv:   Local convolution width. Default 4.
        expand:   Inner dimension expansion. Default 2.
        dt_rank:  Rank of Δ projection. Default "auto" → ceil(d_model/16).
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto"):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(d_model * expand)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        # Input projection: x → (z, x_ssm)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1-D depthwise convolution on x_ssm
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # SSM parameter projections (input-dependent → selective)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # State matrix A: initialized as negative log-spaced (HiPPO-inspired)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # Skip connection D
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        Args:
            x: (B, L, D) input sequence.
        Returns:
            y: (B, L, D) output sequence.
        """
        B, L, D = x.shape

        # Project and split
        xz = self.in_proj(x)                          # (B, L, 2*d_inner)
        x_ssm, z = xz.chunk(2, dim=-1)                # each (B, L, d_inner)

        # Depthwise conv (causal)
        x_ssm = x_ssm.transpose(1, 2)                 # (B, d_inner, L)
        x_ssm = self.conv1d(x_ssm)[:, :, :L]          # causal: trim to L
        x_ssm = x_ssm.transpose(1, 2)                 # (B, L, d_inner)
        x_ssm = F.silu(x_ssm)

        # Input-dependent SSM parameters (selective mechanism)
        x_dbl = self.x_proj(x_ssm)                    # (B, L, dt_rank + 2*N)
        dt, B_param, C_param = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))              # (B, L, d_inner), positive

        # Discretize: Ā = exp(Δ·A), B̄ = Δ·B
        A = -torch.exp(self.A_log)                     # (d_inner, N), negative
        y = self._selective_scan(x_ssm, dt, A, B_param, C_param)

        # Gate with SiLU and project
        y = y * F.silu(z)
        y = self.out_proj(y)
        return y

    def _selective_scan(self, x, dt, A, B, C):
        """Sequential selective scan (reference implementation).

        For production, replace with CUDA parallel scan (mamba_ssm package).

        Args:
            x:  (B, L, d_inner) — input after conv+silu
            dt: (B, L, d_inner) — discretization timestep (positive)
            A:  (d_inner, N)    — state matrix (negative)
            B:  (B, L, N)       — input-dependent B
            C:  (B, L, N)       — input-dependent C
        Returns:
            y:  (B, L, d_inner)
        """
        batch, L, d_inner = x.shape
        N = A.shape[1]

        # Discretize per-step: Ā_k = exp(dt_k · A)
        # dt: (B, L, d_inner) → (B, L, d_inner, 1)
        # A:  (d_inner, N)     → (1, 1, d_inner, N)
        dtA = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)  # (B, L, d_inner, N)
        A_bar = torch.exp(dtA)                                 # Ā

        # B̄ = dt · B → (B, L, d_inner, N)
        dB = dt.unsqueeze(-1) * B.unsqueeze(2)                # (B, L, d_inner, N)

        # Sequential scan
        h = torch.zeros(batch, d_inner, N, device=x.device, dtype=x.dtype)
        ys = []
        for k in range(L):
            h = A_bar[:, k] * h + dB[:, k] * x[:, k].unsqueeze(-1)
            y_k = (h * C[:, k].unsqueeze(1)).sum(dim=-1)      # (B, d_inner)
            ys.append(y_k)

        y = torch.stack(ys, dim=1)                             # (B, L, d_inner)

        # Skip connection
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        return y


class MambaBlock(nn.Module):
    """Single Mamba block with pre-norm residual connection.

    Structure: x → LayerNorm → SelectiveSSM → Dropout → + x
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, drop_path=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        return x + self.drop_path(self.ssm(self.norm(x)))


class DropPath(nn.Module):
    """Stochastic Depth (SD): randomly drops entire residual branches.

    During training, each sample has probability `drop_prob` of being zeroed.
    Scales surviving samples by 1/(1-p) to maintain expected value.
    """

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device, dtype=x.dtype))
        return x * mask / keep
