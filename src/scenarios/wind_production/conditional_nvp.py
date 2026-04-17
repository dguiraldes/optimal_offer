"""Conditional Real-NVP normalizing flow for wind-power scenario generation.

References
----------
Dinh, Sohl-Dickstein & Bengio (2017). Density estimation using Real-NVP.
Dumas, Wehenkel, Dranka, Panciatici & Ernst (2022). A deep generative model
  for probabilistic energy forecasting in power systems: normalizing flows.
  Applied Energy, 305, 117871.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.scenarios.base import N_STEPS, ScenarioModel


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _ConditioningNet(nn.Module):
    """Encode the day-ahead forecast into a fixed-size context vector.

    Two hidden layers (128-wide) with a residual skip connection and GELU.
    Sized to avoid overfitting on ~400 training samples.
    """

    def __init__(self, input_dim: int, context_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, 128)
        self.block = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
        )
        self.out = nn.Linear(128, context_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.proj(x))
        h = h + self.block(h)          # residual
        return self.out(h)


class _ActNorm(nn.Module):
    """Activation normalisation with data-dependent initialisation."""

    def __init__(self, dim: int):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def _initialize(self, x: torch.Tensor) -> None:
        self.bias.data.copy_(-x.mean(0))
        self.log_scale.data.copy_(-torch.log(x.std(0) + 1e-6))
        self.initialized.fill_(True)

    def forward(self, x: torch.Tensor):
        if not self.initialized:
            self._initialize(x)
        y = (x + self.bias) * torch.exp(self.log_scale)
        log_det = self.log_scale.sum().expand(x.shape[0])
        return y, log_det

    def inverse(self, y: torch.Tensor):
        return y * torch.exp(-self.log_scale) - self.bias


class _CouplingLayer(nn.Module):
    """Affine coupling layer conditioned on a context vector.

    Uses a 3-layer residual MLP and soft-clamping of log-scale for more
    stable gradients than hard ``tanh`` clamping.
    """

    CLAMP = 2.0  # soft-clamp range for log_s

    def __init__(
        self,
        dim: int,
        context_dim: int,
        mask: torch.Tensor,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.register_buffer("mask", mask)
        n_fixed = int(mask.sum().item())
        n_transform = dim - n_fixed
        in_dim = n_fixed + context_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * n_transform),
        )
        # Zero-init last layer for near-identity initialisation
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _clamp(self, log_s: torch.Tensor) -> torch.Tensor:
        """Smooth clamping: ``CLAMP * tanh(log_s / CLAMP)``."""
        return self.CLAMP * torch.tanh(log_s / self.CLAMP)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor):
        x_fixed = x[:, self.mask.bool()]
        x_trans = x[:, ~self.mask.bool()]

        params = self.net(torch.cat([x_fixed, ctx], dim=-1))
        log_s, t = params.chunk(2, dim=-1)
        log_s = self._clamp(log_s)

        y_trans = x_trans * torch.exp(log_s) + t
        log_det = log_s.sum(dim=-1)

        y = torch.empty_like(x)
        y[:, self.mask.bool()] = x_fixed
        y[:, ~self.mask.bool()] = y_trans
        return y, log_det

    def inverse(self, y: torch.Tensor, ctx: torch.Tensor):
        y_fixed = y[:, self.mask.bool()]
        y_trans = y[:, ~self.mask.bool()]

        params = self.net(torch.cat([y_fixed, ctx], dim=-1))
        log_s, t = params.chunk(2, dim=-1)
        log_s = self._clamp(log_s)

        x_trans = (y_trans - t) * torch.exp(-log_s)

        x = torch.empty_like(y)
        x[:, self.mask.bool()] = y_fixed
        x[:, ~self.mask.bool()] = x_trans
        return x


def _make_masks(dim: int, n_layers: int) -> list[torch.Tensor]:
    """Create varied binary masks that respect temporal locality.

    Alternates between:
    - even / odd indices (interleaved)
    - first-half / second-half (block)
    This exposes every dimension to transformation from multiple
    complementary partitions without random permutations.
    """
    masks = []
    for i in range(n_layers):
        mask = torch.zeros(dim)
        pattern = i % 4
        if pattern == 0:
            mask[::2] = 1.0                      # even fixed
        elif pattern == 1:
            mask[1::2] = 1.0                     # odd fixed
        elif pattern == 2:
            mask[: dim // 2] = 1.0               # first half fixed
        else:
            mask[dim // 2 :] = 1.0               # second half fixed
        masks.append(mask)
    return masks


class _TemporalGaussian(nn.Module):
    r"""Multivariate Gaussian with exponential temporal correlation.

    The covariance is :math:`\Sigma_{ij} = \exp(-|i-j| / \ell)`,
    which encodes the prior that nearby timesteps are correlated.
    Using this as the base distribution means latent samples are
    already temporally smooth, so the flow only needs to learn
    residual corrections rather than the full correlation structure.
    """

    def __init__(self, dim: int, length_scale: float = 3.0):
        super().__init__()
        idx = torch.arange(dim, dtype=torch.float32)
        C = torch.exp(-torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1)) / length_scale)
        C += 1e-5 * torch.eye(dim)  # numerical stability

        L = torch.linalg.cholesky(C)
        C_inv = torch.cholesky_inverse(L)
        log_det = 2.0 * L.diagonal().log().sum()

        self.register_buffer("L", L)
        self.register_buffer("C_inv", C_inv)
        self.register_buffer("log_det", log_det)
        self.dim = dim
        self._const = dim * np.log(2 * np.pi)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Log-density of the correlated Gaussian."""
        # -0.5 * (z @ C_inv @ z^T  +  log|C|  +  d*log(2π))
        mahal = ((z @ self.C_inv) * z).sum(dim=-1)
        return -0.5 * (mahal + self.log_det + self._const)

    def sample(self, n: int, temperature: float = 1.0) -> torch.Tensor:
        """Draw *n* samples: z = L @ eps,  eps ~ N(0, I)."""
        eps = torch.randn(n, self.dim, device=self.L.device)
        return temperature * (eps @ self.L.T)


# ---------------------------------------------------------------------------
# Full flow
# ---------------------------------------------------------------------------


class ConditionalRealNVP(nn.Module):
    """Conditional Real-NVP normalizing flow for wind-power scenarios.

    Parameters
    ----------
    dim : int
        Dimensionality of each sample (default 96 = 24 h × 4 per hour).
    context_dim : int
        Size of the conditioning vector produced by the forecast encoder.
    n_layers : int
        Number of coupling + ActNorm blocks.
    hidden_dim : int
        Width of the coupling-layer MLPs.
    length_scale : float
        Temporal correlation length (in timesteps) for the base
        distribution.  Higher → smoother latent samples.  Set to 0 or
        ``None`` to fall back to an isotropic standard Gaussian.
    """

    def __init__(
        self,
        dim: int = N_STEPS,
        context_dim: int = 64,
        n_layers: int = 8,
        hidden_dim: int = 128,
        length_scale: float = 3.0,
    ):
        super().__init__()
        self.dim = dim
        self.conditioning = _ConditioningNet(dim, context_dim)

        # Base distribution
        if length_scale and length_scale > 0:
            self.base = _TemporalGaussian(dim, length_scale=length_scale)
        else:
            self.base = None  # fallback to isotropic N(0, I)

        masks = _make_masks(dim, n_layers)

        self.actnorms = nn.ModuleList()
        self.couplings = nn.ModuleList()
        for i in range(n_layers):
            self.actnorms.append(_ActNorm(dim))
            self.couplings.append(
                _CouplingLayer(dim, context_dim, masks[i], hidden_dim)
            )

    # ---- forward / inverse ------------------------------------------------

    def forward(self, x: torch.Tensor, forecast: torch.Tensor):
        """data → latent.  Returns (z, log_det_jacobian)."""
        ctx = self.conditioning(forecast)
        log_det = torch.zeros(x.shape[0], device=x.device)
        z = x
        for actnorm, coupling in zip(self.actnorms, self.couplings):
            z, ld = actnorm(z)
            log_det = log_det + ld
            z, ld = coupling(z, ctx)
            log_det = log_det + ld
        return z, log_det

    def inverse(self, z: torch.Tensor, forecast: torch.Tensor):
        """latent → data  (used for sampling)."""
        ctx = self.conditioning(forecast)
        x = z
        for actnorm, coupling in zip(
            reversed(list(self.actnorms)),
            reversed(list(self.couplings)),
        ):
            x = coupling.inverse(x, ctx)
            x = actnorm.inverse(x)
        return x

    # ---- probability & sampling -------------------------------------------

    def log_prob(self, x: torch.Tensor, forecast: torch.Tensor):
        z, log_det = self.forward(x, forecast)
        if self.base is not None:
            log_pz = self.base.log_prob(z)
        else:
            log_pz = -0.5 * (z**2 + np.log(2 * np.pi)).sum(dim=-1)
        return log_pz + log_det

    @torch.no_grad()
    def sample(
        self,
        forecast: torch.Tensor,
        n_samples: int = 1,
        seed: int | None = None,
        temperature: float = 1.0,
    ):
        """Generate samples conditioned on a single forecast profile."""
        if seed is not None:
            torch.manual_seed(seed)
        device = next(self.parameters()).device
        forecast = forecast.to(device)
        if forecast.dim() == 1:
            forecast = forecast.unsqueeze(0)
        forecast = forecast.expand(n_samples, -1)
        if self.base is not None:
            z = self.base.sample(n_samples, temperature=temperature)
        else:
            z = torch.randn(n_samples, self.dim, device=device) * temperature
        return self.inverse(z, forecast)

    # ---- training helper --------------------------------------------------

    @dataclass
    class TrainResult:
        """Container for training history."""
        train_losses: list[float]
        val_losses: list[float]
        best_epoch: int

    def fit(
        self,
        X_prod: np.ndarray,
        X_fcst: np.ndarray,
        *,
        X_prod_val: np.ndarray | None = None,
        X_fcst_val: np.ndarray | None = None,
        val_fraction: float = 0.15,
        epochs: int = 2000,
        batch_size: int = 32,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        grad_clip: float = 1.0,
        noise_std: float = 0.02,
        patience: int = 150,
        min_delta: float = 0.0,
        verbose: bool = True,
    ) -> "ConditionalRealNVP.TrainResult":
        """Train the flow via maximum likelihood with early stopping.

        Parameters
        ----------
        X_prod : (N, dim)  normalised production windows (train, or full).
        X_fcst : (N, dim)  normalised forecast windows   (train, or full).
        X_prod_val, X_fcst_val : optional explicit validation arrays.
            If not given, ``val_fraction`` of the training data is held out.
        val_fraction : float
            Fraction of *training* data used for validation when explicit
            val arrays are not provided.
        noise_std : float
            Std of additive Gaussian noise applied to production windows
            during training.  Acts as data augmentation / regularisation.
        patience : int
            Number of epochs without improvement before stopping.
        min_delta : float
            Minimum decrease in validation NLL to count as improvement.
        """
        device = next(self.parameters()).device

        # ------ train / val split ------------------------------------------
        if X_prod_val is not None and X_fcst_val is not None:
            prod_t = torch.tensor(X_prod, dtype=torch.float32, device=device)
            fcst_t = torch.tensor(X_fcst, dtype=torch.float32, device=device)
            prod_v = torch.tensor(X_prod_val, dtype=torch.float32, device=device)
            fcst_v = torch.tensor(X_fcst_val, dtype=torch.float32, device=device)
        else:
            full_prod = torch.tensor(X_prod, dtype=torch.float32, device=device)
            full_fcst = torch.tensor(X_fcst, dtype=torch.float32, device=device)
            n_total = full_prod.shape[0]
            n_val = max(1, int(n_total * val_fraction))
            # Shuffle before splitting so validation is not just the tail
            perm = torch.randperm(n_total)
            prod_t = full_prod[perm[n_val:]]
            fcst_t = full_fcst[perm[n_val:]]
            prod_v = full_prod[perm[:n_val]]
            fcst_v = full_fcst[perm[:n_val]]

        n_train = prod_t.shape[0]
        n_val = prod_v.shape[0]

        if verbose:
            print(f"Training on {n_train} samples, validating on {n_val} samples")

        # ------ optimiser --------------------------------------------------
        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=50, min_lr=1e-6,
        )

        train_losses: list[float] = []
        val_losses: list[float] = []
        best_val = float("inf")
        best_epoch = 0
        best_state = None
        wait = 0

        for epoch in range(1, epochs + 1):
            # ---- train step -----------------------------------------------
            self.train()
            idx = torch.randperm(n_train, device=device)
            epoch_loss = 0.0
            for start in range(0, n_train, batch_size):
                batch_idx = idx[start : start + batch_size]
                prod_batch = prod_t[batch_idx]
                # ---- noise augmentation -----------------------------------
                if noise_std > 0:
                    prod_batch = prod_batch + noise_std * torch.randn_like(prod_batch)
                    prod_batch = prod_batch.clamp(0.0, 1.0)
                nll = -self.log_prob(prod_batch, fcst_t[batch_idx]).mean()

                optimizer.zero_grad()
                nll.backward()
                nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                optimizer.step()

                epoch_loss += nll.item() * len(batch_idx)

            avg_train = epoch_loss / n_train
            train_losses.append(avg_train)

            # ---- validation step ------------------------------------------
            self.eval()
            with torch.no_grad():
                val_nll = -self.log_prob(prod_v, fcst_v).mean().item()
            val_losses.append(val_nll)

            scheduler.step(val_nll)

            # ---- early stopping -------------------------------------------
            if val_nll < best_val - min_delta:
                best_val = val_nll
                best_epoch = epoch
                best_state = {k: v.clone() for k, v in self.state_dict().items()}
                wait = 0
            else:
                wait += 1

            if verbose and epoch % 100 == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch:4d}/{epochs} | "
                    f"Train NLL {avg_train:.4f} | "
                    f"Val NLL {val_nll:.4f} | "
                    f"Best {best_val:.4f} (ep {best_epoch}) | "
                    f"LR {lr_now:.1e}"
                )

            if wait >= patience:
                if verbose:
                    print(
                        f"Early stopping at epoch {epoch} "
                        f"(best val NLL {best_val:.4f} at epoch {best_epoch})"
                    )
                break

        # Restore best weights
        if best_state is not None:
            self.load_state_dict(best_state)
            if verbose:
                print(f"Restored best model from epoch {best_epoch}")

        return self.TrainResult(
            train_losses=train_losses,
            val_losses=val_losses,
            best_epoch=best_epoch,
        )


# ---------------------------------------------------------------------------
# ScenarioModel wrapper (for the framework)
# ---------------------------------------------------------------------------


class NormalizingFlowModel(ScenarioModel):
    """Wraps a trained ConditionalRealNVP for the ScenarioGenerator API."""

    name = "conditional_nvp"

    def __init__(
        self,
        model_path: str | Path | None = None,
        flow: ConditionalRealNVP | None = None,
    ):
        if flow is not None:
            self._flow = flow
        elif model_path is not None:
            state = torch.load(
                model_path, map_location="cpu", weights_only=True
            )
            # Infer architecture from checkpoint weights
            context_dim = state["conditioning.out.bias"].shape[0]
            n_layers = sum(1 for k in state if k.startswith("actnorms.") and k.endswith(".bias"))
            hidden_dim = state["couplings.0.net.0.weight"].shape[0]
            self._flow = ConditionalRealNVP(
                context_dim=context_dim,
                n_layers=n_layers,
                hidden_dim=hidden_dim,
            )
            self._flow.load_state_dict(state)
        else:
            raise ValueError("Provide either model_path or flow")
        self._flow.eval()

    @property
    def required_inputs(self) -> list[str]:
        return ["wind_forecast"]

    def generate(
        self,
        n_scenarios: int,
        seed: int | None = None,
        **inputs,
    ) -> np.ndarray:
        forecast = inputs["wind_forecast"]
        if isinstance(forecast, np.ndarray):
            forecast = torch.tensor(forecast, dtype=torch.float32)
        samples = self._flow.sample(forecast, n_samples=n_scenarios, seed=seed)
        return samples.cpu().numpy().clip(0.0, 1.0)
