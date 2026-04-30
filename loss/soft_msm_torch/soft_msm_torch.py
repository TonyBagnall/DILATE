# soft_msm/torch/_soft_msm_torch.py
from __future__ import annotations
import torch
from torch import nn

# -------------------------- helpers (softmins) --------------------------


def _softmin3(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, gamma: float
) -> torch.Tensor:
    """softmin(a,b,c) = -γ logsumexp([-a/γ, -b/γ, -c/γ])"""
    stack = torch.stack((-a / gamma, -b / gamma, -c / gamma), dim=0)
    return -gamma * torch.logsumexp(stack, dim=0)


def _softmin3_scalar(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, gamma: float
) -> torch.Tensor:
    """Scalar softmin3 with fewer ops than stack+logsumexp for inner loop."""
    s1 = -a / gamma
    s2 = -b / gamma
    s3 = -c / gamma
    m = torch.maximum(s1, torch.maximum(s2, s3))
    z = torch.exp(s1 - m) + torch.exp(s2 - m) + torch.exp(s3 - m)
    return -gamma * (torch.log(z) + m)


def _softmin2(a: torch.Tensor, b: torch.Tensor, gamma: float) -> torch.Tensor:
    stack = torch.stack((-a / gamma, -b / gamma), dim=0)
    return -gamma * torch.logsumexp(stack, dim=0)


def _softmin2_vec_scalar_first(
    t1_scalar: torch.Tensor, t2_vec: torch.Tensor, gamma: float
):
    """Vectorized softmin2 when first arg is scalar and second is vector."""
    s1 = -t1_scalar / gamma
    s2 = -t2_vec / gamma
    m = torch.maximum(s1, s2)
    z = torch.exp(s1 - m) + torch.exp(s2 - m)
    return -gamma * (torch.log(z) + m)


# -------------------- parameter-free between-ness gate --------------------


def _between_gate(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Smooth, parameter-free gate g in [0,1].
    a = x - y_prev, b = x - z_other. g≈1 when a*b<0 (between), g≈0 when a*b>0.
    """
    u = a * b
    eps_t = torch.as_tensor(eps, dtype=u.dtype, device=u.device)
    return 0.5 * (1.0 - u / torch.sqrt(u * u + eps_t))


# ----------------------- transition cost (no alpha) -----------------------


def _trans_cost(
    x_val: torch.Tensor,
    y_prev: torch.Tensor,
    z_other: torch.Tensor,
    c: float,
    gamma: float,
) -> torch.Tensor:
    a = x_val - y_prev
    b = x_val - z_other
    g = _between_gate(a, b)
    base = _softmin2(a * a, b * b, gamma)  # ≈ min((x-y)^2, (x-z)^2)
    return c + (1.0 - g) * base


def _trans_cost_row_up(xi, xim1, y_slice, c: float, gamma: float):
    # x_val=xi (scalar), y_prev=xim1 (scalar), z_other=yj (vector)
    a = xi - xim1  # scalar
    b = xi - y_slice  # vector
    g = _between_gate(a, b)
    d_same = a * a  # scalar
    d_cross = b * b  # vector
    base = _softmin2_vec_scalar_first(d_same, d_cross, gamma)  # vector
    return c + (1.0 - g) * base


def _trans_cost_row_left(y_slice, y_prev_slice, xi, c: float, gamma: float):
    # x_val=yj (vector), y_prev=y_{j-1} (vector), z_other=xi (scalar)
    a = y_slice - y_prev_slice  # vector
    b = y_slice - xi  # vector
    g = _between_gate(a, b)
    s1 = -(a * a) / gamma
    s2 = -(b * b) / gamma
    m = torch.maximum(s1, s2)
    z = torch.exp(s1 - m) + torch.exp(s2 - m)
    base = -gamma * (torch.log(z) + m)
    return c + (1.0 - g) * base


# -------------------- 1D core (your kernel) --------------------


def _soft_msm_torch_1d(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    c: float = 1.0,
    gamma: float = 1.0,  # > 0
    window = None,  # Sakoe–Chiba half-width
) -> torch.Tensor:
    """
    Differentiable soft-MSM distance between 1D series x (len n) and y (len m).
    Returns a scalar tensor suitable for .backward().
    """
    if gamma <= 0:
        raise ValueError("gamma must be > 0 for a differentiable soft minimum.")
    if x.dim() != 1 or y.dim() != 1:
        raise ValueError("x and y must be 1D tensors of shape (T,).")
    if not x.is_floating_point() or not y.is_floating_point():
        raise ValueError("x and y must be floating dtype tensors.")

    device, dtype = x.device, x.dtype
    n, m = x.numel(), y.numel()

    # DP table: C[0,0] uses match cost (x0 - y0)^2
    C = torch.full((n, m), float("inf"), device=device, dtype=dtype)
    C[0, 0] = (x[0] - y[0]) ** 2

    def in_band(i: int, j: int) -> bool:
        return True if window is None else (abs(i - j) <= window)

    # First column (vertical)
    for i in range(1, n):
        if in_band(i, 0):
            a = x[i] - x[i - 1]
            b = x[i] - y[0]
            g = _between_gate(a, b)
            s1 = -(a * a) / gamma
            s2 = -(b * b) / gamma
            mm = torch.maximum(s1, s2)
            z = torch.exp(s1 - mm) + torch.exp(s2 - mm)
            base = -gamma * (torch.log(z) + mm)
            trans = c + (1.0 - g) * base
            C[i, 0] = C[i - 1, 0] + trans

    # First row (horizontal)
    for j in range(1, m):
        if in_band(0, j):
            a = y[j] - y[j - 1]
            b = y[j] - x[0]
            g = _between_gate(a, b)
            s1 = -(a * a) / gamma
            s2 = -(b * b) / gamma
            mm = torch.maximum(s1, s2)
            z = torch.exp(s1 - mm) + torch.exp(s2 - mm)
            base = -gamma * (torch.log(z) + mm)
            trans = c + (1.0 - g) * base
            C[0, j] = C[0, j - 1] + trans

    # Main DP (row-wise vectorized costs + scalar recurrence)
    for i in range(1, n):
        j_lo = 1 if window is None else max(1, i - window)
        j_hi = m - 1 if window is None else min(m - 1, i + window)
        if j_lo > j_hi:
            continue

        xi, xim1 = x[i], x[i - 1]
        y_cur = y[j_lo : j_hi + 1]  # [L]
        y_prev = y[j_lo - 1 : j_hi]  # [L]

        up_cost = _trans_cost_row_up(xi, xim1, y_cur, c, gamma)  # [L]
        left_cost = _trans_cost_row_left(y_cur, y_prev, xi, c, gamma)  # [L]
        match = (xi - y_cur).pow(2)  # [L]

        Cijm1 = C[i, j_lo - 1]
        for t in range(y_cur.numel()):
            j = j_lo + t
            d_diag = C[i - 1, j - 1] + match[t]
            d_up = C[i - 1, j] + up_cost[t]
            d_left = Cijm1 + left_cost[t]
            Cij = _softmin3_scalar(d_diag, d_up, d_left, gamma)
            C[i, j] = Cij
            Cijm1 = Cij

    return C[n - 1, m - 1]


# -------------------- batched, multichannel DP --------------------


def _soft_msm_costs_batched(
    x: torch.Tensor,  # (B, C, T)
    y: torch.Tensor,  # (B, C, U)
    c: float,
    gamma: float,
) -> torch.Tensor:
    """
    Run exact DP independently per channel, sum end costs across channels.
    """
    if x.dim() != 3 or y.dim() != 3:
        raise ValueError("x and y must be (B, C, T)/(B, C, U)")
    if x.shape[0] != y.shape[0] or x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have same batch size and channels")

    B, C, _ = x.shape
    costs = torch.zeros(B, dtype=x.dtype, device=x.device)
    for ch in range(C):
        # per-batch loop to reuse your 1D kernel; T/U are small in tests so OK
        for b in range(B):
            costs[b] = costs[b] + _soft_msm_torch_1d(
                x[b, ch], y[b, ch], c=c, gamma=gamma
            )
    return costs  # (B,)


# -------------------- alignment (expected diag match occupancy) --------------------


def _soft_msm_costs_from_M_batched(
    M: torch.Tensor,  # (B, C, T, U) diagonal-match matrix (leaf)
    x: torch.Tensor,  # (B, C, T) detached (for transitions)
    y: torch.Tensor,  # (B, C, U) detached
    c: float,
    gamma: float,
) -> torch.Tensor:
    """
    Same DP but with a provided diagonal-match matrix M instead of (xi-yj)^2;
    used to obtain E = d s / d M via autograd, summed over channels.
    """
    if M.dim() != 4 or x.dim() != 3 or y.dim() != 3:
        raise ValueError("M must be (B,C,T,U), x=(B,C,T), y=(B,C,U)")
    B, C, T, U = M.shape
    costs = torch.zeros(B, dtype=M.dtype, device=M.device)

    for b in range(B):
        for ch in range(C):
            # full DP with M for matches
            cm = torch.full((T, U), float("inf"), dtype=M.dtype, device=M.device)
            cm[0, 0] = M[b, ch, 0, 0]
            # first column
            for i in range(1, T):
                trans_v = _trans_cost(
                    x[b, ch, i], x[b, ch, i - 1], y[b, ch, 0], c=c, gamma=gamma
                )
                cm[i, 0] = cm[i - 1, 0] + trans_v
            # first row
            for j in range(1, U):
                trans_h = _trans_cost(
                    y[b, ch, j], y[b, ch, j - 1], x[b, ch, 0], c=c, gamma=gamma
                )
                cm[0, j] = cm[0, j - 1] + trans_h
            # interior
            for i in range(1, T):
                xi = x[b, ch, i]
                xim1 = x[b, ch, i - 1]
                for j in range(1, U):
                    yj = y[b, ch, j]
                    yjm1 = y[b, ch, j - 1]
                    d1 = cm[i - 1, j - 1] + M[b, ch, i, j]
                    d2 = cm[i - 1, j] + _trans_cost(xi, xim1, yj, c=c, gamma=gamma)
                    d3 = cm[i, j - 1] + _trans_cost(yj, yjm1, xi, c=c, gamma=gamma)
                    cm[i, j] = _softmin3_scalar(d1, d2, d3, gamma)
            costs[b] = costs[b] + cm[T - 1, U - 1]
    return costs  # (B,)


def _device_supports_fp64(t: torch.Tensor) -> bool:
    return t.is_cpu or t.is_cuda  # MPS does not


# ------------------------------- public API --------------------------------


class SoftMSMLoss(nn.Module):
    """
    Soft-MSM loss (batched, multichannel), mirroring Aeon/Numba:
      - exact per-channel DP
      - sum channel end-costs
      - CUDA/CPU: float64-parity (if you feed float64)
      - MPS: value from CPU-float64 (two-step move), gradients from device graph
    """

    def __init__(self, c: float = 1.0, gamma: float = 1.0, reduction: str = "mean"):
        super().__init__()
        if gamma <= 0:
            raise ValueError("gamma must be > 0")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be one of {'mean','sum','none'}")
        self.c = float(c)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # device-native (keeps autograd)
        costs_dev = _soft_msm_costs_batched(
            x.to(x.dtype), y.to(y.dtype), c=self.c, gamma=self.gamma
        )

        if _device_supports_fp64(x):
            costs = costs_dev
        else:
            # MPS path: compute a high-precision reference on CPU float64
            with torch.no_grad():
                # two-step move avoids MPS fp64 conversion error
                x64 = x.detach().to("cpu").to(torch.float64)
                y64 = y.detach().to("cpu").to(torch.float64)
                costs_cpu64 = _soft_msm_costs_batched(
                    x64, y64, c=self.c, gamma=self.gamma
                )
            # value override, gradient from device graph
            costs = costs_cpu64.to(x.device, dtype=costs_dev.dtype) + (
                costs_dev - costs_dev.detach()
            )

        if self.reduction == "mean":
            return costs.mean()
        if self.reduction == "sum":
            return costs.sum()
        return costs


def soft_msm_alignment_matrix(
    x: torch.Tensor,
    y: torch.Tensor,
    c: float = 1.0,
    gamma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Expected diagonal-match occupancy E and Soft-MSM cost (detached).
      E : (B, T, U), summed over channels (matches Aeon)
      s : (B,) float64 (for equivalence)
    """
    # exact equivalence on CPU float64
    x64 = x.detach().to("cpu").to(torch.float64)
    y64 = y.detach().to("cpu").to(torch.float64)
    # Leaf: per-channel diagonal matches
    M = (x64.unsqueeze(-1) - y64.unsqueeze(-2)) ** 2  # (B, C, T, U)
    M.requires_grad_(True)

    s64 = _soft_msm_costs_from_M_batched(M, x64, y64, c=c, gamma=gamma)  # (B,)
    (E_per_channel,) = torch.autograd.grad(
        s64.sum(), M, retain_graph=False, create_graph=False
    )
    E64 = E_per_channel.sum(dim=1)  # (B, T, U)

    # Move back to caller device/dtype (values only; grads not needed)
    E = E64.to(x.device, dtype=x.dtype).detach()
    s = s64.to(x.device, dtype=torch.float64).detach()
    return E, s


@torch.no_grad()
def soft_msm_grad_x(
    x: torch.Tensor,
    y: torch.Tensor,
    c: float = 1.0,
    gamma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gradient of Soft-MSM cost w.r.t. x, computed on CPU float64 for equivalence.

    Returns
    -------
      dx : (B, C, T) in x.dtype on x.device
      s  : (B,) float64 on x.device
    """
    x64 = x.detach().to("cpu").to(torch.float64).clone().requires_grad_(True)
    y64 = y.detach().to("cpu").to(torch.float64)

    s64 = _soft_msm_costs_batched(x64, y64, c=c, gamma=gamma)  # (B,)
    (dx64,) = torch.autograd.grad(
        s64.sum(), x64, retain_graph=False, create_graph=False
    )

    dx = dx64.to(x.device, dtype=x.dtype)
    s = s64.to(x.device, dtype=torch.float64)
    return dx, s
