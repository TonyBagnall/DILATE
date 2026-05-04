# soft_msm/torch/_soft_msm_torch.py
from __future__ import annotations
import torch
from torch import nn

# -------------------------- helpers (softmins) --------------------------


def _softmin3(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, gamma: float
) -> torch.Tensor:
    """Compute softmin(a, b, c)."""
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
    """Compute a smooth, parameter-free gate g in [0, 1].

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
    """Compute differentiable soft-MSM distance between 1D series.

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
    """Run exact DP on channel 0 (matching Aeon's univariate MSM convention)."""
    if x.dim() != 3 or y.dim() != 3:
        raise ValueError("x and y must be (B, C, T)/(B, C, U)")
    if x.shape[0] != y.shape[0] or x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have same batch size and channels")

    B, C, _ = x.shape
    costs = torch.zeros(B, dtype=x.dtype, device=x.device)
    for b in range(B):
        costs[b] = costs[b] + _soft_msm_torch_1d(x[b, 0], y[b, 0], c=c, gamma=gamma)
    return costs  # (B,)


# -------------------- alignment (full path occupancy) --------------------


def _soft_msm_costs_from_M3_batched(
    M_diag: torch.Tensor,  # (B, T, U) diagonal-match costs, leaf
    M_up:   torch.Tensor,  # (B, T, U) up-transition costs, leaf; row 0 unused
    M_left: torch.Tensor,  # (B, T, U) left-transition costs, leaf; col 0 unused
    gamma: float,
) -> torch.Tensor:
    """DP with all three move-type costs as separate differentiable leaves."""
    B, T, U = M_diag.shape
    costs = torch.zeros(B, dtype=M_diag.dtype, device=M_diag.device)

    for b in range(B):
        cm = torch.full((T, U), float("inf"), dtype=M_diag.dtype, device=M_diag.device)
        cm[0, 0] = M_diag[b, 0, 0]
        for i in range(1, T):
            cm[i, 0] = cm[i - 1, 0] + M_up[b, i, 0]
        for j in range(1, U):
            cm[0, j] = cm[0, j - 1] + M_left[b, 0, j]
        for i in range(1, T):
            for j in range(1, U):
                d1 = cm[i - 1, j - 1] + M_diag[b, i, j]
                d2 = cm[i - 1, j]     + M_up[b, i, j]
                d3 = cm[i,     j - 1] + M_left[b, i, j]
                cm[i, j] = _softmin3_scalar(d1, d2, d3, gamma)
        costs[b] = costs[b] + cm[T - 1, U - 1]
    return costs  # (B,)


def _device_supports_fp64(t: torch.Tensor) -> bool:
    return t.is_cpu or t.is_cuda  # MPS does not


# ------------------------------- public API --------------------------------


class SoftMSMLoss(nn.Module):
    """Compute Soft-MSM loss (batched, multichannel), mirroring Aeon/Numba.

    - exact channel-0 DP
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
    """Full path-occupancy matrix E and Soft-MSM cost.

    E[b, i, j] is the expected number of times DP state (i, j) is visited,
    summed over all three move types (diagonal, up, left).  This matches the
    semantics of DILATE's DTW alignment matrix and makes loss_temporal =
    sum(E * omega) a proper temporal-distortion penalty.

    E : (B, T, U), channel 0 only.
    s : (B,) float64.
    """
    x64 = x.detach().to("cpu").to(torch.float64)
    y64 = y.detach().to("cpu").to(torch.float64)
    x0 = x64[:, 0, :]  # (B, T)
    y0 = y64[:, 0, :]  # (B, U)
    B, T = x0.shape
    U = y0.shape[1]

    # --- Leaf 1: diagonal match costs (B, T, U) ---
    M_diag = (x0[:, :, None] - y0[:, None, :]) ** 2
    M_diag.requires_grad_(True)

    # --- Leaf 2: up-transition costs (B, T, U), row 0 padded with zeros ---
    # M_up[b, i, j] = trans_cost(x[i], x[i-1], y[j])  for i >= 1
    a_up = x0[:, 1:, None] - x0[:, :-1, None]         # (B, T-1, 1)
    b_up = x0[:, 1:, None] - y0[:, None, :]            # (B, T-1, U)
    g_up = _between_gate(a_up, b_up)
    s1_up = -(a_up * a_up) / gamma
    s2_up = -(b_up * b_up) / gamma
    mm_up = torch.maximum(s1_up, s2_up)
    base_up = -gamma * (torch.log(torch.exp(s1_up - mm_up) + torch.exp(s2_up - mm_up)) + mm_up)
    M_up = torch.cat([
        torch.zeros(B, 1, U, dtype=x64.dtype),
        c + (1.0 - g_up) * base_up,
    ], dim=1)  # (B, T, U)
    M_up.requires_grad_(True)

    # --- Leaf 3: left-transition costs (B, T, U), col 0 padded with zeros ---
    # M_left[b, i, j] = trans_cost(y[j], y[j-1], x[i])  for j >= 1
    a_left = y0[:, None, 1:] - y0[:, None, :-1]        # (B, 1, U-1)
    b_left = y0[:, None, 1:] - x0[:, :, None]          # (B, T, U-1)
    g_left = _between_gate(a_left, b_left)
    s1_left = -(a_left * a_left) / gamma
    s2_left = -(b_left * b_left) / gamma
    mm_left = torch.maximum(s1_left, s2_left)
    base_left = -gamma * (torch.log(torch.exp(s1_left - mm_left) + torch.exp(s2_left - mm_left)) + mm_left)
    M_left = torch.cat([
        torch.zeros(B, T, 1, dtype=x64.dtype),
        c + (1.0 - g_left) * base_left,
    ], dim=2)  # (B, T, U)
    M_left.requires_grad_(True)

    # --- DP through all three leaves ---
    s64 = _soft_msm_costs_from_M3_batched(M_diag, M_up, M_left, gamma=gamma)

    # --- Full occupancy: sum gradients over all three move types ---
    E_diag, E_up, E_left = torch.autograd.grad(
        s64.sum(), [M_diag, M_up, M_left], retain_graph=False, create_graph=False
    )
    A_full = E_diag + E_up + E_left  # (B, T, U)

    E = A_full.to(x.device, dtype=x.dtype).detach()
    s_dtype = torch.float64 if _device_supports_fp64(x) else x.dtype
    s = s64.to(x.device, dtype=s_dtype).detach()
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
      s  : (B,) float64 on CPU/CUDA, or x.dtype on devices without float64
    """
    with torch.enable_grad():
        x64 = x.detach().to("cpu").to(torch.float64).clone().requires_grad_(True)
        y64 = y.detach().to("cpu").to(torch.float64)

        s64 = _soft_msm_costs_batched(x64, y64, c=c, gamma=gamma)  # (B,)
        (dx64,) = torch.autograd.grad(
            s64.sum(), x64, retain_graph=False, create_graph=False
        )

    dx = dx64.to(x.device, dtype=x.dtype)
    s_dtype = torch.float64 if _device_supports_fp64(x) else x.dtype
    s = s64.to(x.device, dtype=s_dtype)
    return dx, s
