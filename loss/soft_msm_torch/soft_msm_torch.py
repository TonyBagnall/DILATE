# soft_msm/torch/_soft_msm_torch.py
from __future__ import annotations

import torch
from torch import nn


# -------------------------- helpers: soft minima --------------------------


def _softmin3(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Stable soft minimum of three tensors."""
    stack = torch.stack((-a / gamma, -b / gamma, -c / gamma), dim=0)
    return -gamma * torch.logsumexp(stack, dim=0)


def _softmin3_scalar(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Stable scalar softmin3 used in the inner DP recurrence."""
    s1 = -a / gamma
    s2 = -b / gamma
    s3 = -c / gamma
    m = torch.maximum(s1, torch.maximum(s2, s3))
    z = torch.exp(s1 - m) + torch.exp(s2 - m) + torch.exp(s3 - m)
    return -gamma * (torch.log(z) + m)


def _softmin2(a: torch.Tensor, b: torch.Tensor, gamma: float) -> torch.Tensor:
    """Stable soft minimum of two broadcast-compatible tensors.

    torch.stack does not broadcast, so this implementation is required for
    terms such as a.shape == (B, T-1, 1) and b.shape == (B, T-1, U).
    """
    s1 = -a / gamma
    s2 = -b / gamma
    m = torch.maximum(s1, s2)
    z = torch.exp(s1 - m) + torch.exp(s2 - m)
    return -gamma * (torch.log(z) + m)


# -------------------- parameter-free between-ness gate --------------------


def _between_gate(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Smooth gate g in [0, 1].

    a = x - y_prev, b = x - z_other.
    g is close to 1 when a*b < 0, meaning x lies between y_prev and z_other,
    and close to 0 when a*b > 0.
    """
    u = a * b
    eps_t = torch.as_tensor(eps, dtype=u.dtype, device=u.device)
    return 0.5 * (1.0 - u / torch.sqrt(u * u + eps_t))


# ----------------------- transition costs -----------------------


def _trans_cost(
    x_val: torch.Tensor,
    y_prev: torch.Tensor,
    z_other: torch.Tensor,
    c: float,
    gamma: float,
) -> torch.Tensor:
    """Soft-MSM split/merge transition cost."""
    a = x_val - y_prev
    b = x_val - z_other
    g = _between_gate(a, b)
    base = _softmin2(a * a, b * b, gamma)
    return c + (1.0 - g) * base


def _trans_cost_row_up(
    xi: torch.Tensor,
    xim1: torch.Tensor,
    y_slice: torch.Tensor,
    c: float,
    gamma: float,
) -> torch.Tensor:
    """Vectorised up-transition costs for a row.

    M_up[i, j] = trans_cost(x[i], x[i-1], y[j]).
    """
    a = xi - xim1
    b = xi - y_slice
    g = _between_gate(a, b)
    d_same = a * a
    d_cross = b * b

    s1 = -d_same / gamma
    s2 = -d_cross / gamma
    m = torch.maximum(s1, s2)
    z = torch.exp(s1 - m) + torch.exp(s2 - m)
    base = -gamma * (torch.log(z) + m)
    return c + (1.0 - g) * base


def _trans_cost_row_left(
    y_slice: torch.Tensor,
    y_prev_slice: torch.Tensor,
    xi: torch.Tensor,
    c: float,
    gamma: float,
) -> torch.Tensor:
    """Vectorised left-transition costs for a row.

    M_left[i, j] = trans_cost(y[j], y[j-1], x[i]).
    """
    a = y_slice - y_prev_slice
    b = y_slice - xi
    g = _between_gate(a, b)

    s1 = -(a * a) / gamma
    s2 = -(b * b) / gamma
    m = torch.maximum(s1, s2)
    z = torch.exp(s1 - m) + torch.exp(s2 - m)
    base = -gamma * (torch.log(z) + m)
    return c + (1.0 - g) * base


# -------------------- 1D Soft-MSM DP --------------------


def _soft_msm_torch_1d(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    c: float = 1.0,
    gamma: float = 1.0,
    window: int | None = None,
) -> torch.Tensor:
    """Compute differentiable Soft-MSM distance between two 1D series.

    Parameters
    ----------
    x, y : torch.Tensor, shape (T,) and (U,)
        Floating tensors on any PyTorch device.
    c : float, default=1.0
        MSM split/merge constant.
    gamma : float, default=1.0
        Soft-min smoothing parameter. Must be positive.
    window : int or None, default=None
        Optional Sakoe-Chiba half-width.

    Returns
    -------
    torch.Tensor
        Scalar differentiable Soft-MSM cost.
    """
    if gamma <= 0:
        raise ValueError("gamma must be > 0 for a differentiable soft minimum.")
    if x.dim() != 1 or y.dim() != 1:
        raise ValueError("x and y must be 1D tensors of shape (T,) and (U,).")
    if not x.is_floating_point() or not y.is_floating_point():
        raise ValueError("x and y must be floating dtype tensors.")

    device, dtype = x.device, x.dtype
    n, m = x.numel(), y.numel()

    C = torch.full((n, m), float("inf"), device=device, dtype=dtype)
    C[0, 0] = (x[0] - y[0]) ** 2

    def in_band(i: int, j: int) -> bool:
        return True if window is None else abs(i - j) <= window

    for i in range(1, n):
        if in_band(i, 0):
            C[i, 0] = C[i - 1, 0] + _trans_cost(x[i], x[i - 1], y[0], c, gamma)

    for j in range(1, m):
        if in_band(0, j):
            C[0, j] = C[0, j - 1] + _trans_cost(y[j], y[j - 1], x[0], c, gamma)

    for i in range(1, n):
        j_lo = 1 if window is None else max(1, i - window)
        j_hi = m - 1 if window is None else min(m - 1, i + window)
        if j_lo > j_hi:
            continue

        xi = x[i]
        xim1 = x[i - 1]
        y_cur = y[j_lo : j_hi + 1]
        y_prev = y[j_lo - 1 : j_hi]

        up_cost = _trans_cost_row_up(xi, xim1, y_cur, c, gamma)
        left_cost = _trans_cost_row_left(y_cur, y_prev, xi, c, gamma)
        match_cost = (xi - y_cur).pow(2)

        Cijm1 = C[i, j_lo - 1]
        for t in range(y_cur.numel()):
            j = j_lo + t
            d_diag = C[i - 1, j - 1] + match_cost[t]
            d_up = C[i - 1, j] + up_cost[t]
            d_left = Cijm1 + left_cost[t]
            Cij = _softmin3_scalar(d_diag, d_up, d_left, gamma)
            C[i, j] = Cij
            Cijm1 = Cij

    return C[n - 1, m - 1]


# -------------------- batched, channel-0 Soft-MSM --------------------


def _soft_msm_costs_batched(
    x: torch.Tensor,
    y: torch.Tensor,
    c: float,
    gamma: float,
    window: int | None = None,
) -> torch.Tensor:
    """Run exact Soft-MSM DP on channel 0 for batched data.

    x : (B, C, T)
    y : (B, C, U)
    """
    if x.dim() != 3 or y.dim() != 3:
        raise ValueError("x and y must be (B, C, T) and (B, C, U).")
    if x.shape[0] != y.shape[0] or x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have the same batch size and channel count.")
    if gamma <= 0:
        raise ValueError("gamma must be > 0.")

    B = x.shape[0]
    costs = torch.empty(B, dtype=x.dtype, device=x.device)
    for b in range(B):
        costs[b] = _soft_msm_torch_1d(
            x[b, 0],
            y[b, 0],
            c=c,
            gamma=gamma,
            window=window,
        )
    return costs


# -------------------- alignment from three move-cost tensors --------------------


def _soft_msm_costs_from_M3_batched(
    M_diag: torch.Tensor,
    M_up: torch.Tensor,
    M_left: torch.Tensor,
    gamma: float,
    window: int | None = None,
) -> torch.Tensor:
    """Soft-MSM DP using separate differentiable costs for each move type.

    Parameters
    ----------
    M_diag : torch.Tensor, shape (B, T, U)
        Cost added on diagonal moves into state (i, j).
    M_up : torch.Tensor, shape (B, T, U)
        Cost added on vertical/up moves into state (i, j). Row 0 is unused.
    M_left : torch.Tensor, shape (B, T, U)
        Cost added on horizontal/left moves into state (i, j). Column 0 is unused.
    gamma : float
        Soft-min smoothing parameter.
    window : int or None, default=None
        Optional Sakoe-Chiba half-width.

    Returns
    -------
    torch.Tensor, shape (B,)
        Soft-MSM costs.
    """
    if M_diag.dim() != 3 or M_up.dim() != 3 or M_left.dim() != 3:
        raise ValueError("M_diag, M_up and M_left must all be 3D tensors.")
    if M_diag.shape != M_up.shape or M_diag.shape != M_left.shape:
        raise ValueError("M_diag, M_up and M_left must have the same shape.")
    if gamma <= 0:
        raise ValueError("gamma must be > 0.")

    B, T, U = M_diag.shape
    costs = torch.empty(B, dtype=M_diag.dtype, device=M_diag.device)

    def in_band(i: int, j: int) -> bool:
        return True if window is None else abs(i - j) <= window

    for b in range(B):
        cm = torch.full((T, U), float("inf"), dtype=M_diag.dtype, device=M_diag.device)
        cm[0, 0] = M_diag[b, 0, 0]

        for i in range(1, T):
            if in_band(i, 0):
                cm[i, 0] = cm[i - 1, 0] + M_up[b, i, 0]

        for j in range(1, U):
            if in_band(0, j):
                cm[0, j] = cm[0, j - 1] + M_left[b, 0, j]

        for i in range(1, T):
            j_lo = 1 if window is None else max(1, i - window)
            j_hi = U - 1 if window is None else min(U - 1, i + window)
            if j_lo > j_hi:
                continue

            for j in range(j_lo, j_hi + 1):
                d_diag = cm[i - 1, j - 1] + M_diag[b, i, j]
                d_up = cm[i - 1, j] + M_up[b, i, j]
                d_left = cm[i, j - 1] + M_left[b, i, j]
                cm[i, j] = _softmin3_scalar(d_diag, d_up, d_left, gamma)

        costs[b] = cm[T - 1, U - 1]

    return costs


# -------------------- full move-cost construction --------------------


def _make_M3_costs(
    x: torch.Tensor,
    y: torch.Tensor,
    c: float,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct diagonal, up and left Soft-MSM move-cost tensors.

    x and y are channel-0 tensors with shapes (B, T) and (B, U).
    The returned tensors remain connected to x and y, so they are suitable for
    differentiable alignment penalties.
    """
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError("x and y must be 2D tensors of shape (B, T) and (B, U).")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same batch size.")

    B, T = x.shape
    U = y.shape[1]
    device, dtype = x.device, x.dtype

    # Diagonal move into (i, j): match x[i] with y[j].
    M_diag = (x[:, :, None] - y[:, None, :]).pow(2)

    # Up move into (i, j), i >= 1: split/merge x[i] against x[i-1] and y[j].
    a_up = x[:, 1:, None] - x[:, :-1, None]
    b_up = x[:, 1:, None] - y[:, None, :]
    g_up = _between_gate(a_up, b_up)
    base_up = _softmin2(a_up.pow(2), b_up.pow(2), gamma)
    M_up_body = c + (1.0 - g_up) * base_up
    M_up = torch.cat(
        [torch.zeros(B, 1, U, dtype=dtype, device=device), M_up_body],
        dim=1,
    )

    # Left move into (i, j), j >= 1: split/merge y[j] against y[j-1] and x[i].
    a_left = y[:, None, 1:] - y[:, None, :-1]
    b_left = y[:, None, 1:] - x[:, :, None]
    g_left = _between_gate(a_left, b_left)
    base_left = _softmin2(a_left.pow(2), b_left.pow(2), gamma)
    M_left_body = c + (1.0 - g_left) * base_left
    M_left = torch.cat(
        [torch.zeros(B, T, 1, dtype=dtype, device=device), M_left_body],
        dim=2,
    )

    return M_diag, M_up, M_left


# ------------------------------- public API --------------------------------


class SoftMSMLoss(nn.Module):
    """Compute Soft-MSM loss for batched tensors.

    Inputs are expected to have shape (B, C, T) and (B, C, U). The current
    implementation follows the existing Aeon-style univariate convention and
    uses channel 0.
    """

    def __init__(
        self,
        c: float = 1.0,
        gamma: float = 1.0,
        reduction: str = "mean",
        window: int | None = None,
    ):
        super().__init__()
        if gamma <= 0:
            raise ValueError("gamma must be > 0.")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be one of {'mean', 'sum', 'none'}.")
        self.c = float(c)
        self.gamma = float(gamma)
        self.reduction = reduction
        self.window = window

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        costs = _soft_msm_costs_batched(
            x,
            y,
            c=self.c,
            gamma=self.gamma,
            window=self.window,
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
    *,
    differentiable: bool = False,
    window: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return full Soft-MSM path occupancy and Soft-MSM cost.

    Parameters
    ----------
    x, y : torch.Tensor
        Tensors of shape (B, C, T) and (B, C, U). Channel 0 is used.
    c : float, default=1.0
        MSM split/merge constant.
    gamma : float, default=1.0
        Soft-min smoothing parameter.
    differentiable : bool, default=False
        If False, returns detached diagnostic values. If True, returns an
        alignment matrix that remains connected to x and y, so
        sum(A * omega) can backpropagate to model outputs. This requires a
        second-order graph and is substantially more memory-expensive.
    window : int or None, default=None
        Optional Sakoe-Chiba half-width.

    Returns
    -------
    A : torch.Tensor, shape (B, T, U)
        Full expected state-occupancy matrix, summed over diagonal, up and left
        move types.
    s : torch.Tensor, shape (B,)
        Soft-MSM costs.
    """
    if x.dim() != 3 or y.dim() != 3:
        raise ValueError("x and y must be (B, C, T) and (B, C, U).")
    if x.shape[0] != y.shape[0] or x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have the same batch size and channel count.")
    if gamma <= 0:
        raise ValueError("gamma must be > 0.")

    if differentiable:
        x_work = x
        y_work = y
        create_graph = True
        retain_graph = True
    else:
        # Diagnostic/evaluation path. Keeps the old stable behaviour, but does
        # not allow gradients through the returned alignment.
        x_work = x.detach()
        y_work = y.detach()
        create_graph = False
        retain_graph = False

    x0 = x_work[:, 0, :]
    y0 = y_work[:, 0, :]

    M_diag, M_up, M_left = _make_M3_costs(x0, y0, c=c, gamma=gamma)

    # Ensure autograd can take derivatives with respect to all three cost
    # tensors. These tensors remain connected to x/y in differentiable mode.
    M_diag.requires_grad_(True)
    M_up.requires_grad_(True)
    M_left.requires_grad_(True)

    s = _soft_msm_costs_from_M3_batched(
        M_diag,
        M_up,
        M_left,
        gamma=gamma,
        window=window,
    )

    E_diag, E_up, E_left = torch.autograd.grad(
        s.sum(),
        [M_diag, M_up, M_left],
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=False,
    )
    A_full = E_diag + E_up + E_left

    if differentiable:
        return A_full, s

    return A_full.detach().to(device=x.device, dtype=x.dtype), s.detach().to(
        device=x.device,
        dtype=x.dtype,
    )


@torch.no_grad()
def soft_msm_grad_x(
    x: torch.Tensor,
    y: torch.Tensor,
    c: float = 1.0,
    gamma: float = 1.0,
    window: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gradient of Soft-MSM cost with respect to x.

    This is intended for diagnostics/evaluation, not for training. It runs on
    the input device and dtype.

    Returns
    -------
    dx : torch.Tensor, shape (B, C, T)
        Gradient in x.dtype on x.device.
    s : torch.Tensor, shape (B,)
        Soft-MSM costs in x.dtype on x.device.
    """
    with torch.enable_grad():
        x_work = x.detach().clone().requires_grad_(True)
        y_work = y.detach()
        s = _soft_msm_costs_batched(
            x_work,
            y_work,
            c=c,
            gamma=gamma,
            window=window,
        )
        (dx,) = torch.autograd.grad(
            s.sum(),
            x_work,
            retain_graph=False,
            create_graph=False,
        )

    return dx.detach(), s.detach()
