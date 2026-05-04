from loss.soft_msm_torch.soft_msm_torch import (
    SoftMSMLoss,
    soft_msm_alignment_matrix,
)
import torch


def soft_msm_loss(target, outputs, gamma, device):
    # DILATE tensors are (B, T, 1); SoftMSM expects (B, C, T)
    target_bct = target.transpose(1, 2).contiguous()
    outputs_bct = outputs.transpose(1, 2).contiguous()

    loss_shape = SoftMSMLoss(c=1.0, gamma=gamma, reduction="mean")(
        outputs_bct, target_bct
    )
    zero = torch.tensor(0.0, device=outputs.device, dtype=outputs.dtype)
    return loss_shape, loss_shape, zero


def soft_msm_dilate_loss(target, outputs, alpha, gamma, device=None, c=1.0):
    # DILATE tensors are (B, T, 1); SoftMSM expects (B, C, T)
    target_bct = target.transpose(1, 2).contiguous()
    outputs_bct = outputs.transpose(1, 2).contiguous()

    # Full Soft-MSM path occupancy, including diagonal, up and left moves.
    # differentiable=True is essential, otherwise the temporal term will not train.
    A, soft_msm_costs = soft_msm_alignment_matrix(
        outputs_bct,
        target_bct,
        c=c,
        gamma=gamma,
        differentiable=True,
    )

    # Shape loss: use the same Soft-MSM computation used to obtain A.
    # This avoids doing the Soft-MSM DP twice.
    loss_shape = soft_msm_costs.mean()

    # Temporal loss: DILATE-style temporal distortion over the Soft-MSM path occupancy.
    T = target.shape[1]
    idx = torch.arange(T, dtype=outputs.dtype, device=outputs.device)
    omega = (idx[:, None] - idx[None, :]).pow(2) / (T * T)

    loss_temporal = torch.sum(A * omega[None, :, :]) / A.shape[0]

    loss = alpha * loss_shape + (1.0 - alpha) * loss_temporal
    return loss, loss_shape, loss_temporal