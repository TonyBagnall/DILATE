from loss.soft_msm_torch.soft_msm_torch import (
    SoftMSMLoss,
    soft_msm_alignment_matrix,
)
import torch


def soft_msm_loss(target, outputs, gamma, device):
    # DILATE tensors are (B, T, 1); SoftMSM expects (B, C, T)
    target = target.transpose(1, 2).contiguous()
    outputs = outputs.transpose(1, 2).contiguous()

    loss_shape = SoftMSMLoss(c=1.0, gamma=gamma, reduction="mean")(outputs, target)
    zero = torch.tensor(0.0, device=device)
    return loss_shape, loss_shape, zero

def soft_msm_dilate_loss(target, outputs, alpha, gamma, device, c=1.0):
    target_bct = target.transpose(1, 2).contiguous()
    outputs_bct = outputs.transpose(1, 2).contiguous()

    loss_shape = SoftMSMLoss(c=c, gamma=gamma, reduction="mean")(
        outputs_bct, target_bct
    )

    A, _ = soft_msm_alignment_matrix(outputs_bct, target_bct, c=c, gamma=gamma)
    # A is (B, T, T)

    T = target.shape[1]
    idx = torch.arange(T, dtype=outputs.dtype, device=device)
    omega = (idx[:, None] - idx[None, :]) ** 2 / (T * T)

    loss_temporal = torch.sum(A * omega[None, :, :]) / A.shape[0]

    loss = alpha * loss_shape + (1.0 - alpha) * loss_temporal
    return loss, loss_shape, loss_temporal