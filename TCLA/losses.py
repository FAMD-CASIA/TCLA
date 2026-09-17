import torch
import torch.nn.functional as F


def latent_dynamics_regularizer(z, cfg):
    """Weighted latent amplitude penalty."""
    l2_beta = cfg.training.get("latent_l2_beta", 0.0)

    l2_loss = torch.mean(z**2)

    return l2_beta * l2_loss


def behavior_curvature_regularizer(behavior_pred, behavior_target):
    """Second-order loss on predicted behavior curvature."""
    if behavior_pred.shape[-1] < 3:
        return torch.tensor(0.0, device=behavior_pred.device, dtype=behavior_pred.dtype)

    pred_curv = behavior_pred[:, :, 2:] - 2.0 * behavior_pred[:, :, 1:-1] + behavior_pred[:, :, :-2]
    target_curv = behavior_target[:, :, 2:] - 2.0 * behavior_target[:, :, 1:-1] + behavior_target[:, :, :-2]
    return F.mse_loss(pred_curv, target_curv, reduction="mean")


def mmd_multikernel(source, target, device, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """Multi-kernel MMD between two latent sample matrices."""
    if source is None or target is None or source.numel() == 0 or target.numel() == 0:
        return torch.tensor(0.0, device=device)

    n_s = source.size(0)
    n_t = target.size(0)
    n_samples = n_s + n_t
    if n_samples <= 2:
        return torch.tensor(0.0, device=device)

    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(n_samples, n_samples, total.size(1))
    total1 = total.unsqueeze(1).expand(n_samples, n_samples, total.size(1))
    l2_distance_sq = ((total0 - total1) ** 2).sum(2)

    if fix_sigma is not None:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(l2_distance_sq.detach()) / (n_samples**2 - n_samples)

    bandwidth = torch.clamp(bandwidth, min=1e-6)
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernels = sum(torch.exp(-l2_distance_sq / torch.clamp(bw, min=1e-6)) for bw in bandwidth_list)

    xx = kernels[:n_s, :n_s].mean()
    yy = kernels[n_s:, n_s:].mean()
    xy = kernels[:n_s, n_s:].mean()
    return xx + yy - 2 * xy


def maybe_subsample_rows(x, max_rows):
    if max_rows is None or x.size(0) <= max_rows:
        return x
    idx = torch.randperm(x.size(0), device=x.device)[:max_rows]
    return x[idx]


def get_condition_latents(z, labels, condition_ids, max_samples_per_condition=None):
    out = {}
    labels = labels.to(z.device)
    for condition in condition_ids:
        mask = labels == condition
        if mask.sum() == 0:
            out[int(condition.item())] = None
            continue
        z_condition = z[mask]
        z_condition = z_condition.permute(0, 2, 1).contiguous().view(-1, z_condition.size(1))
        z_condition = maybe_subsample_rows(z_condition, max_samples_per_condition)
        out[int(condition.item())] = z_condition
    return out


def conditional_mmd_loss(
    z_source,
    labels_source,
    z_target,
    labels_target,
    device,
    kernel_mul=2.0,
    kernel_num=5,
    max_samples_per_condition=None,
):
    """Conditional MMD summed over labels shared by source and target batches."""
    labels_source = labels_source.to(device)
    labels_target = labels_target.to(device)
    condition_ids = torch.unique(torch.cat([labels_source, labels_target], dim=0))

    source_latents = get_condition_latents(
        z_source,
        labels_source,
        condition_ids,
        max_samples_per_condition=max_samples_per_condition,
    )
    target_latents = get_condition_latents(
        z_target,
        labels_target,
        condition_ids,
        max_samples_per_condition=max_samples_per_condition,
    )

    loss = torch.tensor(0.0, device=device)
    for condition in condition_ids:
        key = int(condition.item())
        z_s_d = source_latents.get(key)
        z_t_d = target_latents.get(key)
        if z_s_d is None or z_t_d is None:
            continue
        loss = loss + mmd_multikernel(
            z_s_d,
            z_t_d,
            device=device,
            kernel_mul=kernel_mul,
            kernel_num=kernel_num,
        )

    return loss
