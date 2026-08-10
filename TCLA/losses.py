import torch


def latent_dynamics_regularizer(z, cfg):
    """Weighted latent amplitude and second-order curvature penalty."""
    l2_beta = cfg.training.get("latent_l2_beta", 0.0)
    curvature_beta = cfg.training.get("latent_curvature_beta", 0.0)

    l2_loss = torch.mean(z**2)
    z_curv = z[:, :, 2:] - 2.0 * z[:, :, 1:-1] + z[:, :, :-2]
    curvature_loss = torch.mean(z_curv**2)

    return l2_beta * l2_loss + curvature_beta * curvature_loss
