import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir(os.path.dirname(os.getcwd()))
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from omegaconf import OmegaConf
import yaml
import accelerate
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from einops import rearrange

# 导入我们的项目模块
from ldns.data.data_multisession import get_session_data_and_configs, get_left_out_session_dataloader, SingleSessionDataset
from ldns.networks import CountWrapper, MultiSessionAutoEncoder

# ==============================================================================
#                      --- 1. 全局配置 ---
# ==============================================================================
LEFT_OUT_SESSION = 'session_0'
SOURCE_SESSION = 'session_4'

cfg_yaml = """
ae_model:
  C_shared: 30
  C: 256
  C_latent: 16
  num_blocks: 4
  num_blocks_decoder: 0
  num_lin_per_mlp: 2
  bidirectional: False

dataset:
  all_sessions: ['session_0', 'session_4']
  datapath_root: /data0/user/cyzhao/LD-AE_Multi_cross_n_fold_with_label/data
  signal_length: 320
  num_folds: 5

# ----------------- 微调训练参数 -----------------
alignment:
  lr: 0.01
  num_epochs: 1000
  num_warmup_epochs: 10
  batch_size: 64
  grad_clip: 1.0
  random_seed: 42
  precision: "no"
  mask_prob: 0.5
  eta_mmd_q1: 10
  eta_mmd_q2: 10
  eta_mmd_q3: 10
  eta_mmd_q4: 10
  kernel_mul: 2.0
  kernel_num: 5

# ----------------- 实验和日志参数 -----------------
exp:
  pretrain_ae_dir: /data0/user/cyzhao/LD-AE_Multi_cross_n_fold_with_label/exp/AE_model/model_train_on_4_left_0/model/full_ae_model_300.pt
  save_dir: /data0/user/cyzhao/LD-AE_Multi_cross_n_fold/exp/alignment/model_session_4_and_0/MK_MMD/model/
"""
config = OmegaConf.create(yaml.safe_load(cfg_yaml))

# ==============================================================================
#                     --- 2. 多核 MMD 损失函数 ---
# ==============================================================================
# code from https://github.com/jxygithub123/TDMNN/blob/main/untitled_deap.py

# def mmd_multikernel(source, target, device, kernel_mul=2.0, kernel_num=5, fix_sigma=None): # 32 9.1%  64 36.2% 128 爆炸
#     if source is None or target is None or len(source) == 0 or len(target) == 0:
#         return torch.tensor(0.0, device=device)

#     n_s = source.size(0)
#     n_t = target.size(0)
#     n_samples = n_s + n_t
    
#     total = torch.cat([source, target], dim=0)
    
#     total0 = total.unsqueeze(0).expand(n_samples, n_samples, total.size(1)) # [4160, 4160, 16]
#     total1 = total.unsqueeze(1).expand(n_samples, n_samples, total.size(1)) # [4160, 4160]
    
#     L2_distance_sq = ((total0 - total1)**2).sum(2) # [4160, 4160]
    
#     if fix_sigma:
#         bandwidth = fix_sigma
#     else:
#         bandwidth = torch.sum(L2_distance_sq.data) / (n_samples**2 - n_samples)
    
#     bandwidth /= kernel_mul ** (kernel_num // 2)
#     bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    
#     kernel_val = [torch.exp(-L2_distance_sq / band) for band in bandwidth_list]
#     kernels = sum(kernel_val) # [4160, 4160]
#     import pdb; pdb.set_trace()
#     XX = kernels[:n_s, :n_s].mean()
#     YY = kernels[n_s:, n_s:].mean()
#     XY = kernels[:n_s, n_s:].mean()
    
#     loss = XX + YY - 2 * XY
#     return loss

def mmd_multikernel(source, target, device, kernel_mul=2.0, kernel_num=5, fix_sigma=None): # 32 5.9%  64 15%  128 49.9%
    """
    优化了显存占用。
    使用展开公式 ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y> 避免生成 (N, N, D) 的中间张量。
    """
    if source is None or target is None or len(source) == 0 or len(target) == 0:
        return torch.tensor(0.0, device=device)

    n_s = source.size(0)
    n_t = target.size(0)
    n_samples = n_s + n_t
    
    # [N, D]
    total = torch.cat([source, target], dim=0) # [4160, 16]
    
    # --- 显存优化核心部分开始 ---
    # 1. 计算 squared norms: ||x||^2
    # total_sq: [N, 1]
    total_sq = total.pow(2).sum(dim=1, keepdim=True) # [4160, 1]
    
    # 2. 计算 dot product: <x, y>
    # xy: [N, N]
    xy = torch.mm(total, total.t()) # [4160, 4160]
    
    # 3. 组合计算 L2 距离矩阵: ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
    # 利用广播机制: [N, 1] + [1, N] - [N, N] -> [N, N]
    # 这里的内存占用仅为 [N, N]，而不是之前的 [N, N, D]
    L2_distance_sq = total_sq + total_sq.t() - 2 * xy # [4160, 4160]
    
    # 4. 数值稳定性处理 (防止浮点误差导致负值)
    L2_distance_sq = torch.clamp(L2_distance_sq, min=0.0)
    # --- 显存优化核心部分结束 ---

    # 计算带宽 sigma
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        # 使用 detach() 避免梯度回传到 sigma 的计算中，增加稳定性
        if n_samples < 2:
            bandwidth = torch.tensor(1.0, device=device)
        else:
            bandwidth = torch.sum(L2_distance_sq.detach()) / (n_samples**2 - n_samples)
    
    # 防止带宽过小
    bandwidth = torch.maximum(bandwidth, torch.tensor(1e-9, device=device))

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    
    # 计算高斯核
    kernel_val = [torch.exp(-L2_distance_sq / band) for band in bandwidth_list]
    kernels = sum(kernel_val) # [N, N]
    
    # 计算 MMD: E[K(x,x)] + E[K(y,y)] - 2E[K(x,y)]
    XX = kernels[:n_s, :n_s].mean()
    YY = kernels[n_s:, n_s:].mean()
    XY = kernels[:n_s, n_s:].mean()
    # import pdb; pdb.set_trace()
    loss = XX + YY - 2 * XY
    return loss
    
# ==============================================================================
#                            --- 3. 主微调函数 ---
# ==============================================================================
def alignment_unsupervised_mmd_batch(cfg, target_session_id, source_session_id):
    accelerator = accelerate.Accelerator(mixed_precision=cfg.alignment.precision)
    device = accelerator.device
    
    # --- Step 1: 准备 Source Session 的模型和无限数据加载器 ---
    torch.manual_seed(cfg.alignment.random_seed)
    np.random.seed(cfg.alignment.random_seed)

    _, source_configs = get_session_data_and_configs([source_session_id], cfg.dataset.datapath_root)
    shared_ae_config = {
        'C_in': cfg.ae_model.C_shared, 'C': cfg.ae_model.C, 'C_latent': cfg.ae_model.C_latent, 
        'L': cfg.dataset.signal_length, 'num_blocks': cfg.ae_model.num_blocks, 
        'bidirectional': cfg.ae_model.bidirectional, 'num_blocks_decoder': cfg.ae_model.num_blocks_decoder, 
        'num_lin_per_mlp': cfg.ae_model.num_lin_per_mlp
    }
    source_ae = CountWrapper(MultiSessionAutoEncoder({source_session_id: source_configs[source_session_id]}, shared_ae_config))
    source_ae.load_state_dict(torch.load(cfg.exp.pretrain_ae_dir, map_location='cpu'))
    source_ae.to(device).eval()
    
    source_data, _ = get_session_data_and_configs([source_session_id], cfg.dataset.datapath_root)
    source_dataset = SingleSessionDataset(
        spikes=source_data[source_session_id]['spikes'],
        behavior=source_data[source_session_id]['behavior'],
        session_id=source_session_id
    )
    source_loader = DataLoader(source_dataset, batch_size=cfg.alignment.batch_size, shuffle=True, drop_last=True)
    def infinite_loader(dataloader):
        while True:
            for data in dataloader: yield data
    source_loader_infinite = infinite_loader(source_loader)
    
    shared_ae_state_dict = {k.replace("ae_net.shared_ae.", ""): v for k, v in source_ae.state_dict().items() if k.startswith("ae_net.shared_ae.")}
    
    # --- Step 2: K-Fold 循环 ---
    target_data, target_configs = get_session_data_and_configs([target_session_id], cfg.dataset.datapath_root)
    for fold_k in range(cfg.dataset.num_folds):
        print(f"\n{'#'*20} FINETUNING WITH MMD (QUADRANT-BATCH) FOR FOLD {fold_k + 1}/{cfg.dataset.num_folds} {'#'*20}")
        
        target_train_loader, _ = get_left_out_session_dataloader(
            target_session_id, target_data, cfg.dataset.num_folds, fold_k, 
            cfg.alignment.batch_size, 4, cfg.alignment.random_seed
        )

        target_ae = CountWrapper(MultiSessionAutoEncoder({target_session_id: target_configs[target_session_id]}, shared_ae_config))
        target_ae.ae_net.shared_ae.load_state_dict(shared_ae_state_dict)
        target_ae.ae_net.freeze_shared_ae()

        alignment_params = target_ae.ae_net.get_alignment_parameters(target_session_id)
        optimizer = torch.optim.AdamW(alignment_params, lr=cfg.alignment.lr)
        criterion_poisson = nn.PoissonNLLLoss(log_input=False, reduction="mean")
        lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=len(target_train_loader) * cfg.alignment.num_warmup_epochs, num_training_steps=len(target_train_loader) * cfg.alignment.num_epochs)

        target_ae, optimizer, target_train_loader, lr_scheduler, prepared_source_loader, source_ae = accelerator.prepare(
            target_ae, optimizer, target_train_loader, lr_scheduler, source_loader_infinite, source_ae
        )

        fold_save_dir = os.path.join(cfg.exp.save_dir, f"fold_{fold_k}")
        os.makedirs(fold_save_dir, exist_ok=True)

        pbar = tqdm(range(cfg.alignment.num_epochs), disable=not accelerator.is_local_main_process, desc=f"Finetuning Fold {fold_k+1}")
        for epoch in pbar:
            target_ae.train()
            for target_batch in target_train_loader:
                source_batch = next(prepared_source_loader)
                
                with accelerator.accumulate(target_ae):
                    optimizer.zero_grad()
                    
                    # Target Path (Trainable)
                    input_signal_tgt = target_batch['signal'] * (torch.rand_like(target_batch['signal']) > cfg.alignment.mask_prob).float() / (1 - cfg.alignment.mask_prob + 1e-8)
                    _, z_target = target_ae(input_signal_tgt, target_session_id)
                    recon_loss = criterion_poisson(target_ae.decode(z_target, target_session_id), target_batch['signal'])
                    
                    # Source Path (Frozen, for MMD target)
                    with torch.no_grad():
                        _, z_source = source_ae(source_batch['signal'].to(device), source_session_id)
                    
                    # --- 按象限计算 MMD 损失 ---
                    def get_quadrant_latents(z, behavior):
                        endpoints = behavior[:, :, -1]
                        masks = {
                            'q1': (endpoints[:, 0] >= 0) & (endpoints[:, 1] >= 0),
                            'q2': (endpoints[:, 0] <  0) & (endpoints[:, 1] >= 0),
                            'q3': (endpoints[:, 0] <  0) & (endpoints[:, 1] <  0),
                            'q4': (endpoints[:, 0] >= 0) & (endpoints[:, 1] <  0),
                        }
                        
                        quadrant_latents = {}
                        for q_name, mask in masks.items():
                            if mask.sum() > 0:
                                # 将 trial 级别的 latents [B_q, C, L] 展平为 [B_q*L, C]
                                quad_z = z[mask]
                                quadrant_latents[q_name] = rearrange(quad_z, 'b c l -> (b l) c')
                            else:
                                quadrant_latents[q_name] = None
                        return quadrant_latents

                    target_latents_by_quad = get_quadrant_latents(z_target, target_batch['behavior'])
                    source_latents_by_quad = get_quadrant_latents(z_source, source_batch['behavior'])
                    
                    mmd_losses = {}
                    total_mmd_loss = torch.tensor(0.0, device=device)
                    
                    for q_name in ['q1', 'q2', 'q3', 'q4']:
                        z_tgt_q = target_latents_by_quad.get(q_name)
                        z_src_q = source_latents_by_quad.get(q_name)
                        
                        # 只有当两个batch都包含该象限的数据时，才计算loss
                        loss_val = mmd_multikernel(
                            z_src_q, z_tgt_q,
                            device=device,
                            kernel_mul=cfg.alignment.kernel_mul,
                            kernel_num=cfg.alignment.kernel_num
                        )
                        mmd_losses[q_name] = loss_val
                        
                        weight_name = f'eta_mmd_{q_name}'
                        total_mmd_loss += cfg.alignment.get(weight_name, 0.0) * loss_val

                    # 组合损失
                    total_loss = recon_loss + total_mmd_loss
                    
                    accelerator.backward(total_loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(alignment_params, cfg.alignment.grad_clip)
                    optimizer.step(); lr_scheduler.step()
                
                if accelerator.is_main_process:
                    pbar.set_postfix({
                        "Recon": recon_loss.item(), 
                        "MMD_Q1": mmd_losses.get('q1', torch.tensor(0)).item(), "MMD_Q2": mmd_losses.get('q2', torch.tensor(0)).item(),
                        "MMD_Q3": mmd_losses.get('q3', torch.tensor(0)).item(), "MMD_Q4": mmd_losses.get('q4', torch.tensor(0)).item(),
                    })
        
            # if accelerator.is_local_main_process and (epoch + 1) % 10 == 0:
            #     unwrapped_model = accelerator.unwrap_model(target_ae)
            #     save_path = os.path.join(fold_save_dir, f"alignmentd_ae_model_{epoch + 1}.pt")
            #     torch.save(unwrapped_model.state_dict(), save_path)

if __name__ == '__main__':
    alignment_unsupervised_mmd_batch(config, LEFT_OUT_SESSION, SOURCE_SESSION)