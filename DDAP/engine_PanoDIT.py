import math
import sys
import os
import torch
import torch.nn.functional as F
import torchvision
from diffusers import DDIMScheduler
import torchmetrics

import util.misc as misc
from util.odi_processing import embed_patches_to_pano_gpu

def train_one_epoch(model, vae, noise_scheduler, text_encoder, data_loader, 
                    optimizer, device, epoch, loss_scaler, lat_weights, args):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    header = f'Epoch: [{epoch}]'
    
    optimizer.zero_grad()
    
    for data_iter_step, (views, angles, weights) in enumerate(metric_logger.log_every(data_loader, 20, header)):
        # views: [B, N, 3, H, W], angles: [B, N, 2]
        views = views.to(device, non_blocking=True)
        angles = angles.to(device, non_blocking=True)
        B, N, C, H, W = views.shape
        
        with torch.no_grad():
            # 将输入转为 VAE 的对应精度 bfloat16
            views_flat = views.view(B * N, C, H, W).to(dtype=torch.bfloat16)
            
            # ✨ 手动分块 Encode (彻底杜绝 VAE OOM)
            chunk_size = 128  # 每次最多处理 128 个 Patch (可根据显存微调)
            latents_list = []
            for i in range(0, B * N, chunk_size):
                chunk = views_flat[i:i+chunk_size]
                z = vae.encode(chunk).latent_dist.sample() * 0.18215
                latents_list.append(z)
            
            latent_flat = torch.cat(latents_list, dim=0)
            
            # 重新变回序列格式 [B, N, 4, 16, 16]
            latent_patch_size = latent_flat.shape[-1]
            clean_latent_patches = latent_flat.view(B, N, 4, latent_patch_size, latent_patch_size)
            
            # 2. Text 特征 (假设全景图均用同一占位 Prompt，如果 Dataset 有真实 Prompt 请替换)
            text_feat = text_encoder(["A 360 degree panoramic view"] * B)
            
        # 3. 采样噪声与时间步
        noise = torch.randn_like(clean_latent_patches)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=device).long()
        noisy_patches = noise_scheduler.add_noise(clean_latent_patches, noise, timesteps)
        
        # 4. DiT 前向预测
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            pred_noise = model(noisy_patches, angles, timesteps, text_feat)
            
            # 使用提取出的纬度权重进行损失加权 (保持球面一致性)
            loss_per_patch = ((pred_noise - noise) ** 2).mean(dim=[2, 3, 4]) # [B, N]
            loss = (loss_per_patch * lat_weights.unsqueeze(0)).mean()
            
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss /= args.accum_iter
        loss_scaler(loss, optimizer, clip_grad=args.clip_grad, parameters=model.parameters(), update_grad=(data_iter_step + 1) % args.accum_iter == 0)
        
        if (data_iter_step + 1) % args.accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, vae, text_encoder, data_loader, device, epoch, args):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = f'Evaluate Epoch: [{epoch}]'
    
    val_psnr = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(device)
    val_ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    # 初始化 DDIM 加速采样器 (仅用于评估)
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(50) # 50 步足以评估生成质量
    
    for batch_idx, (views, angles, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        views = views.to(device, non_blocking=True)
        angles = angles.to(device, non_blocking=True)
        B, N, C, H, W = views.shape
        latent_patch_size = args.patch_size // 8

        # --- 1. CFG (Classifier-Free Guidance) 准备 ---
        text_feat = text_encoder(["A 360 degree panoramic view"] * B)
        null_feat = text_encoder([""] * B)
        guidance_scale = 4.0
        
        # --- 2. 纯噪声起点 ---
        noisy_latents = torch.randn(B, N, 4, latent_patch_size, latent_patch_size, device=device)
        
        # --- 3. DDIM 逆向去噪 ---
        for t in scheduler.timesteps:
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred_text = model(noisy_latents, angles, t_batch, text_feat)
                pred_null = model(noisy_latents, angles, t_batch, null_feat)
            
            # CFG 计算
            pred_noise = pred_null + guidance_scale * (pred_text - pred_null)
            noisy_latents = scheduler.step(pred_noise, t, noisy_latents).prev_sample
            
        # --- 4. VAE 解码回到像素空间 ---
        clean_latents_flat = noisy_latents.view(B * N, 4, latent_patch_size, latent_patch_size) / 0.18215
        clean_latents_flat = clean_latents_flat.to(dtype=torch.bfloat16) # 匹配 VAE 精度
        
        # ✨ 手动分块 Decode
        chunk_size = 128
        pred_views_list = []
        for i in range(0, B * N, chunk_size):
            chunk = clean_latents_flat[i:i+chunk_size]
            decoded_chunk = vae.decode(chunk).sample
            pred_views_list.append(decoded_chunk)
            
        pred_views_flat = torch.cat(pred_views_list, dim=0)
        
        # 转回 float32 继续后面的计算
        pred_views = pred_views_flat.view(B, N, 3, H, W).to(torch.float32)
        
        # 将生成图像规范化到 [0, 1] 范围
        pred_views = (pred_views / 2 + 0.5).clamp(0, 1)
        gt_views = (views / 2 + 0.5).clamp(0, 1) # 假设 GT dataloader 输出在 [-1, 1]
        
        # --- 5. 计算 PSNR 和 SSIM (在 Patch 级别计算) ---
        val_psnr.update(pred_views.view(B*N, 3, H, W), gt_views.view(B*N, 3, H, W))
        val_ssim.update(pred_views.view(B*N, 3, H, W), gt_views.view(B*N, 3, H, W))
        
        # --- 6. 生成可视化图像 (缝合为全景图) ---
        if batch_idx == 0 and misc.is_main_process():
            # 取 Batch 中的第一张图
            h_fov = 360.0 / (2 * args.grid_height)
            v_fov = 180.0 / args.grid_height
            
            # 使用你原有的底层函数缝合
            pano_rec = embed_patches_to_pano_gpu(
                pred_views[0], args.pano_h, args.pano_w, 
                2 * args.grid_height, args.grid_height, h_fov, v_fov, angles[0]
            )
            pano_gt = embed_patches_to_pano_gpu(
                gt_views[0], args.pano_h, args.pano_w, 
                2 * args.grid_height, args.grid_height, h_fov, v_fov, angles[0]
            )
            
            # 上下拼接：上面是 Ground Truth，下面是生成结果
            final_img = torch.cat([pano_gt, pano_rec], dim=1)
            save_path = os.path.join(args.output_dir, f'val_generation_epoch_{epoch}.png')
            torchvision.utils.save_image(final_img, save_path)

    psnr_value = val_psnr.compute().item()
    ssim_value = val_ssim.compute().item()
    metric_logger.update(psnr=psnr_value, ssim=ssim_value)
    
    print('Averaged validation stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}