import torch
from diffusers import DDIMScheduler
# ✅ 直接使用你的底层拼接代码
from util.odi_processing import embed_patches_to_pano_gpu 

@torch.no_grad()
def sample_panorama(dit, vae, text_encoder, text_prompt, dataset_config, device='cuda'):
    B = 1
    N = dataset_config.u_num * dataset_config.v_num
    
    # 你的角度网格生成 (复用你 Dataset 中的角度)
    angles_deg = dataset_config.angles_deg.unsqueeze(0).to(device) 
    
    # 从纯噪声开始
    noisy_latents = torch.randn(B, N, 4, 16, 16, device=device)
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(50)
    
    text_feat = text_encoder([text_prompt])
    
    # DDIM 循环
    for t in scheduler.timesteps:
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        pred_noise = dit(noisy_latents, angles_deg, t_batch, text_feat)
        noisy_latents = scheduler.step(pred_noise, t, noisy_latents).prev_sample
        
    # ✨ 1. VAE 解码每个 Latent Patch 回到 Pixel Space (16x16 -> 128x128)
    noisy_latents_flat = noisy_latents.view(B * N, 4, 16, 16) / 0.18215
    pixel_patches_flat = vae.decode(noisy_latents_flat).sample # [N, 3, 128, 128]
    
    pixel_patches = (pixel_patches_flat / 2 + 0.5).clamp(0, 1) # 归一化到 [0, 1]
    
    # ✨ 2. 调用你的核心代码缝合全景图
    pano_canvas = embed_patches_to_pano_gpu(
        patches_tensor=pixel_patches, 
        pano_h=512, pano_w=1024,
        h_num=dataset_config.u_num, 
        v_num=dataset_config.v_num, 
        h_fov_deg=dataset_config.h_fov_deg, 
        v_fov_deg=dataset_config.v_fov_deg, 
        angles=angles_deg[0]
    )
    
    return pano_canvas