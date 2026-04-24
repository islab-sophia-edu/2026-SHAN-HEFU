import os
import torch
from diffusers import AutoencoderKL
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

def precompute_latents(image_dir, output_dir, pano_h=1024, pano_w=2048,
                        patch_size=128, grid_height=8, device='cuda'):
    """
    预计算所有图片的VAE latent patches，保存到output_dir。
    训练时直接load，完全跳过VAE encode。
    """
    os.makedirs(output_dir, exist_ok=True)
    
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", torch_dtype=torch.bfloat16
    ).to(device)
    vae.eval()

    # 复用你现有的Dataset来生成patches
    from util.datasets_PanoMAE import PanoramicDataset
    dataset = PanoramicDataset(
        root_dir=image_dir, grid_height=grid_height,
        pano_h=pano_h, pano_w=pano_w,
        img_size=patch_size, is_training=False  # 关闭augmentation
    )
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, 
                                          num_workers=4, shuffle=False)
    
    latent_patch_size = patch_size // 8  # 128 → 16
    
    with torch.no_grad():
        for idx, (views, angles, weights) in enumerate(tqdm(loader)):
            fname = dataset.image_files[idx]
            save_path = os.path.join(output_dir, fname.replace('.jpg', '.pt').replace('.png', '.pt'))
            
            if os.path.exists(save_path):
                continue
            
            # views: [1, N, 3, H, W]
            B, N, C, H, W = views.shape
            views_flat = views.view(B*N, C, H, W).to(device, dtype=torch.bfloat16)
            
            # 分块encode防OOM
            latents_list = []
            chunk_size = 64
            for i in range(0, B*N, chunk_size):
                z = vae.encode(views_flat[i:i+chunk_size]).latent_dist.sample() * 0.18215
                latents_list.append(z.cpu())
            
            latent_flat = torch.cat(latents_list, dim=0)  # [N, 4, 16, 16]
            latent_patches = latent_flat.view(N, 4, latent_patch_size, latent_patch_size)
            
            torch.save({
                'latents': latent_patches.half(),  # fp16存储节省磁盘
                'angles': angles[0],               # [N, 2]
                'weights': weights[0],             # [N]
            }, save_path)
    
    print(f"Done. Latents saved to {output_dir}")

if __name__ == '__main__':
    precompute_latents(
        image_dir='/home/shanhefu/hybrid/train',
        output_dir='/home/shanhefu/hybrid/train_latents',
        grid_height=16
    )