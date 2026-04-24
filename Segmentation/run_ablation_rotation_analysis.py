import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# 导入你的组件
import util.odi_processing as odi 
from models_segmentation_2dcnn import vit_huge_patch14 
from evaluate_advanced_stability import ERPRotator, FeatureExtractor, calculate_weighted_equiv_error 

# ==========================================
# 1. 更新消融实验配置 (新增两个关键消融项)
# ==========================================
ABLATION_CONFIGS = {
    # 完整版本 (Dynamic Ratio 0.6-0.9 + Adaptive Masking)
    "Full_PanoMAE": {
        "ckpt": "/media/data_hdd1/shanhefu/outputs/finetune/segmentation_CVRG_Pano/sagementation_mask0.6-0.9_16*32_3.2e-3_decay_huge_2dcnn_16_CE_4_view/checkpoint-99.pth"
    },
    # 消融项 1: 使用动态比例 0.6-0.9，但去掉了自适应掩码 (No Adaptive Masking)
    "w/o_AdaptiveMasking": {
        "ckpt": "/media/data_hdd1/shanhefu/outputs/finetune/segmentation_CVRG_Pano/sagementation_mask0.6-0.9_16*32_3.2e-3_decay_huge_cnn_16_CE_4_noadaptivemasking/checkpoint-80.pth"
    },
    # 消融项 2: 使用固定掩码比例 0.75 (Fixed Ratio 0.75)
    "Fixed_Ratio_0.75": {
        "ckpt": "/media/data_hdd1/shanhefu/outputs/finetune/segmentation_CVRG_Pano/sagementation_mask0.75_16*32_3.2e-3_decay_huge_cnn_16_CE_4/checkpoint-99.pth"
    }
}

# ==========================================
# 2. 自动化消融运行逻辑
# ==========================================
@torch.no_grad()
def run_b3_ablation(img_path, device, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    input_size = (832, 1664)
    u_steps, v_steps = 32, 16
    h_fov, v_fov = 360.0/u_steps, 180.0/v_steps
    target_patch_size = 52 
    
    rotator = ERPRotator(h=832, w=1664, device=device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()
    raw_img = np.array(Image.open(img_path).convert('RGB').resize((input_size[1], input_size[0]), Image.BICUBIC))

    # 预计算角度中心
    v_centers = torch.linspace(90 - v_fov/2, -90 + v_fov/2, v_steps)
    u_centers = torch.linspace(-180, 180, u_steps + 1)[:-1]
    v_grid, u_grid = torch.meshgrid(v_centers, u_centers, indexing='ij')
    angle_centers = torch.stack([u_grid.flatten(), v_grid.flatten()], dim=1).unsqueeze(0).to(device)

    step_angle = 360.0 / u_steps
    # 扫描关键角度：0, 45, 90, 180
    test_angles = [0, step_angle * 4, step_angle * 8, step_angle * 16] 
    
    all_summary = []

    for name, cfg in ABLATION_CONFIGS.items():
        print(f"\n>>> Evaluating Ablation Study: {name}")
        
        # 实例化模型
        model = vit_huge_patch14(img_size=52, patch_size=52, num_classes=8, grid_height=16).to(device)
        
        # 加载并修复权重名
        checkpoint = torch.load(cfg['ckpt'], map_location=device, weights_only=False)
        ckpt_state_dict = checkpoint['model']
        new_state_dict = {}

        for k, v in ckpt_state_dict.items():
            k_new = k.replace('encoder.', '') if k.startswith('encoder.') else k
            k_new = k_new.replace('patch_embed', 'view_embed')
            if k_new == 'view_embed.proj.weight': k_new = 'view_embed.proj.proj.weight'
            if k_new == 'view_embed.proj.bias': k_new = 'view_embed.proj.proj.bias'
            new_state_dict[k_new] = v

        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"  Load Message: {msg}")
        
        model.eval()
        extractor = FeatureExtractor(model)

        # 获取基准特征 (0度)
        patches_0 = odi.extract_patches_from_pano(raw_img, u_steps, v_steps, h_fov, v_fov, target_patch_size)
        tensors_0 = torch.stack([normalize(to_tensor(p)) for p in patches_0]).unsqueeze(0).to(device)
        base_patches, _ = extractor.forward_patch_and_cls(tensors_0, angle_centers)
        base_grid = base_patches.view(1, v_steps, u_steps, -1)

        # 运行旋转扫描
        for i, ang in enumerate(test_angles):
            grid_shift = int(ang / step_angle)
            rot_img = rotator.rotate_image(raw_img, yaw=ang)
            
            curr_patches_raw = odi.extract_patches_from_pano(rot_img, u_steps, v_steps, h_fov, v_fov, target_patch_size)
            curr_tensors = torch.stack([normalize(to_tensor(p)) for p in curr_patches_raw]).unsqueeze(0).to(device)
            curr_patches, _ = extractor.forward_patch_and_cls(curr_tensors, angle_centers)
    
            # 准备循环位移的期望值
            expected_grid = torch.roll(base_grid, shifts=grid_shift, dims=2) 
            
            # 调用新写的加权函数
            error = calculate_weighted_equiv_error(
                curr_patches, 
                expected_grid.view(1, -1, curr_patches.shape[-1]), 
                angle_centers, v_steps, u_steps
            )
            
            all_summary.append({
                'Config': name,
                'Angle': ang,
                'Equiv_Error': error.item()
            })
            print(f"  Angle {ang:>5}° | Error: {error.item():.6f}")

    # 保存最终结果
    result_df = pd.DataFrame(all_summary)
    result_df.to_csv(os.path.join(output_dir, "b3_masking_ablation_results.csv"), index=False)
    print(f"\n[B-3 Complete] Summary saved to {output_dir}")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_b3_ablation(
        img_path="/home/shanhefu/CVRG-Pano/test/rgb/img-7.png", 
        device=device,
        output_dir="/media/data_hdd1/shanhefu/outputs/comparison/B-3/"
    )