import torch
import torch.nn.functional as F
import argparse
import sys
from tqdm import tqdm

# 请确保你的模型和数据集定义在搜索路径中
from dataset_stanford_normal import build_normal_dataset
from models_normal_cnn import vit_huge_patch14 

def get_args():
    parser = argparse.ArgumentParser()
    # 路径保持和你的一致
    parser.add_argument('--resume', default='/media/data_hdd1/shanhefu/outputs/finetune/normal_stanford/normal_mask0.6-0.9_16*32_1.5e-4_decay0.65_huge_1024_nocnn_16_cl_4_hybrid_augmention/checkpoint-99.pth', type=str)
    parser.add_argument('--val_data_path', default='/home/shanhefu/Stanford2D3D/', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--pano_h', default=1024, type=int)
    parser.add_argument('--pano_w', default=2048, type=int)
    parser.add_argument('--grid_height', default=16, type=int)
    parser.add_argument('--batch_size', default=8, type=int) # 调大点跑得快
    return parser.parse_args()

@torch.no_grad()
def main():
    args = get_args()
    device = torch.device(args.device)

    # 1. 准备模型
    u_steps, v_steps = 2 * args.grid_height, args.grid_height
    patch_size = (args.pano_h // v_steps, args.pano_w // u_steps)
    model = vit_huge_patch14(img_size=patch_size, patch_size=patch_size, in_chans=3, 
                             grid_height=args.grid_height, output_size=(args.pano_h, args.pano_w))
    
    print(f">>> Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model', checkpoint)
    new_dict = {k.replace('module.', '').replace('backbone.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_dict, strict=False)
    model.to(device).eval()

    # 2. 准备数据
    dataset = build_normal_dataset(is_train=False, args=args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 3. 核心变换候选方案 (增加到 12 种，覆盖更多旋转可能性)
    test_cases = {
        "0. Original (X,Y,Z)":    lambda p: p,
        "1. Swap X-Z (Z,Y,X)":    lambda p: p[:, [2, 1, 0], :, :],
        "2. Swap Y-Z (X,Z,Y)":    lambda p: p[:, [0, 2, 1], :, :],
        "3. Flip Z (X,Y,-Z)":     lambda p: torch.stack([p[:,0], p[:,1], -p[:,2]], dim=1),
        "4. Flip X (-X,Y,Z)":     lambda p: torch.stack([-p[:,0], p[:,1], p[:,2]], dim=1),
        "5. Swap X-Z & Flip Z":   lambda p: torch.stack([p[:,2], p[:,1], -p[:,0]], dim=1),
        "6. Swap X-Z & Flip X":   lambda p: torch.stack([-p[:,2], p[:,1], p[:,0]], dim=1),
        "7. Swap X-Z & Flip XZ":  lambda p: torch.stack([-p[:,2], p[:,1], -p[:,0]], dim=1),
        "8. Permute (Y,Z,X)":     lambda p: p[:, [1, 2, 0], :, :],
        "9. Permute (Z,X,Y)":     lambda p: p[:, [2, 0, 1], :, :],
        "10. Flip X, Flip Z":     lambda p: torch.stack([-p[:,0], p[:,1], -p[:,2]], dim=1),
        "11. All Flip (-X,-Y,-Z)": lambda p: -p,
    }

    # 初始化计数器
    case_results = {name: {"correct": 0, "total": 0} for name in test_cases.keys()}

    print(f"\n>>> Starting Global Val Scan (Total {len(dataloader)} batches)...")
    
    for views, angles, targets, mask in tqdm(dataloader):
        views, targets, mask = views.to(device), targets.to(device), mask.to(device)
        
        # 推理
        logits = model(views, angles.to(device))
        B, N, C, H, W = logits.shape
        
        # 统一尺寸
        logits_up = F.interpolate(logits.reshape(B*N, C, H, W).contiguous(), size=patch_size, mode='bilinear', align_corners=False)
        targets_up = F.interpolate(targets.reshape(B*N, 3, targets.shape[-2], targets.shape[-1]).contiguous(), size=patch_size, mode='bilinear', align_corners=False)
        mask_up = F.interpolate(mask.reshape(B*N, 1, mask.shape[-2], mask.shape[-1]).contiguous(), size=patch_size, mode='nearest')
        
        pred = F.normalize(logits_up, dim=1).float()
        gt = F.normalize(targets_up, dim=1).float()
        valid_mask = (mask_up.squeeze(1) > 0.5)
        
        # 遍历每种情况
        for name, func in test_cases.items():
            p_test = func(pred)
            dot = torch.sum(p_test * gt, dim=1).clamp(-1.0, 1.0)
            angle = torch.rad2deg(torch.acos(dot))
            
            # 计算当前 batch 的 p30 命中像素数
            correct_pixels = (angle[valid_mask] < 30).sum().item()
            total_pixels = valid_mask.sum().item()
            
            case_results[name]["correct"] += correct_pixels
            case_results[name]["total"] += total_pixels

    # 4. 输出最终统计
    print("\n" + "="*60)
    print(f"{'Global Test Case (Whole Val Set)':<35} | {'Mean P30':<12}")
    print("-" * 60)

    best_p30, best_name = -1.0, ""

    for name in test_cases.keys():
        res = case_results[name]
        final_p30 = (res["correct"] / res["total"] * 100) if res["total"] > 0 else 0
        print(f"{name:<35} | {final_p30:>8.2f}%")
        
        if final_p30 > best_p30:
            best_p30 = final_p30
            best_name = name

    print("="*60)
    print(f"\n🏆 GLOBAL WINNER: {best_name}")
    print(f"📈 FINAL P30 SCORE: {best_p30:.2f}%")
    print("\n[Recommendation] Modify your evaluate function to use this transformation.")
    print("="*60)

if __name__ == "__main__":
    main()