import argparse
import os
import sys
import numpy as np
import torch
from PIL import Image

# --------------------------------------------------------------------------
# 导入我们需要的“组件”
# --------------------------------------------------------------------------

# 1. 从您的主脚本导入 *get_args_parser*
try:
    import main_pretrain_PanoMAE as main_script
except ImportError:
    print("错误：无法导入 'main_pretrain_PanoMAE.py'。")
    print("请确保此脚本与您的主训练脚本 'main_pretrain_PanoMAE.py' 在同一目录中。")
    sys.exit(1)

# 2. 直接导入数据集定义
try:
    from util.datasets_PanoMAE_legacy import PanoramicDataset
except ImportError:
    print("错误：无法从 'util.datasets_PanoMAE' 导入 'PanoramicDataset'。")
    sys.exit(1)

# 3. 直接导入模型定义
try:
    from wLoss.models_PanoMAE_wloss import vit_base_patch16, vit_large_patch16, vit_huge_patch14
except ImportError:
    print("错误：无法从 'models_PanoMAE' 导入模型 (vit_base_patch16 等)。")
    sys.exit(1)

# 4. 导入辅助工具
import util.misc as misc
import util.odi_processing as odi_helper

# --------------------------------------------------------------------------
# 图像网格创建函数 (无变化)
# --------------------------------------------------------------------------
def create_patch_grid_image(patches_list, grid_width_in_patches, padding=4, bg_color=(0, 0, 0)):
    """Creates a grid image of all patches with spacing between them."""
    if not patches_list: return None
    
    first_patch_uint8 = patches_list[0]
    if first_patch_uint8.dtype != np.uint8:
        if first_patch_uint8.max() <= 1.0: first_patch_uint8 = (first_patch_uint8 * 255).astype(np.uint8)
        else: first_patch_uint8 = first_patch_uint8.astype(np.uint8)
    patch_height, patch_width, channels = first_patch_uint8.shape
    
    num_patches = len(patches_list)
    grid_height_in_patches = (num_patches + grid_width_in_patches - 1) // grid_width_in_patches

    canvas_height = grid_height_in_patches * (patch_height + padding) + padding
    canvas_width = grid_width_in_patches * (patch_width + padding) + padding
    
    canvas = np.full((canvas_height, canvas_width, channels), bg_color, dtype=np.uint8)

    for i, patch in enumerate(patches_list):
        row = i // grid_width_in_patches
        col = i % grid_width_in_patches
        y_start = padding + row * (patch_height + padding)
        x_start = padding + col * (patch_width + padding)
        
        patch_uint8 = patch
        if patch_uint8.dtype != np.uint8:
            if patch_uint8.max() <= 1.0: patch_uint8 = (patch_uint8 * 255).astype(np.uint8)
            else: patch_uint8 = patch_uint8.astype(np.uint8)
                
        canvas[y_start : y_start + patch_height, x_start : x_start + patch_width] = patch_uint8
        
    return Image.fromarray(canvas)

# --------------------------------------------------------------------------
# 核心可视化函数 (无变化)
# --------------------------------------------------------------------------
@torch.no_grad()
def generate_visualization(model, data_loader, device, args):
    """
    Runs inference on a single batch and saves the 5-part visualization.
    """
    model.eval()
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)

    print("Loading one batch from validation set for visualization...")
    try:
        views, angles, original_sizes = next(iter(data_loader))
    except StopIteration:
        print("错误：数据加载器 (data_loader) 为空。")
        return

    views, angles = views.to(device, non_blocking=True), angles.to(device, non_blocking=True)
    current_mask_ratio = args.mask_ratio
    
    print(f"Running model inference with mask ratio: {current_mask_ratio}")
    # (假设您的模型支持 float16 推理)
    with torch.amp.autocast(device_type=args.device, dtype=torch.float16):
        loss, pred, mask = model(views, angles, mask_ratio=current_mask_ratio)

    pred_views = pred.view_as(views)
    views_denorm = torch.clamp(views * std + mean, 0, 1)
    pred_denorm = torch.clamp(pred_views * std + mean, 0, 1)

    # --- 图像生成逻辑 ---
    size_tensor = original_sizes[0]
    pano_width = size_tensor[0].item()
    pano_height = pano_width // 2
    pano_size = (pano_height, pano_width)
    v_steps, u_steps = args.grid_height, 2 * args.grid_height
    h_fov, v_fov = 360.0 / u_steps, 180.0 / v_steps

    all_patches_np_denorm = [(p.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) for p in views_denorm[0]]
    patch_grid_img = create_patch_grid_image(all_patches_np_denorm, u_steps, padding=4)
    
    grid_aspect_ratio = patch_grid_img.height / patch_grid_img.width
    resized_grid_height = int(pano_width * grid_aspect_ratio)
    patch_grid_img = patch_grid_img.resize((pano_width, resized_grid_height), Image.Resampling.LANCZOS)

    mask_for_viz = mask[0].cpu().numpy()
    gray_patch_np = np.full_like(all_patches_np_denorm[0], fill_value=128)
    masked_input_patches_list_np = [
        all_patches_np_denorm[i] if mask_for_viz[i] < 0.5 else gray_patch_np
        for i in range(len(all_patches_np_denorm))
    ]
    masked_grid_img = create_patch_grid_image(masked_input_patches_list_np, u_steps, padding=4)
    masked_grid_img = masked_grid_img.resize(patch_grid_img.size, Image.Resampling.NEAREST)

    reconstructed_patches_np_denorm = [(p.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) for p in pred_denorm[0]]

    blended_reconstruction_patches_np = [
        all_patches_np_denorm[i] if mask_for_viz[i] < 0.5 else reconstructed_patches_np_denorm[i]
        for i in range(len(all_patches_np_denorm))
    ]
    reconstructed_grid_img = create_patch_grid_image(blended_reconstruction_patches_np, u_steps, padding=4)
    reconstructed_grid_img = reconstructed_grid_img.resize(patch_grid_img.size, Image.Resampling.LANCZOS)

    def stitch_pano_from_list(patches_np_list_input):
        patches_float = [p.astype(np.float32) / 255.0 for p in patches_np_list_input]
        pano_np = odi_helper.embed_patches_to_pano(
            patches_np_list=patches_float, pano_size=pano_size,
            h_num=u_steps, v_num=v_steps, h_fov_deg=h_fov, v_fov_deg=v_fov
        )
        return (pano_np * 255).astype(np.uint8)

    reconstructed_pano_img = Image.fromarray(stitch_pano_from_list(blended_reconstruction_patches_np))
    original_pano_img = Image.fromarray(stitch_pano_from_list(all_patches_np_denorm))

    final_img_width = pano_width
    final_img_height = (resized_grid_height * 3) + (original_pano_img.height * 2)
    combined_img = Image.new('RGB', (final_img_width, final_img_height))
    
    y_offset = 0
    combined_img.paste(patch_grid_img, (0, y_offset)); y_offset += patch_grid_img.height
    combined_img.paste(masked_grid_img, (0, y_offset)); y_offset += masked_grid_img.height
    combined_img.paste(reconstructed_grid_img, (0, y_offset)); y_offset += reconstructed_grid_img.height 
    combined_img.paste(reconstructed_pano_img, (0, y_offset)); y_offset += reconstructed_pano_img.height
    combined_img.paste(original_pano_img, (0, y_offset))

    base_name = os.path.basename(args.resume)
    file_name = os.path.splitext(base_name)[0]
    save_path = os.path.join(args.output_dir, f'visualization_{file_name}.png')
    
    combined_img.save(save_path)
    print(f"--- 5部分可视化图像已保存到: {save_path} ---")

# --------------------------------------------------------------------------
# 主函数 (已修复)
# --------------------------------------------------------------------------
def main(args):
    misc.init_distributed_mode(args)
    device = torch.device(args.device)

    # === 1. 构建数据集和数据加载器 (无变化) ===
    v_steps = args.grid_height
    dynamic_img_size = 512 // v_steps 

    val_path = args.val_data_path if args.val_data_path else args.data_path
    print(f"Loading validation data from: {val_path}")

    dataset_val = PanoramicDataset(
        root_dir=val_path, 
        grid_height=args.grid_height,
        img_size=dynamic_img_size, 
        multiscale_sampling=False
    )
    
    sampler_val = torch.utils.data.DistributedSampler(
        dataset_val, num_replicas=misc.get_world_size(), rank=misc.get_rank(), shuffle=False
    )
    
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # === 2. 构建模型 (无变化) ===
    print(f"Building model: {args.model}")
    model = globals()[args.model](
        img_size = dynamic_img_size,
        norm_pix_loss=args.norm_pix_loss,
        geometric_bias=args.geometric_bias,
        adaptive_masking=args.adaptive_masking
    )
    model.to(device)

    # === 3. 加载 Checkpoint (已修复) ===
    if args.resume:
        print(f"正在加载 checkpoint: {args.resume}")
        
        # *** 已修复: 添加 weights_only=False ***
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        
        state_dict = checkpoint.get('model', checkpoint)
            
        model.load_state_dict(state_dict, strict=True)
        print("Checkpoint 加载成功。")
    else:
        print("错误：必须通过 --resume 指定一个 checkpoint 文件。")
        sys.exit(1)

    # --- 4. 运行可视化 (无变化) ---
    os.makedirs(args.output_dir, exist_ok=True)
    generate_visualization(model, data_loader_val, device, args)

if __name__ == '__main__':
    # 从您的主脚本获取参数解析器
    parser = main_script.get_args_parser()
    args = parser.parse_args()
    
    main(args)