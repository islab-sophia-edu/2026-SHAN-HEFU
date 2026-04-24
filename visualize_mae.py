import sys
import argparse
import torch
import numpy as np
from PIL import Image
import os
from pathlib import Path

# --- Model and utility imports ---
import models_mae
from util.pos_embed import interpolate_pos_embed
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image

def get_args_parser():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser('MAE Visualization Script', add_help=False)
    
    # Required arguments
    parser.add_argument('--image_path', required=True, type=str, help='Path to the input image.')
    parser.add_argument('--checkpoint_path', required=True, type=str, help='Path to the model checkpoint (.pth) file.')
    
    # Model parameters (must match the checkpoint)
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL', help='Name of the model.')
    parser.add_argument('--input_size', default=224, type=int, help='Image input size.')
    
    # MAE parameters
    parser.add_argument('--mask_ratio', default=0.75, type=float, help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', action='store_true', help='Use normalized pixel values as targets for computing loss.')
    parser.set_defaults(norm_pix_loss=False)

    # Other settings
    parser.add_argument('--device', default='cuda', help='Device to use for computation (e.g., "cuda" or "cpu").')
    
    # Output Directory
    parser.add_argument('--output_dir', default='./generation', type=str, help='Directory to save the output image.')
    
    return parser

def build_transform(input_size):
    """Builds the image transformation pipeline."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    transform = transforms.Compose([
        transforms.Resize(input_size, interpolation=Image.Resampling.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform

def tensor_to_pil(tensor):
    """Converts a PyTorch tensor to a PIL Image after denormalization."""
    # Move tensor to CPU and clone it to avoid modifying the original tensor
    tensor = tensor.cpu().clone()
    
    # Denormalize the tensor
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    tensor = tensor * std[:, None, None] + mean[:, None, None]
    
    # Clamp values to [0, 1] range and convert to PIL Image
    tensor = torch.clamp(tensor, 0, 1)
    return to_pil_image(tensor)

def run_visualization(model, image, transform, mask_ratio, device, output_dir, filename):
    """
    Runs the model on a single image and saves a side-by-side comparison image.
    """
    print("  Running inference...")
    
    # --- Pre-process the image ---
    x = transform(image)
    # Add a batch dimension: (C, H, W) -> (1, C, H, W)
    x = x.unsqueeze(0)

    # --- Run the model's forward pass ---
    with torch.no_grad():
        _, y, mask = model(x.to(device), mask_ratio=mask_ratio)
    
    # The model outputs patches, so we need to unpatchify the reconstruction
    y = model.unpatchify(y)
    y = y.detach().cpu().squeeze(0)

    # --- Prepare the three images for visualization ---
    # 1. Original Image
    original_tensor = x.squeeze(0)

    # 2. Masked Image
    mask = mask.detach().cpu()
    # Expand mask to the size of image patches
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * 3)
    mask = model.unpatchify(mask).squeeze(0)
    masked_tensor = original_tensor * (1 - mask)

    # 3. Reconstructed Image
    reconstructed_tensor = y

    # --- Convert tensors to PIL Images ---
    img_original = tensor_to_pil(original_tensor)
    img_masked = tensor_to_pil(masked_tensor)
    img_reconstructed = tensor_to_pil(reconstructed_tensor)

    # --- Create a new composite image ---
    width, height = img_original.size
    composite_image = Image.new('RGB', (width * 3, height))

    # Paste the three images side-by-side
    composite_image.paste(img_masked, (0, 0))
    composite_image.paste(img_reconstructed, (width, 0))
    composite_image.paste(img_original, (width * 2, 0))
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # Use the filename passed as an argument
    save_path = os.path.join(output_dir, filename)
    composite_image.save(save_path)
    
    print(f"\n✅ Visualization complete. Comparison image saved to: {save_path}")


if __name__ == '__main__':
    # --- Setup and Argument Parsing ---
    args = get_args_parser().parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # --- 1. Build Model ---
    print(f"Building model: {args.model} (input size: {args.input_size}x{args.input_size})")
    model = models_mae.__dict__[args.model](
        img_size=args.input_size, 
        norm_pix_loss=args.norm_pix_loss
    )
    model.to(device)
    
    # --- 2. Load Checkpoint ---
    print(f"Loading checkpoint from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    checkpoint_model = checkpoint.get('model', checkpoint)
    
    # --- 3. Interpolate Position Embeddings (handles different input sizes) ---
    interpolate_pos_embed(model, checkpoint_model)
    
    # --- 4. Load model weights ---
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    # Ensure all weights were loaded
    assert len(msg.missing_keys) == 0, "Model layers are missing from the checkpoint. Check if the model name is correct."
    
    model.eval()
    
    # --- 5. Load and Transform Image ---
    print(f"Loading image from: {args.image_path}")
    img = Image.open(args.image_path).convert('RGB')
    transform = build_transform(args.input_size)
    checkpoint_basename = os.path.basename(args.checkpoint_path)
    # Remove the extension and add .png (e.g., "checkpoint-0.png")
    output_filename = os.path.splitext(checkpoint_basename)[0] + '.png'

    # Pass the new output_filename to the visualization function
    run_visualization(model, img, transform, args.mask_ratio, device, args.output_dir, output_filename)
    