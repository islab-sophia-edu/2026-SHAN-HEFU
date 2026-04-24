import os
import time
import math
import numpy as np
from PIL import Image

# ==========================================
# Tool Functions
# ==========================================
def rnd(x):
    if type(x) is np.ndarray:
        return (x + 0.5).astype(int)
    else:
        return round(x)

def polar(cord):
    if cord.ndim == 1:
        P = np.linalg.norm(cord)
    else:
        P = np.linalg.norm(cord, axis=0)
    phi = np.arcsin(cord[2] / P)
    theta_positive = np.arccos(cord[0] / np.sqrt(cord[0]**2 + cord[1]**2))
    theta_negative = - np.arccos(cord[0] / np.sqrt(cord[0]**2 + cord[1]**2))
    theta = (cord[1] > 0) * theta_negative + (cord[1] <= 0) * theta_positive
    return [theta, phi]

def limit_values(x, r, xcopy=1):
    if xcopy == 1:
        ret = x.copy()
    else:
        ret = x
    ret[ret < r[0]] = r[0]
    ret[ret > r[1]] = r[1]
    return ret

# ==========================================
# Camera & OmniImage Classes
# ==========================================
class CameraPrm:
    def __init__(self, camera_angle, image_plane_size=None, view_angle=None, L=None):
        self.camera_angle = camera_angle
        
        if view_angle is None:
            self.image_plane_size = image_plane_size
            self.L = L
            self.view_angle = 2.0 * np.arctan(np.array(image_plane_size) / (2.0 * L))
        elif image_plane_size is None:
            self.view_angle = view_angle
            self.L = L
            self.image_plane_size = 2.0 * L * np.tan(np.array(view_angle) / 2.0)
        else:
            self.image_plane_size = image_plane_size
            self.view_angle = view_angle
            L = (np.array(image_plane_size) / 2.0) / np.tan(np.array(view_angle) / 2.0)
            if rnd(L[0]) != rnd(L[1]):
                self.L = L[0]
            else:
                self.L = L[0]
        
        self.nc = np.array([
                np.cos(camera_angle[1]) * np.cos(camera_angle[0]), 
                -np.cos(camera_angle[1]) * np.sin(camera_angle[0]), 
                np.sin(camera_angle[1])
            ])
        
        self.c0 = self.L * self.nc

        self.xn = np.array([
                -np.sin(camera_angle[0]), 
                -np.cos(camera_angle[0]),
                0
            ])
        self.yn = np.array([
                -np.sin(camera_angle[1]) * np.cos(camera_angle[0]), 
                np.sin(camera_angle[1]) * np.sin(camera_angle[0]),
                np.cos(camera_angle[1])
            ])
        
        [c1, r1] = np.meshgrid(np.arange(0, rnd(self.image_plane_size[0])), np.arange(0, rnd(self.image_plane_size[1])))
        img_cord = [c1 - self.image_plane_size[0] / 2.0, -r1 + self.image_plane_size[1] / 2.0]
        self.p = self.get_3Dcordinate(img_cord)
        self.polar_omni_cord = polar(self.p)
        
    def get_3Dcordinate(self, c):
        [xp, yp] = c
        if type(xp) is np.ndarray:
            return xp * self.xn.reshape((3,1,1)) + yp * self.yn.reshape((3,1,1)) + np.ones(xp.shape) * self.c0.reshape((3,1,1))
        else:
            return xp * self.xn + yp * self.yn + self.c0

class OmniImage:
    def __init__(self, omni_image):
        if type(omni_image) is str:
            import matplotlib.pyplot as plt
            self.omni_image = plt.imread(omni_image)
        else:
            self.omni_image = omni_image
    
    def extract(self, camera_prm):
        c2 = (camera_prm.polar_omni_cord[0] / (2.0 * np.pi) + 1.0 / 2.0) * self.omni_image.shape[1] - 0.5
        r2 = (-camera_prm.polar_omni_cord[1] / np.pi + 1.0/2.0) * self.omni_image.shape[0] - 0.5
        c2_int = limit_values(rnd(c2), (0, self.omni_image.shape[1]-1), 0)
        r2_int = limit_values(rnd(r2), (0, self.omni_image.shape[0]-1), 0)
        return self.omni_image[r2_int, c2_int]

def extract_patches_from_pano(pano_np, h_num, v_num, h_fov_deg, v_fov_deg, out_size):
    patches = []
    imc = OmniImage(pano_np)
    
    thetas = np.linspace(-np.pi, np.pi, h_num, endpoint=False)
    v_fov_rad = np.radians(v_fov_deg)
    start_phi = np.pi / 2.0 - v_fov_rad / 2.0
    end_phi = -np.pi / 2.0 + v_fov_rad / 2.0
    phis = np.linspace(start_phi, end_phi, v_num)

    for phi_c in phis:
        for theta_c in thetas:
            prm = CameraPrm(camera_angle=[theta_c, phi_c], view_angle=[np.radians(h_fov_deg), np.radians(v_fov_deg)], image_plane_size=[out_size, out_size])
            imp = imc.extract(prm)
            patches.append(imp)
            
    return patches

# ==========================================
# Evaluation Script
# ==========================================
def run_benchmark():
    # Modification: Path converted to base directory
    test_cases = [
        {"desc": "512*1024 (SUN360)", "size": (1024, 512), "dir": "/media/data_hdd/shanhefu/sun360/sun360_outdoor/train/"},
        {"desc": "832*1664 (CVRG-Pano)", "size": (1664, 832), "dir": "/media/data_hdd2/shanhefu/CVRG-Pano/train/rgb/"},
        {"desc": "1024*2048 (Matterport3D)", "size": (2048, 1024), "dir": "/media/data_hdd2/shanhefu/Matterport3D/train/"},
        {"desc": "2048*4096 (Stanford2D3D)", "size": (4096, 2048), "dir": "/media/data_hdd2/shanhefu/Stanford2D3D/area_1/pano/rgb/"}
    ]

    # Extraction configurations
    grid_height = 16
    h_num = grid_height * 2 
    v_num = grid_height       
    h_fov_deg = 360.0 / h_num
    v_fov_deg = 180.0 / v_num
    target_img_count = 100

    print("=" * 60)
    print(f"Extraction Parameter: Grid={v_num}x{h_num} (Total {v_num*h_num} patches)")
    print(f"FOV: H={h_fov_deg}°, V={v_fov_deg}°")
    print(f"Target Images per Dataset: {target_img_count}")
    print("=" * 60)

    for case in test_cases:
        w, h = case["size"]
        img_dir = case["dir"]
        out_size = min(w, h) // 16 
        
        # Image path gathering
        image_paths = []
        if os.path.isdir(img_dir):
            valid_exts = ('.png', '.jpg', '.jpeg')
            all_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(valid_exts)]
            image_paths = all_files[:target_img_count]
        
        actual_real_images = len(image_paths)
        
        # Warm-up (Compilation cache)
        dummy_pano = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        _ = extract_patches_from_pano(dummy_pano, 2, 1, h_fov_deg, v_fov_deg, out_size)

        # Batch Timing
        start_time = time.perf_counter()
        
        for i in range(target_img_count):
            if i < actual_real_images:
                try:
                    img_pil = Image.open(image_paths[i]).convert('RGB').resize((w, h))
                    pano_np = np.array(img_pil)
                except Exception:
                    pano_np = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            else:
                pano_np = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

            # Extraction
            _ = extract_patches_from_pano(pano_np, h_num, v_num, h_fov_deg, v_fov_deg, out_size)
            
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Results Output
        print(f"[{case['desc']}]")
        print(f"  - Source: {actual_real_images} Disk Images, {target_img_count - actual_real_images} Dummy (Resized to {w}x{h})")
        print(f"  - Output Patch Size: {out_size}x{out_size}")
        print(f"  - Total Extraction Time ({target_img_count} imgs): {total_time:.4f} seconds")
        print(f"  - Average Time per Image: {total_time/target_img_count:.4f} seconds")
        print("-" * 60)

if __name__ == '__main__':
    run_benchmark()