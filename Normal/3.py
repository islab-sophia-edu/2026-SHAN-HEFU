import numpy as np
import torch
import torch.nn.functional as F

# =========================================================
# 1. 严格复刻你的 Dataset 几何逻辑 (Z-up)
# =========================================================
class DatasetCameraPrm:
    def __init__(self, camera_angle):
        ca = camera_angle # [theta, phi]
        # nc (Forward / Z)
        self.nc = np.array([
            np.cos(ca[1]) * np.cos(ca[0]), 
            -np.cos(ca[1]) * np.sin(ca[0]), 
            np.sin(ca[1])
        ])
        # xn (Right / X)
        self.xn = np.array([
            -np.sin(ca[0]), 
            -np.cos(ca[0]),
            0
        ])
        # yn (Up-vector base)
        self.yn = np.array([
            -np.sin(ca[1]) * np.cos(ca[0]), 
            np.sin(ca[1]) * np.sin(ca[0]),
            np.cos(ca[1])
        ])

def get_rotation_matrix(prm):
    # Dataset 逻辑: R = [Right, Up, Forward]
    # 注意：Up 是 -prm.yn
    R = np.stack([prm.xn, -prm.yn, prm.nc], axis=0)
    return R

# =========================================================
# 2. 定义你的 Surgery 变换 (你之前提出的那个)
# =========================================================
def apply_user_surgery(v_in):
    """
    v_in: [X, Y, Z] 向量 (Camera Space)
    根据你的 Surgery 逻辑:
    New_X = -Old_Y
    New_Y = Old_Z
    New_Z = Old_X
    """
    v_out = np.zeros_like(v_in)
    v_out[0] = v_in[1] # Target X = -Old Y
    v_out[1] = -v_in[2]  # Target Y = Old Z (对齐垂直轴)
    v_out[2] = -v_in[0]  # Target Z = Old X
    return v_out

# =========================================================
# 3. 闭环测试流程
# =========================================================
def run_consistency_test():
    print(">>> 启动几何逻辑闭环测试...\n")
    
    # 模拟一个世界坐标系下的法线 (Stanford2D3D 是 Y-up)
    # 假设我们看一堵位于正前方的墙，在 Y-up 下，正前方通常是 Z 轴
    # 我们测试一个稍微复杂的向量: [0.5, 0.5, 0.707] (归一化后)
    v_world_gt = np.array([0.5, 0.3, 0.812]) 
    v_world_gt /= np.linalg.norm(v_world_gt)
    
    print(f"[1] 原始 GT (World, Y-up):   {v_world_gt}")

    # 选取一个相机角度 (例如：水平 30度, 仰角 20度)
    angle = [np.radians(30), np.radians(20)]
    prm = DatasetCameraPrm(angle)
    R = get_rotation_matrix(prm)

    # A. 模拟 Dataset 生成过程: World -> Camera (Z-up 投影)
    # v_cam = R @ v_world
    v_cam_raw = np.dot(R, v_world_gt)
    print(f"[2] 投影到 Camera Space:      {v_cam_raw}")

    # B. 模拟模型推理 + 你的权重手术: Apply Surgery
    v_cam_after_surgery = apply_user_surgery(v_cam_raw)
    print(f"[3] 应用 Surgery 变换后:       {v_cam_after_surgery}")

    # C. 模拟可视化/评估过程: Camera -> World (逆旋转)
    # v_world_rec = R.T @ v_cam_after_surgery
    v_world_rec = np.dot(R.T, v_cam_after_surgery)
    v_world_rec /= np.linalg.norm(v_world_rec)
    print(f"[4] 还原回 World Space:       {v_world_rec}")

    # D. 计算误差
    cos_sim = np.dot(v_world_gt, v_world_rec)
    angle_err = np.degrees(np.arccos(np.clip(cos_sim, -1, 1)))
    
    print(f"\n[结果分析]")
    print(f"角度偏差: {angle_err:.4f} 度")
    if angle_err < 1e-3:
        print("✅ 逻辑闭环成功！Surgery 变换与 Dataset 几何逻辑完美匹配。")
    else:
        print("❌ 逻辑失败！你的 Surgery 变换在当前 Dataset 几何定义下是不正确的。")
        
        # 自动搜索正确的变换方案
        find_correct_mapping(v_cam_raw, v_world_gt, R)

def find_correct_mapping(v_cam, v_gt, R):
    print("\n>>> 正在搜索正确的物理映射方案...")
    axes = [0, 1, 2] # X, Y, Z
    signs = [1.0, -1.0]
    best_err = 1000
    best_map = None

    import itertools
    # 穷举所有通道置换和符号翻转组合
    for p in itertools.permutations(axes):
        for s in itertools.product(signs, repeat=3):
            v_test_cam = np.array([v_cam[p[0]]*s[0], v_cam[p[1]]*s[1], v_cam[p[2]]*s[2]])
            v_test_world = np.dot(R.T, v_test_cam)
            v_test_world /= np.linalg.norm(v_test_world)
            err = np.degrees(np.arccos(np.clip(np.dot(v_gt, v_test_world), -1, 1)))
            if err < best_err:
                best_err = err
                best_map = (p, s)

    p, s = best_map
    print(f"找到的最优映射方案 (误差: {best_err:.4e}):")
    mapping_str = [
        f"New_X = {s[0]:+.1f} * Old_{['X','Y','Z'][p[0]]}",
        f"New_Y = {s[1]:+.1f} * Old_{['X','Y','Z'][p[1]]}",
        f"New_Z = {s[2]:+.1f} * Old_{['X','Y','Z'][p[2]]}"
    ]
    for line in mapping_str: print(f"  {line}")

if __name__ == "__main__":
    run_consistency_test()