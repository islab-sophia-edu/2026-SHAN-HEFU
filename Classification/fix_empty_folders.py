import os
from PIL import Image

# 您的测试集路径
test_dir = "/media/data_hdd/shanhefu/sun360/sun360_outdoor_classification/test"

# 缺失的类别列表 (导致报错的那些)
missing_classes = ["airport", "garden", "gulch", "highway", "skatepark", "underwater"]

def create_dummy_image(folder_path):
    """在指定文件夹内创建一个 1x1 的黑色占位图"""
    dummy_path = os.path.join(folder_path, "dummy_placeholder.jpg")
    if not os.path.exists(dummy_path):
        # 创建一个 1x1 的 RGB 黑色图片
        img = Image.new('RGB', (1, 1), color='black')
        img.save(dummy_path)
        print(f"Created dummy image at: {dummy_path}")
    else:
        print(f"Dummy image already exists at: {dummy_path}")

def main():
    print(f"Fixing empty folders in: {test_dir}")
    
    for class_name in missing_classes:
        class_path = os.path.join(test_dir, class_name)
        
        # 1. 如果文件夹不存在，先创建文件夹
        if not os.path.exists(class_path):
            os.makedirs(class_path)
            print(f"Created directory: {class_path}")
        
        # 2. 放入占位图
        create_dummy_image(class_path)
        
    print("\n[Done] All missing classes now have a placeholder image.")
    print("You can now run your training script.")

if __name__ == "__main__":
    main()