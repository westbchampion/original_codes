import numpy as np
import pandas as pd
import cv2
import os
from PIL import Image

# 只用改下面两处
hearder_path = r"C:\Users\dell\Desktop"  # database路径
main_folder = os.path.join(hearder_path, 'TUNEL')   # 自行修改mask处理的指标：CD3, EGFR, HP, PCNA, Tunel
# 此处的"image_name"是每张图像名称的占位符
image_original_path = os.path.join(main_folder, 'image_name', 'images', 'image_name.tif')  # TIFF图像路径格式
coordinate_original_path = os.path.join(main_folder, 'image_name', 'coordinate')  # 坐标路径格式
mask_original_path = os.path.join(main_folder, 'image_name', 'masks')  # mask路径格式

# 获取main_folder路径下所有文件夹的名称列表
file_names = os.listdir(main_folder)

# 遍历文件夹列表，对于每个文件夹执行以下操作：
for file_name in file_names:
    image_path = image_original_path.replace('image_name', file_name)  # TIFF图像路径
    image = Image.open(image_path)  # 打开对应的TIFF格式图像
    width, height = image.size  # 获取当前打开图像尺寸信息

    # 创建一个与图像大小相同的全白图像（白色RGB）
    img2 = np.full((height, width, 3), (255, 255, 255), np.uint8)

    txt_list = coordinate_original_path.replace('image_name', file_name)   # 坐标路径
    files = os.listdir(txt_list)  # 获取该文件夹下所有的坐标文件名列表

    mask_folder = mask_original_path.replace('image_name', file_name)  # mask导出路径
    print(f"是否创建了目标文件夹：{os.path.exists(mask_folder)}")
    # 创建保存掩膜图像的文件夹
    os.makedirs(mask_folder, exist_ok=True)

    # 清空mask_folder路径下所有图片
    for file in os.listdir(mask_folder):
        os.remove(os.path.join(mask_folder, file))

    countours = []  # 存储标注坐标列表

    # 遍历坐标文件列表，每读取一个坐标文件中的数据，就进行以下操作：
    for idx, file in enumerate(files):
        position = os.path.join(txt_list, file)  # 构建坐标文件的完整路径
        with open(position, "r", encoding='utf-8') as f:  # 打开文件
            data = pd.read_table(f)  # 读取坐标数据
            countour = data.apply(lambda x: tuple(x), axis=1).values.tolist()  # 将数据转换为坐标列表
            countours.append(countour)  # 将坐标列表添加到总列表中
            mask = np.zeros(img2.shape[:2], dtype=np.uint8)  # 根据坐标生成掩膜
            mask_position = [np.array(countour, np.int32)]  # 将坐标转换为掩膜位置
            cv2.fillPoly(mask, mask_position, (255, 255, 255))  # 在掩膜图像中填充对应区域为白色

            # 构建掩膜图像保存的文件路径
            output_file = os.path.join(mask_folder, f'{idx}.png')

            cv2.imshow('masks', mask)  # 显示生成的掩膜图像
            cv2.waitKey(1)  # 等待100ms
            if cv2.imwrite(output_file, mask):
                print(f"成功保存图像到：{output_file}")
            else:
                print(f"图像保存失败：{output_file}")
