import cv2
import numpy as np

def overlay_images(base_image_path, overlay_image_path, output_image_path):
    # 加载基础图像（图像B）和覆盖图像（图像A）
    base_image = cv2.imread(base_image_path, cv2.IMREAD_UNCHANGED)
    base_image = cv2.cvtColor(base_image, cv2.COLOR_BGRA2BGR)

    overlay_image = cv2.imread(overlay_image_path, cv2.IMREAD_UNCHANGED)

    # 裁剪覆盖图像以确保它不大于基础图像
    overlay_image = overlay_image[:base_image.shape[0], :base_image.shape[1]]

    # 检查通道数，确保它们相同
    base_channels = base_image.shape[2]
    overlay_channels = overlay_image.shape[2]

    # 如果覆盖图像有透明度通道，而基础图像没有，则移除透明度通道
    if base_channels == 3 and overlay_channels == 4:
        overlay_image = overlay_image[:, :, :3]

    # 如果通道数仍然不匹配，打印错误信息
    if base_channels != overlay_image.shape[2]:
        print("图像通道数不匹配，无法进行覆盖。")
        return

    # 创建掩码以识别非黑色像素
    # 非黑色定义为任何颜色不等于[0, 0, 0]
    #non_black_mask = np.all(overlay_image[:, :, :3] != [0, 0, 0], axis=-1)

  
    # 应用掩码以将非黑色像素从覆盖图像复制到基础图像
    #base_image[non_black_mask, :3] = overlay_image[non_black_mask, :3]
    # 转为绿色
    #base_image[non_black_mask] = [0, 255, 0]
    overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2GRAY)
    #unique, counts = np.unique(overlay_image, return_counts=True)
    #print(unique)
    #print(counts)
     # 使用阈值处理生成病变区域的二值化遮罩
    _, mask = cv2.threshold(overlay_image, 30, 255, cv2.THRESH_BINARY)
    mask=np.where(mask>0)
    overlay_fill = np.zeros_like(base_image, dtype=np.uint8)
    overlay_fill[mask] = [0, 255, 0]

    # Combine the overlay fill with the base image
    result = cv2.addWeighted(base_image, 1, overlay_fill, 0.2, 0)

    # 将遮罩应用于原始图像，只修改病变区域
    #base_image[np.where(mask > 0)] = [0, 0, 255]  # 将病变区域修改为红色
    
    #base_image[np.where(overlay_image > 30)] = [0, 0, 255]
    # 保存结果
    cv2.imwrite(output_image_path, result)
    print(f"已保存修改后的图像到 {output_image_path}")



# 使用示例
# 第一张为原始图像png，第二张为mask图像，第三张为生成的结果图像。
overlay_images('result_png.jpg','re.jpg','output_image.jpg')
