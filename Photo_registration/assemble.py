import cv2
import numpy as np


def apply_color(image, mask, color, alpha=0.5):
    """
    在图像的指定区域赋予颜色，并处理颜色重叠。
    :param image: 原始图像。
    :param mask: 指定区域的掩码，其中非零部分表示应用颜色的区域。
    :param color: 要应用的颜色，格式为(B, G, R)。
    :param alpha: 混合系数，范围从0到1，0表示完全不透明，1表示完全透明。
    :return: 处理后的图像。
    """
    # 将颜色转换为与图像相同的数据类型
    color = np.array(color, dtype=image.dtype)
    
    # 复制原始图像以避免修改原始数据
    new_image = image.copy()
    
    # 检查掩码中非零元素的位置
    mask_indices = np.where(mask > 0)
    
    # 计算混合颜色
    new_image[mask_indices] = (image[mask_indices] * (1 - alpha)) + (color * alpha)
    
    return new_image

def overlay_images(base_image_path, overlay_image_path, output_image_path):
    # 加载基础图像（图像B）和覆盖图像（图像A）
    base_image = cv2.imread(base_image_path, cv2.IMREAD_UNCHANGED)
    print('base_img shape:', base_image.shape)
    base_image = cv2.cvtColor(base_image, cv2.COLOR_BGRA2BGR)

    overlay_image = cv2.imread(overlay_image_path, cv2.IMREAD_UNCHANGED)
    print('overlay_img shape:', overlay_image.shape)
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
    _, mask = cv2.threshold(overlay_image, 30,255, cv2.THRESH_BINARY)
    #mask=np.where(mask>0)
    #overlay_fill = np.zeros_like(base_image, dtype=np.uint8)
    #overlay_fill[mask] = [0, 255, 0]

    # Combine the overlay fill with the base image
    #result = cv2.addWeighted(base_image, 1, overlay_fill, 0.2, 0)

    # 将遮罩应用于原始图像，只修改病变区域
    
    image_colored = apply_color(base_image, mask, (0, 255, 255),0.5)  

    #CD11b(255, 0, 0)蓝色；IL-6(0, 255, 0)绿色；MPO(0, 0, 255)红色；PCNA(255, 255, 0)靛青色；TNF-a(128, 0, 128)紫色

    #HE(0,0,0)黑;CD11b(255,0,0)蓝;IL-6（0，255，0）绿；MPO(0,0,255)红色；PCNA(255,255,0)靛青色；TNF-a(0,255,255)黄色
    #base_image[np.where(overlay_image > 30)] = [0, 0, 255]
    # 保存结果
    cv2.imwrite(output_image_path, image_colored)
    print(f"已保存修改后的图像到 {output_image_path}")



# 使用示例
# 第一张为原始图像png，第二张为mask图像，第三张为生成的结果图像。
img_path='./photos/small_photos/counter/half1_In-L15-4-D14-PCNA.jpg'
re_path='./photos/small_photos/temp2/half1_In-L15-4-D14-TNF-a.jpg'
output_path='./photos/small_photos/counter/half1_In-L15-4-D14-TNF-a.jpg'
overlay_images(img_path,re_path,output_path)
