import cv2
import numpy as np

def caculate_positive_rate(image_path):
    
# 读取图像
    image = cv2.imread(image_path)

    # 将图像转换为HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义红色范围
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # 创建红色掩码
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)

    # 计算红色区域的面积
    red_area = cv2.countNonZero(red_mask)

    # 计算总面积
    total_area = image.shape[0] * image.shape[1]

    # 计算除白色部分的面积
    white_mask = cv2.inRange(image, (200, 200, 200), (255, 255, 255))
    white_area = cv2.countNonZero(white_mask)
    non_white_area = total_area - white_area

    # 计算阳性率
    positivity_rate = red_area / non_white_area

    return positivity_rate
    

if __name__=="__main__":
    
    image="TNF-a.png"
    positive_rate=caculate_positive_rate(image)
    
    print(positive_rate)



