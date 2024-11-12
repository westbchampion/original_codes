import cv2
import numpy as np

import matplotlib.pyplot as plt

# 读取图像
path='./photos/small_photos/result/half1_In-L15-4-D14-HE.jpg'

# 读取图像
image = cv2.imread(path)

# 转换为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用边缘检测
edges = cv2.Canny(gray, 100, 200)

# 找到轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 创建一个全黑的图像
#black_image = np.zeros_like(image)
white_image = np.ones_like(image) * 255
# 在全黑图像上绘制红色轮廓
cv2.drawContours(white_image, contours, -1, (0, 0, 0), thickness=cv2.FILLED)

# 显示和保存结果
fig = plt.figure(figsize=(15, 10))
plt.subplot(1, 1, 1)
plt.imshow(cv2.cvtColor(white_image, cv2.COLOR_BGR2RGB))
plt.title('Red Contours on Black Background')
plt.show()

cv2.imwrite('./photos/small_photos/temp/half1_In-L15-4-D14-HE.jpg', white_image)
