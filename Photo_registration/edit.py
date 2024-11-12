import cv2
import numpy as np
from PIL import Image
from collections import Counter
if __name__ == "__main__":
    img_path = "./photos2/big_photos/result/half1_In-L15-1-D7-1-TNF-a.jpg"  # 需要修改的图像路径
    output_path = "./photos2/big_photos/result/SP/half1_In-L15-1-D7-1-TNF-a.jpg" # 输出路径

    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    shape = img.shape
    value = img[int(shape[0]/2), int(shape[1]/2)]
    img = np.where(img == 0, value, img)
    cv2.imwrite(output_path, img)








