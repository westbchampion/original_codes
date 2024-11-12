
import openslide
import numpy as np

def get_ndpi_data(path):
	k = 1
	source = openslide.open_slide(path)
	downsamples=source.level_downsamples
	[w,h]=source.level_dimensions[0]
	size1=int(w*(downsamples[0]/downsamples[k]))
	size2=int(h*(downsamples[0]/downsamples[k]))
	region=np.array(source.read_region((0,0),k,(size1,size2)))[:, :, :3]
	return region




import cv2
import numpy as np
#----------------------------------------------------#
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#----------------------------------------------------#
import time
import os
import cv2
import numpy as np
from PIL import Image

from unet import Unet
from tqdm import tqdm



mode = "dir_predict"

 

unet = Unet()

import cv2
import numpy as np

def pad_image(img, target_size):
    """
    函数用于在图像的右侧和底部添加填充，使其尺寸成为target_size的倍数
    """
    rows, cols = img.shape[:2]
    pad_right = target_size - (cols % target_size) if cols % target_size != 0 else 0
    pad_bottom = target_size - (rows % target_size) if rows % target_size != 0 else 0

    return cv2.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0, 0])

def sliding_window(image, window_size, step_size):
    """
    实现了滑动窗口的功能，它在图像上按指定的窗口大小和步长滑动，并生成窗口内容。
    """
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def process_image(img):
    """
    读取图像，应用填充，然后使用滑动窗口和UNet模型对图像进行处理。
    处理的结果是一个拼接后的图像，其中每个窗口的中心部分被提取出来并重新组合。
    """
    # Constants
    WINDOW_SIZE = (830, 830)
    STEP_SIZE = 170
    CROP_SIZE = 170

    # Pad the image
    padded_img = pad_image(img, WINDOW_SIZE[0])

    # Initialize an empty array for the stitched image
    stitched_image = np.zeros_like(padded_img)

    # Calculate the total number of windows to process (for progress bar)
    total_windows = ((padded_img.shape[0] - WINDOW_SIZE[0]) // STEP_SIZE + 1) * \
                    ((padded_img.shape[1] - WINDOW_SIZE[1]) // STEP_SIZE + 1)

    # Process the image using sliding window
    with tqdm(total=total_windows, desc="Processing Windows") as pbar:
        for x, y, window in sliding_window(padded_img, WINDOW_SIZE, STEP_SIZE):
            # Check if the window meets the window size requirement
            if window.shape[0] == WINDOW_SIZE[0] and window.shape[1] == WINDOW_SIZE[1]:
                # Process the image through UNet

                window = Image.fromarray(cv2.cvtColor(window, cv2.COLOR_BGR2RGB))

                processed_window = unet.detect_image(window)
                # img to cv2
                processed_window = cv2.cvtColor(np.array(processed_window), cv2.COLOR_RGB2BGR)
                # Crop the center 170x170 part of the processed window
                center_crop = processed_window[
                    (WINDOW_SIZE[0] - CROP_SIZE) // 2 : (WINDOW_SIZE[0] + CROP_SIZE) // 2,
                    (WINDOW_SIZE[1] - CROP_SIZE) // 2 : (WINDOW_SIZE[1] + CROP_SIZE) // 2
                ]
                #print('center_crop:',center_crop.shape)
                #print('stitched_image:',stitched_image.shape)
                center_crop = cv2.cvtColor(center_crop, cv2.COLOR_BGR2BGRA)
                # Place the cropped part into the stitched image
                stitched_image[
                    y + (WINDOW_SIZE[0] - CROP_SIZE) // 2 : y + (WINDOW_SIZE[0] + CROP_SIZE) // 2,
                    x + (WINDOW_SIZE[1] - CROP_SIZE) // 2 : x + (WINDOW_SIZE[1] + CROP_SIZE) // 2
                ] = center_crop

                pbar.update(1)
    return stitched_image

# load pic
# ndarry
#path_ndpi = "/home/hudian/upload-unet-GC-carafe/In-B-1-D7-1-IL-6.ndpi"
#nd_img = get_ndpi_data(path_ndpi)
#nd_img = ndpi2ndarry(path_ndpi)
#img_opencv = cv2.imdecode(np.ascontiguousarray(nd_img),)

# load pic
# ndarry
path_ndpi = r'CAB000058_4923.jpg'
nd_img=cv2.imread(path_ndpi)
#nd_img = ndpi2ndarry(path_ndpi)
#img_opencv = cv2.imdecode(np.ascontiguousarray(nd_img),)
img_opencv = cv2.cvtColor(np.ascontiguousarray(nd_img), cv2.COLOR_RGB2BGR)
print('img shape:',img_opencv.shape)
img_opencv = cv2.cvtColor(img_opencv, cv2.COLOR_BGR2BGRA)
print('img shape:',img_opencv.shape)
result_img = process_image(img_opencv)
cv2.imwrite('CAB_re.jpg',result_img)






