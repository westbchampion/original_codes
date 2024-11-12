import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms
import torch
from tqdm import tqdm

def crop(img, target_shape, shift_x=0, shift_y=0):
    shape = img.shape
    start_x = (shape[1] - target_shape[1]) // 2 + shift_x
    start_y = (shape[0] - target_shape[0]) // 2 + shift_y
    end_x = start_x + target_shape[1]
    end_y = start_y + target_shape[0]

    cropped_img = np.zeros((target_shape[0], target_shape[1], 3), dtype=np.uint8)

    valid_start_x = max(0, -start_x)
    valid_start_y = max(0, -start_y)
    valid_end_x = min(target_shape[1], shape[1] - start_x)
    valid_end_y = min(target_shape[0], shape[0] - start_y)

    img_start_x = max(start_x, 0)
    img_start_y = max(start_y, 0)
    img_end_x = img_start_x + (valid_end_x - valid_start_x)
    img_end_y = img_start_y + (valid_end_y - valid_start_y)

    cropped_img[valid_start_y:valid_end_y, valid_start_x:valid_end_x] = img[img_start_y:img_end_y, img_start_x:img_end_x]

    return cropped_img

def fill(img, shape_oct):
    shape = img.shape
    value1 = float(img[1][1][0])
    value2 = float(img[1][1][1])
    value3 = float(img[1][1][2])
    value = [value1, value2, value3]
    add_width = int((shape_oct[0] - shape[0]) / 2)
    add_height = int((shape_oct[1] - shape[1]) / 2)
    if add_width < 0:
        add_width = 0
    if add_height < 0:
        add_height = 0
    img_out = cv2.copyMakeBorder(img, add_width, add_width, add_height, add_height,
                                 cv2.BORDER_CONSTANT, value=value)
    return img_out

if __name__ == "__main__":
    small_dir="./photos2/small_photos"
    image2_path="./photos2/small_photos/half1_In-L15-1-D7-1-HE.jpg"
    content_dir="./photos2/small_photos/content"
    big_dir="./photos2/big_photos"
    resize_dir="./photos2/big_photos/resize"
    result_dir="./photos2/big_photos/result"
    base_big_photo="./photos2/big_photos/half1_CL-L15-6-D5-2-HE.jpg"
    file_name="half1_CL-L15-6-D5-2-TNF-a.jpg"
    shift_values = [(100,-500), (0,0), (-1400,1000), (-1150, -1750), (-500, -1400), (-800, -1400)]
    index=0
    image1_path=os.path.join(small_dir,file_name)
    output_path_small=os.path.join(content_dir,file_name)
    target_big_photo=os.path.join(big_dir,file_name) #目标大图路径
    output_path_big=os.path.join(resize_dir,file_name)
    output_result=os.path.join(result_dir,file_name)

    # #小图调整
    # image1 = cv2.imread(image1_path)
    # image2 = cv2.imread(image2_path)
    # if image1 is None:
    #             raise FileNotFoundError(f"Error: Could not load image from {image1_path}")
    # if image2 is None:
    #             raise FileNotFoundError(f"Error: Could not load image from {image2_path}")

    # target_shape = (min(image1.shape[0], image2.shape[0]), min(image1.shape[1], image2.shape[1]))
    # shift_x, shift_y = shift_values[index]
    # index += 1
    # image1_cropped = crop(image1, target_shape, shift_x=shift_x, shift_y=shift_y)
    # image2_cropped = crop(image2, target_shape)


    #大图调整

    image2 = cv2.imread(base_big_photo)
    image1= cv2.imread(target_big_photo)
    if image1 is None:
        raise FileNotFoundError(f"Error: Could not load image from {target_big_photo}")
    if image2 is None:
        raise FileNotFoundError(f"Error: Could not load image from {base_big_photo}")
    print("开始调整")
    target_shape = (min(image1.shape[0], image2.shape[0]), min(image1.shape[1], image2.shape[1]))
    shift_x, shift_y = shift_values[index]

    shape_ref_big=target_shape
    crop_img_big = crop(image1, shape_ref_big,shift_x=shift_x, shift_y=shift_y)
    img=fill(crop_img_big, shape_ref_big)
    success = cv2.imwrite(output_path_big, img)
    if not success:
        raise Exception(f"Error: Failed to save image to {output_path_big}")
    image2_cropped = crop(image2, target_shape)
    output=cv2.imread(output_path_big)
    overlay_small = cv2.addWeighted(image2_cropped, 0.4, output, 0.6, 0)
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(overlay_small, cv2.COLOR_BGR2RGB))
    plt.title(f'Overlay of Cropped and Shifted Image 1 and Image 2: {file_name}')
    plt.show()
    cv2.imwrite(output_result,output)