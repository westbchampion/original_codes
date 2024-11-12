import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import numpy as np
import math
from skimage.morphology import disk, rectangle, binary_erosion, binary_closing, binary_opening, binary_dilation, remove_small_objects, label
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

def getmask(im, outpath):

    img = 255 - im
    thred, imgcopy = cv2.threshold(np.uint8(img), 30, 255, cv2.THRESH_BINARY)
    label_image = remove_small_objects(label(imgcopy), 1000)
    label_image = np.where(label_image > 0, 255, label_image)

    output = label_image.copy()
    midimg = label_image.copy()

    #二值化
    midimg = midimg > 0

    # #腐蚀去噪点
    # selem = disk(3)
    # midimg = binary_erosion(midimg, selem)
    #
    # #闭运算去除边缘空洞
    # selem = disk(4)
    # midimg = binary_closing(midimg, selem)
    #
    # #开运算
    # selem = rectangle(1, 8)
    # midimg = binary_opening(midimg, selem)

    #水漫外围，再取反
    midimg = fill_water(midimg)
    h, w = midimg.shape[:2]
    binary = np.zeros([h,w], np.uint8)
    binary[midimg == 0] = 1

    #去除小目标。主要用于去除最下面的奇怪异物
    #label_image = remove_small_objects(label(binary), 15000)
    #binary = label_image > 0
    output[binary==0] = 0

    midimg = np.where(midimg==1, 0, 255)
    cv2.imwrite(outpath, midimg)

def fill_water(image):
    copyimg = image.copy()
    copyimg.astype(np.float32) #cv2要求必须是float32

    #mask必须行和列都加2，且必须为uint8单通道阵列
    h, w=image.shape[:2]
    mask1 = np.zeros([h+2, w+2], np.uint8)
    mask2 = mask1.copy()

    #由于下方出现的弧线可能把图片分割成上下两部分，故同时从上下两个方向进行水漫
    cv2.floodFill(np.float32(copyimg), mask1, (1,1), 1)
    a = h-1
    b = w-1
    #cv2.floodFill(np.float32(copyimg), mask2, (a,b), 1)
    mask = mask1#|mask2

    #还原为原来的尺寸
    output = mask[1:-1, 1:-1]
    return output

def findmask(path, outpath):

    img = cv2.imread(path, 0)
    img = 255 - img
    thred, imgcopy = cv2.threshold(np.uint8(img), 30, 255, cv2.THRESH_BINARY)
    label_image = remove_small_objects(label(imgcopy), 1000)
    label_image = np.where(label_image > 0, 255, label_image)
    # imgcopy = 255 -imgcopy
    # mask = lungsegmentation(imgcopy)
    # imgout = mask * imgcopy
    cv2.imwrite(outpath, label_image)

def findTransform(img_ref, img_mov):
    feature_params = dict(maxCorners=5000, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(10, 10), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    img_ref_pts = cv2.goodFeaturesToTrack(img_ref, mask=None, **feature_params)
    img_mov_pts, st, err = cv2.calcOpticalFlowPyrLK(img_ref, img_mov, img_ref_pts, None, **lk_params)
    good_mov = img_mov_pts[st == 1]
    good_ref = img_ref_pts[st == 1]
    good_mov = np.float32(good_mov) # * 8
    good_ref = np.float32(good_ref) # * 8
    M = cv2.estimateAffinePartial2D(good_mov, good_ref, cv2.RANSAC)  # cv2.estimateAffinePartial2D
    return M[0]

def crop(imgpath1, shape_oct):
    image1 = cv2.imread(imgpath1, cv2.IMREAD_COLOR)
    shape = image1.shape
    if shape[0] > shape_oct[0]:
        start_index = int((shape[0] - shape_oct[0]) / 2)
        image1 = image1[start_index:start_index + shape_oct[0], :, :]
    if shape[1] > shape_oct[1]:
        start_index = int((shape[1] - shape_oct[1]) / 2)
        image1 = image1[:, start_index:start_index + shape_oct[1], :]
    return image1

def fill(img, shape_oct, outpath):
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
    cv2.imwrite(outpath, img_out)

def find2points(img, shape):
    ref_x1 = 0
    ref_y1 = 0
    for j in range(shape[1]):
        for i in range(shape[0]):
            if img[shape[0]-i-1, shape[1]-j-1] > 50:
                ref_x1 = shape[0]-i-1
                ref_y1 = shape[1]-j-1
                if ref_x1 < shape[0] / 2:
                    break
        if ref_x1 != 0 or ref_y1 != 0:
            if ref_x1 < shape[0] / 2:
                break
    return ref_x1, ref_y1

def rigidtanslation(output_path_big, mask_ref, mask_mov, angle, outputpath,ref_img):
    img=cv2.imread(ref_img)
    img_mov = Image.open(output_path_big)
    rotated_img = img_mov.rotate(-angle)

    center = (mask_mov.shape[1] / 2, mask_mov.shape[0] / 2)
    M_angle = cv2.getRotationMatrix2D(center, -angle, 1.0)
    retated_img_mask = cv2.warpAffine(mask_mov, M_angle, (mask_mov.shape[1], mask_mov.shape[0]))
    transx1, transy1 = find2points(mask_ref, mask_ref.shape)
    transx2, transy2 = find2points(retated_img_mask, mask_mov.shape)
    X = int((transx1 - transx2) * (img_mov.size[0] / mask_ref.shape[1]))
    Y = int((transy1 - transy2) * (img_mov.size[1] / mask_ref.shape[0]))

    translated_image = Image.new("RGB", (rotated_img.size[0], rotated_img.size[1]))
    translated_image.paste(rotated_img, (Y, X))
    translated_image_np = np.array(translated_image)

# 确保translated_image_np是正确的颜色通道顺序，OpenCV期望的是BGR
    if translated_image_np.shape[2] == 3:
        translated_image_np = translated_image_np[:, :, ::-1] 
    overlay1 = cv2.addWeighted(img, 0.5, translated_image_np, 0.5, 0)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(overlay1, cv2.COLOR_BGR2RGB))
    plt.show()
    translated_image.save(outputpath)

if __name__=='__main__':

   # 所有需要修改的路径
   big_photos_dir = "./photos2/big_photos/"
   small_photos_dir = "./photos2/small_photos/"
   ref_img_name = "half1_D-L15-4-D7-1-HE.jpg"
   mov_img_name = "half1_D-L15-4-D7-1-IL-6.jpg"

   path_ref_big = big_photos_dir + ref_img_name # "C:/task/big_photos/In-C-6-D7-2-HE.jpg"  # 参考大图路径
   path_ref_small = small_photos_dir + ref_img_name #  "C:/task/small_photos/In-C-6-D7-2-HE.jpg" # 参考小图路径
   img_mov_big = big_photos_dir + mov_img_name # "C:/task/big_photos/In-C-6-D7-2-CD11b.jpg"  # 目标大图路径
   output_path_big = big_photos_dir + "resize/" + mov_img_name     # "C:/task/big_photos/resize/In-C-6-D7-2-CD11b.jpg"  # 调整size后目标大图输出路径
   img_mov_samll = small_photos_dir + mov_img_name # "C:/task/small_photos/In-C-6-D7-2-CD11b.jpg"  # 目标小图路径
   output_path_small = small_photos_dir + "resize/" + mov_img_name   # "C:/task/small_photos/resize/In-C-6-D7-2-CD11b.jpg"  # 调整size后目标小图输出路径
   maskpath_mov = small_photos_dir + "mask/" + mov_img_name  # "C:/task/small_photos/mask/In-C-6-D7-2-CD11b_ostu.jpg"  # 小图掩膜的输出路径
   maskpath_ref = small_photos_dir + "mask/" + ref_img_name  # "C:/task/small_photos/mask/In-C-6-D7-2-HE_ostu.jpg"  # 小图掩膜的输出路径
   outputpath =  big_photos_dir + "result/" + mov_img_name  # "C:/task/big_photos/result/In-C-6-D7-2-CD11b_result.jpg" # 最终输出路径

   # 改变图像大小，保持一致
   print("开始图像resize")
   if os.path.exists(output_path_big) == False:
       img_ref_big = cv2.imread(path_ref_big, 0)  # 参考大图
       shape_ref_big = img_ref_big.shape  # 参考大图size
       crop_img_big = crop(img_mov_big, shape_ref_big)
       fill(crop_img_big, shape_ref_big, output_path_big)

   # 获取小图的mask
   print("图像resize结束，开始获取掩膜")
   img_ref_small = cv2.imread(path_ref_small, 0)  # 参考小图
   shape_ref_small = img_ref_small.shape  # 参考小图size
   crop_img_small = crop(img_mov_samll, shape_ref_small)
   fill(crop_img_small, shape_ref_small, output_path_small)
#   findmask(path_ref_small, maskpath_ref)  # 如果结果不好，可以注释掉getmask，患者findmask
#   findmask(output_path_small, maskpath_mov)  # 如果结果不好，可以注释掉getmask，患者findmask
   getmask(img_ref_small, maskpath_ref)
   img_mov_small1 = cv2.imread(output_path_small, 0)  # 目标小图
   getmask(img_mov_small1, maskpath_mov)


   # 寻找变换
   print("掩膜获取结束，开始寻找旋转角度")
   mask_ref = cv2.imread(maskpath_ref, 0) # 参考小图掩膜
   mask_mov = cv2.imread(maskpath_mov, 0) # 目标小图掩膜
   M = findTransform(mask_ref, mask_mov)
   angle = math.atan2(M[1][0], M[0][0])
   angle = math.degrees(angle)

   print("寻找旋转角度结束，开始配准")
   rigidtanslation(output_path_big, mask_ref, mask_mov, angle, outputpath,path_ref_big)
   print("配准完成")



