
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms
import torch
from tqdm import tqdm
from SuperPointPretrainedNetwork.demo_superpoint import SuperPointFrontend

description_method = 'superpoint'

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


def register_images_superpoint(base_image, target_image, max_features=1000):
  
    superpoint_model = SuperPointFrontend(weights_path='SuperPointPretrainedNetwork\superpoint_v1.pth',
                          nms_dist=4,
                          conf_thresh=0.3,
                          nn_thresh=0.7,
                          cuda=False)
    
    base_image_gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
    target_image_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    
    base_image_gray = (base_image_gray.astype('float32') / 255.)
    target_image_gray = (target_image_gray.astype('float32') / 255.)
    
    keypoints1, descriptors1, _ = superpoint_model.run(base_image_gray)
    keypoints2, descriptors2, _ = superpoint_model.run(target_image_gray)
    
    descriptors1 = descriptors1[:, :max_features]
    descriptors2 = descriptors2[:, :max_features]

    # Use BFMatcher for matching descriptors
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1.T, descriptors2.T)
    matches = sorted(matches, key=lambda x: x.distance)
    
    
    return keypoints1, keypoints2, matches
    

def register_images_orb(base_image, target_image, max_features=1000):
    # Convert images to grayscale for feature detection
    base_image_gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
    target_image_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

    # Feature extraction and matching using ORB
    orb = cv2.ORB_create(nfeatures=max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(base_image_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(target_image_gray, None)

    # Use BFMatcher for matching descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # matches = bf.match(descriptors1.T, descriptors2.T)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    return keypoints1, keypoints2, matches
    

def register_images(base_image, target_image, method=description_method, max_matches=500, save_H=False):
    if method == 'superpoint':
        register_images_func = register_images_superpoint
    elif method == 'orb':
        register_images_func = register_images_orb
    else: 
        raise NotImplementedError(f"The method {method} is still not implemented")
        
    keypoints1, keypoints2, matches = register_images_func(base_image, target_image)

    if method =='superpoint':
        def convert_superpoint_keypoints(keypoints):
            keypoints_cv = [cv2.KeyPoint(x=keypoints[0, i], y=keypoints[1, i], size=1) for i in range(keypoints.shape[1])]
            return keypoints_cv

        keypoints1_cv = convert_superpoint_keypoints(keypoints1)
        keypoints2_cv = convert_superpoint_keypoints(keypoints2)
        
        image_matches = cv2.drawMatches(base_image, keypoints1_cv, target_image, keypoints2_cv, matches[:40], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    else: 
        image_matches = cv2.drawMatches(base_image, keypoints1, target_image, keypoints2, matches[:40], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    

    # Draw matches

    # Filter good matches
    good_matches = matches[:max_matches]  # Use top 80 matches

    if len(good_matches) > 4:
        # Extract location of good matches
        points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
        points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

        for i, match in enumerate(good_matches):
            if method == 'orb':
                points1[i, :] = keypoints1[match.queryIdx].pt
                points2[i, :] = keypoints2[match.trainIdx].pt 
            else:
                points1[i, :] = keypoints1[:2, match.queryIdx]
                points2[i, :] = keypoints2[:2, match.trainIdx]
                
        
        # Find homography
        H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)

        # Warp image
        registered_image = cv2.warpPerspective(base_image, H, (target_image.shape[1], target_image.shape[0]))
        if save_H:
            return registered_image, H ,points1,points2# Return the registered image and the homography matrix
        else:
            return registered_image
    else:
        print("Error: Not enough good matches found to calculate homography.")
        return None


def adjust_H_for_big_image(H, small_shape, big_shape):
    s_x = big_shape[1] / small_shape[1]
    s_y = big_shape[0] / small_shape[0]

    # Scaling matrix from small to big image coordinates
    S = np.array([
        [s_x, 0,   0],
        [0,   s_y, 0],
        [0,   0,   1]
    ])

    # Inverse scaling matrix
    S_inv = np.linalg.inv(S)

    # Adjusted homography matrix for the big images
    H_big = S @ H @ S_inv


    return H_big

# 应用变换到大图
def calculate_error(points1, points2, H):
    points1_transformed = cv2.perspectiveTransform(np.expand_dims(points1, axis=1), H)
    # 计算变换后的点与目标点之间的差异
    errors = np.sqrt((points1_transformed - np.expand_dims(points2, axis=1))**2).sum(axis=2)
    mean_error = np.mean(errors)
    mse = np.mean(errors**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))
    return mean_error, mse, rmse, mae


if __name__=="__main__":


    small_dir="./photos2/small_photos"
    image2_path="./photos2/small_photos/half1_In-L15-1-D7-1-HE.jpg"
    content_dir="./photos2/small_photos/content"
    big_dir="./photos2/big_photos"
    resize_dir="./photos2/big_photos/resize"
    result_dir="./photos2/big_photos/result"
    base_big_photo="./photos2/big_photos/half1_In-L15-1-D7-1-HE.jpg"
    
    file_name="half1_In-L15-1-D7-1-MPO.jpg"

    image1_path=os.path.join(small_dir,file_name)
    output_path_small=os.path.join(content_dir,file_name)
    target_big_photo=os.path.join(big_dir,file_name) #目标大图路径
    output_path_big=os.path.join(resize_dir,file_name)
    output_result=os.path.join(result_dir,file_name)
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    if image1 is None:
            print(f"Error: Could not load image from {image1_path}")
    if image2 is None:
            print(f"Error: Could not load image from {image2_path}")
    interp = cv2.INTER_AREA
    
    height, width, _ = image2.shape
    print("开始图像resize")
    if os.path.exists(output_path_small) == False:
            img_ref_big = cv2.imread(image2_path, 0)  # 参考大图
            shape_ref_big = img_ref_big.shape  # 参考大图size
            crop_img_big = crop(image1_path, shape_ref_big)
            fill(crop_img_big, shape_ref_big, output_path_small)
    if os.path.exists(output_path_big) == False:
            img_ref_big = cv2.imread(base_big_photo, 0)  # 参考大图
            shape_ref_big = img_ref_big.shape  # 参考大图size
            crop_img_big = crop(target_big_photo, shape_ref_big)
            fill(crop_img_big, shape_ref_big, output_path_big)
       
    output_path_small=cv2.imread(output_path_small)
        
    registered_image1, H1,points1,points2 = register_images(output_path_small, image2, save_H=True)
    #cv2.imwrite("./photos/big_photos/reuslt/half1_In-L15-4-D3-2-CD11b_small.jpg",registered_image1)
    overlay_small = cv2.addWeighted(image2, 0.4, registered_image1, 0.6, 0)
    #cv2.imwrite('./photos/small_photos/result/photo2.jpg',overlay_small)
    fig = plt.figure(figsize=(15, 10))
    plt.subplot(1, 1, 1)
    plt.imshow(cv2.cvtColor(overlay_small, cv2.COLOR_BGR2RGB))
    plt.title('Overlay of Registered Small Image 1 and Base Big Image')
    plt.show()
    base_big_photo2 = cv2.imread(base_big_photo)
    big_photo = cv2.imread(output_path_big)
    H1_big = adjust_H_for_big_image(H1, image2.shape, base_big_photo2.shape)
    # # #H1_big=adjust_homography_for_image_scaling_and_rotation(image2.shape, base_big_photo2.shape,H1)
    registered_big_image = cv2.warpPerspective(big_photo, H1_big, (base_big_photo2.shape[1], base_big_photo2.shape[0]))
    
    # #overlay_big1 = cv2.addWeighted(base_big_photo2, 0.4, registered_big_image, 0.6, 0)
    cv2.imwrite('./photos2/big_photos/result/half1_In-L15-1-D7-1-MPO.jpg',registered_big_image)
    # print('done')
    # fig = plt.figure(figsize=(15, 10))
    # plt.subplot(1, 2, 1)
    # plt.imshow(cv2.cvtColor(overlay_big1, cv2.COLOR_BGR2RGB))
    # plt.title('Overlay of Registered Big Image 1 and Base Big Image')
    # plt.subplot(1, 2, 2)
    # plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    # plt.title('HE Image')
    # plt.show()
    # fig = plt.figure(figsize=(15, 10))
    # plt.subplot(1, 1, 1)
    # plt.imshow(cv2.cvtColor(registered_big_image, cv2.COLOR_BGR2RGB))
    # plt.title('Overlay of Registered Big Image 1 and Base Big Image')
    # plt.show()

# 保存图像
    #cv2.imwrite(output_result, registered_big_image)
#     cv2.imwrite('half1_In-L15-4-D3-2-CD11b.jpg', registered_big_image)
#     overlay_big = cv2.addWeighted(base_big_photo, 0.4, registered_big_image, 0.6, 0)

# # 用matplotlib显示结果
#     fig = plt.figure(figsize=(15, 10))
#     plt.subplot(1, 1, 1)
#     plt.imshow(cv2.cvtColor(overlay_big, cv2.COLOR_BGR2RGB))
#     plt.title('Overlay of Registered Big Image 1 and Base Big Image')
#     plt.show()
#     print("done")