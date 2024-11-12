import cv2
import os
import numpy as np

if __name__=='__main__':

   big_photo_dir = "./photos/big_photos/full"
   big_photo_out_dir = "./photos/big_photos/half"
   small_photo_dir = "./photos/small_photos/full"
   small_photo_out_dir = "./photos/small_photos/half"
   """
   print("开始大图分割")
   num = 0
   for filename in os.listdir(big_photo_dir):
       pathname = os.path.join(big_photo_dir, filename)
       img_big = cv2.imread(pathname, cv2.IMREAD_COLOR)
       shape = img_big.shape
       half_num = int(shape[1]/2)
       half1 = img_big[:, :half_num-4800, :]
       half2 = img_big[:, half_num+4800:, :]
       cv2.imwrite(big_photo_out_dir + "/half1_" + filename, half1)
       cv2.imwrite(big_photo_out_dir + "/half2_" + filename, half2)
       num = num + 1
       print("完成大图进度：", num)
    
   print("所有大图分割完成，开始小图分割")
   """
   print("开始小图分割")
   num = 0
   for filename in os.listdir(small_photo_dir):
       pathname = os.path.join(small_photo_dir, filename)
       img_small = cv2.imread(pathname, cv2.IMREAD_COLOR)
       shape = img_small.shape
       half_num = int(shape[1]/2)
       half1 = img_small[:, :half_num-600, :]
       #half2 = img_small[:, half_num+600:, :]
       cv2.imwrite(small_photo_out_dir + "/half1_" + filename, half1)
       #cv2.imwrite(small_photo_out_dir + "/half2_" + filename, half2)
       num = num + 1
       print("完成小图进度：", num)
    







