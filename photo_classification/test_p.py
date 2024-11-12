import os
from glob import glob

import numpy as np
import cv2
import copy
import openslide

from PIL import Image, ImageDraw, ImageFont
import predict_big
from predict_big import predict
from tqdm import tqdm

#ytes.decode("utf-8", "ignore")

def get_ndpi_data(path):
	k = 1
	source = openslide.open_slide(path)
	downsamples=source.level_downsamples
	[w,h]=source.level_dimensions[0]
	size1=int(w*(downsamples[0]/downsamples[k]))
	size2=int(h*(downsamples[0]/downsamples[k]))
	region=np.array(source.read_region((0,0),k,(size1,size2)))[:, :, :3]

	return region


def get_vis_result(cut_image, mask):
	cut_image[np.where(mask == 255)] = [0, 255, 0]

	return cut_image



def pred( imgFile, saveResultFile):
	"""Predict.
	
	Arguments:

		nmodel (MaskRCNN Instance): trained model.
	
		imgFile (str): path to the image file.

		saveMaskFile (str): path to the saved mask image.

		saveResultFile (str): path to the saved the result image overlaid with
		the predicted mask.
		
	Returns:
	
		maxScore (float): prediction probability."""
	ext = os.path.splitext(imgFile)[1]
	print("here")
	try:
		if ext==".ndpi":img = get_ndpi_data(imgFile)
		else:img = cv2.imread(imgFile)[...,::-1]
		#img = get_ndpi_data(imgFile)
		print(img.shape)
  
	except:
		return -1
	H,W,C = img.shape
	split_scale, height, width, index = 128, H, W, 1
	results, preds = np.zeros((H, W), dtype = np.uint8), np.zeros((H, W, 3), dtype = np.uint8)
	row = int(height / split_scale)
	col = int(width / split_scale)
	if height % split_scale == 0:
		row_flag = False
	else:
		row_flag = True
	if width % split_scale == 0:
		col_flag = False
	else:
		col_flag = True

	for idx in tqdm(range(row)):
		for idy in range(col):
			temp_img = img[idx * split_scale : (idx + 1) * split_scale, idy * split_scale : (idy + 1) * split_scale, :]
			cls=predict_big.predict(temp_img)
			
			if cls == 0 or cls==1:
				img[idx * split_scale : (idx + 1) * split_scale, idy * split_scale : (idy + 1) * split_scale, :] = [255, 255, 255]
				temp_img = img[idx * split_scale : (idx + 1) * split_scale, idy * split_scale : (idy + 1) * split_scale, :]
				cut_image = copy.deepcopy(temp_img)
				predMask = np.zeros((split_scale,split_scale), dtype=np.uint8)
				drawonPred = cut_image.copy()
			else:
				
				cut_image = copy.deepcopy(temp_img)
				drawonPred = cut_image.copy()
			preds[idx * split_scale : (idx + 1) * split_scale, idy * split_scale : (idy + 1) * split_scale, :] = drawonPred
			#save_img = get_vis_result(cut_image, predMask)
			#cv2.imwrite(os.path.join(save_cut_res_path, str(index).zfill(3)+'.jpg'), save_img)
			index += 1
		if col_flag:
			temp_img = img[idx * split_scale : (idx + 1) * split_scale, width - split_scale : width, :]
			cls=predict_big.predict(temp_img)
			if cls == 0 or cls==1:
				img[idx * split_scale : (idx + 1) * split_scale, width - split_scale : width, :] = [255, 255, 255]
				temp_img = img[idx * split_scale : (idx + 1) * split_scale, width - split_scale : width, :]
				cut_image = copy.deepcopy(temp_img)
				predMask = np.zeros((split_scale,split_scale), dtype=np.uint8)
				drawonPred = cut_image.copy()
			else:
		
				cut_image = copy.deepcopy(temp_img)
				drawonPred = cut_image.copy()
			
			preds[idx * split_scale : (idx + 1) * split_scale, width - split_scale : width, :] = drawonPred
			#save_img = get_vis_result(cut_image, predMask)
			#cv2.imwrite(os.path.join(save_cut_res_path, str(index).zfill(3)+'.jpg'), save_img)
			index += 1
	if row_flag:
		for idy in range(col):
			temp_img = img[height - split_scale : height, idy * split_scale : (idy + 1) * split_scale, :]
			cls=predict_big.predict(temp_img)
			if cls == 0 or cls==1:
				img[height - split_scale : height, idy * split_scale : (idy + 1) * split_scale, :] = [255, 255, 255]
				temp_img = img[height - split_scale : height, idy * split_scale : (idy + 1) * split_scale, :]
				cut_image = copy.deepcopy(temp_img)
				predMask = np.zeros((split_scale,split_scale), dtype=np.uint8)
				drawonPred = cut_image.copy()
			else:
				
				cut_image = copy.deepcopy(temp_img)
				drawonPred = cut_image.copy()
			preds[height - split_scale : height, idy * split_scale : (idy + 1) * split_scale, :] = drawonPred
			#save_img = get_vis_result(cut_image, predMask)
			#cv2.imwrite(os.path.join(save_cut_res_path, str(index).zfill(3)+'.jpg'), save_img)
			index += 1
		if col_flag:
			temp_img = img[height - split_scale : height, width - split_scale : width, :]
			cls=predict_big.predict(temp_img)
			if cls == 0 or cls==1:
				img[height - split_scale : height, width - split_scale : width, :] = [255, 255, 255]
				temp_img = img[height - split_scale : height, width - split_scale : width, :]
				cut_image = copy.deepcopy(temp_img)
				predMask = np.zeros((split_scale,split_scale), dtype=np.uint8)
				drawonPred = cut_image.copy()
			else:
				
				cut_image = copy.deepcopy(temp_img)
				drawonPred = cut_image.copy()
			preds[height - split_scale : height, width - split_scale : width, :] = drawonPred
			#save_img = get_vis_result(cut_image, predMask)
			#cv2.imwrite(os.path.join(save_cut_res_path, str(index).zfill(3)+'.jpg'), save_img)
			index += 1
	h, w = results.shape 
	#preds = cv2.addWeighted(img, 0.5, preds, 0.5, 0)
	#cv2.imwrite('test.jpg', img)
	#cv2.imwrite(saveMaskFile, results)
	cv2.imwrite(saveResultFile, preds[...,::-1]) #[...,::-1]
	
	





if __name__ == "__main__":
	#n_model=load_models()
	# print('model',n_model)
	img_path="result_png.jpg"
	
	save_result_path="result.jpg"
	
	#save_result_path="ndpi_data//result1.jpg"
	
	pred(img_path,save_result_path)