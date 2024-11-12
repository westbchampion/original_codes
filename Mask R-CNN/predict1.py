# %%
"""Predict."""

import os
from glob import glob

import numpy as np
import cv2
import copy
import tensorflow as tf

import mrcnn_seg as ms
import mrcnn.model as modellib

from utilities import utils as myutils
#ytes.decode("utf-8", "ignore")
# Root directory of the project
ROOT_DIR = os.path.abspath("./")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR,"models//best_model_all.h5")

def load_models():
	"""Load the trained model.
	
	Returns:
	
		nmodel (MaskRCNN Instance): trained model."""
	config = ms.get_config(100, 2, 1024, 'detect')
	DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
	with tf.device(DEVICE):
		nmodel = modellib.MaskRCNN(
			mode="inference", model_dir=MODEL_DIR, config=config
		)
	#weightsPath = glob(f"logs/002/*.h5")[0]
	#if not os.path.isfile(COCO_MODEL_PATH):
	#	return -1
	nmodel.load_weights(COCO_MODEL_PATH, by_name=True)
	return nmodel

def caulate(label, pred):
	t_p = 0
	f_p = 0
	cal_i = 0
	cal_u = 0
	h, w = label.shape
	for row in range(h):
		for col in range(w):
			if label[row, col] == 255 and pred[row, col] == 255:
				cal_i += 1
			if label[row, col] == 255 or pred[row, col] == 255:
				cal_u += 1
	iou = cal_i / cal_u
	num_objects, labels = cv2.connectedComponents(pred)
	print('num_objects',num_objects)
	for i in range(0, num_objects):
		pred_temp = np.zeros(pred.shape, dtype=np.uint16)
		pred_temp[np.where(labels == i)] = 255
		label_temp = copy.deepcopy(label)
		label_temp[np.where(pred_temp == 0)] = 0
		label_temp = label_temp.astype(np.uint16)
		res = label_temp + pred_temp
		numbers, cnts = np.unique(res, return_counts = True)
		for idx in range(len(numbers)):
			print('numbers[idx]n',numbers[idx])
			suc = 0
			sum = 0
			if numbers[idx] == 255:
				sum = cnts[idx]
			if numbers[idx] == 510:
				suc = cnts[idx]
		if suc + sum > 0:
			ratio = suc / (suc + sum)
			if ratio > 0.8:
				t_p += 1
			else:
				f_p += 1
	recall = t_p / (t_p + f_p)

	return iou, recall

		



def pred(nmodel, imgFile, saveMaskFile, saveResultFile):
	"""Predict.
	
	Arguments:

		nmodel (MaskRCNN Instance): trained model.
	
		imgFile (str): path to the image file.

		saveMaskFile (str): path to the saved mask image.

		saveResultFile (str): path to the saved the result image overlaid with
		the predicted mask.
		
	Returns:
	
		maxScore (float): prediction probability."""
	try:
		img = cv2.imread(imgFile)[...,::-1]
		print(img.shape)
		maskFile = imgFile.split(".tif")[0] + '.png'
		maskFile = maskFile.replace("/images/","/masks/")
		org_mask = cv2.imread(maskFile, 0)
  
	except:
		return -1
	H,W,C = img.shape
	results = nmodel.detect([img], verbose=0)
	r = results[0]
	rScores = r['scores']
	if rScores.size==0:
		maxScore = 0
	else:
		maxScore = np.amax(rScores)
	rMasks = r['masks']
	if rMasks.size==0:
		predMask = np.zeros((H,W), dtype=np.uint8)
		drawonPred = img.copy()
	else:
		predMask = np.amax(rMasks, axis=-1).astype(np.uint8)*255
		drawonPred = myutils.add_color_mask(img, predMask, (0,255,0))
	
	h, w = predMask.shape 
	
	iou, recall = caulate(org_mask, predMask)
	

	pred = predMask.ravel()#图像矩阵归一化
	label = org_mask.ravel()
	save_image = []
	
	

 
	# 
	# assert (np.unique(label).tolist == [0, 255]), 'label mask [0, 255]'
	# assert (np.unique(pred).tolist == [0, 255]), 'pred mask [0, 255]'
	# 
	use_img = img.reshape(h * w, -1)
	#print(use_img.shape) (692224,3)
	
	tp = 0
	fp=0
	fn=0
	tn=0
	
	for i in range(len(label)):
		if label[i] == pred[i] and label[i] == 255:
			tp += 1
			save_image.append([255, 0, 0])  #blue
		elif label[i] == pred[i] and label[i] == 0:
			tn+=1
			# import pdb;pdb.set_trace()
			save_image.append(use_img[i][::-1]) #原图
		elif label[i] != pred[i] and label[i] == 255:
			fp+=1
			save_image.append([0, 0, 255]) #red
		elif label[i] != pred[i] and label[i] == 0:
			fn+=1
			save_image.append([0, 255, 0]) #green
		else:
			save_image.append([0, 255, 255]) #yellow
			print('unknow pixel:', label[i], pred[i])
   
	total=tp+tn+fp+fn

	print(np.array(save_image).shape)
 	#print
	save_img = np.reshape(np.array(save_image), (h, w, 3))
	result_path = maskFile.replace('/masks/','/results/')
	print(result_path, iou, recall)
 
	positive_rate = tp / (tp + fp)  #recall
	print('{} positive rate: {}'.format(result_path, positive_rate))
	os.makedirs(result_path.split('.png')[0], exist_ok=True)
	cv2.imwrite(result_path, save_img)
	cv2.imwrite(saveMaskFile, predMask)
	cv2.imwrite(saveResultFile, drawonPred[...,::-1]) #[...,::-1]
	return maxScore





if __name__ == "__main__":
	n_model=load_models()
	# print('model',n_model)
	img_path="test_datatrading_2//test_data_traing//test_data//CD11b//In-8-5-D3-CD11b-4-right//images//In-8-5-D3-CD11b-4_right.tif"
	save_mask_path="test_datatrading_2//test_data_traing//test_data\CD11b\In-8-5-D3-CD11b-4-right//results//005.jpg"
	save_result_path="test_datatrading_2//test_data_traing//test_data\CD11b\In-8-5-D3-CD11b-4-right//results//006.jpg"
	pred(n_model,img_path,save_mask_path,save_result_path)
	

# %%



