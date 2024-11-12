import json
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np


def random_div_train_val(listInput, trainRatio, seed=12345):
	"""Shuffle the list's components and divide it into two lists of training
	and validation sets.
	
	Arguments:
	
		listInput (list): list of input.
		
		firstRatio (float): ratio of the train list's components accounting for
		the total components in the input list.

		seed (int): random seed.
		
	Returns:
	
		list, list: two divided list."""
	imgNum = len(listInput)
	trainNum = int(imgNum*trainRatio)
	idxArr = np.arange(len(listInput))
	np.random.seed(seed)
	np.random.shuffle(idxArr)
	trainList = [listInput[i] for i in idxArr[:trainNum]]
	valList = [listInput[i] for i in idxArr[trainNum:]]
	return trainList, valList


def get_files_by_format(fpath, format):
	"""
	Get all the file names with the specific format `format` in directory `fpath`.
	
	Arguments:
		fpath {str} -- the files' folder path.
		format {str} -- the specific format of the interested file without the prefixed dot.

	Returns:
		list of str -- all the file names.
		int -- the total number of the interested files.
	"""
	s_files = os.listdir(fpath)
	files = [x for x in s_files if x.split('.')[-1].upper()==format.upper()]
	fnum = len(files)
	return files, fnum


def write_json(data, json_file):
	"""
	Write dictionary to json file.

	Arguments:
		data {dict} -- input data.
		json_file {str} -- path to the json file.
	"""
	with open(json_file, 'w') as jf:
		json.dump(data, jf)


def read_json(json_file):
	"""
	Read data from json file.

	Arguments:
		json_file {str} -- path to the json file.

	Returns:
		dict -- data from json file.
	"""
	with open(json_file) as f:
		data = json.load(f)
	return data


def same_dict(dict1, dict2):
	"""
	Compare two dictionaries. If the two dictionaries are the same, return True, else return False.

	Arguments:
		dict1 {dict} -- dictionary one.
		dict2 {dict} -- dictionary two.

	Returns:
		res {bool} -- True if the two dictionaries are the same. False otherwise.
	"""
	if dict1.keys()==dict2.keys():
		keys = dict1.keys()
	else:
		return False
	for key in keys:
		if dict1[key] != dict2[key]:
			return False
	return True


def make_sure_dir_exist(dir_):
	"""
	Make sure the directory exists.

	Arguments:
		dir_ {str} -- path to the directory.

	Returns:
		str -- the path to the directory which exists now.
	"""
	if not os.path.isdir(dir_):
		os.makedirs(dir_)
	return dir_


def get_img_files(img_path):
	"""
	Get all the image file names in directory img_path.

	Arguments: 
		img_path {str} -- path of the image directory.

	Returns:
		list of str -- all the image names. 
		int -- the total number of the image files in the directory. 
	"""
	img_formats = ['JPG','PNG','TIF','BMP','TIFF']
	s_files = os.listdir(img_path)
	img_files = [x for x in s_files if x.split('.')[-1].upper() in img_formats]
	img_num = len(img_files)
	return img_files, img_num


def get_polygon_coords(ann_file, layer_tag, img_size):
	"""
	Get the coordinates of the boundary points of each polygon from an `ImageScope` annotation xml file.
	
	Arguments:
		ann_file {str} -- path of the annotation file.
		layer_tag {str} -- tag name of the layer that contains all the polygon annotations.
		img_size {tuple of ints} -- size of the image where the annotations are drawn on. Format: (height, width).

	Returns:
		dict -- key is the number of each polygon, an int. value is the points coordinates, a 2D list of int, its shape is Nx2, where N is the number of all the points in the polygon. Each point's format: (x,y).
	"""
  # ann_file = os.path.join('./source_img/', name+'.xml')
	H, W = img_size
	tree = ET.parse(ann_file)
	pts = {}
	for ann in tree.iter('Annotation'):
		if ann.get('Name')==layer_tag:
			for reg in ann.iter('Region'):
				Id = int(float(reg.get('Id')))
				pt = []
				for vertex in reg.iter('Vertex'):
					x = min(max(int(float(vertex.get('X'))), 0), W-1)
					y = min(max(int(float(vertex.get('Y'))), 0), H-1)
					pt.append([x,y])
				pts[Id] = pt
	return pts


def fill_polygon_mask(img, contour, color):
	"""
	Make mask for the input image based on the input polygon contour coordinates.

	Argument:
		img {ndarray} - input image. Its rank could be 2 or 3.
		contour {list of ndarray} - array of polygons. Each ndarray is one polygon. Each polygon contains all of its coordinates (x,y).
		color {tuple of int} - color of the foreground of the mask. Its format is (Blue, Green, Red).
	
	Returns:
		{ndarray} - output mask. It has the same shape with the input.
	"""
	mask = np.zeros_like(img)
	mask = cv2.fillPoly(mask, contour, color)
	return mask


def overlay_heatmap(grayImg, img, alpha=0.3, beta=1, threshold=None):
	"""Overlay the grayscale image as heatmap over the image via weighted 
	addition.
	
	Arguments:
	
		grayImg (numpy.ndarray): grayscale image. dtype: numpy.uint8.
		
		img (numpy.ndarray): RGB image.

		alpha (float): weight of heatmap when add the heatmap and the image.
		Default: 0.3.

		beta (float): weight of image when add the heatmap and the image. 
		Default: 1.

		threshold (int): threshold value to determine which pixels should be
		overlay with heatmap. If pixels in `grayImg` greater than this argument,
		it will be overlay with heatmap, otherwise not. If None, overlay the
		whole heatmap on top of the image. Default: None.
		
	Returns:
	
		drawon (numpy.ndarray): input image with heatmap applied."""
	heatMask = cv2.applyColorMap(grayImg, cv2.COLORMAP_JET)
	drawon = img.copy()
	heatMask = cv2.cvtColor(heatMask, cv2.COLOR_BGR2RGB)
	if threshold is not None:
		threshMask = np.greater(grayImg, threshold)
		for i in range(heatMask.shape[-1]):
			heatMask[...,i] = heatMask[...,i] * threshMask
	drawon = cv2.addWeighted(heatMask, alpha, drawon, beta, 0)
	return drawon


def add_color_mask(img, gt, clr, alpha=0.3, is_target_white=False, val_replace_white=128):
	"""
	Add semi-transparent colored mask on one image.
	
	Arguments:

		img {numpy array} -- BGR image, which needs addition of semi-transparent 
		colored mask.

		gt {numpy array} -- rank 2. 8-bit single channel array, that specifies 
		elements of the output array to be changed. The target pixel value is 
		255, the background pixel value is 0.

		clr {tuple} -- 3 int that specifies the color of the added mask, in the 
		color sequence of BGR. Each channel's value range is [0, 255].

		alpha {float} -- control the indensity of the semi-transparent colored 
		mask. Range is [0,1]. 0 means totally transpanrent, 1 means totally 
		opaque.

		is_target_white {bool} -- True: the targets are white pixels. False: 
		the target are not white pixels. If the targets are white pixels, the 
		colored mask cannot be added onto them. Therefore, if the targets are 
		white pixels, these pixels will be set to zero before the addWeighted 
		operation.

		val_replace_white {int} -- value to replace the white pixels. Range is 
		[0,255]. Use this argument and `alpha` together to control the white 
		target's appearance after adding the colored mask.
	
	Returns:

		numpy array -- masked image, has the same shape as the input image.
	"""
	color_img = np.zeros(img.shape, img.dtype)
	color_img[:,:] = clr
	color_mask = cv2.bitwise_and(color_img, color_img, mask=gt)
	if is_target_white:
		target_index = np.nonzero(gt)
		img[target_index] = val_replace_white
	masked_img = cv2.addWeighted(
		color_mask, alpha=alpha, src2=img, beta=1, gamma=0
	)
	return masked_img


