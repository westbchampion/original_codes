"""
Mask R-CNN for one class segmentation.
"""
import shutil
import os
import sys
from glob import glob
from math import ceil
import numpy as np
import skimage.io
from imgaug import augmenters as iaa
# Root directory of the project
ROOT_DIR = os.path.abspath("./")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from utilities import utils as myutils
from keras.callbacks import ModelCheckpoint
from keras import backend as K
K.set_learning_phase(True)
import tensorflow as tf
if tf.__version__ >= '2.0.0':
    tf.compat.v1.disable_eager_execution()
os.environ['CUDA_VISIBLE_DEVICES']='2'

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "pre_trained_model", "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
iter_num = 0
# Ratio of images to be used as training data
TRAIN_RATIO = 0.8235

############################################################
#  Configurations
############################################################

def get_config(imgNum, cpuCount, imgLen, configTag):
	class SegConfig(Config):
		"""Configuration for training on the segmentation dataset."""
		#print(imgNum)
		IMG_NUM = imgNum
		assert IMG_NUM != 0, f'img number is 0, recheck the path'

		# Training image number
		TRAIN_NUM = ceil(IMG_NUM*TRAIN_RATIO)

		# Validation image number
		VAL_NUM = IMG_NUM - TRAIN_NUM

		# Give the configuration a recognizable name
		NAME = "shapes"

		# Number of CPUs to use for `fit_generator` in the situation of
		# multi-thread processing.
		CPU_COUNT = cpuCount

		# Adjust depending on your GPU memory
		IMAGES_PER_GPU = 1

		# Number of classes (including background)
		NUM_CLASSES = 1 + 1  # Background + targets

		# Number of training and validation steps per epoch
		STEPS_PER_EPOCH = ceil((IMG_NUM - VAL_NUM)/IMAGES_PER_GPU)
		print(f'step per epoch for mrcnn seg.py {STEPS_PER_EPOCH}')
		VALIDATION_STEPS = max(1, ceil(VAL_NUM/IMAGES_PER_GPU))

		# Don't exclude based on confidence. Since we have two classes
		# then 0.5 is the minimum anyway as it picks between targets and BG
		DETECTION_MIN_CONFIDENCE = 0

		# Backbone network architecture
		# Supported values are: resnet50, resnet101
		BACKBONE = "resnet50"

		# Input image resizing
		# Random crops of size 512x512
		IMAGE_RESIZE_MODE = "crop"
		IMAGE_MIN_DIM = imgLen
		IMAGE_MAX_DIM = imgLen
		IMAGE_MIN_SCALE = 1.0

		# Length of square anchor side in pixels
		RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

		# ROIs kept after non-maximum supression (training and inference)
		POST_NMS_ROIS_TRAINING = 200
		POST_NMS_ROIS_INFERENCE = 200

		# Non-max suppression threshold to filter RPN proposals.
		# You can increase this during training to generate more propsals.
		RPN_NMS_THRESHOLD = 0.9

		# How many anchors per image to use for RPN training
		RPN_TRAIN_ANCHORS_PER_IMAGE = 64

		# Image mean (RGB)
		MEAN_PIXEL = np.array([165.26, 91.39, 67.67])

		# If enabled, resizes instance masks to a smaller size to reduce
		# memory load. Recommended when using high-resolution images.
		USE_MINI_MASK = True
		MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

		# Number of ROIs per image to feed to classifier/mask heads
		# The Mask RCNN paper uses 512 but often the RPN doesn't generate
		# enough positive proposals to fill this and keep a positive:negative
		# ratio of 1:3. You can increase the number of proposals by adjusting
		# the RPN NMS threshold.
		TRAIN_ROIS_PER_IMAGE = 128

		# Maximum number of ground truth instances to use in one image
		# MAX_GT_INSTANCES = 400
		MAX_GT_INSTANCES = 200

		# Max number of final detections per image
		# DETECTION_MAX_INSTANCES = 400
		DETECTION_MAX_INSTANCES = 200

	class SegInferenceConfig(SegConfig):
		# Set batch size to 1 to run one image at a time
		GPU_COUNT = 1
		IMAGES_PER_GPU = 1
		# Don't resize imager for inferencing
		IMAGE_RESIZE_MODE = "pad64"
		# Non-max suppression threshold to filter RPN proposals.
		# You can increase this during training to generate more propsals.
		RPN_NMS_THRESHOLD = 0.7

	if configTag=='train':
		return SegConfig()
	elif configTag=="detect":
		return SegInferenceConfig()


############################################################
#  Dataset
############################################################

class SegDataset(utils.Dataset):

	def load_seg(self, dataset_dir, subset, seed=12345):
		"""Load a subset of the segmenta dataset.

		Arguments:

			dataset_dir (str): Root directory of the dataset

			subset (str): Subset to load. Either the name of the sub-directory,
					such as stage1_train, stage1_test, ...etc. or, one of:
					* train: stage1_train excluding validation images
					* val: validation images from VAL_IMAGE_IDS
					
			seed (int): Seed for random number generator"""
		# Add classes. We have one class.
		# Naming the dataset lesion, and the class lesion
		self.add_class("shapes", 1, "lesion")
			
		def traverse_folders(path):
			total_dir = []
			# 遍历文件夹
			for subfolders in os.listdir(path):
				for sub_f in os.listdir(os.path.join(path,subfolders)):
					if len(os.listdir(os.path.join('.\\mrcnn_data',subfolders, sub_f,'images')))==0 or len(os.listdir(os.path.join('.\\mrcnn_data',subfolders, sub_f,'masks')))==0:
						shutil.rmtree(os.path.join('.\\mrcnn_data',subfolders, sub_f))
						break
					total_dir.append(os.path.join(subfolders, sub_f))
					print("子文件夹：", os.path.join(subfolders, sub_f))
			return total_dir
		imgIds = []
		# lis = next(os.walk(args.dataset))[1]
		lis = traverse_folders(args.dataset)
		# print(lis)
		for s in lis:
			# # if s.startswith('I'):
			# if s[:-5] == 'masks' or s[:-6] == 'images':
			# 	continue
			imgIds.append(s)
		#noLesionTrainIds, noLesionValIds = myutils.random_div_train_val(
			#noLesionImgIds, TRAIN_RATIO, seed
		trainIds, valIds = myutils.random_div_train_val(
		 imgIds, TRAIN_RATIO, seed
		 )
		#trainIds = lesionTrainIds + noLesionTrainIds
		#valIds = lesionValIds + noLesionValIds
		if subset == "val":
			image_ids = valIds
		elif subset == "train":
			image_ids = trainIds
		# Add images
		#这里没有问题
		for i,image_id in enumerate(image_ids):
			imgPath = f'{dataset_dir}/{image_id}/images'
			imgFiles = glob(f'{imgPath}/*.tif')
			if len(imgFiles)==0:
				print(f'`{imgPath}` has no image.')
				continue
			else:
				for i in range(len(imgFiles)):	
					self.add_image(
					"shapes",
					image_id=image_id,
					path=imgFiles[i])#path=os.path.join(dataset_dir, s, "images/{}.tif".format(image_id)))

	def load_mask(self, image_id):
		info = self.image_info[image_id]
		#print(os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks"))
		mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks/")
		# print(mask_dir)
		# print(next(os.walk(mask_dir))[2])
		mask = []
		for f in next(os.walk(mask_dir))[2]:

			if f.endswith(".png"):
				#print(os.path.join(mask_dir, f))
				m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
				mask.append(m)
		#print(len(mask))
		
		mask = np.stack(mask, axis=-1)
		# Return mask, and array of class IDs of each instance. Since we have
		# one class ID, we return an array of ones
		return mask, np.ones([mask.shape[-1]], dtype=np.int32) 
	def image_reference(self, image_id):
		"""Return the path of the image."""
		info = self.image_info[image_id]
		if info["source"] == "shapes":
			return info["id"]
		else:
			super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################

def train(model, dataset_dir):
	"""Train the model."""
	# Training dataset.
	dataset_train = SegDataset()
	dataset_train.load_seg(dataset_dir, "train")
	dataset_train.prepare()

	# Validation dataset
	dataset_val = SegDataset()
	dataset_val.load_seg(dataset_dir, "val")
	dataset_val.prepare()

	# Image augmentation
	# http://imgaug.readthedocs.io/en/latest/source/augmenters.htmltrainmode
	augmentation = iaa.SomeOf((0, 2), [
		iaa.Fliplr(0.5),
		iaa.Flipud(0.5),
		iaa.OneOf([iaa.Affine(rotate=90),
				   iaa.Affine(rotate=180),
				   iaa.Affine(rotate=270)]),
		iaa.Multiply((0.8, 1.5)),
		iaa.GaussianBlur(sigma=(0.0, 5.0))
	])

	# *** This training schedule is an example. Update to your needs ***

	# If starting from imagenet, train heads only for a bit
	# since they have random weights
	# print("Train network heads")
	# model.train(dataset_train, dataset_val,
	# 			learning_rate=config.LEARNING_RATE,
	# 			epochs=20,
	# 			augmentation=augmentation,
	# 			layers='heads')

	print("Train all layers")
	# checkpoint = ModelCheckpoint('./models/', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
	# mean_average_precision_callback = modellib.MeanAveragePrecisionCallback(model, model_inference, dataset_val, calculate_map_at_every_X_epoch=5, verbose=1)
	
	checkpoint_filepath = './models/best_model_all.h5'

	# Create a custom callback to save the best model
	class SaveBestModelCallback(ModelCheckpoint):
		def __init__(self, filepath):
			super(SaveBestModelCallback, self).__init__(filepath, monitor='val_loss', save_weights_only=True,  save_best_only=True, mode='min', verbose=1)

	model.train(dataset_train, dataset_val,
				learning_rate=config.LEARNING_RATE,
				epochs=25,
				augmentation=augmentation,layers="all",model_path='pre_trained_model\mask_rcnn_coco.h5',
				custom_callbacks = [SaveBestModelCallback(checkpoint_filepath)]
    			)
############################################################
#  Command Line
############################################################

if __name__ == '__main__':
	import argparse
 

	# Parse command line arguments
	parser = argparse.ArgumentParser(
		description='Mask R-CNN for segmentation'
	)
	parser.add_argument(
		"--command",  required=False, metavar="<command>", help="'train' or 'detect'",  type=str, default='train'
	)
	parser.add_argument(
		'--dataset', required=False, metavar="./mrcnn_data", type=str, default='C://code//test//mrcnn_data',
		help='Root directory of the dataset'
	)
	parser.add_argument(
		'--weights', required=False, metavar="/path/to/weights.h5", type=str, default='coco',
		help="Path to weights .h5 file or 'coco'"
	)
	parser.add_argument(
		'--cpuCount', required=False, metavar="CPU_COUNT", default=4, type=int,
		help="CPU number to use for the multi-thread processing"
	)
	parser.add_argument(
		'--imgLen', required=False, metavar="IMAGE_LENGTH", default=512, type=int,
		help="Input image length"
	)
	parser.add_argument(
		'--logs', required=False, default=DEFAULT_LOGS_DIR,
		metavar="/path/to/logs/", type=str,
		help='Logs and checkpoints directory (default=logs/)'
	)
	parser.add_argument(
		'--subset', required=False, metavar="Dataset sub-directory", type=str, default='C://code//test//test_data',
		help="Subset of dataset to run prediction on"
	)
	args = parser.parse_args()

	# Validate arguments
	if args.command == "train":
		assert args.dataset, "Argument --dataset is required for training"
	elif args.command == "detect":
		assert args.subset, "Provide --subset to run prediction on"

	print("Weights: ", args.weights)
	print("Dataset: ", args.dataset)
	if args.subset:
		print("Subset: ", args.subset)
	print("Logs: ", args.logs)

	
	
#这里没有问题
	#print(imageIds)
	def traverse_folders(path):
		total_dir = []
		# 遍历文件夹
		for subfolders in os.listdir(path):
			for sub_f in os.listdir(os.path.join(path,subfolders)):
				total_dir.append(os.path.join(subfolders, sub_f))
				print("子文件夹：", os.path.join(subfolders, sub_f))
		return total_dir

	imageIds=[]
	lis=[]
	lis = traverse_folders(args.dataset)
	# print(lis)
	# # lis=next(os.walk(args.dataset))[1]
	# print(len(lis))
	for s in lis:
		# if s.startswith('I'):
		imageIds.append(s)
	# print(imageIds)
	imgNum = len(imageIds)
	if args.command == "train":
		config = get_config(imgNum, args.cpuCount, args.imgLen, 'train')
	elif args.command == "detect":
		config = get_config(imgNum, args.cpuCount, args.imgLen, 'detect')
	config.display()

	# Create model
	if args.command == "train":
		model = modellib.MaskRCNN(mode="training", config=config,
								  model_dir=args.logs)
	else:
		model = modellib.MaskRCNN(mode="inference", config=config,
								  model_dir=args.logs)

	# Select weights file to load
	if args.weights.lower() == "coco":
		weights_path = COCO_WEIGHTS_PATH
		# Download weights file
		if not os.path.exists(weights_path):
			utils.download_trained_weights(weights_path)
	elif args.weights.lower() == "last":
		# Find last trained weights
		weights_path = model.find_last()
	elif args.weights.lower() == "imagenet":
		# Start from ImageNet trained weights
		weights_path = model.get_imagenet_weights()
	else:
		weights_path = args.weights

	# Load weights
	print("Loading weights ", weights_path)
	if args.weights.lower() == "coco":
		# Exclude the last layers because they require a matching
		# number of classes
		model.load_weights(weights_path, by_name=True, exclude=[
			"mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"])
	else:
		model.load_weights(weights_path, by_name=True)

	# Train or evaluate
	if args.command == "train":
		train(model, args.dataset)
	elif args.command == "detect":
		pass
	else:
		print("'{}' is not recognized. "
			  "Use 'train' or 'detect'".format(args.command))
    # save weights

