
import numpy as np

from seg_models import unetFull
from seg_models import Transformer
from datafunctions import *
from losses import *
from CheckpointSelector import *
from customAccuracy import *

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import tensorflow_probability as tfp

import cvnn.layers as complex_layers


from tqdm import tqdm
import json
import tensorflow as tf


'''
Main class that creates the model and can instantiate a model for training, testing etc.
constructor needs a set of parameters only to create the Baseline Object.

###############
Methods:
train: Method called to train a model. This function will call methods image_generator, get_model and build_model to create the model before calling the fit to train the model.
type_train -  determine if it is a train from scratch or an active learning train.

###############
infer_MCDropOut: Method to produces the MCDropout realizations. This function is called from within the Method predict_drop.
model - model used in the inference
img - numpy array that will be fed to the model
num_inter - number of realization to be done in the MCDropOut

###############
predict_drop: Method that is called to test the generalization test data for the MCDropout inference. the function uses the method infer_MCDropOut and it also calls get_model, build_model and image_generator to produce the results.

###############
finalMap: Method that produces the new training images for an active learning model (tested, but results were not good)
imgT - numpy array of the predicted image
mcluT - numpy array of the MCLU of the prediced image
pred_p - numpy array with the probabilities of the classes in the prediced image
pred_i - numpy array of the probability of the resulting class in the prediced image
stdImgT - numpy array with the standard deviation of the classes in the prediced image
imgTa - reference image in the previous iteration
probimgTa - probability of the class in the previous iteration

###############
save_images: MEthod to save all the images produced by the refine training dataset.
pred_p - numpy array with probability of the predictions
pred_std - numpy array with standard deviation of the predictions
imageName - string with name of the original image
outName - string with name of the output image
outReplaceStr - string for output replacemente
train_valid - to say if it is train or validation.

###############
refineTrainDataset: Method that is called to change the training data based on MCDropout and produce a new refined training data based on the MCDropout reference. (tested, but results were very bad).

###############
test: Method that is called to test the generalization test data for a deterministic result, based on the best weights obtained in the training. Method calls get_model, build_model and image_generator to produce the results.

###############
get_model: Method to define the architecture of the model to be used. This is predefined by the combination in the PARAMS file (UnetParams.py) by deciding the DATA_USAGE and the NETWORK. However it is advised to look into the function since there are more options commented on it.

###############
build_model: Method that sets the loss function, optimization and compiles the model architecture defined in the method get_model. one should check this function to decide which loss function to use, which type of weights for the classes, etc.
Options are commented.
type_train - if the training is from scratch or from a refined dataset (active learning) - tested but not good results

###############
image_generator: Method that generates the batches for the training and validation.
trainData - list of data (training or validation)
type_train - if the training is from scratch or from a refined dataset (active learning) - tested but not good results
isValid -  if it is validation or training
'''


class Baseline:
	def __init__(self, params=None):

		self.params=params

		# TensorFlow allocates all GPU memory up front by default, so turn that off
		gpus = tf.config.list_physical_devices('GPU')
		if gpus:
			try:
    		# Currently, memory growth needs to be the same across GPUs
				for gpu in gpus:
					tf.config.experimental.set_memory_growth(gpu, True)
				logical_gpus = tf.config.list_logical_devices('GPU')
				print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
			except RuntimeError as e:
    		# Memory growth must be set before GPUs have been initialized
				print(e)

	def train (self, type_train):
		train_data = []
		valid_data = []
		if self.params.DATA_USAGE == 'S1_I_only':
			image_paths, valid_img_paths = get_image_paths(self.params, typeImg='S1', isTest=False, type_train=type_train)
			for imgPath in image_paths:
				train_data.append((imgPath, 0, 0))
			for imgPath in valid_img_paths:
				valid_data.append((imgPath, 0, 0))
		elif self.params.DATA_USAGE == 'S1_only':
			image_paths, valid_img_paths = get_image_paths(self.params, typeImg='S1', isTest=False, type_train=type_train)
			for imgPath in image_paths:
				train_data.append((imgPath, 0, 0))
			for imgPath in valid_img_paths:
				valid_data.append((imgPath, 0, 0))
		elif self.params.DATA_USAGE == 'S1_Complex':
			image_paths, valid_img_paths = get_image_paths(self.params, typeImg='S1', isTest=False, type_train=type_train)
			for imgPath in image_paths:
				train_data.append((imgPath, 0, 0))
			for imgPath in valid_img_paths:
				valid_data.append((imgPath, 0, 0))
		elif self.params.DATA_USAGE == 'S2_S1_Complex':
			image_paths, valid_img_paths = get_image_paths(self.params, typeImg='S1', isTest=False, type_train=type_train)
			for imgPathS1 in image_paths:
				imageName = os.path.split(imgPathS1)[-1]
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TRAIN_S2_DIR, imgpathS2)
				train_data.append((imgPathS2, imgPathS1, 0, 0))
			for imgPathS1 in valid_img_paths:
				imageName = os.path.split(imgPathS1)[-1]
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.VALID_S2_DIR, imgpathS2)
				valid_data.append((imgPathS2, imgPathS1, 0, 0))
		elif self.params.DATA_USAGE == 'S2_only':
			image_paths, valid_img_paths = get_image_paths(self.params, typeImg='S2', isTest=False, type_train=type_train)
			for imgPath in image_paths:
				train_data.append((imgPath, 0, 0))
			for imgPath in valid_img_paths:
				valid_data.append((imgPath, 0, 0))
		elif self.params.DATA_USAGE == 'S1S2All':
			image_paths, valid_img_paths = get_image_paths(self.params, typeImg='S1', isTest=False, type_train=type_train)
			for imgPathS1 in image_paths:
				imageName = os.path.split(imgPathS1)[-1]
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TRAIN_S2_DIR, imgpathS2)
				train_data.append((imgPathS2, imgPathS1, 0, 0))
			for imgPathS1 in valid_img_paths:
				imageName = os.path.split(imgPathS1)[-1]
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.VALID_S2_DIR, imgpathS2)
				valid_data.append((imgPathS2, imgPathS1, 0, 0))
		elif self.params.DATA_USAGE == 'S2S1IAll':
			image_paths, valid_img_paths = get_image_paths(self.params, typeImg='S1', isTest=False, type_train=type_train)
			for imgPathS1 in image_paths:
				imageName = os.path.split(imgPathS1)[-1]
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TRAIN_S2_DIR, imgpathS2)
				train_data.append((imgPathS2, imgPathS1, 0, 0))
			for imgPathS1 in valid_img_paths:
				imageName = os.path.split(imgPathS1)[-1]
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.VALID_S2_DIR, imgpathS2)
				valid_data.append((imgPathS2, imgPathS1, 0, 0))
		elif self.params.DATA_USAGE == 'S1_S2_Sep':
			image_paths, valid_img_paths = get_image_paths(self.params, typeImg='S1', isTest=False, type_train=type_train)
			for imgPathS1 in image_paths:
				imageName = os.path.split(imgPathS1)[-1]
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TRAIN_S2_DIR, imgpathS2)
				train_data.append((imgPathS2, imgPathS1, 0, 0))
			for imgPathS1 in valid_img_paths:
				imageName = os.path.split(imgPathS1)[-1]
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.VALID_S2_DIR, imgpathS2)
				valid_data.append((imgPathS2, imgPathS1, 0, 0))
		elif self.params.DATA_USAGE == 'S1I_S2_Sep':
			image_paths, valid_img_paths = get_image_paths(self.params, typeImg='S1', isTest=False, type_train=type_train)
			for imgPathS1 in image_paths:
				imageName = os.path.split(imgPathS1)[-1]
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TRAIN_S2_DIR, imgpathS2)
				train_data.append((imgPathS2, imgPathS1, 0, 0))
			for imgPathS1 in valid_img_paths:
				imageName = os.path.split(imgPathS1)[-1]
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.VALID_S2_DIR, imgpathS2)
				valid_data.append((imgPathS2, imgPathS1, 0, 0))
		elif self.params.DATA_USAGE == 'S1IV_S2_Sep':
			image_paths, valid_img_paths = get_image_paths(self.params, typeImg='S1', isTest=False, type_train=type_train)
			for imgPathS1 in image_paths:
				imageName = os.path.split(imgPathS1)[-1]
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TRAIN_S2_DIR, imgpathS2)
				train_data.append((imgPathS2, imgPathS1, 0, 0))
			for imgPathS1 in valid_img_paths:
				imageName = os.path.split(imgPathS1)[-1]
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.VALID_S2_DIR, imgpathS2)
				valid_data.append((imgPathS2, imgPathS1, 0, 0))
		elif self.params.DATA_USAGE == 'S2F_only':
			image_paths, valid_img_paths = get_image_paths(self.params, typeImg='S2', isTest=False, type_train=type_train)
			for imgPath in image_paths:
				train_data.append((imgPath, 0, 0))
			for imgPath in valid_img_paths:
				valid_data.append((imgPath, 0, 0))
		elif self.params.DATA_USAGE == 'S1F_only':
			image_paths, valid_img_paths = get_image_paths(self.params, typeImg='S1', isTest=False, type_train=type_train)
			for imgPath in image_paths:
				train_data.append((imgPath, 0, 0))
			for imgPath in valid_img_paths:
				valid_data.append((imgPath, 0, 0))
		elif self.params.DATA_USAGE == 'S1IF_only':
			image_paths, valid_img_paths = get_image_paths(self.params, typeImg='S1', isTest=False, type_train=type_train)
			for imgPath in image_paths:
				train_data.append((imgPath, 0, 0))
			for imgPath in valid_img_paths:
				valid_data.append((imgPath, 0, 0))
		elif self.params.DATA_USAGE == 'S1IVF_only':
			image_paths, valid_img_paths = get_image_paths(self.params, typeImg='S1', isTest=False, type_train=type_train)
			for imgPath in image_paths:
				train_data.append((imgPath, 0, 0))
			for imgPath in valid_img_paths:
				valid_data.append((imgPath, 0, 0))
		elif self.params.DATA_USAGE == 'S1IV_only':
			image_paths, valid_img_paths = get_image_paths(self.params, typeImg='S1', isTest=False, type_train=type_train)
			for imgPath in image_paths:
				train_data.append((imgPath, 0, 0))
			for imgPath in valid_img_paths:
				valid_data.append((imgPath, 0, 0))
		elif self.params.DATA_USAGE == 'S1IS2FAll':
			image_paths, valid_img_paths = get_image_paths(self.params, typeImg='S1', isTest=False, type_train=type_train)
			for imgPathS1 in image_paths:
				imageName = os.path.split(imgPathS1)[-1]
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TRAIN_S2_DIR, imgpathS2)
				train_data.append((imgPathS2, imgPathS1, 0, 0))
			for imgPathS1 in valid_img_paths:
				imageName = os.path.split(imgPathS1)[-1]
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.VALID_S2_DIR, imgpathS2)
				valid_data.append((imgPathS2, imgPathS1, 0, 0))
		elif self.params.DATA_USAGE == 'S1IFS2FAll':
			image_paths, valid_img_paths = get_image_paths(self.params, typeImg='S1', isTest=False, type_train=type_train)
			for imgPathS1 in image_paths:
				imageName = os.path.split(imgPathS1)[-1]
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TRAIN_S2_DIR, imgpathS2)
				train_data.append((imgPathS2, imgPathS1, 0, 0))
			for imgPathS1 in valid_img_paths:
				imageName = os.path.split(imgPathS1)[-1]
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.VALID_S2_DIR, imgpathS2)
				valid_data.append((imgPathS2, imgPathS1, 0, 0))
		elif self.params.DATA_USAGE == 'S1S2FAll':
			image_paths, valid_img_paths = get_image_paths(self.params, typeImg='S1', isTest=False, type_train=type_train)
			for imgPathS1 in image_paths:
				imageName = os.path.split(imgPathS1)[-1]
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TRAIN_S2_DIR, imgpathS2)
				train_data.append((imgPathS2, imgPathS1, 0, 0))
			for imgPathS1 in valid_img_paths:
				imageName = os.path.split(imgPathS1)[-1]
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.VALID_S2_DIR, imgpathS2)
				valid_data.append((imgPathS2, imgPathS1, 0, 0))
		elif self.params.DATA_USAGE == 'S1FS2FAll':
			image_paths, valid_img_paths = get_image_paths(self.params, typeImg='S1', isTest=False, type_train=type_train)
			for imgPathS1 in image_paths:
				imageName = os.path.split(imgPathS1)[-1]
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TRAIN_S2_DIR, imgpathS2)
				train_data.append((imgPathS2, imgPathS1, 0, 0))
			for imgPathS1 in valid_img_paths:
				imageName = os.path.split(imgPathS1)[-1]
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.VALID_S2_DIR, imgpathS2)
				valid_data.append((imgPathS2, imgPathS1, 0, 0))
		elif self.params.DATA_USAGE == 'S1_S2F_Sep':
			image_paths, valid_img_paths = get_image_paths(self.params, typeImg='S1', isTest=False, type_train=type_train)
			for imgPathS1 in image_paths:
				imageName = os.path.split(imgPathS1)[-1]
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TRAIN_S2_DIR, imgpathS2)
				train_data.append((imgPathS2, imgPathS1, 0, 0))
			for imgPathS1 in valid_img_paths:
				imageName = os.path.split(imgPathS1)[-1]
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.VALID_S2_DIR, imgpathS2)
				valid_data.append((imgPathS2, imgPathS1, 0, 0))
		elif self.params.DATA_USAGE == 'S1I_S2F_Sep':
			image_paths, valid_img_paths = get_image_paths(self.params, typeImg='S1', isTest=False, type_train=type_train)
			for imgPathS1 in image_paths:
				imageName = os.path.split(imgPathS1)[-1]
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TRAIN_S2_DIR, imgpathS2)
				train_data.append((imgPathS2, imgPathS1, 0, 0))
			for imgPathS1 in valid_img_paths:
				imageName = os.path.split(imgPathS1)[-1]
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.VALID_S2_DIR, imgpathS2)
				valid_data.append((imgPathS2, imgPathS1, 0, 0))
		elif self.params.DATA_USAGE == 'S1IF_S2F_Sep':
			image_paths, valid_img_paths = get_image_paths(self.params, typeImg='S1', isTest=False, type_train=type_train)
			for imgPathS1 in image_paths:
				imageName = os.path.split(imgPathS1)[-1]
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TRAIN_S2_DIR, imgpathS2)
				train_data.append((imgPathS2, imgPathS1, 0, 0))
			for imgPathS1 in valid_img_paths:
				imageName = os.path.split(imgPathS1)[-1]
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.VALID_S2_DIR, imgpathS2)
				valid_data.append((imgPathS2, imgPathS1, 0, 0))
		elif self.params.DATA_USAGE == 'S1IVF_S2F_Sep':
			image_paths, valid_img_paths = get_image_paths(self.params, typeImg='S1', isTest=False, type_train=type_train)
			for imgPathS1 in image_paths:
				imageName = os.path.split(imgPathS1)[-1]
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TRAIN_S2_DIR, imgpathS2)
				train_data.append((imgPathS2, imgPathS1, 0, 0))
			for imgPathS1 in valid_img_paths:
				imageName = os.path.split(imgPathS1)[-1]
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.VALID_S2_DIR, imgpathS2)
				valid_data.append((imgPathS2, imgPathS1, 0, 0))
		elif self.params.DATA_USAGE == 'S1IVF_S2_Sep':
			image_paths, valid_img_paths = get_image_paths(self.params, typeImg='S1', isTest=False, type_train=type_train)
			for imgPathS1 in image_paths:
				imageName = os.path.split(imgPathS1)[-1]
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TRAIN_S2_DIR, imgpathS2)
				train_data.append((imgPathS2, imgPathS1, 0, 0))
			for imgPathS1 in valid_img_paths:
				imageName = os.path.split(imgPathS1)[-1]
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.VALID_S2_DIR, imgpathS2)
				valid_data.append((imgPathS2, imgPathS1, 0, 0))
		elif self.params.DATA_USAGE == 'S1F_S2F_Sep':
			image_paths, valid_img_paths = get_image_paths(self.params, typeImg='S1', isTest=False, type_train=type_train)
			for imgPathS1 in image_paths:
				imageName = os.path.split(imgPathS1)[-1]
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TRAIN_S2_DIR, imgpathS2)
				train_data.append((imgPathS2, imgPathS1, 0, 0))
			for imgPathS1 in valid_img_paths:
				imageName = os.path.split(imgPathS1)[-1]
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.VALID_S2_DIR, imgpathS2)
				valid_data.append((imgPathS2, imgPathS1, 0, 0))
		elif self.params.DATA_USAGE == 'S1IV_S2Some':
			image_paths, valid_img_paths = get_image_paths(self.params, typeImg='S1', isTest=False, type_train=type_train)
			for imgPathS1 in image_paths:
				imageName = os.path.split(imgPathS1)[-1]
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TRAIN_S2_DIR, imgpathS2)
				train_data.append((imgPathS2, imgPathS1, 0, 0))
			for imgPathS1 in valid_img_paths:
				imageName = os.path.split(imgPathS1)[-1]
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.VALID_S2_DIR, imgpathS2)
				valid_data.append((imgPathS2, imgPathS1, 0, 0))
		elif self.params.DATA_USAGE == 'S1I_S2Some':
			image_paths, valid_img_paths = get_image_paths(self.params, typeImg='S1', isTest=False, type_train=type_train)
			for imgPathS1 in image_paths:
				imageName = os.path.split(imgPathS1)[-1]
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TRAIN_S2_DIR, imgpathS2)
				train_data.append((imgPathS2, imgPathS1, 0, 0))
			for imgPathS1 in valid_img_paths:
				imageName = os.path.split(imgPathS1)[-1]
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.VALID_S2_DIR, imgpathS2)
				valid_data.append((imgPathS2, imgPathS1, 0, 0))
		else:
			raise ValueError("Incorrect DATA_USAGE definition. Correct in unetParams")


		train_datagen = self.image_generator(train_data, type_train=type_train)
		validation_datagen = self.image_generator(valid_data, isValid = True, type_train=type_train)
		model = self.build_model(type_train=type_train)
		checkpoint = ModelCheckpoint(filepath=self.params.CHECKPOINT_PATH, monitor='loss', verbose=0, save_best_only=False,save_weights_only=False, mode='auto', save_freq=self.params.MODEL_SAVE_PERIOD)
		history_checkpoint = CSVLogger(self.params.LOGFILE, separator=",", append=True)
		tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.params.CHECKPOINT_DIR,  profile_batch=2)
		if len(train_data) <= 0:
			raise ValueError("No training data found. Update unetParams.py accordingly")

		if len(valid_data) <= 0:
			raise ValueError("No Validation data found. Update unetParams.py accordingly")


	        # train model
		model.summary()
		
		if self.params.NETWORK =='calibration':
			hist = model.fit(validation_datagen, steps_per_epoch=int(len(valid_data)/self.params.BATCH_SZ),epochs=self.params.NUM_EPOCHS, callbacks=[checkpoint, history_checkpoint,tensorboard_callback])
		else:
			hist = model.fit(train_datagen, steps_per_epoch=int(len(train_data)/self.params.BATCH_SZ),epochs=self.params.NUM_EPOCHS, callbacks=[checkpoint, history_checkpoint,tensorboard_callback], validation_data=validation_datagen, validation_steps=int(len(valid_data)/self.params.BATCH_SZ))
			
	
	def infer_MCDropOut(self, model, img, num_inter):
		infer = []
		for i in range(num_inter):
			if self.params.NETWORK == 'labelUnet':
				output30, output10 = model(img, training=True)
				#print(output10)
				#infer.append(output10[0,:,:,:])
				infer.append(output10[0,:,:,0:self.params.NUM_CATEGORIES])
			else: 
				infer.append(model(img, training=True)[0,:,:,:])

		if self.params.NETWORK == 'labelUnet':
			output30, output10 = model(img, training=False)
			#infer.append(output10[0,:,:,:])
			infer.append(output10[0,:,:,0:self.params.NUM_CATEGORIES])
		else:
			infer.append(model(img, training=False)[0,:,:,:])
		infer = np.asarray(infer)
		pred1 = np.mean(infer, axis=0)
		pred2 = np.argmax(infer, axis=-1)
		pred2 = voting(self.params, pred2)
		pred3 = np.median(infer, axis=0)
		pred_std = np.std(infer, axis=0)
		pred_perc97 = np.percentile(infer, 97.5, axis=0)
		#print(pred_perc97.shape)
		pred_perc2 = np.percentile(infer, 2.5, axis=0)
		#print(pred_perc2.shape)
		delta_pred = pred_perc97-pred_perc2
		return pred1, pred2, pred3, pred_perc97, delta_pred, pred_std



	def predict_drop(self):

		numPredChannels = self.params.NUM_CATEGORIES
		outReplaceStr = self.params.CLSPRED_FILE_STR
		model = self.build_model()
		selector = CheckpointSel(self.params.LOGFILE)
		test_model_num = selector.findBest(self.params, train=False)
		test_model_name = 'weights.'+str(test_model_num).zfill(2)+self.params.SAVE_CHECK_EXT
		print(test_model_name)
		model.load_weights(os.path.join(self.params.TEST_MODEL_DIR,test_model_name), by_name=True)
		num_inter = self.params.NUM_DROPOUT_ITER
		model.summary()

		if self.params.DATA_USAGE == 'S1_I_only':
			
			imgPaths = get_image_paths(self.params, typeImg='S1', isTest=True)
			print(len(imgPaths))
			print('Number of files = ', len(imgPaths))
			for imgPath in tqdm(imgPaths):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				img = load_img(imgPath)
				img = np.stack((img[:,:,0],img[:,:,3]), axis=2)
				if self.params.S1_IN_DB:
					img=dBConvert(img)
				img = np.expand_dims(img, axis=0).astype('float32') #Dropar aluns 
				pred_p, pred_v, pred_std = self.infer_MCDropOut(model, img, num_inter)
				mclu_ = mclu(pred_p)
				entropy_ = entropy(pred_p)
				pred = np.argmax(pred_p[:,:,1:], axis=2).astype('uint8')+1
				pred_p, pred_std = getProbandStdArgMax(pred, pred_p, pred_std)
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_STD_STR)), pred_std)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_ENT_STR)), entropy_)

		elif self.params.DATA_USAGE == 'S1_only':
			
			imgPaths = get_image_paths(self.params, typeImg='S1', isTest=True)
			print('Number of files = ', len(imgPaths))
			for imgPath in tqdm(imgPaths):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				img = load_img(imgPath)
				img = np.expand_dims(img[:,:,:4], axis=0).astype('float32') #Dropar aluns 
				pred_p, pred_v, pred_std = self.infer_MCDropOut(model, img, num_inter)
				mclu_ = mclu(pred_p)
				entropy_ = entropy(pred_p)
				pred = np.argmax(pred_p[:,:,1:], axis=2).astype('uint8')+1
				pred_p, pred_std = getProbandStdArgMax(pred, pred_p, pred_std)
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_STD_STR)), pred_std)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_ENT_STR)), entropy_)


		elif self.params.DATA_USAGE == 'S1_Complex':

			imgPaths = get_image_paths(self.params, typeImg='S1', isTest=True)
			print('Number of files = ', len(imgPaths))
			for imgPath in tqdm(imgPaths):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				img = load_img(imgPath)
				if self.params.S1_IN_DB:
					img =  dBConvert(img)
				img = img.astype(np.complex64)
				img = np.stack((img[:,:,0],img[:,:,3], img[:,:,1]+1j*img[:,:,2]), axis=2)
				img = np.expand_dims(img, axis=0).astype(np.complex64) #Dropar aluns 
				pred_p, pred_v, pred_std = self.infer_MCDropOut(model, img, num_inter)
				mclu_ = mclu(pred_p)
				entropy_ = entropy(pred_p)
				pred = np.argmax(pred_p[:,:,1:], axis=2).astype('uint8')+1
				pred_p, pred_std = getProbandStdArgMax(pred, pred_p, pred_std)
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_STD_STR)), pred_std)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_ENT_STR)), entropy_)

		elif self.params.DATA_USAGE == 'S2_S1_Complex':

			imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=True)
			imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=True)
			
			if(len(imgPathsS1)!=len(imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files. Correct the amount in the folder.")
			
			print('Number of files = ', len(imgPathsS1))


			for imgPath in tqdm(imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				if self.params.S1_IN_DB:
					imgS1 =  dBConvert(imgS1)
				imgS1 = imgS1.astype(np.complex64)
				imgS1 = np.stack((imgS1[:,:,0],imgS1[:,:,3], imgS1[:,:,1]+1j*imgS1[:,:,2]), axis=2)
				imgS1 = np.expand_dims(imgS1, axis=0).astype(np.complex64) #Dropar aluns
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TEST_DIR, imgpathS2)
				imgS2 = load_img(imgPathS2)
				#imgS2 = imgS2.astype(np.complex64)
				imgS2 = np.expand_dims(imgS2[:,:,:4], axis=0).astype(np.complex64)
				img = [imgS2,imgS1]
				pred_p, pred_v, pred_std = self.infer_MCDropOut(model, img, num_inter)
				mclu_ = mclu(pred_p)
				entropy_ = entropy(pred_p)
				pred = np.argmax(pred_p[:,:,1:], axis=2).astype('uint8')+1
				pred_p, pred_std = getProbandStdArgMax(pred, pred_p, pred_std)
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_STD_STR)), pred_std)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_ENT_STR)), entropy_)

		elif self.params.DATA_USAGE == 'S2_only':
			
			imgPaths = get_image_paths(self.params, typeImg='S2', isTest=True)
			print('Number of files = ', len(imgPaths))
			for imgPath in tqdm(imgPaths):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S2_FILE_STR, outReplaceStr)
				img = load_img(imgPath)
				img = np.expand_dims(img[:,:,:4], axis=0).astype('float32') #Dropar aluns 
				pred_p, pred_v, pred_std = self.infer_MCDropOut(model, img, num_inter)
				mclu_ = mclu(pred_p)
				entropy_ = entropy(pred_p)
				pred = np.argmax(pred_p[:,:,1:], axis=2).astype('uint8')+1
				pred_p, pred_std = getProbandStdArgMax(pred, pred_p, pred_std)
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_STD_STR)), pred_std)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_ENT_STR)), entropy_)

		elif self.params.DATA_USAGE == 'S1S2All':
			
			imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=True)
			imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=True)
			
			if(len(imgPathsS1)!=len(imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files. Correct the amount in the folder.")
			
			print('Number of files = ', len(imgPathsS1))

			for imgPath in tqdm(imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				imgS1 = np.expand_dims(imgS1[:,:,:4], axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TEST_DIR, imgpathS2)
				imgS2 = load_img(imgPathS2)
				imgS2 = np.expand_dims(imgS2[:,:,:4], axis=0).astype('float32')
				img = np.concatenate((imgS2,imgS1),axis=3)
				pred_p, pred_v, pred_std = self.infer_MCDropOut(model, img, num_inter)
				mclu_ = mclu(pred_p)
				entropy_ = entropy(pred_p)
				pred = np.argmax(pred_p[:,:,1:], axis=2).astype('uint8')+1
				pred_p, pred_std = getProbandStdArgMax(pred, pred_p, pred_std)
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_STD_STR)), pred_std)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_ENT_STR)), entropy_)

		elif self.params.DATA_USAGE == 'S2S1IAll':
			
			imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=True)
			imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=True)
			
			if(len(imgPathsS1)!=len(imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files. Correct the amount in the folder.")
			
			print('Number of files = ', len(imgPathsS1))

			for imgPath in tqdm(imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				imgS1 = np.stack((imgS1[:,:,0],imgS1[:,:,3]), axis=2)
				imgS1 = np.expand_dims(imgS1, axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TEST_DIR, imgpathS2)
				imgS2 = load_img(imgPathS2)
				imgS2 = np.expand_dims(imgS2[:,:,:4], axis=0).astype('float32')
				img = np.concatenate((imgS2,imgS1),axis=3)
				pred_p, pred_v, pred_std = self.infer_MCDropOut(model, img, num_inter)
				mclu_ = mclu(pred_p)
				entropy_ = entropy(pred_p)
				pred = np.argmax(pred_p[:,:,1:], axis=2).astype('uint8')+1
				pred_p, pred_std = getProbandStdArgMax(pred, pred_p, pred_std)
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_STD_STR)), pred_std)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_ENT_STR)), entropy_)

		elif self.params.DATA_USAGE == 'S1IV_S2Some':
			imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=True)
			imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=True)
			
			if(len(imgPathsS1)!=len(imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files. Correct the amount in the folder.")

			print('Number of files = ', len(imgPathsS1))

			for imgPath in tqdm(imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				vol = np.divide(imgS1[:,:,0], imgS1[:,:,3], out=np.zeros_like(imgS1[:,:,0]), where=imgS1[:,:,3]!=0)
				vol = np.nan_to_num(vol, nan=0.0)
				if np.max(vol)>np.min(vol):
					vol = (vol - np.min(vol))/(np.max(vol)-np.min(vol))
				imgS1 = np.stack((imgS1[:,:,0],imgS1[:,:,3],vol), axis=2)
				imgS1 = np.expand_dims(imgS1, axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TEST_DIR, imgpathS2)
				imgS2 = load_img(imgPathS2)
				imgS2 = np.concatenate((imgS2[:,:,5:], imgS2[:,:,8:]), axis=2)
				imgS2 = np.expand_dims(imgS2, axis=0).astype('float32')
				pred_p, pred_v, pred_std = self.infer_MCDropOut(model, [imgS2,imgS1], num_inter)
				mclu_ = mclu(pred_p)
				entropy_ = entropy(pred_p)
				pred = np.argmax(pred_p[:,:,1:], axis=2).astype('uint8')+1
				pred_p, pred_std = getProbandStdArgMax(pred, pred_p, pred_std)
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_STD_STR)), pred_std)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_ENT_STR)), entropy_)

		elif self.params.DATA_USAGE == 'S1I_S2Some':
			imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=True)
			imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=True)
			
			if(len(imgPathsS1)!=len(imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files. Correct the amount in the folder.")

			print('Number of files = ', len(imgPathsS1))

			for imgPath in tqdm(imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				imgS1 = np.stack((imgS1[:,:,0],imgS1[:,:,3]), axis=2)
				imgS1 = np.expand_dims(imgS1, axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TEST_DIR, imgpathS2)
				imgS2 = load_img(imgPathS2)
				imgS2 = np.concatenate((imgS2[:,:,5:], imgS2[:,:,8:]), axis=2)
				imgS2 = np.expand_dims(imgS2, axis=0).astype('float32')
				pred_p, pred_v, pred_std = self.infer_MCDropOut(model, [imgS2,imgS1], num_inter)
				mclu_ = mclu(pred_p)
				entropy_ = entropy(pred_p)
				pred = np.argmax(pred_p[:,:,1:], axis=2).astype('uint8')+1
				pred_p, pred_std = getProbandStdArgMax(pred, pred_p, pred_std)
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_STD_STR)), pred_std)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_ENT_STR)), entropy_)

		elif self.params.DATA_USAGE == 'S1_S2_Sep':
			
			imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=True)
			imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=True)
			
			if(len(imgPathsS1)!=len(imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files. Correct the amount in the folder.")

			print('Number of files = ', len(imgPathsS1))

			for imgPath in tqdm(imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				imgS1 = np.expand_dims(imgS1[:,:,:4], axis=0).astype('float32') #Dropar aluns  
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TEST_DIR, imgpathS2)
				imgS2 = load_img(imgPathS2)
				imgS2 = np.expand_dims(imgS2[:,:,:4], axis=0).astype('float32')
				pred_p, pred_v, pred_std = self.infer_MCDropOut(model, [imgS2,imgS1], num_inter)
				mclu_ = mclu(pred_p)
				entropy_ = entropy(pred_p)
				pred = np.argmax(pred_p[:,:,1:], axis=2).astype('uint8')+1
				pred_p, pred_std = getProbandStdArgMax(pred, pred_p, pred_std)
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_STD_STR)), pred_std)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_ENT_STR)), entropy_)

		elif self.params.DATA_USAGE == 'S1I_S2_Sep':
			
			imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=True)
			imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=True)
			
			if(len(imgPathsS1)!=len(imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files. Correct the amount in the folder.")

			print('Number of files = ', len(imgPathsS1))

			for imgPath in tqdm(imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				if self.params.S1_IN_DB:
					imgS1=dBConvert(imgS1)
				imgS1 = np.stack((imgS1[:,:,0],imgS1[:,:,3]), axis=2)
				imgS1 = np.expand_dims(imgS1, axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TEST_DIR, imgpathS2)
				imgS2 = load_img(imgPathS2)
				imgS2 = np.expand_dims(imgS2[:,:,:4], axis=0).astype('float32')
				#pred1, pred2, pred3, pred_perc97, delta_pred, pred_std
				pred_p, pred_v, pred_m, pred_perc, delta_pred, pred_std = self.infer_MCDropOut(model, [imgS2,imgS1], num_inter)
            	#Pred_Mean:
				mclu_ = mclu(pred_p[:,:,1:])
				entropy_ = entropy(pred_p[:,:,1:], self.params)
				pred = np.argmax(pred_p[:,:,1:], axis=2).astype('uint8')+1
				pred_k, pred_s = getProbandStdArgMax(pred, pred_p, pred_std)
				pred_k, delta_p = getProbandStdArgMax(pred, pred_p, delta_pred)
				
				if not os.path.isdir(self.params.OUTPUT_DIR+'mean/'):
					os.makedirs(self.params.OUTPUT_DIR+'mean/')
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR+'mean/', outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR+'mean/', outName.replace(outReplaceStr,self.params.LABEL_STD_STR)), pred_s)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR+'mean/', outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR+'mean/', outName.replace(outReplaceStr,self.params.LABEL_ENT_STR)), entropy_)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR+'mean/', outName.replace(outReplaceStr,self.params.LABEL_DELPE_STR)), delta_p)#LABEL_DELPE_STR
				
            	#Pred_Median:
				mclu_ = mclu(pred_m[:,:,1:])
				entropy_ = entropy(pred_m[:,:,1:])
				pred = np.argmax(pred_m[:,:,1:], axis=2).astype('uint8')+1
				pred_k, pred_s = getProbandStdArgMax(pred, pred_m, pred_std)
				pred_k, delta_p = getProbandStdArgMax(pred, pred_m, delta_pred)
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				if not os.path.isdir(self.params.OUTPUT_DIR+'median/'):
					os.makedirs(self.params.OUTPUT_DIR+'median/')
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR+'median/', outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR+'median/', outName.replace(outReplaceStr,self.params.LABEL_STD_STR)), pred_s)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR+'median/', outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR+'median/', outName.replace(outReplaceStr,self.params.LABEL_ENT_STR)), entropy_)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR+'median/', outName.replace(outReplaceStr,self.params.LABEL_DELPE_STR)), delta_p)#LABEL_DELPE_STR

            	#Pred_Voting:
				mclu_ = mclu(pred_v[:,:,1:])
				entropy_ = entropy(pred_v[:,:,1:])
				pred = np.argmax(pred_v[:,:,1:], axis=2).astype('uint8')+1
				pred_k, pred_s = getProbandStdArgMax(pred, pred_v, pred_std)
				pred_k, delta_p = getProbandStdArgMax(pred, pred_v, delta_pred)
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.

				if not os.path.isdir(self.params.OUTPUT_DIR+'voting/'):
					os.makedirs(self.params.OUTPUT_DIR+'voting/')
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR+'voting/', outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR+'voting/', outName.replace(outReplaceStr,self.params.LABEL_STD_STR)), pred_s)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR+'voting/', outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR+'voting/', outName.replace(outReplaceStr,self.params.LABEL_ENT_STR)), entropy_)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR+'voting/', outName.replace(outReplaceStr,self.params.LABEL_DELPE_STR)), delta_p)#LABEL_DELPE_STR

            	#Pred_Percentil:
				mclu_ = mclu(pred_perc[:,:,1:])
				entropy_ = entropy(pred_perc[:,:,1:])
				pred = np.argmax(pred_perc[:,:,1:], axis=2).astype('uint8')+1
				pred_k, pred_s = getProbandStdArgMax(pred, pred_perc, pred_std)
				pred_k, delta_p = getProbandStdArgMax(pred, pred_perc, delta_pred)
				#Ver se preciso fazer o delta_perd para todos os outros.
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				if not os.path.isdir(self.params.OUTPUT_DIR+'percentile/'):
					os.makedirs(self.params.OUTPUT_DIR+'percentile/')
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR+'percentile/', outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR+'percentile/', outName.replace(outReplaceStr,self.params.LABEL_STD_STR)), pred_s)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR+'percentile/', outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR+'percentile/', outName.replace(outReplaceStr,self.params.LABEL_ENT_STR)), entropy_)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR+'percentile/', outName.replace(outReplaceStr,self.params.LABEL_DELPE_STR)), delta_p)#LABEL_DELPE_STR
			
		elif self.params.DATA_USAGE == 'S1IV_S2_Sep':
			imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=True)
			imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=True)
			
			if(len(imgPathsS1)!=len(imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files. Correct the amount in the folder.")

			print('Number of files = ', len(imgPathsS1))

			for imgPath in tqdm(imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				vol = np.divide(imgS1[:,:,0], imgS1[:,:,3], out=np.zeros_like(imgS1[:,:,0]), where=imgS1[:,:,3]!=0)
				vol = np.nan_to_num(vol, nan=0.0)
				if np.max(vol)>np.min(vol):
					vol = (vol - np.min(vol))/(np.max(vol)-np.min(vol))
				imgS1 = np.stack((imgS1[:,:,0],imgS1[:,:,3],vol), axis=2)
				imgS1 = np.expand_dims(imgS1, axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TEST_DIR, imgpathS2)
				imgS2 = load_img(imgPathS2)
				imgS2 = np.expand_dims(imgS2[:,:,:4], axis=0).astype('float32')
				pred_p, pred_v, pred_std = self.infer_MCDropOut(model, [imgS2,imgS1], num_inter)
				mclu_ = mclu(pred_p)
				entropy_ = entropy(pred_p)
				pred = np.argmax(pred_p[:,:,1:], axis=2).astype('uint8')+1
				pred_p, pred_std = getProbandStdArgMax(pred, pred_p, pred_std)
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_STD_STR)), pred_std)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_ENT_STR)), entropy_)
		elif self.params.DATA_USAGE == 'S2F_only':
			imgPaths = get_image_paths(self.params, typeImg='S2', isTest=True)
			print('Number of files = ', len(imgPaths))
			for imgPath in tqdm(imgPaths):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S2_FILE_STR, outReplaceStr)
				img = np.expand_dims(load_img(imgPath), axis=0).astype('float32') #Dropar aluns 
				pred_p, pred_v, pred_std = self.infer_MCDropOut(model, img, num_inter)
				mclu_ = mclu(pred_p)
				entropy_ = entropy(pred_p)
				pred = np.argmax(pred_p[:,:,1:], axis=2).astype('uint8')+1
				pred_p, pred_std = getProbandStdArgMax(pred, pred_p, pred_std)
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_STD_STR)), pred_std)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_ENT_STR)), entropy_)
		elif self.params.DATA_USAGE == 'S1F_only':
			imgPaths = get_image_paths(self.params, typeImg='S1', isTest=True)
			print(len(imgPaths))
			print('Number of files = ', len(imgPaths))
			for imgPath in tqdm(imgPaths):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				img = np.expand_dims(load_img(imgPath), axis=0).astype('float32') #Dropar aluns 
				pred_p, pred_v, pred_std = self.infer_MCDropOut(model, img, num_inter)
				mclu_ = mclu(pred_p)
				entropy_ = entropy(pred_p)
				pred = np.argmax(pred_p[:,:,1:], axis=2).astype('uint8')+1
				pred_p, pred_std = getProbandStdArgMax(pred, pred_p, pred_std)
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_STD_STR)), pred_std)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_ENT_STR)), entropy_)
		elif self.params.DATA_USAGE == 'S1IF_only':
			imgPaths = get_image_paths(self.params, typeImg='S1', isTest=True)
			print(len(imgPaths))
			print('Number of files = ', len(imgPaths))
			for imgPath in tqdm(imgPaths):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				img = load_img(imgPath)
				imgT = np.stack((img[:,:,0],img[:,:,3]), axis=2)
				img = np.concatenate((imgT, img[:,:,4:]), axis=2)
				img = np.expand_dims(img, axis=0).astype('float32') #Dropar aluns 
				pred_p, pred_v, pred_std = self.infer_MCDropOut(model, img, num_inter)
				mclu_ = mclu(pred_p)
				entropy_ = entropy(pred_p)
				pred = np.argmax(pred_p[:,:,1:], axis=2).astype('uint8')+1
				pred_p, pred_std = getProbandStdArgMax(pred, pred_p, pred_std)
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_STD_STR)), pred_std)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_ENT_STR)), entropy_)
		elif self.params.DATA_USAGE == 'S1IVF_only':
			imgPaths = get_image_paths(self.params, typeImg='S1', isTest=True)
			print(len(imgPaths))
			print('Number of files = ', len(imgPaths))
			for imgPath in tqdm(imgPaths):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				img = load_img(imgPath)
				vol = np.divide(imgS1[:,:,0], imgS1[:,:,3], out=np.zeros_like(imgS1[:,:,0]), where=imgS1[:,:,3]!=0)
				vol = np.nan_to_num(vol, nan=0.0)
				if np.max(vol)>np.min(vol):
					vol = (vol - np.min(vol))/(np.max(vol)-np.min(vol))
				imgT = np.stack((img[:,:,0],img[:,:,3],vol), axis=2)
				img = np.concatenate((imgT, img[:,:,4:]), axis=2)
				img = np.expand_dims(img, axis=0).astype('float32') #Dropar aluns 
				pred_p, pred_v, pred_std = self.infer_MCDropOut(model, img, num_inter)
				mclu_ = mclu(pred_p)
				entropy_ = entropy(pred_p)
				pred = np.argmax(pred_p[:,:,1:], axis=2).astype('uint8')+1
				pred_p, pred_std = getProbandStdArgMax(pred, pred_p, pred_std)
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_STD_STR)), pred_std)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_ENT_STR)), entropy_)
		elif self.params.DATA_USAGE == 'S1IV_only':
			imgPaths = get_image_paths(self.params, typeImg='S1', isTest=True)
			print(len(imgPaths))
			print('Number of files = ', len(imgPaths))
			for imgPath in tqdm(imgPaths):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				img = load_img(imgPath)
				vol = np.divide(imgS1[:,:,0], imgS1[:,:,3], out=np.zeros_like(imgS1[:,:,0]), where=imgS1[:,:,3]!=0)
				vol = np.nan_to_num(vol, nan=0.0)
				if np.max(vol)>np.min(vol):
					vol = (vol - np.min(vol))/(np.max(vol)-np.min(vol))
				img = np.stack((img[:,:,0],img[:,:,3],vol), axis=2)
				img = np.expand_dims(img, axis=0).astype('float32') #Dropar aluns 
				pred_p, pred_v, pred_std = self.infer_MCDropOut(model, img, num_inter)
				mclu_ = mclu(pred_p)
				entropy_ = entropy(pred_p)
				pred = np.argmax(pred_p[:,:,1:], axis=2).astype('uint8')+1
				pred_p, pred_std = getProbandStdArgMax(pred, pred_p, pred_std)
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_STD_STR)), pred_std)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_ENT_STR)), entropy_)
		elif self.params.DATA_USAGE == 'S1IS2FAll':
			imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=True)
			imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=True)
			
			if(len(imgPathsS1)!=len(imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files. Correct the amount in the folder.")
			
			print('Number of files = ', len(imgPathsS1))

			for imgPath in tqdm(imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				imgS1 = np.stack((imgS1[:,:,0],imgS1[:,:,3]), axis=2)
				imgS1 = np.expand_dims(imgS1, axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TEST_DIR, imgpathS2)
				imgS2 = np.expand_dims(load_img(imgPathS2), axis=0).astype('float32')
				img = np.concatenate((imgS2,imgS1),axis=3)
				pred_p, pred_v, pred_std = self.infer_MCDropOut(model, img, num_inter)
				mclu_ = mclu(pred_p)
				entropy_ = entropy(pred_p)
				pred = np.argmax(pred_p[:,:,1:], axis=2).astype('uint8')+1
				pred_p, pred_std = getProbandStdArgMax(pred, pred_p, pred_std)
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_STD_STR)), pred_std)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_ENT_STR)), entropy_)
		elif self.params.DATA_USAGE == 'S1IFS2FAll':
			imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=True)
			imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=True)
			
			if(len(imgPathsS1)!=len(imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files. Correct the amount in the folder.")
			
			print('Number of files = ', len(imgPathsS1))

			for imgPath in tqdm(imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				imgS1T = np.stack((imgS1[:,:,0],imgS1[:,:,3]), axis=2)
				imgS1 =np.concatenate((imgS1T, imgS1[:,:,4:]), axis=2)
				imgS1 = np.expand_dims(imgS1, axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TEST_DIR, imgpathS2)
				imgS2 = np.expand_dims(load_img(imgPathS2), axis=0).astype('float32')
				img = np.concatenate((imgS2,imgS1),axis=3)
				pred_p, pred_v, pred_std = self.infer_MCDropOut(model, img, num_inter)
				mclu_ = mclu(pred_p)
				entropy_ = entropy(pred_p)
				pred = np.argmax(pred_p[:,:,1:], axis=2).astype('uint8')+1
				pred_p, pred_std = getProbandStdArgMax(pred, pred_p, pred_std)
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_STD_STR)), pred_std)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_ENT_STR)), entropy_)

		elif self.params.DATA_USAGE == 'S1S2FAll':
			imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=True)
			imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=True)
			
			if(len(imgPathsS1)!=len(imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files. Correct the amount in the folder.")
			
			print('Number of files = ', len(imgPathsS1))

			for imgPath in tqdm(imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				imgS1 = np.expand_dims(imgS1[:,:,:4], axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TEST_DIR, imgpathS2)
				imgS2 = np.expand_dims(load_img(imgPathS2), axis=0).astype('float32')
				img = np.concatenate((imgS2,imgS1),axis=3)
				pred_p, pred_v, pred_std = self.infer_MCDropOut(model, img, num_inter)
				mclu_ = mclu(pred_p)
				entropy_ = entropy(pred_p)
				pred = np.argmax(pred_p[:,:,1:], axis=2).astype('uint8')+1
				pred_p, pred_std = getProbandStdArgMax(pred, pred_p, pred_std)
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_STD_STR)), pred_std)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_ENT_STR)), entropy_)
		elif self.params.DATA_USAGE == 'S1FS2FAll':
			imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=True)
			imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=True)
			
			if(len(imgPathsS1)!=len(imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files. Correct the amount in the folder.")
			
			print('Number of files = ', len(imgPathsS1))

			for imgPath in tqdm(imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = np.expand_dims(load_img(imgPath), axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TEST_DIR, imgpathS2)
				imgS2 = np.expand_dims(load_img(imgPathS2), axis=0).astype('float32')
				img = np.concatenate((imgS2,imgS1),axis=3)
				pred_p, pred_v, pred_std = self.infer_MCDropOut(model, img, num_inter)
				mclu_ = mclu(pred_p)
				entropy_ = entropy(pred_p)
				pred = np.argmax(pred_p[:,:,1:], axis=2).astype('uint8')+1
				pred_p, pred_std = getProbandStdArgMax(pred, pred_p, pred_std)
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_STD_STR)), pred_std)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_ENT_STR)), entropy_)
		elif self.params.DATA_USAGE == 'S1_S2F_Sep':
			imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=True)
			imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=True)
			
			if(len(imgPathsS1)!=len(imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files. Correct the amount in the folder.")

			print('Number of files = ', len(imgPathsS1))

			for imgPath in tqdm(imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				imgS1 = np.expand_dims(imgS1[:,:,:4], axis=0).astype('float32') #Dropar aluns  
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TEST_DIR, imgpathS2)
				imgS2 = np.expand_dims(load_img(imgPathS2), axis=0).astype('float32')
				pred_p, pred_v, pred_std = self.infer_MCDropOut(model, [imgS2,imgS1], num_inter)
				mclu_ = mclu(pred_p)
				entropy_ = entropy(pred_p)
				pred = np.argmax(pred_p[:,:,1:], axis=2).astype('uint8')+1
				pred_p, pred_std = getProbandStdArgMax(pred, pred_p, pred_std)
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_STD_STR)), pred_std)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_ENT_STR)), entropy_)
		elif self.params.DATA_USAGE == 'S1I_S2F_Sep':
			imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=True)
			imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=True)
			
			if(len(imgPathsS1)!=len(imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files. Correct the amount in the folder.")

			print('Number of files = ', len(imgPathsS1))

			for imgPath in tqdm(imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				imgS1 = np.stack((imgS1[:,:,0],imgS1[:,:,3]), axis=2)
				imgS1 = np.expand_dims(imgS1, axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TEST_DIR, imgpathS2)
				imgS2 = np.expand_dims(load_img(imgPathS2), axis=0).astype('float32')
				pred_p, pred_v, pred_std = self.infer_MCDropOut(model, [imgS2,imgS1], num_inter)
				mclu_ = mclu(pred_p)
				entropy_ = entropy(pred_p)
				pred = np.argmax(pred_p[:,:,1:], axis=2).astype('uint8')+1
				pred_p, pred_std = getProbandStdArgMax(pred, pred_p, pred_std)
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_STD_STR)), pred_std)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_ENT_STR)), entropy_)
		elif self.params.DATA_USAGE == 'S1IF_S2F_Sep':
			imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=True)
			imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=True)
			
			if(len(imgPathsS1)!=len(imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files. Correct the amount in the folder.")

			print('Number of files = ', len(imgPathsS1))

			for imgPath in tqdm(imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				imgS1T = np.stack((imgS1[:,:,0],imgS1[:,:,3]), axis=2)
				imgS1 = np.concatenate((imgS1T, imgS1[:,:,4:]), axis=2)
				imgS1 = np.expand_dims(imgS1, axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TEST_DIR, imgpathS2)
				imgS2 = np.expand_dims(load_img(imgPathS2), axis=0).astype('float32')
				pred_p, pred_v, pred_std = self.infer_MCDropOut(model, [imgS2,imgS1], num_inter)
				mclu_ = mclu(pred_p)
				entropy_ = entropy(pred_p)
				pred = np.argmax(pred_p[:,:,1:], axis=2).astype('uint8')+1
				pred_p, pred_std = getProbandStdArgMax(pred, pred_p, pred_std)
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_STD_STR)), pred_std)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_ENT_STR)), entropy_)
		elif self.params.DATA_USAGE == 'S1F_S2F_Sep':
			imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=True)
			imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=True)
			
			if(len(imgPathsS1)!=len(imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files. Correct the amount in the folder.")

			print('Number of files = ', len(imgPathsS1))

			for imgPath in tqdm(imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = np.expand_dims(load_img(imgPath), axis=0).astype('float32') #Dropar aluns  
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TEST_DIR, imgpathS2)
				imgS2 = np.expand_dims(load_img(imgPathS2), axis=0).astype('float32')
				pred_p, pred_v, pred_std = self.infer_MCDropOut(model, [imgS2,imgS1], num_inter)
				mclu_ = mclu(pred_p)
				entropy_ = entropy(pred_p)
				pred = np.argmax(pred_p[:,:,1:], axis=2).astype('uint8')+1
				pred_p, pred_std = getProbandStdArgMax(pred, pred_p, pred_std)
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_STD_STR)), pred_std)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_ENT_STR)), entropy_)
		else:
			raise ValueError("Incorrect DATA_USAGE definition. Correct in unetParams")

	def finalMap(self, imgT, mcluT, pred_p, pred_i, stdImgT, imgTa, probimgTa):
		imgFinal = np.zeros_like(imgT)
		probImgT = np.zeros_like(imgT)
		if probimgTa is None:
			#means that imgTa is GT original (in other words, it is the first refinement of the dataset)
			A = mcluT > 0.8 #pensar se o melhor sao esses valores.
			B = stdImgT < 0.15
			A=A*B
			imgFinal = imgT*A
			probImgT = pred_i*A
			probimgTa = np.ones_like(imgT)*0.907 #esse valor eh a exatidao global do mapa
		else:
			#means that it is not the first refinement of the dataset
			A = mcluT > 0.8 #pensar se o melhor sao esses valores.
			B = stdImgT < 0.15
			A=A*B
			imgFinal = imgT*A
			probImgT = pred_i*A

		for i in range(imgFinal.shape[0]):
			for j in range(imgFinal.shape[1]):
				if imgFinal[i,j]==0:
					classCount = np.ones(self.params.NUM_CATEGORIES)
					sumProb = pred_p[i,j,:]
					classCount[int(imgT[i,j])]=classCount[int(imgT[i,j])]-1
					sumProb[imgT[i,j]]=sumProb[int(imgT[i,j])]-pred_p[i,j,int(imgT[i,j])]
					classCount[int(imgTa[i,j])]=classCount[int(imgTa[i,j])]+1
					sumProb[int(imgTa[i,j])] = sumProb[int(imgTa[i,j])] + probimgTa[i,j]
					for k in range(max(0, i-1),min(i+2,self.params.IMG_SZ[0]),1):
						for l in range(max(0, j-1),min(j+2,self.params.IMG_SZ[0]),1):
							classCount[int(imgT[k,l])]=classCount[int(imgT[k,l])]+1
							sumProb[int(imgT[k,l])]=sumProb[int(imgT[k,l])]+pred_i[k,l]
					sumProb = sumProb/classCount
					imgFinal[i,j] = np.argmax(sumProb)
					probImgT[i,j] = sumProb[int(imgFinal[i,j])]




		return imgFinal, probImgT

	def save_images(self, pred_p, pred_std, imageName, outName, outReplaceStr, train_valid):


		if train_valid == 'train':
			mclu_ = mclu(pred_p)
			entropy_ = entropy(pred_p)
			pred = np.argmax(pred_p, axis=2).astype('uint8')
			pred_i, pred_std = getProbandStdArgMax(pred, pred_p, pred_std)
			#Funcao para juntar essa inferência com o GT e a inferencia anterior.
			#Abrir GT e dado anterior (se existir)
			label_name = imageName.replace(self.params.S1_FILE_STR, self.params.LABEL_FILE_STR)
			gt_img = load_img(os.path.join(self.params.TRAIN_LABEL_DIR, label_name))
			#Abrir dataset novo anterior:
			img_path=os.path.join(self.params.TRAIN_LABEL_NEW, label_name)
			prob_name = label_name.replace(self.params.LABEL_FILE_STR, self.params.LABEL_PREC_STR)
			std_name = label_name.replace(self.params.LABEL_FILE_STR, self.params.LABEL_STD_STR)
			if (os.path.exists(img_path)):
				img_Ta = load_img(img_path)
				img_Ta_p = load_img(os.path.join(self.params.TRAIN_LABEL_NEW, prob_name))
			else:
				img_Ta = gt_img
				img_Ta_p = None
			pred, pred_i = self.finalMap(pred, mclu_, pred_p, pred_i, pred_std, img_Ta, img_Ta_p)

			tifffile.imsave(os.path.join(self.params.TRAIN_LABEL_NEW, outName), pred)
			tifffile.imsave(os.path.join(self.params.TRAIN_LABEL_NEW, outName.replace(outReplaceStr,self.params.LABEL_PREC_STR)), pred_i)
			tifffile.imsave(os.path.join(self.params.TRAIN_LABEL_NEW, outName.replace(outReplaceStr,self.params.LABEL_STD_STR)), pred_std)
			tifffile.imsave(os.path.join(self.params.TRAIN_LABEL_NEW, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)
			tifffile.imsave(os.path.join(self.params.TRAIN_LABEL_NEW, outName.replace(outReplaceStr,self.params.LABEL_ENT_STR)), entropy_)

		elif train_valid == 'valid':
			mclu_ = mclu(pred_p)
			#entropy_ = entropy(pred_p)
			pred = np.argmax(pred_p, axis=2).astype('uint8')
			pred_i, pred_std = getProbandStdArgMax(pred, pred_p, pred_std)
			#Funcao para juntar essa inferência com o GT e a inferencia anterior.
			#Abrir GT e dado anterior (se existir)
			label_name = imageName.replace(self.params.S1_FILE_STR, self.params.LABEL_FILE_STR)
			gt_img = load_img(os.path.join(self.params.VALID_LABEL_DIR, label_name))

			#Abrir dataset novo anterior:
			img_path=os.path.join(self.params.VALID_LABEL_NEW, label_name)
			prob_name = label_name.replace(self.params.LABEL_FILE_STR, self.params.LABEL_PREC_STR)
			std_name = label_name.replace(self.params.LABEL_FILE_STR, self.params.LABEL_STD_STR)
			if (os.path.exists(img_path)):
				img_Ta = load_img(img_path)
				img_Ta_p = load_img(os.path.join(self.params.VALID_LABEL_NEW, prob_name))
			else:
				img_Ta = gt_img
				img_Ta_p = None
				img_ta_std = None
				
			pred, pred_i = self.finalMap(pred, mclu_, pred_p, pred_i, pred_std, img_Ta, img_Ta_p)

			tifffile.imsave(os.path.join(self.params.VALID_LABEL_NEW, outName), pred)
			tifffile.imsave(os.path.join(self.params.VALID_LABEL_NEW, outName.replace(outReplaceStr,self.params.LABEL_PREC_STR)), pred_i)
			tifffile.imsave(os.path.join(self.params.VALID_LABEL_NEW, outName.replace(outReplaceStr,self.params.LABEL_STD_STR)), pred_std)
			tifffile.imsave(os.path.join(self.params.VALID_LABEL_NEW, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)
			tifffile.imsave(os.path.join(self.params.VALID_LABEL_NEW, outName.replace(outReplaceStr,self.params.LABEL_ENT_STR)), entropy_)

		else:
			raise ValueError("Value must be train or valid to refine the dataset")



	def refineTrainDataset(self):

		numPredChannels = self.params.NUM_CATEGORIES
		outReplaceStr = self.params.LABEL_FILE_STR
		model = self.build_model()
		selector = CheckpointSel(self.params.LOGFILE)
		test_model_num = selector.findBest(self.params, train=False)
		test_model_name = 'weights.'+str(test_model_num).zfill(2)+self.params.SAVE_CHECK_EXT
		print(test_model_name)
		model.load_weights(os.path.join(self.params.TEST_MODEL_DIR,test_model_name), by_name=True)
		num_inter = self.params.NUM_DROPOUT_ITER
		model.summary()

		if self.params.DATA_USAGE == 'S1_I_only':
			
			train_imgPaths, valid_imgPaths = get_image_paths(self.params, typeImg='S1', isTest=False)
			print('Number of files = ', len(train_imgPaths))
			for imgPath in tqdm(train_imgPaths):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				img = load_img(imgPath)
				img = np.stack((img[:,:,0],img[:,:,3]), axis=2)
				img = np.expand_dims(img, axis=0).astype('float32') #Dropar aluns 
				pred_p, pred_std = self.infer_MCDropOut(model, img, num_inter)
				self.save_images(pred_p, pred_std, imageName, outName, outReplaceStr, 'train')

			print('Number of files = ', len(valid_imgPaths))
			for imgPath in tqdm(valid_imgPaths):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				img = load_img(imgPath)
				img = np.stack((img[:,:,0],img[:,:,3]), axis=2)
				img = np.expand_dims(img, axis=0).astype('float32') #Dropar aluns 
				pred_p, pred_std = self.infer_MCDropOut(model, img, num_inter)
				self.save_images(pred_p, pred_std, imageName,outName, outReplaceStr, 'valid')

		elif self.params.DATA_USAGE == 'S1_only':
			
			train_imgPaths, valid_imgPaths = get_image_paths(self.params, typeImg='S1', isTest=False)
			print('Number of files = ', len(train_imgPaths))
			for imgPath in tqdm(train_imgPaths):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				img = load_img(imgPath)
				img = np.expand_dims(img[:,:,:4], axis=0).astype('float32') #Dropar aluns 
				pred_p, pred_std = self.infer_MCDropOut(model, img, num_inter)
				self.save_images(pred_p, pred_std, imageName,outName, outReplaceStr, 'train')

			print('Number of files = ', len(valid_imgPaths))
			for imgPath in tqdm(valid_imgPaths):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				img = load_img(imgPath)
				img = np.expand_dims(img[:,:,:4], axis=0).astype('float32')
				pred_p, pred_std = self.infer_MCDropOut(model, img, num_inter)
				self.save_images(pred_p, pred_std, imageName,outName, outReplaceStr, 'valid')

		elif self.params.DATA_USAGE == 'S2_only':
			
			train_imgPaths, valid_imgPaths = get_image_paths(self.params, typeImg='S2', isTest=False)
			print('Number of files = ', len(train_imgPaths))
			for imgPath in tqdm(train_imgPaths):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S2_FILE_STR, outReplaceStr)
				img = load_img(imgPath)
				img = np.expand_dims(img[:,:,:4], axis=0).astype('float32') #Dropar aluns 
				pred_p, pred_std = self.infer_MCDropOut(model, img, num_inter)
				self.save_images(pred_p, pred_std, imageName,outName, outReplaceStr, 'train')

			print('Number of files = ', len(valid_imgPaths))
			for imgPath in tqdm(valid_imgPaths):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S2_FILE_STR, outReplaceStr)
				img = load_img(imgPath)
				img = np.expand_dims(img[:,:,:4], axis=0).astype('float32') #Dropar aluns 
				pred_p, pred_std = self.infer_MCDropOut(model, img, num_inter)
				self.save_images(pred_p, pred_std, imageName,outName, outReplaceStr, 'valid')

		elif self.params.DATA_USAGE == 'S1S2All':
			
			train_imgPathsS1, valid_imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=False)
			train_imgPathsS2, valid_imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=False)
			
			if(len(train_imgPathsS1)!=len(train_imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files on training. Correct the amount in the folder.")
			if(len(valid_imgPathsS1)!=len(valid_imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files on validation. Correct the amount in the folder.")
			
			print('Number of files = ', len(train_imgPathsS1))

			for imgPath in tqdm(train_imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				imgS1 = np.expand_dims(imgS1[:,:,:4], axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TRAIN_S2_DIR, imgpathS2)
				imgS2 = load_img(imgPathS2)
				imgS2 = np.expand_dims(imgS2[:,:,:4], axis=0).astype('float32')
				img = np.concatenate((imgS2,imgS1),axis=3)
				pred_p, pred_std = self.infer_MCDropOut(model, img, num_inter)
				self.save_images(pred_p, pred_std, imageName,outName, outReplaceStr, 'train')

			print('Number of files = ', len(valid_imgPathsS1))

			for imgPath in tqdm(valid_imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				imgS1 = np.expand_dims(imgS1[:,:,:4], axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.VALID_S2_DIR, imgpathS2)
				imgS2 = load_img(imgPathS2)
				imgS2 = np.expand_dims(imgS2[:,:,:4], axis=0).astype('float32')
				img = np.concatenate((imgS2,imgS1),axis=3)
				pred_p, pred_std = self.infer_MCDropOut(model, img, num_inter)
				self.save_images(pred_p, pred_std, imageName,outName, outReplaceStr, 'valid')

		elif self.params.DATA_USAGE == 'S2S1IAll':
			
			train_imgPathsS1, valid_imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=False)
			train_imgPathsS2, valid_imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=False)
			
			if(len(train_imgPathsS1)!=len(train_imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files on training. Correct the amount in the folder.")
			if(len(valid_imgPathsS1)!=len(valid_imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files on validation. Correct the amount in the folder.")
			
			print('Number of files = ', len(train_imgPathsS1))

			for imgPath in tqdm(train_imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				imgS1 = np.stack((imgS1[:,:,0],imgS1[:,:,3]), axis=2)
				imgS1 = np.expand_dims(imgS1, axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TRAIN_S2_DIR, imgpathS2)
				imgS2 = load_img(imgPathS2)
				imgS2 = np.expand_dims(imgS2[:,:,:4], axis=0).astype('float32')
				img = np.concatenate((imgS2,imgS1),axis=3)
				pred_p, pred_std = self.infer_MCDropOut(model, img, num_inter)
				self.save_images(pred_p, pred_std, imageName,outName, outReplaceStr, 'train')

			print('Number of files = ', len(valid_imgPathsS1))

			for imgPath in tqdm(valid_imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				imgS1 = np.stack((imgS1[:,:,0],imgS1[:,:,3]), axis=2)
				imgS1 = np.expand_dims(imgS1, axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.VALID_S2_DIR, imgpathS2)
				imgS2 = load_img(imgPathS2)
				imgS2 = np.expand_dims(imgS2[:,:,:4], axis=0).astype('float32')
				img = np.concatenate((imgS2,imgS1),axis=3)
				pred_p, pred_std = self.infer_MCDropOut(model, img, num_inter)
				self.save_images(pred_p, pred_std, imageName,outName, outReplaceStr, 'valid')

		elif self.params.DATA_USAGE == 'S1_S2_Sep':
			
			train_imgPathsS1, valid_imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=False)
			train_imgPathsS2, valid_imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=False)
			
			if(len(train_imgPathsS1)!=len(train_imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files on training. Correct the amount in the folder.")
			if(len(valid_imgPathsS1)!=len(valid_imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files on validation. Correct the amount in the folder.")

			print('Number of files = ', len(train_imgPathsS1))

			for imgPath in tqdm(train_imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				imgS1 = np.expand_dims(imgS1[:,:,:4], axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TRAIN_S2_DIR, imgpathS2)
				imgS2 = load_img(imgPathS2)
				imgS2 = np.expand_dims(imgS2[:,:,:4], axis=0).astype('float32')
				pred_p, pred_std = self.infer_MCDropOut(model, [imgS2,imgS1], num_inter)
				self.save_images(pred_p, pred_std, imageName,outName, outReplaceStr, 'train')

			print('Number of files = ', len(valid_imgPathsS1))

			for imgPath in tqdm(valid_imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				imgS1 = np.expand_dims(imgS1[:,:,:4], axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.VALID_S2_DIR, imgpathS2)
				imgS2 = load_img(imgPathS2)
				imgS2 = np.expand_dims(imgS2[:,:,:4], axis=0).astype('float32')
				pred_p, pred_std = self.infer_MCDropOut(model, [imgS2,imgS1], num_inter)
				self.save_images(pred_p, pred_std, imageName, outName, outReplaceStr,'valid')

		elif self.params.DATA_USAGE == 'S1I_S2_Sep':
			
			train_imgPathsS1, valid_imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=False)
			train_imgPathsS2, valid_imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=False)
			
			if(len(train_imgPathsS1)!=len(train_imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files on training. Correct the amount in the folder.")
			if(len(valid_imgPathsS1)!=len(valid_imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files on validation. Correct the amount in the folder.")

			print('Number of files = ', len(train_imgPathsS1))

			for imgPath in tqdm(train_imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				imgS1 = np.stack((imgS1[:,:,0],imgS1[:,:,3]), axis=2)
				imgS1 = np.expand_dims(imgS1, axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TRAIN_S2_DIR, imgpathS2)
				imgS2 = load_img(imgPathS2)
				imgS2 = np.expand_dims(imgS2[:,:,:4], axis=0
					).astype('float32')
				pred_p, pred_std = self.infer_MCDropOut(model, [imgS2,imgS1], num_inter)
				self.save_images(pred_p, pred_std, imageName,outName, outReplaceStr, 'train')
            

			print('Number of files = ', len(valid_imgPathsS1))

			for imgPath in tqdm(valid_imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				imgS1 = np.stack((imgS1[:,:,0],imgS1[:,:,3]), axis=2)
				imgS1 = np.expand_dims(imgS1, axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.VALID_S2_DIR, imgpathS2)
				imgS2 = load_img(imgPathS2)
				imgS2 = np.expand_dims(imgS2[:,:,:4], axis=0).astype('float32')
				pred_p, pred_std = self.infer_MCDropOut(model, [imgS2,imgS1], num_inter)
				self.save_images(pred_p, pred_std, imageName,outName, outReplaceStr, 'valid')


		elif self.params.DATA_USAGE == 'S1IV_S2_Sep':
			
			train_imgPathsS1, valid_imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=False)
			train_imgPathsS2, valid_imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=False)
			
			if(len(train_imgPathsS1)!=len(train_imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files on training. Correct the amount in the folder.")
			if(len(valid_imgPathsS1)!=len(valid_imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files on validation. Correct the amount in the folder.")

			print('Number of files = ', len(train_imgPathsS1))

			for imgPath in tqdm(train_imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				vol = np.divide(imgS1[:,:,0], imgS1[:,:,3], out=np.zeros_like(imgS1[:,:,0]), where=imgS1[:,:,3]!=0)
				vol = np.nan_to_num(vol, nan=0.0)
				if np.max(vol)>np.min(vol):
					vol = (vol - np.min(vol))/(np.max(vol)-np.min(vol))
				imgS1 = np.stack((imgS1[:,:,0],imgS1[:,:,3],vol), axis=2)
				imgS1 = np.expand_dims(imgS1, axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TRAIN_S2_DIR, imgpathS2)
				imgS2 = load_img(imgPathS2)
				imgS2 = np.expand_dims(imgS2[:,:,:4], axis=0).astype('float32')
				pred_p, pred_std = self.infer_MCDropOut(model, [imgS2,imgS1], num_inter)
				self.save_images(pred_p, pred_std, imageName, outName, outReplaceStr, 'train')

			print('Number of files = ', len(valid_imgPathsS1))

			for imgPath in tqdm(valid_imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				vol = np.divide(imgS1[:,:,0], imgS1[:,:,3], out=np.zeros_like(imgS1[:,:,0]), where=imgS1[:,:,3]!=0)
				vol = np.nan_to_num(vol, nan=0.0)
				if np.max(vol)>np.min(vol):
					vol = (vol - np.min(vol))/(np.max(vol)-np.min(vol))
				imgS1 = np.stack((imgS1[:,:,0],imgS1[:,:,3],vol), axis=2)
				imgS1 = np.expand_dims(imgS1, axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.VALID_S2_DIR, imgpathS2)
				imgS2 = load_img(imgPathS2)
				imgS2 = np.expand_dims(imgS2[:,:,:4], axis=0).astype('float32')
				pred_p, pred_std = self.infer_MCDropOut(model, [imgS2,imgS1], num_inter)
				self.save_images(pred_p, pred_std, imageName, outName, outReplaceStr, 'valid')
		elif self.params.DATA_USAGE == 'S2F_only':

			train_imgPaths, valid_imgPaths = get_image_paths(self.params, typeImg='S2', isTest=False)
			print('Number of files = ', len(train_imgPaths))
			for imgPath in tqdm(train_imgPaths):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S2_FILE_STR, outReplaceStr)
				img = np.expand_dims(load_img(imgPath), axis=0).astype('float32') #Dropar aluns 
				pred_p, pred_std = self.infer_MCDropOut(model, img, num_inter)
				self.save_images(pred_p, pred_std, imageName,outName, outReplaceStr, 'train')

			print('Number of files = ', len(valid_imgPaths))
			for imgPath in tqdm(valid_imgPaths):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S2_FILE_STR, outReplaceStr)
				img = np.expand_dims(load_img(imgPath), axis=0).astype('float32') #Dropar aluns 
				pred_p, pred_std = self.infer_MCDropOut(model, img, num_inter)
				self.save_images(pred_p, pred_std, imageName,outName, outReplaceStr, 'valid')
		elif self.params.DATA_USAGE == 'S1F_only':
			train_imgPaths, valid_imgPaths = get_image_paths(self.params, typeImg='S1', isTest=False)
			print('Number of files = ', len(train_imgPaths))
			for imgPath in tqdm(train_imgPaths):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				img = np.expand_dims(load_img(imgPath), axis=0).astype('float32') #Dropar aluns 
				pred_p, pred_std = self.infer_MCDropOut(model, img, num_inter)
				self.save_images(pred_p, pred_std, imageName,outName, outReplaceStr, 'train')

			print('Number of files = ', len(valid_imgPaths))
			for imgPath in tqdm(valid_imgPaths):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				img = np.expand_dims(load_img(imgPath), axis=0).astype('float32')
				pred_p, pred_std = self.infer_MCDropOut(model, img, num_inter)
				self.save_images(pred_p, pred_std, imageName,outName, outReplaceStr, 'valid')
		elif self.params.DATA_USAGE == 'S1IF_only':
			train_imgPaths, valid_imgPaths = get_image_paths(self.params, typeImg='S1', isTest=False)
			print('Number of files = ', len(train_imgPaths))
			for imgPath in tqdm(train_imgPaths):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				img = load_img(imgPath)
				imgT = np.stack((img[:,:,0],img[:,:,3]), axis=2)
				img = np.concatenate((imgT, img[:,:,4:]), axis=2)
				img = np.expand_dims(img, axis=0).astype('float32') #Dropar aluns 
				pred_p, pred_std = self.infer_MCDropOut(model, img, num_inter)
				self.save_images(pred_p, pred_std, imageName, outName, outReplaceStr, 'train')

			print('Number of files = ', len(valid_imgPaths))
			for imgPath in tqdm(valid_imgPaths):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				img = load_img(imgPath)
				img = np.stack((img[:,:,0],img[:,:,3], img[:,:,4:]), axis=2)
				img = np.expand_dims(img, axis=0).astype('float32') #Dropar aluns 
				pred_p, pred_std = self.infer_MCDropOut(model, img, num_inter)
				self.save_images(pred_p, pred_std, imageName,outName, outReplaceStr, 'valid')
		elif self.params.DATA_USAGE == 'S1IS2FAll':
			train_imgPathsS1, valid_imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=False)
			train_imgPathsS2, valid_imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=False)
			
			if(len(train_imgPathsS1)!=len(train_imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files on training. Correct the amount in the folder.")
			if(len(valid_imgPathsS1)!=len(valid_imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files on validation. Correct the amount in the folder.")
			
			print('Number of files = ', len(train_imgPathsS1))

			for imgPath in tqdm(train_imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				imgS1 = np.stack((imgS1[:,:,0],imgS1[:,:,3]), axis=2)
				imgS1 = np.expand_dims(imgS1, axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TRAIN_S2_DIR, imgpathS2)
				imgS2 = np.expand_dims(load_img(imgPathS2), axis=0).astype('float32')
				img = np.concatenate((imgS2,imgS1),axis=3)
				pred_p, pred_std = self.infer_MCDropOut(model, img, num_inter)
				self.save_images(pred_p, pred_std, imageName,outName, outReplaceStr, 'train')

			print('Number of files = ', len(valid_imgPathsS1))

			for imgPath in tqdm(valid_imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				imgS1 = np.stack((imgS1[:,:,0],imgS1[:,:,3]), axis=2)
				imgS1 = np.expand_dims(imgS1, axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.VALID_S2_DIR, imgpathS2)
				imgS2 = np.expand_dims(load_img(imgPathS2), axis=0).astype('float32')
				img = np.concatenate((imgS2,imgS1),axis=3)
				pred_p, pred_std = self.infer_MCDropOut(model, img, num_inter)
				self.save_images(pred_p, pred_std, imageName,outName, outReplaceStr, 'valid')
		elif self.params.DATA_USAGE == 'S1IFS2FAll':
			train_imgPathsS1, valid_imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=False)
			train_imgPathsS2, valid_imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=False)
			
			if(len(train_imgPathsS1)!=len(train_imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files on training. Correct the amount in the folder.")
			if(len(valid_imgPathsS1)!=len(valid_imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files on validation. Correct the amount in the folder.")
			
			print('Number of files = ', len(train_imgPathsS1))

			for imgPath in tqdm(train_imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				imgS1T = np.stack((imgS1[:,:,0],imgS1[:,:,3]), axis=2)
				imgS1 = np.concatenate((imgS1T, imgS1[:,:,4:]), axis=2)
				imgS1 = np.expand_dims(imgS1, axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TRAIN_S2_DIR, imgpathS2)
				imgS2 = np.expand_dims(load_img(imgPathS2), axis=0).astype('float32')
				img = np.concatenate((imgS2,imgS1),axis=3)
				pred_p, pred_std = self.infer_MCDropOut(model, img, num_inter)
				self.save_images(pred_p, pred_std, imageName,outName, outReplaceStr, 'train')

			print('Number of files = ', len(valid_imgPathsS1))

			for imgPath in tqdm(valid_imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				imgS1T = np.stack((imgS1[:,:,0],imgS1[:,:,3]), axis=2)
				imgS1 = np.concatenate((imgS1T, imgS1[:,:,4:]), axis=2)
				imgS1 = np.expand_dims(imgS1, axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.VALID_S2_DIR, imgpathS2)
				imgS2 = np.expand_dims(load_img(imgPathS2), axis=0).astype('float32')
				img = np.concatenate((imgS2,imgS1),axis=3)
				pred_p, pred_std = self.infer_MCDropOut(model, img, num_inter)
				self.save_images(pred_p, pred_std, imageName,outName, outReplaceStr, 'valid')
		elif self.params.DATA_USAGE == 'S1S2FAll':
			train_imgPathsS1, valid_imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=False)
			train_imgPathsS2, valid_imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=False)
			
			if(len(train_imgPathsS1)!=len(train_imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files on training. Correct the amount in the folder.")
			if(len(valid_imgPathsS1)!=len(valid_imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files on validation. Correct the amount in the folder.")
			
			print('Number of files = ', len(train_imgPathsS1))

			for imgPath in tqdm(train_imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				imgS1 = np.expand_dims(imgS1[:,:,:4], axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TRAIN_S2_DIR, imgpathS2)
				imgS2 = np.expand_dims(load_img(imgPathS2), axis=0).astype('float32')
				img = np.concatenate((imgS2,imgS1),axis=3)
				pred_p, pred_std = self.infer_MCDropOut(model, img, num_inter)
				self.save_images(pred_p, pred_std, imageName,outName, outReplaceStr, 'train')

			print('Number of files = ', len(valid_imgPathsS1))

			for imgPath in tqdm(valid_imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				imgS1 = np.expand_dims(imgS1[:,:,:4], axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.VALID_S2_DIR, imgpathS2)
				imgS2 = np.expand_dims(load_img(imgPathS2), axis=0).astype('float32')
				img = np.concatenate((imgS2,imgS1),axis=3)
				pred_p, pred_std = self.infer_MCDropOut(model, img, num_inter)
				self.save_images(pred_p, pred_std, imageName,outName, outReplaceStr, 'valid')
		elif self.params.DATA_USAGE == 'S1FS2FAll':
			train_imgPathsS1, valid_imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=False)
			train_imgPathsS2, valid_imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=False)
			
			if(len(train_imgPathsS1)!=len(train_imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files on training. Correct the amount in the folder.")
			if(len(valid_imgPathsS1)!=len(valid_imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files on validation. Correct the amount in the folder.")
			
			print('Number of files = ', len(train_imgPathsS1))

			for imgPath in tqdm(train_imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = np.expand_dims(load_img(imgPath), axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TRAIN_S2_DIR, imgpathS2)
				imgS2 = np.expand_dims(load_img(imgPathS2), axis=0).astype('float32')
				img = np.concatenate((imgS2,imgS1),axis=3)
				pred_p, pred_std = self.infer_MCDropOut(model, img, num_inter)
				self.save_images(pred_p, pred_std, imageName,outName, outReplaceStr, 'train')

			print('Number of files = ', len(valid_imgPathsS1))

			for imgPath in tqdm(valid_imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = np.expand_dims(load_img(imgPath), axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.VALID_S2_DIR, imgpathS2)
				imgS2 = np.expand_dims(load_img(imgPathS2), axis=0).astype('float32')
				img = np.concatenate((imgS2,imgS1),axis=3)
				pred_p, pred_std = self.infer_MCDropOut(model, img, num_inter)
				self.save_images(pred_p, pred_std, imageName,outName, outReplaceStr, 'valid')
		elif self.params.DATA_USAGE == 'S1_S2F_Sep':
			train_imgPathsS1, valid_imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=False)
			train_imgPathsS2, valid_imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=False)
			
			if(len(train_imgPathsS1)!=len(train_imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files on training. Correct the amount in the folder.")
			if(len(valid_imgPathsS1)!=len(valid_imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files on validation. Correct the amount in the folder.")

			print('Number of files = ', len(train_imgPathsS1))

			for imgPath in tqdm(train_imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				imgS1 = np.expand_dims(imgS1[:,:,:4], axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TRAIN_S2_DIR, imgpathS2)
				imgS2 = np.expand_dims(load_img(imgPathS2), axis=0).astype('float32')
				pred_p, pred_std = self.infer_MCDropOut(model, [imgS2,imgS1], num_inter)
				self.save_images(pred_p, pred_std, imageName,outName, outReplaceStr, 'train')

			print('Number of files = ', len(valid_imgPathsS1))

			for imgPath in tqdm(valid_imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				imgS1 = np.expand_dims(imgS1[:,:,:4], axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.VALID_S2_DIR, imgpathS2)
				imgS2 = np.expand_dims(load_img(imgPathS2), axis=0).astype('float32')
				pred_p, pred_std = self.infer_MCDropOut(model, [imgS2,imgS1], num_inter)
				self.save_images(pred_p, pred_std, imageName, outName, outReplaceStr,'valid')
		elif self.params.DATA_USAGE == 'S1F_S2F_Sep':
			train_imgPathsS1, valid_imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=False)
			train_imgPathsS2, valid_imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=False)
			
			if(len(train_imgPathsS1)!=len(train_imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files on training. Correct the amount in the folder.")
			if(len(valid_imgPathsS1)!=len(valid_imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files on validation. Correct the amount in the folder.")

			print('Number of files = ', len(train_imgPathsS1))

			for imgPath in tqdm(train_imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = np.expand_dims(load_img(imgPath), axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TRAIN_S2_DIR, imgpathS2)
				imgS2 = np.expand_dims(load_img(imgPathS2), axis=0).astype('float32')
				pred_p, pred_std = self.infer_MCDropOut(model, [imgS2,imgS1], num_inter)
				self.save_images(pred_p, pred_std, imageName,outName, outReplaceStr, 'train')

			print('Number of files = ', len(valid_imgPathsS1))

			for imgPath in tqdm(valid_imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = np.expand_dims(load_img(imgPath), axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.VALID_S2_DIR, imgpathS2)
				imgS2 = np.expand_dims(load_img(imgPathS2), axis=0).astype('float32')
				pred_p, pred_std = self.infer_MCDropOut(model, [imgS2,imgS1], num_inter)
				self.save_images(pred_p, pred_std, imageName, outName, outReplaceStr,'valid')
		elif self.params.DATA_USAGE == 'S1I_S2F_Sep':
			train_imgPathsS1, valid_imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=False)
			train_imgPathsS2, valid_imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=False)
			
			if(len(train_imgPathsS1)!=len(train_imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files on training. Correct the amount in the folder.")
			if(len(valid_imgPathsS1)!=len(valid_imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files on validation. Correct the amount in the folder.")

			print('Number of files = ', len(train_imgPathsS1))

			for imgPath in tqdm(train_imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				imgS1 = np.stack((imgS1[:,:,0],imgS1[:,:,3]), axis=2)
				imgS1 = np.expand_dims(imgS1, axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TRAIN_S2_DIR, imgpathS2)
				imgS2 = np.expand_dims(load_img(imgPathS2), axis=0).astype('float32')
				pred_p, pred_std = self.infer_MCDropOut(model, [imgS2,imgS1], num_inter)
				self.save_images(pred_p, pred_std, imageName,outName, outReplaceStr, 'train')
            

			print('Number of files = ', len(valid_imgPathsS1))

			for imgPath in tqdm(valid_imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				imgS1 = np.stack((imgS1[:,:,0],imgS1[:,:,3]), axis=2)
				imgS1 = np.expand_dims(imgS1, axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.VALID_S2_DIR, imgpathS2)
				imgS2 = np.expand_dims(load_img(imgPathS2), axis=0).astype('float32')
				pred_p, pred_std = self.infer_MCDropOut(model, [imgS2,imgS1], num_inter)
				self.save_images(pred_p, pred_std, imageName,outName, outReplaceStr, 'valid')
		elif self.params.DATA_USAGE == 'S1IF_S2F_Sep':
			train_imgPathsS1, valid_imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=False)
			train_imgPathsS2, valid_imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=False)
			
			if(len(train_imgPathsS1)!=len(train_imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files on training. Correct the amount in the folder.")
			if(len(valid_imgPathsS1)!=len(valid_imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files on validation. Correct the amount in the folder.")

			print('Number of files = ', len(train_imgPathsS1))

			for imgPath in tqdm(train_imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				imgS1T = np.stack((imgS1[:,:,0],imgS1[:,:,3]), axis=2)
				imgS1 = np.concatenate((imgS1T, imgS1[:,:,4:]), axis=2)
				imgS1 = np.expand_dims(imgS1, axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TRAIN_S2_DIR, imgpathS2)
				imgS2 = np.expand_dims(load_img(imgPathS2), axis=0).astype('float32')
				pred_p, pred_std = self.infer_MCDropOut(model, [imgS2,imgS1], num_inter)
				self.save_images(pred_p, pred_std, imageName,outName, outReplaceStr, 'train')
            

			print('Number of files = ', len(valid_imgPathsS1))

			for imgPath in tqdm(valid_imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				imgS1 = np.stack((imgS1[:,:,0],imgS1[:,:,3], imgS1[:,:,4:]), axis=2)
				imgS1 = np.expand_dims(imgS1, axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.VALID_S2_DIR, imgpathS2)
				imgS2 = np.expand_dims(load_img(imgPathS2), axis=0).astype('float32')
				pred_p, pred_std = self.infer_MCDropOut(model, [imgS2,imgS1], num_inter)
				self.save_images(pred_p, pred_std, imageName,outName, outReplaceStr, 'valid')
		else:
			raise ValueError("Incorrect DATA_USAGE definition. Correct in unetParams")


	def test (self):

		numPredChannels = self.params.NUM_CATEGORIES
		outReplaceStr = self.params.CLSPRED_FILE_STR
		model = self.build_model()
		selector = CheckpointSel(self.params.LOGFILE)
		test_model_num = selector.findBest(self.params, train=False)
		test_model_name = 'weights.'+str(test_model_num).zfill(2)+self.params.SAVE_CHECK_EXT
		print(test_model_name)
		model.load_weights(os.path.join(self.params.TEST_MODEL_DIR,test_model_name), by_name=True)

		model.summary()


		if self.params.DATA_USAGE == 'S1_I_only':
			
			imgPaths = get_image_paths(self.params, typeImg='S1', isTest=True)
			print(len(imgPaths))
			print('Number of files = ', len(imgPaths))
			for imgPath in tqdm(imgPaths):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				img = load_img(imgPath)
				if self.params.S1_IN_DB:
					img = dBConvert(img)
				img = np.stack((img[:,:,0],img[:,:,3]), axis=2)
				img = np.expand_dims(img, axis=0).astype('float32') #Dropar aluns 
				if self.params.NETWORK == 'labelUnet':
					output30, output10 = model.predict(img)
					pred = output10[0,:,:,:]
				else:
					pred = model.predict(img)[0,:,:,:]
				mclu_ = mclu(pred)
				pred = np.argmax(pred[:,:,1:], axis=2).astype('uint8')+1
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)

		elif self.params.DATA_USAGE == 'S1_only':
			
			imgPaths = get_image_paths(self.params, typeImg='S1', isTest=True)
			print('Number of files = ', len(imgPaths))
			for imgPath in tqdm(imgPaths):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				img = load_img(imgPath)
				if self.params.S1_IN_DB:
					img=dBConvert(img)
				img = np.expand_dims(img[:,:,:4], axis=0).astype('float32') #Dropar aluns 
				if self.params.NETWORK == 'labelUnet':
					output30, output10 = model.predict(img)
					pred = output10[0,:,:,:]
				else:
					pred = model.predict(img)[0,:,:,:]
				mclu_ = mclu(pred)
				pred = np.argmax(pred[:,:,1:], axis=2).astype('uint8')+1
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)

		elif self.params.DATA_USAGE == 'S1_Complex':

			imgPaths = get_image_paths(self.params, typeImg='S1', isTest=True)
			print('Number of files = ', len(imgPaths))
			for imgPath in tqdm(imgPaths):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				img = load_img(imgPath)
				if self.params.S1_IN_DB:
					img =  dBConvert(img)
				img = img.astype(np.complex64)
				img = np.stack((img[:,:,0],img[:,:,3], img[:,:,1]+1j*img[:,:,2]), axis=2)
				img = np.expand_dims(img, axis=0).astype(np.complex64) #Dropar aluns
				if self.params.NETWORK == 'labelUnet':
					output30, output10 = model.predict(img)
					pred = output10[0,:,:,:]
				else:
					pred = model.predict(img)[0,:,:,:]
				mclu_ = mclu(pred)
				pred = np.argmax(pred[:,:,1:], axis=2).astype('uint8')+1
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)


		elif self.params.DATA_USAGE == 'S2_S1_Complex':

			imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=True)
			imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=True)
			
			if(len(imgPathsS1)!=len(imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files. Correct the amount in the folder.")
			
			print('Number of files = ', len(imgPathsS1))


			for imgPath in tqdm(imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				if self.params.S1_IN_DB:
					imgS1 =  dBConvert(imgS1)
				imgS1 = imgS1.astype(np.complex64)
				imgS1 = np.stack((imgS1[:,:,0],imgS1[:,:,3], imgS1[:,:,1]+1j*imgS1[:,:,2]), axis=2)
				imgS1 = np.expand_dims(imgS1, axis=0).astype(np.complex64) #Dropar aluns
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TEST_DIR, imgpathS2)
				imgS2 = load_img(imgPathS2)
				#imgS2 = imgS2.astype(np.complex64)
				imgS2 = np.expand_dims(imgS2[:,:,:4], axis=0).astype(np.complex64)
				img = [imgS2,imgS1]
				if self.params.NETWORK == 'labelUnet':
					output30, output10 = model.predict(img)
					pred = output10[0,:,:,:]
				else:
					pred = model.predict(img)[0,:,:,:]
				mclu_ = mclu(pred)
				pred = np.argmax(pred[:,:,1:], axis=2).astype('uint8')+1
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)

		elif self.params.DATA_USAGE == 'S2_only':
			
			imgPaths = get_image_paths(self.params, typeImg='S2', isTest=True)
			print('Number of files = ', len(imgPaths))
			for imgPath in tqdm(imgPaths):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S2_FILE_STR, outReplaceStr)
				img = load_img(imgPath)
				img = np.expand_dims(img[:,:,:4], axis=0).astype('float32') #Dropar aluns 
				if self.params.NETWORK == 'labelUnet':
					output30, output10 = model.predict(img)
					#print(output10.shape)
					pred = output10[0,:,:,0:self.params.NUM_CATEGORIES]
					#pred = output10[0,:,:,:]
				else:
					pred = model.predict(img)[0,:,:,:]
				mclu_ = mclu(pred)
				pred = np.argmax(pred[:,:,1:], axis=2).astype('uint8')+1
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)

		elif self.params.DATA_USAGE == 'S1S2All':
			
			imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=True)
			imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=True)
			
			if(len(imgPathsS1)!=len(imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files. Correct the amount in the folder.")
			
			print('Number of files = ', len(imgPathsS1))

			for imgPath in tqdm(imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				if self.params.S1_IN_DB:
					imgS1=dBConvert(imgS1)
				imgS1 = np.expand_dims(imgS1[:,:,:4], axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TEST_DIR, imgpathS2)
				imgS2 = load_img(imgPathS2)
				imgS2 = np.expand_dims(imgS2[:,:,:4], axis=0).astype('float32')
				img = np.concatenate((imgS2,imgS1),axis=3)
				pred = model.predict(img)[0,:,:,:]
				mclu_ = mclu(pred)
				pred = np.argmax(pred[:,:,1:], axis=2).astype('uint8')+1
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)

		elif self.params.DATA_USAGE == 'S2S1IAll':
			
			imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=True)
			imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=True)
			
			if(len(imgPathsS1)!=len(imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files. Correct the amount in the folder.")
			
			print('Number of files = ', len(imgPathsS1))

			for imgPath in tqdm(imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				if self.params.S1_IN_DB:
					imgS1=dBConvert(imgS1)
				imgS1 = np.stack((imgS1[:,:,0],imgS1[:,:,3]), axis=2)
				imgS1 = np.expand_dims(imgS1, axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TEST_DIR, imgpathS2)
				imgS2 = load_img(imgPathS2)
				imgS2 = np.expand_dims(imgS2[:,:,:4], axis=0).astype('float32')
				img = np.concatenate((imgS2,imgS1),axis=3)
				pred = model.predict(img)[0,:,:,:]
				mclu_ = mclu(pred)
				pred = np.argmax(pred[:,:,1:], axis=2).astype('uint8')+1
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)

		elif self.params.DATA_USAGE == 'S1_S2_Sep':
			
			imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=True)
			imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=True)
			
			if(len(imgPathsS1)!=len(imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files. Correct the amount in the folder.")

			print('Number of files = ', len(imgPathsS1))

			for imgPath in tqdm(imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1=load_img(imgPath)
				if self.params.S1_IN_DB:
					imgS1=dBConvert(imgS1)
				imgS1 = np.expand_dims(imgS1[:,:,:4], axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TEST_DIR, imgpathS2)
				imgS2 = load_img(imgPathS2)
				imgS2 = np.expand_dims(imgS2[:,:,:4], axis=0).astype('float32')
				pred = model.predict([imgS2,imgS1])[0,:,:,:]
				mclu_ = mclu(pred)
				pred = np.argmax(pred[:,:,1:], axis=2).astype('uint8')+1
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)

		elif self.params.DATA_USAGE == 'S1I_S2_Sep':
			
			imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=True)
			imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=True)
			
			if(len(imgPathsS1)!=len(imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files. Correct the amount in the folder.")

			print('Number of files = ', len(imgPathsS1))

			for imgPath in tqdm(imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				if self.params.S1_IN_DB:
					imgS1=dBConvert(imgS1)
				imgS1 = np.stack((imgS1[:,:,0],imgS1[:,:,3]), axis=2)
				imgS1 = np.expand_dims(imgS1, axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TEST_DIR, imgpathS2)
				imgS2 = load_img(imgPathS2)
				imgS2 = np.expand_dims(imgS2[:,:,:4], axis=0).astype('float32')
				
				if self.params.NETWORK == 'labelUnet':
					output30, output10 = model.predict([imgS2,imgS1])
					pred = output10[0,:,:,:]
				else:
					pred = model.predict([imgS2,imgS1])[0,:,:,:]
				mclu_ = mclu(pred)
				pred = np.argmax(pred[:,:,1:], axis=2).astype('uint8')+1
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)

		elif self.params.DATA_USAGE == 'S1IV_S2_Sep':
			imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=True)
			imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=True)
			
			if(len(imgPathsS1)!=len(imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files. Correct the amount in the folder.")

			print('Number of files = ', len(imgPathsS1))

			for imgPath in tqdm(imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				vol = np.divide(imgS1[:,:,0], imgS1[:,:,3], out=np.zeros_like(imgS1[:,:,0]), where=imgS1[:,:,3]!=0)
				vol = np.nan_to_num(vol, nan=0.0)
				if self.params.S1_IN_DB:
					imgS1=dBConvert(imgS1)
				if np.max(vol)>np.min(vol):
					vol = (vol - np.min(vol))/(np.max(vol)-np.min(vol))
				imgS1 = np.stack((imgS1[:,:,0],imgS1[:,:,3],vol), axis=2)
				imgS1 = np.expand_dims(imgS1, axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TEST_DIR, imgpathS2)
				imgS2 = load_img(imgPathS2)
				imgS2 = np.expand_dims(imgS2[:,:,:4], axis=0).astype('float32')
				if self.params.NETWORK == 'labelUnet':
					output30, output10 = model.predict([imgS2,imgS1])
					pred = output10[0,:,:,:]
				else:
					pred = model.predict([imgS2,imgS1])[0,:,:,:]
				mclu_ = mclu(pred)
				pred = np.argmax(pred[:,:,1:], axis=2).astype('uint8')+1
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)

		elif self.params.DATA_USAGE == 'S2F_only':
			imgPaths = get_image_paths(self.params, typeImg='S2', isTest=True)
			print('Number of files = ', len(imgPaths))
			for imgPath in tqdm(imgPaths):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S2_FILE_STR, outReplaceStr)
				img = np.expand_dims(load_img(imgPath), axis=0).astype('float32') #Dropar aluns 
				if self.params.NETWORK == 'labelUnet':
					output30, output10 = model.predict(img)
					#print(output10.shape)
					pred = output10[0,:,:,0:self.params.NUM_CATEGORIES]
					#pred = output10[0,:,:,:]
				else:
					pred = model.predict(img)[0,:,:,:]
				mclu_ = mclu(pred)
				pred = np.argmax(pred[:,:,1:], axis=2).astype('uint8')+1
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)

		elif self.params.DATA_USAGE == 'S1F_only':
			imgPaths = get_image_paths(self.params, typeImg='S1', isTest=True)
			print('Number of files = ', len(imgPaths))
			for imgPath in tqdm(imgPaths):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1=load_img(imgPath)
				if self.params.S1_IN_DB:
					imgS1=dBConvert(imgS1)
				img = np.expand_dims(imgS1, axis=0).astype('float32') #Dropar aluns 
				pred = model.predict(img)[0,:,:,:]
				mclu_ = mclu(pred)
				pred = np.argmax(pred[:,:,1:], axis=2).astype('uint8')+1
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)

		elif self.params.DATA_USAGE == 'S1IF_only':
			imgPaths = get_image_paths(self.params, typeImg='S1', isTest=True)
			print(len(imgPaths))
			print('Number of files = ', len(imgPaths))
			for imgPath in tqdm(imgPaths):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				img = load_img(imgPath)
				if self.params.S1_IN_DB:
					img = dBConvert(img)
				imgT = np.stack((img[:,:,0],img[:,:,3]), axis=2)
				img = np.concatenate((imgT, img[:,:,4:]), axis=2)
				img = np.expand_dims(img, axis=0).astype('float32') #Dropar aluns 
				pred = model.predict(img)[0,:,:,:]
				mclu_ = mclu(pred)
				pred = np.argmax(pred[:,:,1:], axis=2).astype('uint8')+1
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)

		elif self.params.DATA_USAGE == 'S1IVF_only':
			imgPaths = get_image_paths(self.params, typeImg='S1', isTest=True)
			print(len(imgPaths))
			print('Number of files = ', len(imgPaths))
			for imgPath in tqdm(imgPaths):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				img = load_img(imgPath)
				if self.params.S1_IN_DB:
					img = dBConvert(img)
				vol = np.divide(img[:,:,0], img[:,:,3], out=np.zeros_like(img[:,:,0]), where=img[:,:,3]!=0)
				vol = np.nan_to_num(vol, nan=0.0)
				if np.max(vol)>np.min(vol):
					vol = (vol - np.min(vol))/(np.max(vol)-np.min(vol))
				imgT = np.stack((img[:,:,0],img[:,:,3],vol), axis=2)
				img = np.concatenate((imgT, img[:,:,4:]), axis=2)
				img = np.expand_dims(img, axis=0).astype('float32') #Dropar aluns 
				pred = model.predict(img)[0,:,:,:]
				mclu_ = mclu(pred)
				pred = np.argmax(pred[:,:,1:], axis=2).astype('uint8')+1
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)

		elif self.params.DATA_USAGE == 'S1IV_only':
			imgPaths = get_image_paths(self.params, typeImg='S1', isTest=True)
			print(len(imgPaths))
			print('Number of files = ', len(imgPaths))
			for imgPath in tqdm(imgPaths):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				img = load_img(imgPath)
				if self.params.S1_IN_DB:
					img = dBConvert(img)
				vol = np.divide(img[:,:,0], img[:,:,3], out=np.zeros_like(imgS1[:,:,0]), where=img[:,:,3]!=0)
				vol = np.nan_to_num(vol, nan=0.0)
				if np.max(vol)>np.min(vol):
					vol = (vol - np.min(vol))/(np.max(vol)-np.min(vol))
				img = np.stack((img[:,:,0],img[:,:,3],vol), axis=2)
				img = np.expand_dims(img, axis=0).astype('float32') #Dropar aluns 
				pred = model.predict(img)[0,:,:,:]
				mclu_ = mclu(pred)
				pred = np.argmax(pred[:,:,1:], axis=2).astype('uint8')+1
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)

		elif self.params.DATA_USAGE == 'S1IS2FAll':
			imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=True)
			imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=True)
			
			if(len(imgPathsS1)!=len(imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files. Correct the amount in the folder.")
			
			print('Number of files = ', len(imgPathsS1))

			for imgPath in tqdm(imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				if self.params.S1_IN_DB:
					imgS1=dBConvert(imgS1)
				imgS1 = np.stack((imgS1[:,:,0],imgS1[:,:,3]), axis=2)
				imgS1 = np.expand_dims(imgS1, axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TEST_DIR, imgpathS2)
				imgS2 = np.expand_dims(load_img(imgPathS2), axis=0).astype('float32')
				img = np.concatenate((imgS2,imgS1),axis=3)
				pred = model.predict(img)[0,:,:,:]
				mclu_ = mclu(pred)
				pred = np.argmax(pred[:,:,1:], axis=2).astype('uint8')+1
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)

		elif self.params.DATA_USAGE == 'S1IFS2FAll':
			imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=True)
			imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=True)
			
			if(len(imgPathsS1)!=len(imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files. Correct the amount in the folder.")
			
			print('Number of files = ', len(imgPathsS1))

			for imgPath in tqdm(imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				if self.params.S1_IN_DB:
					imgS1=dBConvert(imgS1)
				imgS1T = np.stack((imgS1[:,:,0],imgS1[:,:,3]), axis=2)
				imgS1 = np.concatenate((imgS1T, imgS1[:,:,4:]), axis=2)
				imgS1 = np.expand_dims(imgS1, axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TEST_DIR, imgpathS2)
				imgS2 = np.expand_dims(load_img(imgPathS2), axis=0).astype('float32')
				img = np.concatenate((imgS2,imgS1),axis=3)
				pred = model.predict(img)[0,:,:,:]
				mclu_ = mclu(pred)
				pred = np.argmax(pred[:,:,1:], axis=2).astype('uint8')+1
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)

		elif self.params.DATA_USAGE == 'S1S2FAll':
			imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=True)
			imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=True)
			
			if(len(imgPathsS1)!=len(imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files. Correct the amount in the folder.")
			
			print('Number of files = ', len(imgPathsS1))

			for imgPath in tqdm(imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				if self.params.S1_IN_DB:
					imgS1=dBConvert(imgS1)
				imgS1 = np.expand_dims(imgS1[:,:,:4], axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TEST_DIR, imgpathS2)
				imgS2 = np.expand_dims(load_img(imgPathS2), axis=0).astype('float32')
				img = np.concatenate((imgS2,imgS1),axis=3)
				pred = model.predict(img)[0,:,:,:]
				mclu_ = mclu(pred)
				pred = np.argmax(pred[:,:,1:], axis=2).astype('uint8')+1
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)

		elif self.params.DATA_USAGE == 'S1FS2FAll':
			imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=True)
			imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=True)
			
			if(len(imgPathsS1)!=len(imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files. Correct the amount in the folder.")
			
			print('Number of files = ', len(imgPathsS1))

			for imgPath in tqdm(imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1=load_img(imgPath)
				if self.params.S1_IN_DB:
					imgS1=dBConvert(imgS1)
				imgS1 = np.expand_dims(imgS1, axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TEST_DIR, imgpathS2)
				imgS2 = np.expand_dims(load_img(imgPathS2), axis=0).astype('float32')
				img = np.concatenate((imgS2,imgS1),axis=3)
				pred = model.predict(img)[0,:,:,:]
				mclu_ = mclu(pred)
				pred = np.argmax(pred[:,:,1:], axis=2).astype('uint8')+1
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)

		elif self.params.DATA_USAGE == 'S1_S2F_Sep':
			imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=True)
			imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=True)
			
			if(len(imgPathsS1)!=len(imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files. Correct the amount in the folder.")

			print('Number of files = ', len(imgPathsS1))

			for imgPath in tqdm(imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1=load_img(imgPath)
				if self.params.S1_IN_DB:
					imgS1=dBConvert(imgS1)
				imgS1 = np.expand_dims(imgS1[:,:,:4], axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TEST_DIR, imgpathS2)
				imgS2 = np.expand_dims(load_img(imgPathS2), axis=0).astype('float32')
				pred = model.predict([imgS2,imgS1])[0,:,:,:]
				mclu_ = mclu(pred)
				pred = np.argmax(pred[:,:,1:], axis=2).astype('uint8')+1
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)

		elif self.params.DATA_USAGE == 'S1I_S2F_Sep':
			imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=True)
			imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=True)
			
			if(len(imgPathsS1)!=len(imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files. Correct the amount in the folder.")

			print('Number of files = ', len(imgPathsS1))

			for imgPath in tqdm(imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				if self.params.S1_IN_DB:
					imgS1=dBConvert(imgS1)
				imgS1 = np.stack((imgS1[:,:,0],imgS1[:,:,3]), axis=2)
				imgS1 = np.expand_dims(imgS1, axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TEST_DIR, imgpathS2)
				imgS2 = np.expand_dims(load_img(imgPathS2), axis=0).astype('float32')
				
				if self.params.NETWORK == 'labelUnet':
					output30, output10 = model.predict([imgS2,imgS1])
					pred = output10[0,:,:,:]
				else:
					pred = model.predict([imgS2,imgS1])[0,:,:,:]
				mclu_ = mclu(pred)
				pred = np.argmax(pred[:,:,1:], axis=2).astype('uint8')+1
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)

		elif self.params.DATA_USAGE == 'S1IF_S2F_Sep':
			imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=True)
			imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=True)
			
			if(len(imgPathsS1)!=len(imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files. Correct the amount in the folder.")

			print('Number of files = ', len(imgPathsS1))

			for imgPath in tqdm(imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				if self.params.S1_IN_DB:
					imgS1=dBConvert(imgS1)
				imgS1T = np.stack((imgS1[:,:,0],imgS1[:,:,3]), axis=2)
				imgS1 = np.concatenate((imgS1T, imgS1[:,:,4:]), axis=2)
				imgS1 = np.expand_dims(imgS1, axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TEST_DIR, imgpathS2)
				imgS2 = np.expand_dims(load_img(imgPathS2), axis=0).astype('float32')
				
				if self.params.NETWORK == 'labelUnet':
					output30, output10 = model.predict([imgS2,imgS1])
					pred = output10[0,:,:,:]
				else:
					pred = model.predict([imgS2,imgS1])[0,:,:,:]
				mclu_ = mclu(pred)
				pred = np.argmax(pred[:,:,1:], axis=2).astype('uint8')+1
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)


		elif self.params.DATA_USAGE == 'S1IVF_S2F_Sep':
			imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=True)
			imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=True)
			
			if(len(imgPathsS1)!=len(imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files. Correct the amount in the folder.")

			print('Number of files = ', len(imgPathsS1))

			for imgPath in tqdm(imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				vol = np.divide(imgS1[:,:,0], imgS1[:,:,3], out=np.zeros_like(imgS1[:,:,0]), where=imgS1[:,:,3]!=0)
				vol = np.nan_to_num(vol, nan=0.0)
				if self.params.S1_IN_DB:
					imgS1=dBConvert(imgS1)
				if np.max(vol)>np.min(vol):
					vol = (vol - np.min(vol))/(np.max(vol)-np.min(vol))
				imgS1T = np.stack((imgS1[:,:,0],imgS1[:,:,3],vol), axis=2)
				imgS1 = np.concatenate((imgS1T, imgS1[:,:,4:]), axis=2)
				imgS1 = np.expand_dims(imgS1, axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TEST_DIR, imgpathS2)
				imgS2 = np.expand_dims(load_img(imgPathS2), axis=0).astype('float32')
				
				if self.params.NETWORK == 'labelUnet':
					output30, output10 = model.predict([imgS2,imgS1])
					pred = output10[0,:,:,:]
				else:
					pred = model.predict([imgS2,imgS1])[0,:,:,:]
				mclu_ = mclu(pred)
				pred = np.argmax(pred[:,:,1:], axis=2).astype('uint8')+1
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)

		elif self.params.DATA_USAGE == 'S1IVF_S2_Sep':
			imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=True)
			imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=True)
			
			if(len(imgPathsS1)!=len(imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files. Correct the amount in the folder.")

			print('Number of files = ', len(imgPathsS1))

			for imgPath in tqdm(imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				vol = np.divide(imgS1[:,:,0], imgS1[:,:,3], out=np.zeros_like(imgS1[:,:,0]), where=imgS1[:,:,3]!=0)
				vol = np.nan_to_num(vol, nan=0.0)
				if self.params.S1_IN_DB:
					imgS1=dBConvert(imgS1)
				if np.max(vol)>np.min(vol):
					vol = (vol - np.min(vol))/(np.max(vol)-np.min(vol))
				imgS1T = np.stack((imgS1[:,:,0],imgS1[:,:,3],vol), axis=2)
				imgS1 = np.concatenate((imgS1T, imgS1[:,:,4:]), axis=2)
				imgS1 = np.expand_dims(imgS1, axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TEST_DIR, imgpathS2)
				imgS2 = load_img(imgPathS2)
				imgS2 = np.expand_dims(imgS2[:,:,:4], axis=0).astype('float32')
				if self.params.NETWORK == 'labelUnet':
					output30, output10 = model.predict([imgS2,imgS1])
					pred = output10[0,:,:,:]
				else:
					pred = model.predict([imgS2,imgS1])[0,:,:,:]
				mclu_ = mclu(pred)
				pred = np.argmax(pred[:,:,1:], axis=2).astype('uint8')+1
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)

		elif self.params.DATA_USAGE == 'S1I_S2Some':
			imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=True)
			imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=True)
			
			if(len(imgPathsS1)!=len(imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files. Correct the amount in the folder.")

			print('Number of files = ', len(imgPathsS1))

			for imgPath in tqdm(imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				if self.params.S1_IN_DB:
					imgS1=dBConvert(imgS1)
				imgS1 = np.stack((imgS1[:,:,0],imgS1[:,:,3]), axis=2)
				imgS1 = np.expand_dims(imgS1, axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TEST_DIR, imgpathS2)
				imgS2 = load_img(imgPathS2)
				imgS2 = np.concatenate((imgS2[:,:,5:], imgS2[:,:,8:]), axis=2)
				imgS2 = np.expand_dims(imgS2, axis=0).astype('float32')
				if self.params.NETWORK == 'labelUnet':
					output30, output10 = model.predict([imgS2,imgS1])
					pred = output10[0,:,:,:]
				else:
					pred = model.predict([imgS2,imgS1])[0,:,:,:]
				mclu_ = mclu(pred)
				pred = np.argmax(pred[:,:,1:], axis=2).astype('uint8')+1
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)

		elif self.params.DATA_USAGE == 'S1IV_S2Some':
			imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=True)
			imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=True)
			
			if(len(imgPathsS1)!=len(imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files. Correct the amount in the folder.")

			print('Number of files = ', len(imgPathsS1))

			for imgPath in tqdm(imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1 = load_img(imgPath)
				vol = np.divide(imgS1[:,:,0], imgS1[:,:,3], out=np.zeros_like(imgS1[:,:,0]), where=imgS1[:,:,3]!=0)
				vol = np.nan_to_num(vol, nan=0.0)
				if self.params.S1_IN_DB:
					imgS1=dBConvert(imgS1)
				if np.max(vol)>np.min(vol):
					vol = (vol - np.min(vol))/(np.max(vol)-np.min(vol))
				imgS1 = np.stack((imgS1[:,:,0],imgS1[:,:,3],vol), axis=2)
				imgS1 = np.expand_dims(imgS1, axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TEST_DIR, imgpathS2)
				imgS2 = load_img(imgPathS2)
				imgS2 = np.concatenate((imgS2[:,:,5:], imgS2[:,:,8:]), axis=2)
				imgS2 = np.expand_dims(imgS2, axis=0).astype('float32')
				if self.params.NETWORK == 'labelUnet':
					output30, output10 = model.predict([imgS2,imgS1])
					pred = output10[0,:,:,:]
				else:
					pred = model.predict([imgS2,imgS1])[0,:,:,:]
				mclu_ = mclu(pred)
				pred = np.argmax(pred[:,:,1:], axis=2).astype('uint8')+1
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)

		elif self.params.DATA_USAGE == 'S1F_S2F_Sep':
			imgPathsS1 = get_image_paths(self.params, typeImg='S1', isTest=True)
			imgPathsS2 = get_image_paths(self.params, typeImg='S2', isTest=True)
			
			if(len(imgPathsS1)!=len(imgPathsS2)):
				raise ValueError("Missing corresponding S1 and S2 files. Correct the amount in the folder.")

			print('Number of files = ', len(imgPathsS1))

			for imgPath in tqdm(imgPathsS1):
				imageName = os.path.split(imgPath)[-1]
				outName = imageName.replace(self.params.S1_FILE_STR, outReplaceStr)
				imgS1=load_img(imgPath)
				if self.params.S1_IN_DB:
					imgS1=dBConvert(imgS1)
				imgS1 = np.expand_dims(imgS1, axis=0).astype('float32') #Dropar aluns 
				imgpathS2 = imageName.replace(self.params.S1_FILE_STR, self.params.S2_FILE_STR)
				imgPathS2 = os.path.join(self.params.TEST_DIR, imgpathS2)
				imgS2 = np.expand_dims(load_img(imgPathS2), axis=0).astype('float32')
				pred = model.predict([imgS2,imgS1])[0,:,:,:]
				mclu_ = mclu(pred)
				pred = np.argmax(pred[:,:,1:], axis=2).astype('uint8')+1
            	#Nao necessariamenente eu salvo a imagem, pois posso precisar montar a imagem total de base do que montar os patches.
            	#########
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred)
				tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName.replace(outReplaceStr,self.params.LABEL_MCLU_STR)), mclu_)
		else:
			raise ValueError("Incorrect DATA_USAGE definition. Correct in unetParams")

	def get_model(self):

		model = None
		#this will depend on both NETWORK and DATAUSAGE
		if self.params.NETWORK == 'FullUnet':
			if self.params.DATA_USAGE == 'S1_I_only':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],2), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],2)
				model = unetFull.unetFull(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1_only':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],4), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4)
				model = unetFull.unetFull(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S2_only':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],4), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4)
				model = unetFull.unetFull(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1S2All':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],4+4), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4+4)
				model = unetFull.unetFull(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S2S1IAll':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],2+4), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],2+4)
				model = unetFull.unetFull(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1_S2_Sep':
				inputTensorS1 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],4), name = 'S1data')
				inputShapeS1 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4)
				inputTensorS2 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],4), name = 'S2data')
				inputShapeS2 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4)
				model = unetFull.UnetDoubleFull(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1I_S2_Sep':
				inputTensorS1 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],2), name = 'S1data')
				inputShapeS1 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],2)
				inputTensorS2 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],4), name = 'S2data')
				inputShapeS2 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4)
				model = unetFull.UnetDoubleFull(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1IV_S2_Sep':
				inputTensorS1 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],3), name = 'S1data')
				inputShapeS1 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],3)
				inputTensorS2 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],4), name = 'S2data')
				inputShapeS2 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4)
				model = unetFull.UnetDoubleFull(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S2F_only':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S2), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S2)
				model = unetFull.unetFull(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1F_only':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S1), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S1)
				model = unetFull.unetFull(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1IF_only':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],5), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],5)
				model = unetFull.unetFull(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1IS2FAll':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],2+self.params.NUM_CHANNELS_S2), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],2+self.params.NUM_CHANNELS_S2)
				model = unetFull.unetFull(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1IFS2FAll':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],5+self.params.NUM_CHANNELS_S2), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],5+self.params.NUM_CHANNELS_S2)
				model = unetFull.unetFull(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1S2FAll':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],4+self.params.NUM_CHANNELS_S2), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4+self.params.NUM_CHANNELS_S2)
				model = unetFull.unetFull(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1FS2FAll':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S1+self.params.NUM_CHANNELS_S2), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S1+self.params.NUM_CHANNELS_S2)
				model = unetFull.unetFull(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1_S2F_Sep':
				inputTensorS1 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],4), name = 'S1data')
				inputShapeS1 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4)
				inputTensorS2 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S2), name = 'S2data')
				inputShapeS2 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S2)
				model = unetFull.UnetDoubleFull(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1I_S2F_Sep':
				inputTensorS1 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],2), name = 'S1data')
				inputShapeS1 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],2)
				inputTensorS2 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S2), name = 'S2data')
				inputShapeS2 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S2)
				model = unetFull.UnetDoubleFull(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1IF_S2F_Sep':
				inputTensorS1 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],5), name = 'S1data')
				inputShapeS1 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],5)
				inputTensorS2 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S2), name = 'S2data')
				inputShapeS2 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S2)
				model = unetFull.UnetDoubleFull(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1F_S2F_Sep':
				inputTensorS1 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S1), name = 'S1data')
				inputShapeS1 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S1)
				inputTensorS2 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S2), name = 'S2data')
				inputShapeS2 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S2)
				model = unetFull.UnetDoubleFull(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1IVF_S2F_Sep':
				inputTensorS1 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1], 6), name = 'S1data')
				inputShapeS1 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1], 6)
				inputTensorS2 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S2), name = 'S2data')
				inputShapeS2 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S2)
				model = unetFull.UnetDoubleFull(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
			else:
				raise ValueError("Incorrect DATA_USAGE definition. Correct in unetParams")
		elif self.params.NETWORK == 'Unet3':
			if self.params.DATA_USAGE == 'S1_I_only':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],2), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],2)
				model = None
			elif self.params.DATA_USAGE == 'S1_only':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],4), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4)
				model = None
			elif elf.params.DATA_USAGE == 'S2_only':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],4), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4)
				model = None
			elif self.params.DATA_USAGE == 'S1S2All':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],4+4), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4+4)
				model = None
			elif self.params.DATA_USAGE == 'S2S1IAll':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],2+4), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],2+4)
				model = None
			elif self.params.DATA_USAGE == 'S1_S2_Sep':
				inputTensorS1 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],4), name = 'S1data')
				inputShapeS1 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4)
				inputTensorS2 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],4), name = 'S2data')
				inputShapeS2 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4)
				model = None
			elif self.params.DATA_USAGE == 'S1I_S2_Sep':
				inputTensorS1 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],2), name = 'S1data')
				inputShapeS1 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],2)
				inputTensorS2 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],4), name = 'S2data')
				inputShapeS2 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4)
				model = None
			elif self.params.DATA_USAGE == 'S1IVF_S2F_Sep':
				inputTensorS1 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1], 6), name = 'S1data')
				inputShapeS1 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1], 6)
				inputTensorS2 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S2), name = 'S2data')
				inputShapeS2 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S2)
				model = None
			else:
				raise ValueError("Incorrect DATA_USAGE definition. Correct in unetParams")
		elif self.params.NETWORK == 'UnetFree':
			if self.params.DATA_USAGE == 'S1_I_only':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],2), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],2)
				model = unetFull.UnetFree(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1_only':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],4), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4)
				model = unetFull.UnetFree(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S2_only':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],4), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4)
				model = unetFull.UnetFree(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1S2All':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],4+4), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4+4)
				model = unetFull.UnetFree(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S2S1IAll':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],2+4), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],2+4)
				model = unetFull.UnetFree(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1_S2_Sep':
				inputTensorS1 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],4), name = 'S1data')
				inputShapeS1 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4)
				inputTensorS2 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],4), name = 'S2data')
				inputShapeS2 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4)
				model = unetFull.UnetDoubleFree(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1I_S2_Sep':
				inputTensorS1 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],2), name = 'S1data')
				inputShapeS1 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],2)
				inputTensorS2 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],4), name = 'S2data')
				inputShapeS2 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4)
				model = unetFull.UnetDoubleFree(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S2F_only':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S2), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S2)
				model = unetFull.UnetFree(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1F_only':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S1), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S1)
				model = unetFull.UnetFree(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1IF_only':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],5), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],5)
				model = unetFull.UnetFree(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1IS2FAll':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],2+self.params.NUM_CHANNELS_S2), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],2+self.params.NUM_CHANNELS_S2)
				model = unetFull.UnetFree(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1IFS2FAll':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],5+self.params.NUM_CHANNELS_S2), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],5+self.params.NUM_CHANNELS_S2)
				model = unetFull.UnetFree(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1S2FAll':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],4+self.params.NUM_CHANNELS_S2), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4+self.params.NUM_CHANNELS_S2)
				model = unetFull.UnetFree(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1FS2FAll':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S1+self.params.NUM_CHANNELS_S2), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S1+self.params.NUM_CHANNELS_S2)
				model = unetFull.UnetFree(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1_S2F_Sep':
				inputTensorS1 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],4), name = 'S1data')
				inputShapeS1 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4)
				inputTensorS2 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S2), name = 'S2data')
				inputShapeS2 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S2)
				model = unetFull.UnetDoubleFree(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1I_S2F_Sep':
				inputTensorS1 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],2), name = 'S1data')
				inputShapeS1 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],2)
				inputTensorS2 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S2), name = 'S2data')
				inputShapeS2 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S2)
				model = unetFull.UnetDoubleFree(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1IF_S2F_Sep':
				inputTensorS1 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],5), name = 'S1data')
				inputShapeS1 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],5)
				inputTensorS2 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S2), name = 'S2data')
				inputShapeS2 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S2)
				model = unetFull.UnetDoubleFree(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1F_S2F_Sep':
				inputTensorS1 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S1), name = 'S1data')
				inputShapeS1 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S1)
				inputTensorS2 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S2), name = 'S2data')
				inputShapeS2 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S2)
				model = unetFull.UnetDoubleFree(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
			else:
				raise ValueError("Incorrect DATA_USAGE definition. Correct in unetParams")
		elif self.params.NETWORK == 'FusionS1S2Unet':
			if self.params.DATA_USAGE == 'S1_S2_Sep':
				inputTensorS1 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],4), name = 'S1data')
				inputShapeS1 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4)
				inputTensorS2 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],4), name = 'S2data')
				inputShapeS2 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4)
				model = None
			elif self.params.DATA_USAGE == 'S1I_S2_Sep':
				inputTensorS1 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],2), name = 'S1data')
				inputShapeS1 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4)
				inputTensorS2 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],4), name = 'S2data')
				inputShapeS2 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4)
				model = None
			else:
				raise ValueError("Incorrect DATA_USAGE definition for this Network. Correct in unetParams")
		elif self.params.NETWORK == 'QuadUnet':
			if self.params.DATA_USAGE == 'S1_I_only':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],2), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],2)
				model = unetFull.quadUnet(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1I_S2_Sep':
				inputTensorS1 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],2), name = 'S1data')
				inputShapeS1 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],2)
				inputTensorS2 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],4), name = 'S2data')
				inputShapeS2 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4)
				model = unetFull.QuadDoubleUnet(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1I_S2F_Sep':
				inputTensorS1 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],2), name = 'S1data')
				inputShapeS1 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],2)
				inputTensorS2 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S2), name = 'S2data')
				inputShapeS2 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S2)
				model = unetFull.QuadDoubleUnet(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
			else:
				raise ValueError("Incorrect DATA_USAGE definition for this Network. Correct in unetParams")
		elif  self.params.NETWORK == 'QuadUnetD':
			if self.params.DATA_USAGE == 'S1_I_only':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],2), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],2)
				model = unetFull.quadUnetDense(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
			else:
				raise ValueError("Incorrect DATA_USAGE definition for this Network. Correct in unetParams")
		elif  self.params.NETWORK == 'FullUnetWave':
			if self.params.DATA_USAGE == 'S1I_S2_Sep':
				inputTensorS1 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],2), name = 'S1data')
				inputShapeS1 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],2)
				inputTensorS2 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],4), name = 'S2data')
				inputShapeS2 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4)
				model = unetFull.UnetDoubleFullWave(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1I_S2F_Sep':
				inputTensorS1 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],2), name = 'S1data')
				inputShapeS1 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],2)
				inputTensorS2 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S2), name = 'S2data')
				inputShapeS2 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S2)
				model = unetFull.UnetDoubleFullWave(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
			else:
				raise ValueError("Incorrect DATA_USAGE definition for this Network. Correct in unetParams")
		elif self.params.NETWORK =='SwinTransStack':
			if self.params.DATA_USAGE == 'S2S1IAll':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],2+4), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],2+4)
				model = Transformer.SwinTransformerStack(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1I_S2_Sep':
				inputTensorS1 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],2), name = 'S1data')
				inputShapeS1 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],2)
				inputTensorS2 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],4), name = 'S2data')
				inputShapeS2 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4)
				model = Transformer.SwinTransformerDoubleStack(input_shape = [inputShapeS2, inputShapeS1], input_tensor =[inputTensorS2, inputTensorS1] , classes = self.params.NUM_CATEGORIES)
			else:
				raise ValueError("Incorrect DATA_USAGE definition for this Network. Correct in unetParams")
		elif self.params.NETWORK == 'SwinUnet':
			if self.params.DATA_USAGE == 'S2S1IAll':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],2+4), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],2+4)
				model = Transformer.SwinUnet(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
			else:
				raise ValueError("Incorrect DATA_USAGE definition for this Network. Correct in unetParams")
		elif self.params.NETWORK =='FPB_Net':
			if self.params.DATA_USAGE == 'S1I_S2_Sep':
				inputTensorS1 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],2), name = 'S1data')
				inputShapeS1 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],2)
				inputTensorS2 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],4), name = 'S2data')
				inputShapeS2 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4)
				model = unetFull.FPB_Network(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
				#model = unetFull.ModelHighLabeling(input_shape = [inputShapeS2, inputShapeS1], num_classes=self.params.NUM_CATEGORIES)
			else:
				raise ValueError("Incorrect DATA_USAGE definition for this Network. Correct in unetParams")
		elif self.params.NETWORK =='labelUnet':
			if self.params.DATA_USAGE == 'S1I_S2_Sep':
				inputTensorS1 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],2), name = 'S1data')
				inputShapeS1 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],2)
				inputTensorS2 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],4), name = 'S2data')
				inputShapeS2 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4)
				#model = unetFull.labelsuperResUnet_pre(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
				#model = unetFull.labelsuperResUnet_early1(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
				#model = unetFull.labelsuperResUnet_early2(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
				#model = unetFull.labelsuperResUnet_early2_mod(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
				#model = unetFull.labelsuperResUnet_early2_mod2(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
				model = unetFull.labelsuperFPB_early2_mod(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
				#model = unetFull.labelsuperFPB_early1_mod(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
				#model = unetFull.labelsuperFPB_middle_mod(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
				#model = unetFull.labelsuperFPB_late_mod(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
				#model = unetFull.labelsuperResUnet_early1_mod(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
				#model = unetFull.labelsuperResUnet_middle_mod(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
				#model = unetFull.labelsuperResUnet_late_mod(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
				#model = unetFull.labelsuperVGGUnet_early2_mod(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
				#model = unetFull.labelsuperResUnet_early2_mod2(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
				#model = unetFull.labelsuperResUnet_early2_sln(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
				#model = unetFull.labelsuperDenseUnet_early2_mod(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
				#model = unetFull.labelsuperResUnet_late(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES) 
				#model = unetFull.labelsuperResUnet(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1_Complex':
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],3)
				inputTensor = complex_layers.complex_input(shape=inputShape, name = 'data')
				model = unetFull.labelsuperResComplex(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S2_S1_Complex':
				inputShapeS1 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],3)
				inputTensorS1 = complex_layers.complex_input(shape=inputShapeS1, name = 'S1data')
				inputShapeS2 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4)
				inputTensorS2 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],4), name = 'S2data')
				#model = unetFull.labelsuperResComplexFusion_early(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2,inputTensorS1], classes = self.params.NUM_CATEGORIES)
				model = unetFull.labelsuperFPBComplexFusion_early(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2,inputTensorS1], classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1IV_S2_Sep':
				inputTensorS1 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],3), name = 'S1data')
				inputShapeS1 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],3)
				inputTensorS2 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],4), name = 'S2data')
				inputShapeS2 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4)
				#model = unetFull.labelsuperResUnet_pre(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
				#model = unetFull.labelsuperResUnet_early1(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
				#model = unetFull.labelsuperResUnet_early2(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
				model = unetFull.labelsuperResUnet_early2_mod(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
				#model = unetFull.labelsuperResUnet_late(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
				#model = unetFull.labelsuperResUnet(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S2_only':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],4), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4)
				#model = unetFull.unet3Depth1(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
				#model = unetFull.unet3Depth2(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
				#model = unetFull.Resunet3Depth1(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
				model = unetFull.labelsuperFPB_1input_mod(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES) #a testar
			elif self.params.DATA_USAGE == 'S1_I_only':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],2), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],2)
				#model = unetFull.unet3Depth1(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
				#model = unetFull.unet3Depth2(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
				model = unetFull.Resunet3Depth1(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1_only':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],4), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4)
				#model = unetFull.unet3Depth1(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
				#model = unetFull.unet3Depth2(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
								model = unetFull.Resunet3Depth1(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1IVF_S2F_Sep':
				inputTensorS1 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1], 6), name = 'S1data')
				inputShapeS1 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1], 6)
				inputTensorS2 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S2), name = 'S2data')
				inputShapeS2 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S2)
				#model = unetFull.labelsuperResUnet(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
				model = unetFull.labelsuperResUnet_pre(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1IVF_S2_Sep':
				inputTensorS1 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1], 6), name = 'S1data')
				inputShapeS1 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1], 6)
				inputTensorS2 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1], 4), name = 'S2data')
				inputShapeS2 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1], 4)
				#model = unetFull.labelsuperResUnet(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
				model = unetFull.labelsuperResUnet_pre(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1IV_S2Some':
				inputTensorS1 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1], 3), name = 'S1data')
				inputShapeS1 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1], 6)
				inputTensorS2 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1], 7), name = 'S2data')
				inputShapeS2 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1], 4)
				#model = unetFull.labelsuperResUnet(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
				#model = unetFull.labelsuperResUnet_early1(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
				#model = unetFull.labelsuperResUnet_early2(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
				model = unetFull.labelsuperResUnet_early2_mod(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
				#model = unetFull.labelsuperResUnet_pre(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1I_S2Some':
				inputTensorS1 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1], 2), name = 'S1data')
				inputShapeS1 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1], 6)
				inputTensorS2 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1], 7), name = 'S2data')
				inputShapeS2 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1], 4)
				#model = unetFull.labelsuperResUnet(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
				#model = unetFull.labelsuperResUnet_early1(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
				#model = unetFull.labelsuperResUnet_early2(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
				model = unetFull.labelsuperResUnet_early2_mod(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
				#model = unetFull.labelsuperResUnet_pre(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
			else:
				raise ValueError("Incorrect DATA_USAGE definition for this Network. Correct in unetParams")
		elif self.params.NETWORK =='calibration':	
			if self.params.DATA_USAGE == 'S1I_S2_Sep':
				inputTensorS1 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],2), name = 'S1data')
				inputShapeS1 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],2)
				inputTensorS2 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],4), name = 'S2data')
				inputShapeS2 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4)
				selector = CheckpointSel(self.params.OGRMODEL2CALIBLOG)
				cont_model_num = selector.findBest(self.params)
				cont_model_name = 'weights.'+str(cont_model_num).zfill(2)+self.params.SAVE_CHECK_EXT
				weights=os.path.join(self.params.OGRMODEL2CALIB,cont_model_name)
				model = unetFull.uncertaintyTemperatureCalibration(weights,input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
		elif self.params.NETWORK =='unet3Depth0':
			if self.params.DATA_USAGE == 'S2_only':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],4), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4)
				#model = unetFull.unet3Depth0(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
				model = unetFull.unet3Resnet(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
				#model = unetFull.FPB_1input_1out_mod(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES) #a testar
			elif self.params.DATA_USAGE == 'S2F_only':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S2), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S2)
				model = unetFull.unet3Resnet(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1_I_only':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],2), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],2)
				model = unetFull.unet3Resnet(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1IV_only':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],3), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],3)
				model = unetFull.unet3Resnet(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1IF_only':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],5), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],5)
				model = unetFull.unet3Resnet(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1IVF_only':
				inputTensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],6), name = 'data')
				inputShape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],6)
				model = unetFull.unet3Resnet(input_shape = inputShape, input_tensor = inputTensor, classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1IVF_S2F_Sep':
				inputTensorS1 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1], 6), name = 'S1data')
				inputShapeS1 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1], 6)
				inputTensorS2 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S2), name = 'S2data')
				inputShapeS2 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS_S2)
				model = unetFull.ResUnet3_double_middle(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1IVF_S2_Sep':
				inputTensorS1 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1], 6), name = 'S1data')
				inputShapeS1 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1], 6)
				inputTensorS2 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1], 4), name = 'S2data')
				inputShapeS2 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1], 4)
				model = unetFull.ResUnet3_double_middle(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1IV_S2Some':
				inputTensorS1 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1], 3), name = 'S1data')
				inputShapeS1 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1], 3)
				inputTensorS2 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1], 7), name = 'S2data')
				inputShapeS2 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1], 7)
				model = unetFull.ResUnet3_double_middle(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
			elif self.params.DATA_USAGE == 'S1I_S2_Sep':
				inputTensorS1 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],2), name = 'S1data')
				inputShapeS1 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],2)
				inputTensorS2 = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],4), name = 'S2data')
				inputShapeS2 = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],4)
				#model = unetFull.FPB_early2_mod(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES)
				model = unetFull.ResUnet_early2_1out(input_shape = [inputShapeS2, inputShapeS1], input_tensor = [inputTensorS2, inputTensorS1], classes = self.params.NUM_CATEGORIES) #a testar
			else:
				raise ValueError("Incorrect DATA_USAGE definition for this Network. Correct in unetParams")
		return model

	def build_model(self, type_train=None):

		model = self.get_model()
		if self.params.CONTINUE_TRAINING or type_train==1:
			selector = CheckpointSel(self.params.LOGFILE)
			cont_model_num = selector.findBest(self.params)
			cont_model_name = 'weights.'+str(cont_model_num).zfill(2)+self.params.SAVE_CHECK_EXT
			model.load_weights(os.path.join(self.params.CONTINUE_MODEL_FILE_DIR,cont_model_name))

		#class_weights = calc_class_weights_meanprop(self.params,type_train=type_train)
			
		if self.params.OPTIMIZER=='Adam':
			initial_learning = 0.002 #when weighted cross entropy
			#initial_learning = 0.0002 #when dual focal loss
			decay_steps=int(get_total_train_data(self.params)/self.params.BATCH_SZ)
			learning_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning,decay_steps=decay_steps,decay_rate=0.92, staircase=True)
			optimizer = tf.keras.optimizers.Adam(learning_rate=learning_schedule, amsgrad=False)
		if (self.params.NETWORK=='labelUnet'):
			#class_weights10 = calc_class_weights_maxprop(self.params,type_train=type_train)
			class_weights10 = calc_class_weights_meanprop(self.params,type_train=type_train)
			print(class_weights10)
			#class_weights30 = calc_class_weights_maxprop(self.params,type_train=type_train, typeGT=30)
			class_weights30 = calc_class_weights_meanprop(self.params,type_train=type_train, typeGT=30)
			print(class_weights30)
			#model.compile(optimizer, loss={'output30':dual_focal_loss(alpha = class_weights30, beta = 2, gamma = 1.2), 'output10':dual_focal_loss(alpha = class_weights10, beta = 2, gamma = 1.2)}, metrics={'output30':'categorical_accuracy','output10':'categori#model.compile(optimizer, loss={'output30':dual_focal_loss(alpha = class_weights30, beta = 2, gamma = 1.2), 'output10':dual_focal_loss(alpha = class_weights10, beta = 2, gamma = 1.2)}, metrics={'output30':'categorical_accuracy','output10':'categorical_accuracy'})cal_accuracy'})
			#model.compile(optimizer, loss={'output30':dual_focal_loss(alpha = class_weights30, beta = 2, gamma = 1.2), 'output10':dual_focal_loss(alpha = class_weights10, beta = 2, gamma = 1.2)}, metrics={'output30':masked_categoricalAcc(),'output10':masked_categoricalAcc()})
			model.compile(optimizer, loss={'output30':cal_dual_focal_loss(alpha = class_weights30, beta = 2, gamma = 1.2), 'output10':cal_dual_focal_loss(alpha = class_weights10, beta = 2, gamma = 1.2)}, metrics={'output30':masked_categoricalAcc(),'output10':masked_categoricalAcc()})
			#model.compile(optimizer, loss={'output30':focal_loss(alpha = class_weights30, beta = 2, gamma_f = 1.2), 'output10':focal_loss(alpha = class_weights10, beta = 2, gamma_f = 1.2)}, metrics={'output30':masked_categoricalAcc(),'output10':masked_categoricalAcc()})
			#model.compile(optimizer, loss={'output30':cal_focal_loss(alpha = class_weights30), 'output10':cal_focal_loss(alpha = class_weights10)}, metrics={'output30':masked_categoricalAcc(),'output10':masked_categoricalAcc()})
			#model.compile(optimizer, loss={'output30':weighted_cross_entropy(weights = class_weights30), 'output10':weighted_cross_entropy(weights = class_weights10)}, metrics={'output30':masked_categoricalAcc(),'output10':masked_categoricalAcc()})
			#model.compile(optimizer, loss={'output30':dual_focal_loss(alpha = class_weights30, beta = 2, gamma = 1.2), 'combined_output':combinedRQLoss()}, metrics={'output30':'categorical_accuracy'})
		elif self.params.NETWORK =='calibration':
			class_weights = calc_class_weights_meanprop(self.params,type_train=type_train)
			print(class_weights)
			model.compile(optimizer, loss=cal_dual_focal_loss(alpha = class_weights, beta = 2, gamma = 1.2))
			#model.compile(optimizer, loss=calibrationLoss(num_bins=10))
		else:
			#class_weights = calc_class_weights_maxprop(self.params,type_train=type_train)
			class_weights = calc_class_weights_meanprop(self.params,type_train=type_train)
			print(class_weights)
			#model.compile(optimizer, loss=dual_focal_loss(alpha = class_weights, beta = 2, gamma = 1.2), metrics=['categorical_accuracy'])
			model.compile(optimizer, loss=weighted_cross_entropy(weights = class_weights), metrics=['categorical_accuracy'])
		
		return model

	def image_generator(self, trainData, type_train, isValid = False):
		"""
		Generates training batches of image data and ground truth from either semantic or depth files
		:param trainData: training paths of CLS files (string replacement to get RGB and depth files) and starting x,y pixel positions (can be      non-zero if blocking is set to happen in params.py)
		:yield: current batch data
		"""
		idx = np.random.permutation(len(trainData))
		while True:
			batchInds = get_batch_inds(idx, self.params)
			for inds in batchInds:
				imgBatch,labelBatch = load_batch(inds, trainData, self.params, isValid,type_train=type_train)
				yield (imgBatch, labelBatch)