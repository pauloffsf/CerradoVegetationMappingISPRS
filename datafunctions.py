import os
import numpy as np
from glob import glob
import tifffile
import cv2
import findpeaks

from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.utils import to_categorical

'''
function that parses th earguments given to NetworkRun.py
'''
def parse_args(argv, params):

	isTrain = None
    
	argOptions1 = '1st argument options: train, test, droptest, refinetraindata or refinetrain.'
	noArgStr = 'No arguments provided.' 
	incorrectArgStr = 'Incorrect argument provided.'
	insufficientArgStr = 'Not enough arguments provided.'
	exampleUsageStr = 'python runBaseline.py train'

	try:
		trainStr = argv[1].lower()
	except:
		raise ValueError('%s %s %s' % (noArgStr,argOptions1,exampleUsageStr))
    
	print(trainStr)
	if trainStr == 'train':
		isTrain = 0
	elif trainStr == 'test':
		isTrain = 1
	elif trainStr == 'droptest':
		isTrain = 2
	elif trainStr == 'refinetraindata':
		isTrain = 3
	elif trainStr == 'refinetrain':
		isTrain = 4
	else:
		raise ValueError('%s %s %s' % (incorrectArgStr,argOptions1,exampleUsageStr))

	return isTrain


'''
function that resizes an image by a scale
image - numpy array to be resized
scale - scale to multipy the dimensions of the image to
return: numpy array with them size
'''
def resize(image, scale):
	width = int(image.shape[1] *scale)
	height =int(image.shape[0] *scale)
	dim = (width, height)
	return cv2.resize(image, dim, interpolation =cv2.INTER_NEAREST)


'''
functionthat obtains the MCLU from a prediction numpy-array
pred_p - numpy array with prediction probabilities
'''
def mclu (pred_p):

	pred_ord = np.sort(pred_p,axis=2)
	maximum = pred_ord[:,:,-1]
	second_maximum = pred_ord[:,:,-2]

	mclu = maximum - second_maximum
	return mclu


'''
Function to convert an numpy array to dB scale
img - numpy array
'''
def dBConvert(img):
	e=0.0000000001
	img1=img[:,:,0]
	img2=img[:,:,3]
	img1 = np.log10(img1+e)
	img2 = np.log10(img2+e)
	img1=(img1-(-10))/(10-(-10)) #scaling between 0-1
	img2=(img2-(-10))/(10-(-10)) #scaling between 0-1
	img[:,:,0]=img1
	img[:,:,3]=img2
	return img

'''
Function to obtain the entropy as an uncertainty measure in a prediction numpy array
pred_p - numpy array with prediction probabilities
params - parameters of the model
'''
def entropy (pred_p, params):

	entropy = - np.sum(pred_p*np.log(pred_p, where=pred_p > 0), axis=2)
	entropy = entropy/np.log(params.NUM_CATEGORIES-1) #scale to force entropy to be betweeen 0 and 1. 

	return entropy


'''
Function to obtain the voting on a group of predictions
infer - numpy array with prediction probabilities
params - parameters of the model
'''
def voting(params, infer):
	size = infer.shape
	pred = np.zeros((size[1],size[2],params.NUM_CATEGORIES))
	for i in range(size[1]):
		for j in range(size[2]):
			pred[i,j,:]= np.bincount(infer[:,i,j], minlength=params.NUM_CATEGORIES)
	pred = pred/size[0]
	return pred

'''
Function to obtain the standard deviation and probability of the predicted class 
pred_p - numpy array with prediction probabilities
pred - numpy array with the predicted classes
pred_std - numpy array with the standard deviation of the prediction probabilities
'''
def getProbandStdArgMax(pred, pred_p, pred_std):
	prob_arg = np.zeros(pred.shape)
	std_arg = np.zeros(pred.shape)

	for i in range(pred.shape[0]):
		for j in range(pred.shape[1]):
			prob_arg[i,j] = pred_p[i,j,pred[i,j]]
			std_arg[i,j] = pred_std[i,j,pred[i,j]]

	return prob_arg, std_arg


'''
Function that returns a list with all the images path in the training or validation folder, with their corresponding reference image path
params - parameters of the model
typeImg - S1 or S2
isTest=None - get data on the test folder
type_train=0 - if it is a training from scratch or training in active learning (tested but didnt get good results.)
'''
def get_image_paths(params, typeImg, isTest=None, type_train=0):

	train_dir=None
	valid_dir=None
	if type_train == 0:
		train_dir=params.TRAIN_LABEL_DIR
		valid_dir=params.VALID_LABEL_DIR
	else:
		train_dir=params.TRAIN_LABEL_NEW
		valid_dir=params.VALID_LABEL_NEW

	if isTest:
		if typeImg == 'S1':
			return glob(os.path.join(params.TEST_DIR, '%s*.%s' % (params.S1_FILE_STR,params.S1_FILE_EXT)))
		elif typeImg == 'S2':
			return glob(os.path.join(params.TEST_DIR, '%s*.%s' % (params.S2_FILE_STR,params.S2_FILE_EXT)))
		else:
			raise ValueError('Error on Data Type. Possible Values are S1 or S2')
	else:
		#Get only the images that has a corresponding Ground Truth.
		img_paths = []
		valid_img_paths = []
		gt_image_path = '%s*.%s' % (params.LABEL_FILE_STR, params.LABEL_FILE_EXT)
		strReplace = params.LABEL_FILE_STR
		glob_path = os.path.join(train_dir, gt_image_path)
		curr_paths = glob(glob_path)
		for currPath in curr_paths:
			image_name = os.path.split(currPath)[-1]
			if typeImg == 'S1':
				image_name = image_name.replace(strReplace, params.S1_FILE_STR)
				image_name = image_name.replace(params.LABEL_FILE_EXT, params.S1_FILE_EXT)
				img_paths.append(os.path.join(params.TRAIN_S1_DIR, image_name))
			elif typeImg == 'S2':
				image_name = image_name.replace(strReplace, params.S2_FILE_STR)
				image_name = image_name.replace(params.LABEL_FILE_EXT, params.S2_FILE_EXT)
				img_paths.append(os.path.join(params.TRAIN_S2_DIR, image_name))
			else:
				raise ValueError('Error on Data Type. Possible Values are S1 or S2')

		gt_image_path = '%s*.%s' % (params.LABEL_FILE_STR, params.LABEL_FILE_EXT)
		strReplace = params.LABEL_FILE_STR
		glob_path = os.path.join(valid_dir, gt_image_path)
		curr_paths = glob(glob_path)
		for currPath in curr_paths:
			image_name = os.path.split(currPath)[-1]
			if typeImg == 'S1':
				image_name = image_name.replace(strReplace, params.S1_FILE_STR)
				image_name = image_name.replace(params.LABEL_FILE_EXT, params.S1_FILE_EXT)
				valid_img_paths.append(os.path.join(params.VALID_S1_DIR, image_name))
			elif typeImg == 'S2':
				image_name = image_name.replace(strReplace, params.S2_FILE_STR)
				image_name = image_name.replace(params.LABEL_FILE_EXT, params.S2_FILE_EXT)
				valid_img_paths.append(os.path.join(params.VALID_S2_DIR, image_name))
			else:
				raise ValueError('Error on Data Type. Possible Values are S1 or S2')
		return img_paths, valid_img_paths


	return None


'''
Load image
:param imgPath: path of the image to load
:return: numpy array of the image
'''
def load_img(imgPath):
	if imgPath.endswith('.tif'):
		img = tifffile.imread(imgPath)
	else:
		img = np.array(cv2.imread(imgPath))
	return img


'''
function to transform the reference image label into a categorical label image.
labelPath - path to the label image
params - parameters of the model
'''
def get_label_mask(labelPath, params):
	currLabel = load_img(labelPath)
	currLabel=currLabel.astype(np.uint8)
	if params.NUM_CATEGORIES > 1:
		currLabel = to_categorical(currLabel, num_classes=params.NUM_CATEGORIES+1)
	return currLabel


'''
function to get the indexes of the images that composes the batches for the training
idx - permuted list of indexes with the size of the training data
params - parameters of the model
'''
def get_batch_inds(idx, params):

	N = len(idx)
	batchInds = []
	idx0 = 0
	toProcess = True
	while toProcess:
		idx1 = idx0 + params.BATCH_SZ
		if idx1 > N:
			idx1 = N
			idx0 = idx1 - params.BATCH_SZ
			toProcess = False
		batchInds.append(idx[idx0:idx1])
		idx0 = idx1
	return batchInds

'''
function that loads the current batch of images to be used for training or validation
inds - list of the image indexes of the current batch
trainData - list with the addresses of all the images
params - parameters of the model
isValid - if it is validation or training.
type_train - if it is a training from scratch or training in active learning (tested but didnt get good results.)
'''
def load_batch(inds, trainData, params, isValid, type_train):
	'''
	Given the batch indices, load the images and ground truth (labels or depth data)
	:param inds: batch indices
	:param trainData: training paths of the image files 
	:param params: input parameters from params.py
	:return: numpy arrays for image and ground truth batch data
	'''

	train_dir=None
	valid_dir=None
	if type_train == 0:
		train_dir=params.TRAIN_LABEL_DIR
		valid_dir=params.VALID_LABEL_DIR
	else:
		train_dir=params.TRAIN_LABEL_NEW
		valid_dir=params.VALID_LABEL_NEW

	batchShape = (params.BATCH_SZ, params.IMG_SZ[0], params.IMG_SZ[1])
	numChannels = params.NUM_CATEGORIES
	labelReplaceStr = params.LABEL_FILE_STR
	if (params.NETWORK=='labelUnet'):

		labelBatch=[]
		labelBatch30 = np.zeros((batchShape[0], int(batchShape[1]/3), int(batchShape[2]/3), numChannels))
		labelBatch10 = np.zeros((batchShape[0], batchShape[1], batchShape[2], numChannels))

		batchInd = 0

		if params.DATA_USAGE == 'S1_I_only':
			imgBatch = np.zeros((batchShape[0], batchShape[1], batchShape[2], 2))
			for i in inds:
				currData = trainData[i]
				imgPath = currData[0]
				if not isValid:
					if train_dir != params.TRAIN_S1_DIR:
						imageName = os.path.split(imgPath)[-1]
						if params.LABEL_FILE_EXT != params.S1_FILE_EXT:
							imageName = imageName.replace('.'+params.S1_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageName.replace(params.S1_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPath.replace(params.S1_FILE_STR, labelReplaceStr)
					currImg = load_img(imgPath)
					if params.S1_IN_DB:
						currImg=dBConvert(currImg)
					currImg = np.stack((currImg[:,:,0],currImg[:,:,3]), axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					imageName = os.path.split(labelPath)[-1]
					imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					labelPath = os.path.join(params.TRAIN_LABEL30_DIR,imageName)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[1:3]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				else:
					if valid_dir != params.VALID_S1_DIR:
						imageName = os.path.split(imgPath)[-1]
						if params.LABEL_FILE_EXT != params.S1_FILE_STR:
							imageName = imageName.replace('.'+params.S1_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S1_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPath.replace(params.S1_FILE_STR, labelReplaceStr)
					currImg = load_img(imgPath)
					if params.S1_IN_DB:
						currImg=dBConvert(currImg)
					currImg = np.stack((currImg[:,:,0],currImg[:,:,3]), axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					imageName = os.path.split(labelPath)[-1]
					imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					labelPath = os.path.join(params.VALID_LABEL30_DIR,imageName)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[1:3]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				batchInd += 1
		elif params.DATA_USAGE == 'S1IV_only':
			imgBatch = np.zeros((batchShape[0], batchShape[1], batchShape[2], 3))
			for i in inds:
				currData = trainData[i]
				imgPath = currData[0]
				if not isValid:
					if train_dir != params.TRAIN_S1_DIR:
						imageName = os.path.split(imgPath)[-1]
						if params.LABEL_FILE_EXT != params.S1_FILE_EXT:
							imageName = imageName.replace('.'+params.S1_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageName.replace(params.S1_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPath.replace(params.S1_FILE_STR, labelReplaceStr)
					currImg = load_img(imgPath)
					vol = np.divide(currImg[:,:,0], currImg[:,:,3], out=np.zeros_like(currImg[:,:,0]), where=currImg[:,:,3]!=0)
					vol = np.nan_to_num(vol, nan=0.0)
					if params.S1_IN_DB:
						currImg=dBConvert(currImg)
					if np.max(vol)>np.min(vol):
						vol = (vol - np.min(vol))/(np.max(vol)-np.min(vol))
					currImg = np.stack((currImg[:,:,0],currImg[:,:,3],vol), axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					imageName = os.path.split(labelPath)[-1]
					imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					labelPath = os.path.join(params.TRAIN_LABEL30_DIR,imageName)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[1:3]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				else:
					if valid_dir != params.VALID_S1_DIR:
						imageName = os.path.split(imgPath)[-1]
						if params.LABEL_FILE_EXT != params.S1_FILE_STR:
							imageName = imageName.replace('.'+params.S1_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S1_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPath.replace(params.S1_FILE_STR, labelReplaceStr)
					currImg = load_img(imgPath)
					vol = np.divide(currImg[:,:,0], currImg[:,:,3], out=np.zeros_like(currImg[:,:,0]), where=currImg[:,:,3]!=0)
					vol = np.nan_to_num(vol, nan=0.0)
					if params.S1_IN_DB:
						currImg=dBConvert(currImg)
					if np.max(vol)>np.min(vol):
						vol = (vol - np.min(vol))/(np.max(vol)-np.min(vol))
					currImg = np.stack((currImg[:,:,0],currImg[:,:,3],vol), axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					imageName = os.path.split(labelPath)[-1]
					imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					labelPath = os.path.join(params.VALID_LABEL30_DIR,imageName)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[1:3]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				batchInd += 1
		elif params.DATA_USAGE == 'S1_only':
			imgBatch = np.zeros((batchShape[0], batchShape[1], batchShape[2], 4))
			for i in inds:
				currData = trainData[i]
				imgPath = currData[0]
				if not isValid:
					if train_dir != params.TRAIN_S1_DIR:
						imageName = os.path.split(imgPath)[-1]
						if params.LABEL_FILE_EXT != params.S1_FILE_STR:
							imageName = imageName.replace('.'+params.S1_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageName.replace(params.S1_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPath.replace(params.S1_FILE_STR, labelReplaceStr)
					currImg = load_img(imgPath)
					if params.S1_IN_DB:
						currImg=dBConvert(currImg)
					currImg = currImg[:,:,:4]
					currLabel10 = get_label_mask(labelPath, params)
					imageName = os.path.split(labelPath)[-1]
					imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					labelPath = os.path.join(params.TRAIN_LABEL30_DIR,imageName)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[1:3]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				else:
					if valid_dir != params.VALID_S1_DIR:
						imageName = os.path.split(imgPath)[-1]
						if params.LABEL_FILE_EXT != params.S1_FILE_STR:
							imageName = imageName.replace('.'+params.S1_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S1_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPath.replace(params.S1_FILE_STR, labelReplaceStr)
					currImg = load_img(imgPath)
					if params.S1_IN_DB:
						currImg=dBConvert(currImg)
					currImg = currImg[:,:,:4]
					currLabel10 = get_label_mask(labelPath, params)
					imageName = os.path.split(labelPath)[-1]
					imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					labelPath = os.path.join(params.VALID_LABEL30_DIR,imageName)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[1:3]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				batchInd += 1
		elif params.DATA_USAGE == 'S1_Complex':
			imgBatch = np.zeros((batchShape[0], batchShape[1], batchShape[2], 3)).astype(np.complex64)
			for i in inds:
				currData = trainData[i]
				imgPath = currData[0]
				if not isValid:
					if train_dir != params.TRAIN_S1_DIR:
						imageName = os.path.split(imgPath)[-1]
						if params.LABEL_FILE_EXT != params.S1_FILE_STR:
							imageName = imageName.replace('.'+params.S1_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageName.replace(params.S1_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPath.replace(params.S1_FILE_STR, labelReplaceStr)
					currImg = load_img(imgPath)
					if params.S1_IN_DB:
						currImg=dBConvert(currImg)
					currImg = currImg.astype(np.complex64)
					currImg = np.stack((currImg[:,:,0],currImg[:,:,3], currImg[:,:,1]+1j*currImg[:,:,2]), axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					imageName = os.path.split(labelPath)[-1]
					imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					labelPath = os.path.join(params.TRAIN_LABEL30_DIR,imageName)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[1:3]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				else:
					if valid_dir != params.VALID_S1_DIR:
						imageName = os.path.split(imgPath)[-1]
						if params.LABEL_FILE_EXT != params.S1_FILE_STR:
							imageName = imageName.replace('.'+params.S1_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S1_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPath.replace(params.S1_FILE_STR, labelReplaceStr)
					currImg = load_img(imgPath)
					if params.S1_IN_DB:
						currImg=dBConvert(currImg)
					currImg = currImg.astype(np.complex64)
					currImg = np.stack((currImg[:,:,0],currImg[:,:,3], currImg[:,:,1]+1j*currImg[:,:,2]), axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					imageName = os.path.split(labelPath)[-1]
					imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					labelPath = os.path.join(params.VALID_LABEL30_DIR,imageName)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[1:3]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				batchInd += 1
		elif params.DATA_USAGE == 'S2_S1_Complex':
			imgBatch = []
			imgBatchS2 = np.zeros((batchShape[0], batchShape[1], batchShape[2], 4))
			imgBatchS1 = np.zeros((batchShape[0], batchShape[1], batchShape[2], 3))
			for i in inds:
				currData = trainData[i]
				imgPathS2 = currData[0]
				imgPathS1 = currData[1]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageNameS2 = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageNameS2 = imageNameS2.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageNameS2.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS2 = currImgS2[:,:,:4]
					#currImgS2 = currImgS2.astype(np.complex64)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = currImgS1.astype(np.complex64)
					currImgS1 = np.stack((currImgS1[:,:,0],currImgS1[:,:,3], currImgS1[:,:,1]+1j*currImgS1[:,:,2]), axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					imageName = os.path.split(labelPath)[-1]
					imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					labelPath = os.path.join(params.TRAIN_LABEL30_DIR,imageName)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					#imgBatch.append([currImgS2, currImgS1])
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS2 = currImgS2[:,:,:4]
					currImgS2 = currImgS2.astype(np.complex64)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = currImgS1.astype(np.complex64)
					currImgS1 = np.stack((currImgS1[:,:,0],currImgS1[:,:,3], currImgS1[:,:,1]+1j*currImgS1[:,:,2]), axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					imageName = os.path.split(labelPath)[-1]
					imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					labelPath = os.path.join(params.VALID_LABEL30_DIR,imageName)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					#imgBatch.append([currImgS2, currImgS1])
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				batchInd += 1
			imgBatch=[imgBatchS2, imgBatchS1]
		elif params.DATA_USAGE == 'S2_only':
			imgBatch = np.zeros((batchShape[0], batchShape[1], batchShape[2], 4))
			for i in inds:
				currData = trainData[i]
				imgPath = currData[0]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageName = os.path.split(imgPath)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPath.replace(params.S2_FILE_STR, labelReplaceStr)
					currImg = load_img(imgPath)
					currImg = currImg[:,:,:4]
					currLabel10 = get_label_mask(labelPath, params)
					imageName = os.path.split(labelPath)[-1]
					imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					labelPath = os.path.join(params.TRAIN_LABEL30_DIR,imageName)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[1:3]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPath)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPath.replace(params.S2_FILE_STR, labelReplaceStr)
					currImg = load_img(imgPath)
					currImg = currImg[:,:,:4]
					currLabel10 = get_label_mask(labelPath, params)
					imageName = os.path.split(labelPath)[-1]
					imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					labelPath = os.path.join(params.VALID_LABEL30_DIR,imageName)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[1:3]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				batchInd += 1
		elif params.DATA_USAGE == 'S1S2All':
			imgBatch = np.zeros((batchShape[0], batchShape[1], batchShape[2], 4+4))
			for i in inds:
				currData = trainData[i]
				imgPathS2 = currData[0]
				imgPathS1 = currData[1]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageNameS2 = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageNameS2 = imageNameS2.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageNameS2.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS2 = currImgS2[:,:,:4]
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = currImgS1[:,:,:4]
					currImg = np.concatenate((currImgS2,currImgS1),axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					imageName = os.path.split(imgPath)[-1]
					imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					labelPath = os.path.join(params.TRAIN_LABEL30_DIR,imageName)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS2 = currImgS2[:,:,:4]
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = currImgS1[:,:,:4]
					currImg = np.concatenate((currImgS2,currImgS1),axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					imageName = os.path.split(labelPath)[-1]
					imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					labelPath = os.path.join(params.VALID_LABEL30_DIR,imageName)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				batchInd += 1
		elif params.DATA_USAGE == 'S2S1IAll':
			imgBatch = np.zeros((batchShape[0], batchShape[1], batchShape[2], 4+2))
			for i in inds:
				currData = trainData[i]
				imgPathS2 = currData[0]
				imgPathS1 = currData[1]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageNameS2 = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageNameS2 = imageNameS2.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageNameS2.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS2 = currImgS2[:,:,:4]
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = np.stack((currImgS1[:,:,0],currImgS1[:,:,3]), axis=2)
					currImg = np.concatenate((currImgS2,currImgS1),axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					imageName = os.path.split(labelPath)[-1]
					imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					labelPath = os.path.join(params.TRAIN_LABEL30_DIR,imageName)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS2 = currImgS2[:,:,:4]
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = np.stack((currImgS1[:,:,0],currImgS1[:,:,3]), axis=2)
					currImg = np.concatenate((currImgS2,currImgS1),axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					imageName = os.path.split(labelPath)[-1]
					imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					labelPath = os.path.join(params.VALID_LABEL30_DIR,imageName)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				batchInd += 1
		elif params.DATA_USAGE == 'S1_S2_Sep':
			imgBatch = []
			imgBatchS2 = np.zeros((batchShape[0], batchShape[1], batchShape[2], 4))
			imgBatchS1 = np.zeros((batchShape[0], batchShape[1], batchShape[2], 4))
			for i in inds:
				currData = trainData[i]
				imgPathS2 = currData[0]
				imgPathS1 = currData[1]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageNameS2 = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageNameS2 = imageNameS2.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageNameS2.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS2 = currImgS2[:,:,:4]
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = currImgS1[:,:,:4]
					currLabel10 = get_label_mask(labelPath, params)
					imageName = os.path.split(labelPath)[-1]
					imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					labelPath = os.path.join(params.TRAIN_LABEL30_DIR,imageName)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					#imgBatch.append([currImgS2, currImgS1])
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS2 = currImgS2[:,:,:4]
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = currImgS1[:,:,:4]
					currLabel10 = get_label_mask(labelPath, params)
					imageName = os.path.split(labelPath)[-1]
					imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					labelPath = os.path.join(params.VALID_LABEL30_DIR,imageName)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					#imgBatch.append([currImgS2, currImgS1])
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				batchInd += 1
			imgBatch=[imgBatchS2, imgBatchS1]
		elif params.DATA_USAGE == 'S1I_S2_Sep':
			imgBatch = []
			imgBatchS2 = np.zeros((batchShape[0], batchShape[1], batchShape[2], 4))
			imgBatchS1 = np.zeros((batchShape[0], batchShape[1], batchShape[2], 2))
			for i in inds:
				currData = trainData[i]
				imgPathS2 = currData[0]
				imgPathS1 = currData[1]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageNameS2 = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageNameS2 = imageNameS2.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageNameS2.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS2 = currImgS2[:,:,:4]
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = np.stack((currImgS1[:,:,0],currImgS1[:,:,3]), axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					imageName = os.path.split(labelPath)[-1]
					imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					labelPath = os.path.join(params.TRAIN_LABEL30_DIR,imageName)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					#imgBatch.append([currImgS2, currImgS1])
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS2 = currImgS2[:,:,:4]
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = np.stack((currImgS1[:,:,0],currImgS1[:,:,3]), axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					imageName = os.path.split(labelPath)[-1]
					imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					labelPath = os.path.join(params.VALID_LABEL30_DIR,imageName)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					#imgBatch.append([currImgS2, currImgS1])
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				batchInd += 1
			imgBatch=[imgBatchS2, imgBatchS1]
		elif params.DATA_USAGE == 'S1IV_S2_Sep':
			imgBatch = []
			imgBatchS2 = np.zeros((batchShape[0], batchShape[1], batchShape[2], 4))
			imgBatchS1 = np.zeros((batchShape[0], batchShape[1], batchShape[2], 3))
			for i in inds:
				currData = trainData[i]
				imgPathS2 = currData[0]
				imgPathS1 = currData[1]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageNameS2 = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageNameS2 = imageNameS2.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageNameS2.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS2 = currImgS2[:,:,:4]
					currImgS1 = load_img(imgPathS1)
					vol = np.divide(currImgS1[:,:,0], currImgS1[:,:,3], out=np.zeros_like(currImgS1[:,:,0]), where=currImgS1[:,:,3]!=0)
					vol = np.nan_to_num(vol, nan=0.0)
					if np.max(vol)>np.min(vol):
						vol = (vol - np.min(vol))/(np.max(vol)-np.min(vol))
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = np.stack((currImgS1[:,:,0],currImgS1[:,:,3],vol), axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					imageName = os.path.split(labelPath)[-1]
					imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					labelPath = os.path.join(params.TRAIN_LABEL30_DIR,imageName)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS2 = currImgS2[:,:,:4]
					currImgS1 = load_img(imgPathS1)
					vol = np.divide(currImgS1[:,:,0], currImgS1[:,:,3], out=np.zeros_like(currImgS1[:,:,0]), where=currImgS1[:,:,3]!=0)
					vol = np.nan_to_num(vol, nan=0.0)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					if np.max(vol)>np.min(vol):
						vol = (vol - np.min(vol))/(np.max(vol)-np.min(vol))
					currImgS1 = np.stack((currImgS1[:,:,0],currImgS1[:,:,3],vol), axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					imageName = os.path.split(labelPath)[-1]
					imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					labelPath = os.path.join(params.VALID_LABEL30_DIR,imageName)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					#imgBatch.append([currImgS2, currImgS1])
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				batchInd += 1
			imgBatch=[imgBatchS2, imgBatchS1]
		elif params.DATA_USAGE == 'S1IVF_S2F_Sep':
			imgBatch = []
			imgBatchS2 = np.zeros((batchShape[0], batchShape[1], batchShape[2], params.NUM_CHANNELS_S2))
			imgBatchS1 = np.zeros((batchShape[0], batchShape[1], batchShape[2], 6))
			for i in inds:
				currData = trainData[i]
				imgPathS2 = currData[0]
				imgPathS1 = currData[1]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageNameS2 = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageNameS2 = imageNameS2.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageNameS2.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					vol = np.divide(currImgS1[:,:,0], currImgS1[:,:,3], out=np.zeros_like(currImgS1[:,:,0]), where=currImgS1[:,:,3]!=0)
					vol = np.nan_to_num(vol, nan=0.0)
					if np.max(vol)>np.min(vol):
						vol = (vol - np.min(vol))/(np.max(vol)-np.min(vol))
					currImgS1T = np.stack((currImgS1[:,:,0],currImgS1[:,:,3],vol), axis=2)
					currImgS1 = np.concatenate((currImgS1T, currImgS1[:,:,4:]), axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					imageName = os.path.split(labelPath)[-1]
					imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					labelPath = os.path.join(params.TRAIN_LABEL30_DIR,imageName)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					vol = np.divide(currImgS1[:,:,0], currImgS1[:,:,3], out=np.zeros_like(currImgS1[:,:,0]), where=currImgS1[:,:,3]!=0)
					vol = np.nan_to_num(vol, nan=0.0)
					if np.max(vol)>np.min(vol):
						vol = (vol - np.min(vol))/(np.max(vol)-np.min(vol))
					currImgS1T = np.stack((currImgS1[:,:,0],currImgS1[:,:,3],vol), axis=2)
					currImgS1 = np.concatenate((currImgS1T, currImgS1[:,:,4:]), axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					imageName = os.path.split(labelPath)[-1]
					imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					labelPath = os.path.join(params.VALID_LABEL30_DIR,imageName)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					#imgBatch.append([currImgS2, currImgS1])
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				batchInd += 1
			imgBatch=[imgBatchS2, imgBatchS1]
		elif params.DATA_USAGE == 'S1IV_S2Some':
			imgBatch = []
			imgBatchS2 = np.zeros((batchShape[0], batchShape[1], batchShape[2], 7))
			imgBatchS1 = np.zeros((batchShape[0], batchShape[1], batchShape[2], 3))
			for i in inds:
				currData = trainData[i]
				imgPathS2 = currData[0]
				imgPathS1 = currData[1]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageNameS2 = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageNameS2 = imageNameS2.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageNameS2.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS2 = np.concatenate((currImgS2[:,:,:5], currImgS2[:,:,8:]), axis=2)
					currImgS1 = load_img(imgPathS1)
					vol = np.divide(currImgS1[:,:,0], currImgS1[:,:,3], out=np.zeros_like(currImgS1[:,:,0]), where=currImgS1[:,:,3]!=0)
					vol = np.nan_to_num(vol, nan=0.0)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					if np.max(vol)>np.min(vol):
						vol = (vol - np.min(vol))/(np.max(vol)-np.min(vol))
					currImgS1 = np.stack((currImgS1[:,:,0],currImgS1[:,:,3],vol), axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					imageName = os.path.split(labelPath)[-1]
					imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					labelPath = os.path.join(params.TRAIN_LABEL30_DIR,imageName)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS2 = np.concatenate((currImgS2[:,:,:5], currImgS2[:,:,8:]), axis=2)
					currImgS1 = load_img(imgPathS1)
					vol = np.divide(currImgS1[:,:,0], currImgS1[:,:,3], out=np.zeros_like(currImgS1[:,:,0]), where=currImgS1[:,:,3]!=0)
					vol = np.nan_to_num(vol, nan=0.0)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					if np.max(vol)>np.min(vol):
						vol = (vol - np.min(vol))/(np.max(vol)-np.min(vol))
					currImgS1 = np.stack((currImgS1[:,:,0],currImgS1[:,:,3],vol), axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					imageName = os.path.split(labelPath)[-1]
					imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					labelPath = os.path.join(params.VALID_LABEL30_DIR,imageName)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					#imgBatch.append([currImgS2, currImgS1])
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				batchInd += 1
			imgBatch=[imgBatchS2, imgBatchS1]
		elif params.DATA_USAGE == 'S1I_S2Some':
			imgBatch = []
			imgBatchS2 = np.zeros((batchShape[0], batchShape[1], batchShape[2], 7))
			imgBatchS1 = np.zeros((batchShape[0], batchShape[1], batchShape[2], 2))
			for i in inds:
				currData = trainData[i]
				imgPathS2 = currData[0]
				imgPathS1 = currData[1]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageNameS2 = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageNameS2 = imageNameS2.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageNameS2.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS2 = np.concatenate((currImgS2[:,:,:5], currImgS2[:,:,8:]), axis=2)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = np.stack((currImgS1[:,:,0],currImgS1[:,:,3]), axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					imageName = os.path.split(labelPath)[-1]
					imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					labelPath = os.path.join(params.TRAIN_LABEL30_DIR,imageName)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS2 = np.concatenate((currImgS2[:,:,:5], currImgS2[:,:,8:]), axis=2)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = np.stack((currImgS1[:,:,0],currImgS1[:,:,3]), axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					imageName = os.path.split(labelPath)[-1]
					imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					labelPath = os.path.join(params.VALID_LABEL30_DIR,imageName)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					#imgBatch.append([currImgS2, currImgS1])
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				batchInd += 1
			imgBatch=[imgBatchS2, imgBatchS1]
		elif params.DATA_USAGE == 'S1IVF_S2_Sep':
			imgBatch = []
			imgBatchS2 = np.zeros((batchShape[0], batchShape[1], batchShape[2], 4))
			imgBatchS1 = np.zeros((batchShape[0], batchShape[1], batchShape[2], 6))
			for i in inds:
				currData = trainData[i]
				imgPathS2 = currData[0]
				imgPathS1 = currData[1]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageNameS2 = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageNameS2 = imageNameS2.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageNameS2.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS2 = currImgS2[:,:,:4]
					currImgS1 = load_img(imgPathS1)
					vol = np.divide(currImgS1[:,:,0], currImgS1[:,:,3], out=np.zeros_like(currImgS1[:,:,0]), where=currImgS1[:,:,3]!=0)
					vol = np.nan_to_num(vol, nan=0.0)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					if np.max(vol)>np.min(vol):
						vol = (vol - np.min(vol))/(np.max(vol)-np.min(vol))
					currImgS1T = np.stack((currImgS1[:,:,0],currImgS1[:,:,3],vol), axis=2)
					currImgS1 = np.concatenate((currImgS1T, currImgS1[:,:,4:]), axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					imageName = os.path.split(labelPath)[-1]
					imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					labelPath = os.path.join(params.TRAIN_LABEL30_DIR,imageName)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS2 = currImgS2[:,:,:4]
					currImgS1 = load_img(imgPathS1)
					vol = np.divide(currImgS1[:,:,0], currImgS1[:,:,3], out=np.zeros_like(currImgS1[:,:,0]), where=currImgS1[:,:,3]!=0)
					vol = np.nan_to_num(vol, nan=0.0)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					if np.max(vol)>np.min(vol):
						vol = (vol - np.min(vol))/(np.max(vol)-np.min(vol))
					currImgS1T = np.stack((currImgS1[:,:,0],currImgS1[:,:,3],vol), axis=2)
					currImgS1 = np.concatenate((currImgS1T, currImgS1[:,:,4:]), axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					imageName = os.path.split(labelPath)[-1]
					imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					labelPath = os.path.join(params.VALID_LABEL30_DIR,imageName)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					#imgBatch.append([currImgS2, currImgS1])
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				batchInd += 1
			imgBatch=[imgBatchS2, imgBatchS1]
		elif params.DATA_USAGE == 'S2F_only':
			imgBatch = np.zeros((batchShape[0], batchShape[1], batchShape[2], params.NUM_CHANNELS_S2))
			for i in inds:
				currData = trainData[i]
				imgPath = currData[0]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageName = os.path.split(imgPath)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPath.replace(params.S2_FILE_STR, labelReplaceStr)
					currImg = load_img(imgPath)
					currLabel10 = get_label_mask(labelPath, params)
					imageName = os.path.split(labelPath)[-1]
					imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					labelPath = os.path.join(params.TRAIN_LABEL30_DIR,imageName)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[1:3]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPath)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPath.replace(params.S2_FILE_STR, labelReplaceStr)
					currImg = load_img(imgPath)
					currLabel10 = get_label_mask(labelPath, params)
					imageName = os.path.split(labelPath)[-1]
					imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					labelPath = os.path.join(params.VALID_LABEL30_DIR,imageName)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[1:3]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				batchInd += 1
		elif params.DATA_USAGE == 'S1F_only':
			imgBatch = np.zeros((batchShape[0], batchShape[1], batchShape[2], params.NUM_CHANNELS_S1))
			for i in inds:
				currData = trainData[i]
				imgPath = currData[0]
				if not isValid:
					if train_dir != params.TRAIN_S1_DIR:
						imageName = os.path.split(imgPath)[-1]
						if params.LABEL_FILE_EXT != params.S1_FILE_STR:
							imageName = imageName.replace('.'+params.S1_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageName.replace(params.S1_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPath.replace(params.S1_FILE_STR, labelReplaceStr)
					currImg = load_img(imgPath)
					if params.S1_IN_DB:
						currImg=dBConvert(currImg)
					currLabel10 = get_label_mask(labelPath, params)
					imageName = os.path.split(labelPath)[-1]
					imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					labelPath = os.path.join(params.TRAIN_LABEL30_DIR,imageName)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[1:3]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				else:
					if valid_dir != params.VALID_S1_DIR:
						imageName = os.path.split(imgPath)[-1]
						if params.LABEL_FILE_EXT != params.S1_FILE_STR:
							imageName = imageName.replace('.'+params.S1_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S1_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPath.replace(params.S1_FILE_STR, labelReplaceStr)
					currImg = load_img(imgPath)
					if params.S1_IN_DB:
						currImg=dBConvert(currImg)
					currLabel10 = get_label_mask(labelPath, params)
					imageName = os.path.split(labelPath)[-1]
					imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					labelPath = os.path.join(params.VALID_LABEL30_DIR,imageName)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[1:3]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				batchInd += 1
		elif params.DATA_USAGE == 'S1IF_only':
			imgBatch = np.zeros((batchShape[0], batchShape[1], batchShape[2], 5))
			for i in inds:
				currData = trainData[i]
				imgPath = currData[0]
				if not isValid:
					if train_dir != params.TRAIN_S1_DIR:
						imageName = os.path.split(imgPath)[-1]
						if params.LABEL_FILE_EXT != params.S1_FILE_EXT:
							imageName = imageName.replace('.'+params.S1_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageName.replace(params.S1_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPath.replace(params.S1_FILE_STR, labelReplaceStr)
					currImg = load_img(imgPath)
					if params.S1_IN_DB:
						currImg=dBConvert(currImg)
					currImgT = np.stack((currImg[:,:,0],currImg[:,:,3]), axis=2)
					currImg = np.concatenate((currImgT, currImg[:,:,4:]), axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					if (train_dir!=params.TRAIN_LABEL30_DIR):
						imageName = os.path.split(labelPath)[-1]
						imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
						labelPath = os.path.join(params.TRAIN_LABEL30_DIR,imageName)
					else:
						labelPath = labelPath.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[1:3]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				else:
					if valid_dir != params.VALID_S1_DIR:
						imageName = os.path.split(imgPath)[-1]
						if params.LABEL_FILE_EXT != params.S1_FILE_STR:
							imageName = imageName.replace('.'+params.S1_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S1_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPath.replace(params.S1_FILE_STR, labelReplaceStr)
					currImg = load_img(imgPath)
					if params.S1_IN_DB:
						currImg=dBConvert(currImg)
					currImgT = np.stack((currImg[:,:,0],currImg[:,:,3]), axis=2)
					currImg = np.concatenate((currImgT, currImg[:,:,4:]), axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					if (valid_dir!=params.VALID_LABEL30_DIR):
						imageName = os.path.split(labelPath)[-1]
						imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
						labelPath = os.path.join(params.VALID_LABEL30_DIR,imageName)
					else:
						labelPath = labelPath.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[1:3]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				batchInd += 1
		elif params.DATA_USAGE == 'S1IVF_only':
			imgBatch = np.zeros((batchShape[0], batchShape[1], batchShape[2], 6))
			for i in inds:
				currData = trainData[i]
				imgPath = currData[0]
				if not isValid:
					if train_dir != params.TRAIN_S1_DIR:
						imageName = os.path.split(imgPath)[-1]
						if params.LABEL_FILE_EXT != params.S1_FILE_EXT:
							imageName = imageName.replace('.'+params.S1_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageName.replace(params.S1_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPath.replace(params.S1_FILE_STR, labelReplaceStr)
					currImg = load_img(imgPath)
					vol = np.divide(currImg[:,:,0], currImg[:,:,3], out=np.zeros_like(currImg[:,:,0]), where=currImg[:,:,3]!=0)
					vol = np.nan_to_num(vol, nan=0.0)
					if params.S1_IN_DB:
						currImg=dBConvert(currImg)
					if np.max(vol)>np.min(vol):
						vol = (vol - np.min(vol))/(np.max(vol)-np.min(vol))
					currImgT = np.stack((currImg[:,:,0],currImg[:,:,3],vol), axis=2)
					currImg = np.concatenate((currImgT, currImg[:,:,4:]), axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					if (train_dir!=params.TRAIN_LABEL30_DIR):
						imageName = os.path.split(labelPath)[-1]
						imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
						labelPath = os.path.join(params.TRAIN_LABEL30_DIR,imageName)
					else:
						labelPath = labelPath.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[1:3]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				else:
					if valid_dir != params.VALID_S1_DIR:
						imageName = os.path.split(imgPath)[-1]
						if params.LABEL_FILE_EXT != params.S1_FILE_STR:
							imageName = imageName.replace('.'+params.S1_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S1_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPath.replace(params.S1_FILE_STR, labelReplaceStr)
					currImg = load_img(imgPath)
					vol = np.divide(currImg[:,:,0], currImg[:,:,3], out=np.zeros_like(currImg[:,:,0]), where=currImg[:,:,3]!=0)
					vol = np.nan_to_num(vol, nan=0.0)
					if params.S1_IN_DB:
						currImg=dBConvert(currImg)
					if np.max(vol)>np.min(vol):
						vol = (vol - np.min(vol))/(np.max(vol)-np.min(vol))
					currImgT = np.stack((currImg[:,:,0],currImg[:,:,3],vol), axis=2)
					currImg = np.concatenate((currImgT, currImg[:,:,4:]), axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					if (valid_dir!=params.VALID_LABEL30_DIR):
						imageName = os.path.split(labelPath)[-1]
						imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
						labelPath = os.path.join(params.VALID_LABEL30_DIR,imageName)
					else:
						labelPath = labelPath.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[1:3]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				batchInd += 1
		elif params.DATA_USAGE == 'S1IS2FAll':
			imgBatch = np.zeros((batchShape[0], batchShape[1], batchShape[2], params.NUM_CHANNELS_S2+2))
			for i in inds:
				currData = trainData[i]
				imgPathS2 = currData[0]
				imgPathS1 = currData[1]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageNameS2 = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageNameS2 = imageNameS2.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageNameS2.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = np.stack((currImgS1[:,:,0],currImgS1[:,:,3]), axis=2)
					currImg = np.concatenate((currImgS2,currImgS1),axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					if (train_dir!=params.TRAIN_LABEL30_DIR):
						imageName = os.path.split(labelPath)[-1]
						imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
						labelPath = os.path.join(params.TRAIN_LABEL30_DIR,imageName)
					else:
						labelPath = labelPath.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = np.stack((currImgS1[:,:,0],currImgS1[:,:,3]), axis=2)
					currImg = np.concatenate((currImgS2,currImgS1),axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					if (valid_dir!=params.VALID_LABEL30_DIR):
						imageName = os.path.split(labelPath)[-1]
						imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
						labelPath = os.path.join(params.VALID_LABEL30_DIR,imageName)
					else:
						labelPath = labelPath.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				batchInd += 1
		elif params.DATA_USAGE == 'S1S2FAll':
			imgBatch = np.zeros((batchShape[0], batchShape[1], batchShape[2], params.NUM_CHANNELS_S2+4))
			for i in inds:
				currData = trainData[i]
				imgPathS2 = currData[0]
				imgPathS1 = currData[1]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageNameS2 = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageNameS2 = imageNameS2.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageNameS2.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = currImgS1[:,:,:4]
					currImg = np.concatenate((currImgS2,currImgS1),axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					if (train_dir!=params.TRAIN_LABEL30_DIR):
						imageName = os.path.split(imgPath)[-1]
						imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
						labelPath = os.path.join(params.TRAIN_LABEL30_DIR,imageName)
					else:
						labelPath = labelPath.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = currImgS1[:,:,:4]
					currImg = np.concatenate((currImgS2,currImgS1),axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					if (valid_dir!=params.VALID_LABEL30_DIR):
						imageName = os.path.split(labelPath)[-1]
						imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
						labelPath = os.path.join(params.VALID_LABEL30_DIR,imageName)
					else:
						labelPath = labelPath.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				batchInd += 1
		elif params.DATA_USAGE == 'S1_S2F_Sep':
			imgBatch = []
			imgBatchS2 = np.zeros((batchShape[0], batchShape[1], batchShape[2], params.NUM_CHANNELS_S2))
			imgBatchS1 = np.zeros((batchShape[0], batchShape[1], batchShape[2], 4))
			for i in inds:
				currData = trainData[i]
				imgPathS2 = currData[0]
				imgPathS1 = currData[1]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageNameS2 = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageNameS2 = imageNameS2.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageNameS2.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = currImgS1[:,:,:4]
					currLabel10 = get_label_mask(labelPath, params)
					if (train_dir!=params.TRAIN_LABEL30_DIR):
						imageName = os.path.split(labelPath)[-1]
						imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
						labelPath = os.path.join(params.TRAIN_LABEL30_DIR,imageName)
					else:
						labelPath = labelPath.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					#imgBatch.append([currImgS2, currImgS1])
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = currImgS1[:,:,:4]
					currLabel10 = get_label_mask(labelPath, params)
					if (valid_dir!=params.VALID_LABEL30_DIR):
						imageName = os.path.split(labelPath)[-1]
						imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
						labelPath = os.path.join(params.VALID_LABEL30_DIR,imageName)
					else:
						labelPath = labelPath.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					#imgBatch.append([currImgS2, currImgS1])
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				batchInd += 1
			imgBatch=[imgBatchS2, imgBatchS1]
		elif params.DATA_USAGE == 'S1I_S2F_Sep':
			imgBatch = []
			imgBatchS2 = np.zeros((batchShape[0], batchShape[1], batchShape[2], params.NUM_CHANNELS_S2))
			imgBatchS1 = np.zeros((batchShape[0], batchShape[1], batchShape[2], 2))
			for i in inds:
				currData = trainData[i]
				imgPathS2 = currData[0]
				imgPathS1 = currData[1]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageNameS2 = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageNameS2 = imageNameS2.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageNameS2.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = np.stack((currImgS1[:,:,0],currImgS1[:,:,3]), axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					if (train_dir!=params.TRAIN_LABEL30_DIR):
						imageName = os.path.split(labelPath)[-1]
						imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
						labelPath = os.path.join(params.TRAIN_LABEL30_DIR,imageName)
					else:
						labelPath = labelPath.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					#imgBatch.append([currImgS2, currImgS1])
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = np.stack((currImgS1[:,:,0],currImgS1[:,:,3]), axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					if (valid_dir!=params.VALID_LABEL30_DIR):
						imageName = os.path.split(labelPath)[-1]
						imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
						labelPath = os.path.join(params.VALID_LABEL30_DIR,imageName)
					else:
						labelPath = labelPath.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					#imgBatch.append([currImgS2, currImgS1])
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				batchInd += 1
			imgBatch=[imgBatchS2, imgBatchS1]
		elif params.DATA_USAGE == 'S1F_S2F_Sep':
			imgBatch = []
			imgBatchS2 = np.zeros((batchShape[0], batchShape[1], batchShape[2], params.NUM_CHANNELS_S2))
			imgBatchS1 = np.zeros((batchShape[0], batchShape[1], batchShape[2], params.NUM_CHANNELS_S1))
			for i in inds:
				currData = trainData[i]
				imgPathS2 = currData[0]
				imgPathS1 = currData[1]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageNameS2 = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageNameS2 = imageNameS2.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageNameS2.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currLabel10 = get_label_mask(labelPath, params)
					if (train_dir!=params.TRAIN_LABEL30_DIR):
						imageName = os.path.split(labelPath)[-1]
						imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
						labelPath = os.path.join(params.TRAIN_LABEL30_DIR,imageName)
					else:
						labelPath = labelPath.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					#imgBatch.append([currImgS2, currImgS1])
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currLabel10 = get_label_mask(labelPath, params)
					if (valid_dir!=params.VALID_LABEL30_DIR):
						imageName = os.path.split(labelPath)[-1]
						imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
						labelPath = os.path.join(params.VALID_LABEL30_DIR,imageName)
					else:
						labelPath = labelPath.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					#imgBatch.append([currImgS2, currImgS1])
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				batchInd += 1
			imgBatch=[imgBatchS2, imgBatchS1]
		elif params.DATA_USAGE == 'S1IFS2FAll':
			imgBatch = np.zeros((batchShape[0], batchShape[1], batchShape[2], params.NUM_CHANNELS_S2+5))
			for i in inds:
				currData = trainData[i]
				imgPathS2 = currData[0]
				imgPathS1 = currData[1]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageNameS2 = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageNameS2 = imageNameS2.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageNameS2.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = np.stack((currImgS1[:,:,0],currImgS1[:,:,3],currImgS1[:,:,4:]), axis=2)
					currImg = np.concatenate((currImgS2,currImgS1),axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					if (train_dir!=params.TRAIN_LABEL30_DIR):
						imageName = os.path.split(labelPath)[-1]
						imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
						labelPath = os.path.join(params.TRAIN_LABEL30_DIR,imageName)
					else:
						labelPath = labelPath.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = np.stack((currImgS1[:,:,0],currImgS1[:,:,3],currImgS1[:,:,4:]), axis=2)
					currImg = np.concatenate((currImgS2,currImgS1),axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					if (valid_dir!=params.VALID_LABEL30_DIR):
						imageName = os.path.split(labelPath)[-1]
						imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
						labelPath = os.path.join(params.VALID_LABEL30_DIR,imageName)
					else:
						labelPath = labelPath.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				batchInd += 1
		elif params.DATA_USAGE == 'S1IF_S2F_Sep':
			imgBatch = []
			imgBatchS2 = np.zeros((batchShape[0], batchShape[1], batchShape[2], params.NUM_CHANNELS_S2))
			imgBatchS1 = np.zeros((batchShape[0], batchShape[1], batchShape[2], 5))
			for i in inds:
				currData = trainData[i]
				imgPathS2 = currData[0]
				imgPathS1 = currData[1]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageNameS2 = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageNameS2 = imageNameS2.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageNameS2.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = np.stack((currImgS1[:,:,0],currImgS1[:,:,3], currImgS1[:,:,4:]), axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					if (train_dir!=params.TRAIN_LABEL30_DIR):
						imageName = os.path.split(labelPath)[-1]
						imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
						labelPath = os.path.join(params.TRAIN_LABEL30_DIR,imageName)
					else:
						labelPath = labelPath.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					#imgBatch.append([currImgS2, currImgS1])
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = np.stack((currImgS1[:,:,0],currImgS1[:,:,3], currImgS1[:,:,4:]), axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					if (valid_dir!=params.VALID_LABEL30_DIR):
						imageName = os.path.split(labelPath)[-1]
						imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
						labelPath = os.path.join(params.VALID_LABEL30_DIR,imageName)
					else:
						labelPath = labelPath.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					#imgBatch.append([currImgS2, currImgS1])
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				batchInd += 1
			imgBatch=[imgBatchS2, imgBatchS1]
		elif params.DATA_USAGE == 'S1FS2FAll':
			imgBatch = np.zeros((batchShape[0], batchShape[1], batchShape[2], params.NUM_CHANNELS_S2+params.NUM_CHANNELS_S1))
			for i in inds:
				currData = trainData[i]
				imgPathS2 = currData[0]
				imgPathS1 = currData[1]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageNameS2 = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageNameS2 = imageNameS2.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageNameS2.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImg = np.concatenate((currImgS2,currImgS1),axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					if (train_dir!=params.TRAIN_LABEL30_DIR):
						imageName = os.path.split(imgPath)[-1]
						imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
						labelPath = os.path.join(params.TRAIN_LABEL30_DIR,imageName)
					else:
						labelPath = labelPath.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImg = np.concatenate((currImgS2,currImgS1),axis=2)
					currLabel10 = get_label_mask(labelPath, params)
					if (valid_dir!=params.VALID_LABEL30_DIR):
						imageName = os.path.split(labelPath)[-1]
						imageName = imageName.replace(labelReplaceStr, params.LABEL30_FILE_STR)
						labelPath = os.path.join(params.VALID_LABEL30_DIR,imageName)
					else:
						labelPath = labelPath.replace(labelReplaceStr, params.LABEL30_FILE_STR)
					currLabel30 = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel10 = currLabel10[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch30[batchInd,:,:,:] = currLabel30[:,:,:params.NUM_CATEGORIES]
					labelBatch10[batchInd,:,:,:] = currLabel10[:,:,:params.NUM_CATEGORIES]
					labelBatch = [labelBatch30, labelBatch10]
				batchInd += 1
	else:

		labelBatch = np.zeros((batchShape[0], batchShape[1], batchShape[2], numChannels))

		batchInd = 0

		if params.DATA_USAGE == 'S1_I_only':
			imgBatch = np.zeros((batchShape[0], batchShape[1], batchShape[2], 2))
			for i in inds:
				currData = trainData[i]
				imgPath = currData[0]
				if not isValid:
					if train_dir != params.TRAIN_S1_DIR:
						imageName = os.path.split(imgPath)[-1]
						if params.LABEL_FILE_EXT != params.S1_FILE_EXT:
							imageName = imageName.replace('.'+params.S1_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageName.replace(params.S1_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPath.replace(params.S1_FILE_STR, labelReplaceStr)
					currImg = load_img(imgPath)
					if params.S1_IN_DB:
						currImg=dBConvert(currImg)
					currImg = np.stack((currImg[:,:,0],currImg[:,:,3]), axis=2)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[1:3]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				else:
					if valid_dir != params.VALID_S1_DIR:
						imageName = os.path.split(imgPath)[-1]
						if params.LABEL_FILE_EXT != params.S1_FILE_STR:
							imageName = imageName.replace('.'+params.S1_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S1_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPath.replace(params.S1_FILE_STR, labelReplaceStr)
					currImg = load_img(imgPath)
					if params.S1_IN_DB:
						currImg=dBConvert(currImg)
					currImg = np.stack((currImg[:,:,0],currImg[:,:,3]), axis=2)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[1:3]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				batchInd += 1
		elif params.DATA_USAGE == 'S1IV_only':
			imgBatch = np.zeros((batchShape[0], batchShape[1], batchShape[2], 3))
			for i in inds:
				currData = trainData[i]
				imgPath = currData[0]
				if not isValid:
					if train_dir != params.TRAIN_S1_DIR:
						imageName = os.path.split(imgPath)[-1]
						if params.LABEL_FILE_EXT != params.S1_FILE_EXT:
							imageName = imageName.replace('.'+params.S1_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageName.replace(params.S1_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPath.replace(params.S1_FILE_STR, labelReplaceStr)
					currImg = load_img(imgPath)
					vol = np.divide(currImg[:,:,0], currImg[:,:,3], out=np.zeros_like(currImg[:,:,0]), where=currImg[:,:,3]!=0)
					vol = np.nan_to_num(vol, nan=0.0)
					if params.S1_IN_DB:
						currImg=dBConvert(currImg)
					if np.max(vol)>np.min(vol):
						vol = (vol - np.min(vol))/(np.max(vol)-np.min(vol))
					currImg = np.stack((currImg[:,:,0],currImg[:,:,3],vol), axis=2)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[1:3]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				else:
					if valid_dir != params.VALID_S1_DIR:
						imageName = os.path.split(imgPath)[-1]
						if params.LABEL_FILE_EXT != params.S1_FILE_STR:
							imageName = imageName.replace('.'+params.S1_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S1_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPath.replace(params.S1_FILE_STR, labelReplaceStr)
					currImg = load_img(imgPath)
					vol = np.divide(currImg[:,:,0], currImg[:,:,3], out=np.zeros_like(currImg[:,:,0]), where=currImg[:,:,3]!=0)
					vol = np.nan_to_num(vol, nan=0.0)
					if params.S1_IN_DB:
						currImg=dBConvert(currImg)
					if np.max(vol)>np.min(vol):
						vol = (vol - np.min(vol))/(np.max(vol)-np.min(vol))
					currImg = np.stack((currImg[:,:,0],currImg[:,:,3],vol), axis=2)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[1:3]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				batchInd += 1
		elif params.DATA_USAGE == 'S1_only':
			imgBatch = np.zeros((batchShape[0], batchShape[1], batchShape[2], 4))
			for i in inds:
				currData = trainData[i]
				imgPath = currData[0]
				if not isValid:
					if train_dir != params.TRAIN_S1_DIR:
						imageName = os.path.split(imgPath)[-1]
						if params.LABEL_FILE_EXT != params.S1_FILE_STR:
							imageName = imageName.replace('.'+params.S1_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageName.replace(params.S1_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPath.replace(params.S1_FILE_STR, labelReplaceStr)
					currImg = load_img(imgPath)
					if params.S1_IN_DB:
						currImg=dBConvert(currImg)
					currImg = currImg[:,:,:4]
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[1:3]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				else:
					if valid_dir != params.VALID_S1_DIR:
						imageName = os.path.split(imgPath)[-1]
						if params.LABEL_FILE_EXT != params.S1_FILE_STR:
							imageName = imageName.replace('.'+params.S1_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S1_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPath.replace(params.S1_FILE_STR, labelReplaceStr)
					currImg = load_img(imgPath)
					if params.S1_IN_DB:
						currImg=dBConvert(currImg)
					currImg = currImg[:,:,:4]
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[1:3]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				batchInd += 1
		elif params.DATA_USAGE == 'S2_only':
			imgBatch = np.zeros((batchShape[0], batchShape[1], batchShape[2], 4))
			for i in inds:
				currData = trainData[i]
				imgPath = currData[0]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageName = os.path.split(imgPath)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPath.replace(params.S2_FILE_STR, labelReplaceStr)
					currImg = load_img(imgPath)
					currImg = currImg[:,:,:4]
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[1:3]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPath)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPath.replace(params.S2_FILE_STR, labelReplaceStr)
					currImg = load_img(imgPath)
					currImg = currImg[:,:,:4]
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[1:3]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				batchInd += 1
		elif params.DATA_USAGE == 'S1S2All':
			imgBatch = np.zeros((batchShape[0], batchShape[1], batchShape[2], 4+4))
			for i in inds:
				currData = trainData[i]
				imgPathS2 = currData[0]
				imgPathS1 = currData[1]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageNameS2 = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageNameS2 = imageNameS2.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageNameS2.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS2 = currImgS2[:,:,:4]
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = currImgS1[:,:,:4]
					currImg = np.concatenate((currImgS2,currImgS1),axis=2)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS2 = currImgS2[:,:,:4]
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = currImgS1[:,:,:4]
					currImg = np.concatenate((currImgS2,currImgS1),axis=2)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				batchInd += 1
		elif params.DATA_USAGE == 'S2S1IAll':
			imgBatch = np.zeros((batchShape[0], batchShape[1], batchShape[2], 4+2))
			for i in inds:
				currData = trainData[i]
				imgPathS2 = currData[0]
				imgPathS1 = currData[1]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageNameS2 = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageNameS2 = imageNameS2.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageNameS2.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS2 = currImgS2[:,:,:4]
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = np.stack((currImgS1[:,:,0],currImgS1[:,:,3]), axis=2)
					currImg = np.concatenate((currImgS2,currImgS1),axis=2)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS2 = currImgS2[:,:,:4]
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = np.stack((currImgS1[:,:,0],currImgS1[:,:,3]), axis=2)
					currImg = np.concatenate((currImgS2,currImgS1),axis=2)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				batchInd += 1
		elif params.DATA_USAGE == 'S1_S2_Sep':
			imgBatch = []
			imgBatchS2 = np.zeros((batchShape[0], batchShape[1], batchShape[2], 4))
			imgBatchS1 = np.zeros((batchShape[0], batchShape[1], batchShape[2], 4))
			for i in inds:
				currData = trainData[i]
				imgPathS2 = currData[0]
				imgPathS1 = currData[1]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageNameS2 = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageNameS2 = imageNameS2.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageNameS2.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS2 = currImgS2[:,:,:4]
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = currImgS1[:,:,:4]
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					#imgBatch.append([currImgS2, currImgS1])
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS2 = currImgS2[:,:,:4]
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = currImgS1[:,:,:4]
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					#imgBatch.append([currImgS2, currImgS1])
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				batchInd += 1
			imgBatch=[imgBatchS2, imgBatchS1]
		elif params.DATA_USAGE == 'S1I_S2_Sep':
			imgBatch = []
			imgBatchS2 = np.zeros((batchShape[0], batchShape[1], batchShape[2], 4))
			imgBatchS1 = np.zeros((batchShape[0], batchShape[1], batchShape[2], 2))
			for i in inds:
				currData = trainData[i]
				imgPathS2 = currData[0]
				imgPathS1 = currData[1]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageNameS2 = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageNameS2 = imageNameS2.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageNameS2.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS2 = currImgS2[:,:,:4]
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = np.stack((currImgS1[:,:,0],currImgS1[:,:,3]), axis=2)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					#imgBatch.append([currImgS2, currImgS1])
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS2 = currImgS2[:,:,:4]
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = np.stack((currImgS1[:,:,0],currImgS1[:,:,3]), axis=2)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					#imgBatch.append([currImgS2, currImgS1])
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				batchInd += 1
			imgBatch=[imgBatchS2, imgBatchS1]
		elif params.DATA_USAGE == 'S1IV_S2_Sep':
			imgBatch = []
			imgBatchS2 = np.zeros((batchShape[0], batchShape[1], batchShape[2], 4))
			imgBatchS1 = np.zeros((batchShape[0], batchShape[1], batchShape[2], 3))
			for i in inds:
				currData = trainData[i]
				imgPathS2 = currData[0]
				imgPathS1 = currData[1]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageNameS2 = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageNameS2 = imageNameS2.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageNameS2.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS2 = currImgS2[:,:,:4]
					currImgS1 = load_img(imgPathS1)
					vol = np.divide(currImgS1[:,:,0], currImgS1[:,:,3], out=np.zeros_like(currImgS1[:,:,0]), where=currImgS1[:,:,3]!=0)
					vol = np.nan_to_num(vol, nan=0.0)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					if np.max(vol)>np.min(vol):
						vol = (vol - np.min(vol))/(np.max(vol)-np.min(vol))
					currImgS1 = np.stack((currImgS1[:,:,0],currImgS1[:,:,3],vol), axis=2)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS2 = currImgS2[:,:,:4]
					currImgS1 = load_img(imgPathS1)
					vol = np.divide(currImgS1[:,:,0], currImgS1[:,:,3], out=np.zeros_like(currImgS1[:,:,0]), where=currImgS1[:,:,3]!=0)
					vol = np.nan_to_num(vol, nan=0.0)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					if np.max(vol)>np.min(vol):
						vol = (vol - np.min(vol))/(np.max(vol)-np.min(vol))
					currImgS1 = np.stack((currImgS1[:,:,0],currImgS1[:,:,3],vol), axis=2)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					#imgBatch.append([currImgS2, currImgS1])
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				batchInd += 1
			imgBatch=[imgBatchS2, imgBatchS1]
		elif params.DATA_USAGE == 'S1IV_S2Some':
			imgBatch = []
			imgBatchS2 = np.zeros((batchShape[0], batchShape[1], batchShape[2], 7))
			imgBatchS1 = np.zeros((batchShape[0], batchShape[1], batchShape[2], 3))
			for i in inds:
				currData = trainData[i]
				imgPathS2 = currData[0]
				imgPathS1 = currData[1]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageNameS2 = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageNameS2 = imageNameS2.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageNameS2.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS2 = np.concatenate((currImgS2[:,:,:5], currImgS2[:,:,8:]), axis=2)
					currImgS1 = load_img(imgPathS1)
					vol = np.divide(currImgS1[:,:,0], currImgS1[:,:,3], out=np.zeros_like(currImgS1[:,:,0]), where=currImgS1[:,:,3]!=0)
					vol = np.nan_to_num(vol, nan=0.0)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					if np.max(vol)>np.min(vol):
						vol = (vol - np.min(vol))/(np.max(vol)-np.min(vol))
					currImgS1 = np.stack((currImgS1[:,:,0],currImgS1[:,:,3],vol), axis=2)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS2 = np.concatenate((currImgS2[:,:,:5], currImgS2[:,:,8:]), axis=2)
					currImgS1 = load_img(imgPathS1)
					vol = np.divide(currImgS1[:,:,0], currImgS1[:,:,3], out=np.zeros_like(currImgS1[:,:,0]), where=currImgS1[:,:,3]!=0)
					vol = np.nan_to_num(vol, nan=0.0)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					if np.max(vol)>np.min(vol):
						vol = (vol - np.min(vol))/(np.max(vol)-np.min(vol))
					currImgS1 = np.stack((currImgS1[:,:,0],currImgS1[:,:,3],vol), axis=2)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					#imgBatch.append([currImgS2, currImgS1])
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				batchInd += 1
			imgBatch=[imgBatchS2, imgBatchS1]
		elif params.DATA_USAGE == 'S1I_S2Some':
			imgBatch = []
			imgBatchS2 = np.zeros((batchShape[0], batchShape[1], batchShape[2], 7))
			imgBatchS1 = np.zeros((batchShape[0], batchShape[1], batchShape[2], 2))
			for i in inds:
				currData = trainData[i]
				imgPathS2 = currData[0]
				imgPathS1 = currData[1]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageNameS2 = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageNameS2 = imageNameS2.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageNameS2.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS2 = np.concatenate((currImgS2[:,:,:5], currImgS2[:,:,8:]), axis=2)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = np.stack((currImgS1[:,:,0],currImgS1[:,:,3]), axis=2)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS2 = np.concatenate((currImgS2[:,:,:5], currImgS2[:,:,8:]), axis=2)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = np.stack((currImgS1[:,:,0],currImgS1[:,:,3]), axis=2)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					#imgBatch.append([currImgS2, currImgS1])
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				batchInd += 1
			imgBatch=[imgBatchS2, imgBatchS1]
		elif params.DATA_USAGE == 'S1IVF_S2_Sep':
			imgBatch = []
			imgBatchS2 = np.zeros((batchShape[0], batchShape[1], batchShape[2], 4))
			imgBatchS1 = np.zeros((batchShape[0], batchShape[1], batchShape[2], 6))
			for i in inds:
				currData = trainData[i]
				imgPathS2 = currData[0]
				imgPathS1 = currData[1]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageNameS2 = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageNameS2 = imageNameS2.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageNameS2.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS2 = currImgS2[:,:,:4]
					currImgS1 = load_img(imgPathS1)
					vol = np.divide(currImgS1[:,:,0], currImgS1[:,:,3], out=np.zeros_like(currImgS1[:,:,0]), where=currImgS1[:,:,3]!=0)
					vol = np.nan_to_num(vol, nan=0.0)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					if np.max(vol)>np.min(vol):
						vol = (vol - np.min(vol))/(np.max(vol)-np.min(vol))
					currImgS1T = np.stack((currImgS1[:,:,0],currImgS1[:,:,3],vol), axis=2)
					currImgS1 = np.concatenate((currImgS1T, currImgS1[:,:,4:]), axis=2)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS2 = currImgS2[:,:,:4]
					currImgS1 = load_img(imgPathS1)
					vol = np.divide(currImgS1[:,:,0], currImgS1[:,:,3], out=np.zeros_like(currImgS1[:,:,0]), where=currImgS1[:,:,3]!=0)
					vol = np.nan_to_num(vol, nan=0.0)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					if np.max(vol)>np.min(vol):
						vol = (vol - np.min(vol))/(np.max(vol)-np.min(vol))
					currImgS1T = np.stack((currImgS1[:,:,0],currImgS1[:,:,3],vol), axis=2)
					currImgS1 = np.concatenate((currImgS1T, currImgS1[:,:,4:]), axis=2)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					#imgBatch.append([currImgS2, currImgS1])
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				batchInd += 1
			imgBatch=[imgBatchS2, imgBatchS1]
		elif params.DATA_USAGE == 'S1IVF_S2F_Sep':
			imgBatch = []
			imgBatchS2 = np.zeros((batchShape[0], batchShape[1], batchShape[2], params.NUM_CHANNELS_S2))
			imgBatchS1 = np.zeros((batchShape[0], batchShape[1], batchShape[2], 6))
			for i in inds:
				currData = trainData[i]
				imgPathS2 = currData[0]
				imgPathS1 = currData[1]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageNameS2 = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageNameS2 = imageNameS2.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageNameS2.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS1 = load_img(imgPathS1)
					vol = np.divide(currImgS1[:,:,0], currImgS1[:,:,3], out=np.zeros_like(currImgS1[:,:,0]), where=currImgS1[:,:,3]!=0)
					vol = np.nan_to_num(vol, nan=0.0)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					if np.max(vol)>np.min(vol):
						vol = (vol - np.min(vol))/(np.max(vol)-np.min(vol))
					currImgS1T = np.stack((currImgS1[:,:,0],currImgS1[:,:,3],vol), axis=2)
					currImgS1 = np.concatenate((currImgS1T, currImgS1[:,:,4:]), axis=2)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS1 = load_img(imgPathS1)
					vol = np.divide(currImgS1[:,:,0], currImgS1[:,:,3], out=np.zeros_like(currImgS1[:,:,0]), where=currImgS1[:,:,3]!=0)
					vol = np.nan_to_num(vol, nan=0.0)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					if np.max(vol)>np.min(vol):
						vol = (vol - np.min(vol))/(np.max(vol)-np.min(vol))
					currImgS1T = np.stack((currImgS1[:,:,0],currImgS1[:,:,3],vol), axis=2)
					currImgS1 = np.concatenate((currImgS1T, currImgS1[:,:,4:]), axis=2)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					#imgBatch.append([currImgS2, currImgS1])
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				batchInd += 1
			imgBatch=[imgBatchS2, imgBatchS1]
		elif params.DATA_USAGE == 'S2F_only':
			imgBatch = np.zeros((batchShape[0], batchShape[1], batchShape[2], params.NUM_CHANNELS_S2))
			for i in inds:
				currData = trainData[i]
				imgPath = currData[0]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageName = os.path.split(imgPath)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPath.replace(params.S2_FILE_STR, labelReplaceStr)
					currImg = load_img(imgPath)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[1:3]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPath)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPath.replace(params.S2_FILE_STR, labelReplaceStr)
					currImg = load_img(imgPath)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[1:3]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				batchInd += 1
		elif params.DATA_USAGE == 'S1F_only':
			imgBatch = np.zeros((batchShape[0], batchShape[1], batchShape[2], params.NUM_CHANNELS_S1))
			for i in inds:
				currData = trainData[i]
				imgPath = currData[0]
				if not isValid:
					if train_dir != params.TRAIN_S1_DIR:
						imageName = os.path.split(imgPath)[-1]
						if params.LABEL_FILE_EXT != params.S1_FILE_STR:
							imageName = imageName.replace('.'+params.S1_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageName.replace(params.S1_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPath.replace(params.S1_FILE_STR, labelReplaceStr)
					currImg = load_img(imgPath)
					if params.S1_IN_DB:
						currImg=dBConvert(currImg)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[1:3]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				else:
					if valid_dir != params.VALID_S1_DIR:
						imageName = os.path.split(imgPath)[-1]
						if params.LABEL_FILE_EXT != params.S1_FILE_STR:
							imageName = imageName.replace('.'+params.S1_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S1_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPath.replace(params.S1_FILE_STR, labelReplaceStr)
					currImg = load_img(imgPath)
					if params.S1_IN_DB:
						currImg=dBConvert(currImg)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[1:3]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				batchInd += 1
		elif params.DATA_USAGE == 'S1IF_only':
			imgBatch = np.zeros((batchShape[0], batchShape[1], batchShape[2], 5))
			for i in inds:
				currData = trainData[i]
				imgPath = currData[0]
				if not isValid:
					if train_dir != params.TRAIN_S1_DIR:
						imageName = os.path.split(imgPath)[-1]
						if params.LABEL_FILE_EXT != params.S1_FILE_EXT:
							imageName = imageName.replace('.'+params.S1_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageName.replace(params.S1_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPath.replace(params.S1_FILE_STR, labelReplaceStr)
					currImg = load_img(imgPath)
					if params.S1_IN_DB:
						currImg=dBConvert(currImg)
					currImgT = np.stack((currImg[:,:,0],currImg[:,:,3]), axis=2)
					currImg = np.concatenate((currImgT, currImg[:,:,4:]), axis=2)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[1:3]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				else:
					if valid_dir != params.VALID_S1_DIR:
						imageName = os.path.split(imgPath)[-1]
						if params.LABEL_FILE_EXT != params.S1_FILE_STR:
							imageName = imageName.replace('.'+params.S1_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S1_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPath.replace(params.S1_FILE_STR, labelReplaceStr)
					currImg = load_img(imgPath)
					if params.S1_IN_DB:
						currImg=dBConvert(currImg)
					currImgT = np.stack((currImg[:,:,0],currImg[:,:,3]), axis=2)
					currImg = np.concatenate((currImgT, currImg[:,:,4:]), axis=2)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[1:3]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				batchInd += 1
		elif params.DATA_USAGE == 'S1IVF_only':
			imgBatch = np.zeros((batchShape[0], batchShape[1], batchShape[2], 6))
			for i in inds:
				currData = trainData[i]
				imgPath = currData[0]
				if not isValid:
					if train_dir != params.TRAIN_S1_DIR:
						imageName = os.path.split(imgPath)[-1]
						if params.LABEL_FILE_EXT != params.S1_FILE_EXT:
							imageName = imageName.replace('.'+params.S1_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageName.replace(params.S1_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPath.replace(params.S1_FILE_STR, labelReplaceStr)
					currImg = load_img(imgPath)
					vol = np.divide(currImg[:,:,0], currImg[:,:,3], out=np.zeros_like(currImg[:,:,0]), where=currImg[:,:,3]!=0)
					vol = np.nan_to_num(vol, nan=0.0)
					if params.S1_IN_DB:
						currImg=dBConvert(currImg)
					if np.max(vol)>np.min(vol):
						vol = (vol - np.min(vol))/(np.max(vol)-np.min(vol))
					currImgT = np.stack((currImg[:,:,0],currImg[:,:,3],vol), axis=2)
					currImg = np.concatenate((currImgT, currImg[:,:,4:]), axis=2)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[1:3]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				else:
					if valid_dir != params.VALID_S1_DIR:
						imageName = os.path.split(imgPath)[-1]
						if params.LABEL_FILE_EXT != params.S1_FILE_STR:
							imageName = imageName.replace('.'+params.S1_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S1_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPath.replace(params.S1_FILE_STR, labelReplaceStr)
					currImg = load_img(imgPath)
					vol = np.divide(currImg[:,:,0], currImg[:,:,3], out=np.zeros_like(currImg[:,:,0]), where=currImg[:,:,3]!=0)
					vol = np.nan_to_num(vol, nan=0.0)
					if params.S1_IN_DB:
						currImg=dBConvert(currImg)
					if np.max(vol)>np.min(vol):
						vol = (vol - np.min(vol))/(np.max(vol)-np.min(vol))
					currImgT = np.stack((currImg[:,:,0],currImg[:,:,3],vol), axis=2)
					currImg = np.concatenate((currImgT, currImg[:,:,4:]), axis=2)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[1:3]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				batchInd += 1
		elif params.DATA_USAGE == 'S1IS2FAll':
			imgBatch = np.zeros((batchShape[0], batchShape[1], batchShape[2], params.NUM_CHANNELS_S2+2))
			for i in inds:
				currData = trainData[i]
				imgPathS2 = currData[0]
				imgPathS1 = currData[1]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageNameS2 = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageNameS2 = imageNameS2.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageNameS2.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = np.stack((currImgS1[:,:,0],currImgS1[:,:,3]), axis=2)
					currImg = np.concatenate((currImgS2,currImgS1),axis=2)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = np.stack((currImgS1[:,:,0],currImgS1[:,:,3]), axis=2)
					currImg = np.concatenate((currImgS2,currImgS1),axis=2)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				batchInd += 1
		elif params.DATA_USAGE == 'S1S2FAll':
			imgBatch = np.zeros((batchShape[0], batchShape[1], batchShape[2], params.NUM_CHANNELS_S2+4))
			for i in inds:
				currData = trainData[i]
				imgPathS2 = currData[0]
				imgPathS1 = currData[1]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageNameS2 = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageNameS2 = imageNameS2.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageNameS2.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = currImgS1[:,:,:4]
					currImg = np.concatenate((currImgS2,currImgS1),axis=2)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = currImgS1[:,:,:4]
					currImg = np.concatenate((currImgS2,currImgS1),axis=2)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				batchInd += 1
		elif params.DATA_USAGE == 'S1FS2FAll':
			imgBatch = np.zeros((batchShape[0], batchShape[1], batchShape[2], params.NUM_CHANNELS_S2+params.NUM_CHANNELS_S1))
			for i in inds:
				currData = trainData[i]
				imgPathS2 = currData[0]
				imgPathS1 = currData[1]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageNameS2 = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageNameS2 = imageNameS2.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageNameS2.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImg = np.concatenate((currImgS2,currImgS1),axis=2)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImg = np.concatenate((currImgS2,currImgS1),axis=2)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				batchInd += 1
		elif params.DATA_USAGE == 'S1IFS2FAll':
			imgBatch = np.zeros((batchShape[0], batchShape[1], batchShape[2], params.NUM_CHANNELS_S2+5))
			for i in inds:
				currData = trainData[i]
				imgPathS2 = currData[0]
				imgPathS1 = currData[1]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageNameS2 = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageNameS2 = imageNameS2.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageNameS2.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = np.stack((currImgS1[:,:,0],currImgS1[:,:,3],currImgS1[:,:,4:]), axis=2)
					currImg = np.concatenate((currImgS2,currImgS1),axis=2)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = np.stack((currImgS1[:,:,0],currImgS1[:,:,3], currImgS1[:,:,4:]), axis=2)
					currImg = np.concatenate((currImgS2,currImgS1),axis=2)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImg = currImg[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatch[batchInd,:,:,:] = currImg
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				batchInd += 1
		elif params.DATA_USAGE == 'S1_S2F_Sep':
			imgBatch = []
			imgBatchS2 = np.zeros((batchShape[0], batchShape[1], batchShape[2], params.NUM_CHANNELS_S2))
			imgBatchS1 = np.zeros((batchShape[0], batchShape[1], batchShape[2], 4))
			for i in inds:
				currData = trainData[i]
				imgPathS2 = currData[0]
				imgPathS1 = currData[1]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageNameS2 = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageNameS2 = imageNameS2.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageNameS2.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = currImgS1[:,:,:4]
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					#imgBatch.append([currImgS2, currImgS1])
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = currImgS1[:,:,:4]
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					#imgBatch.append([currImgS2, currImgS1])
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				batchInd += 1
			imgBatch=[imgBatchS2, imgBatchS1]
		elif params.DATA_USAGE == 'S1F_S2F_Sep':
			imgBatch = []
			imgBatchS2 = np.zeros((batchShape[0], batchShape[1], batchShape[2], params.NUM_CHANNELS_S2))
			imgBatchS1 = np.zeros((batchShape[0], batchShape[1], batchShape[2], params.NUM_CHANNELS_S1))
			for i in inds:
				currData = trainData[i]
				imgPathS2 = currData[0]
				imgPathS1 = currData[1]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageNameS2 = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageNameS2 = imageNameS2.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageNameS2.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					#imgBatch.append([currImgS2, currImgS1])
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					#imgBatch.append([currImgS2, currImgS1])
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				batchInd += 1
			imgBatch=[imgBatchS2, imgBatchS1]
		elif params.DATA_USAGE == 'S1I_S2F_Sep':
			imgBatch = []
			imgBatchS2 = np.zeros((batchShape[0], batchShape[1], batchShape[2], params.NUM_CHANNELS_S2))
			imgBatchS1 = np.zeros((batchShape[0], batchShape[1], batchShape[2], 2))
			for i in inds:
				currData = trainData[i]
				imgPathS2 = currData[0]
				imgPathS1 = currData[1]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageNameS2 = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageNameS2 = imageNameS2.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageNameS2.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = np.stack((currImgS1[:,:,0],currImgS1[:,:,3]), axis=2)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					#imgBatch.append([currImgS2, currImgS1])
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = np.stack((currImgS1[:,:,0],currImgS1[:,:,3]), axis=2)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					#imgBatch.append([currImgS2, currImgS1])
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				batchInd += 1
			imgBatch=[imgBatchS2, imgBatchS1]
		elif params.DATA_USAGE == 'S1IF_S2F_Sep':
			imgBatch = []
			imgBatchS2 = np.zeros((batchShape[0], batchShape[1], batchShape[2], params.NUM_CHANNELS_S2))
			imgBatchS1 = np.zeros((batchShape[0], batchShape[1], batchShape[2], 5))
			for i in inds:
				currData = trainData[i]
				imgPathS2 = currData[0]
				imgPathS1 = currData[1]
				if not isValid:
					if train_dir != params.TRAIN_S2_DIR:
						imageNameS2 = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageNameS2 = imageNameS2.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(train_dir, imageNameS2.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = np.stack((currImgS1[:,:,0],currImgS1[:,:,3], currImgS1[:,:,4:]), axis=2)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					#imgBatch.append([currImgS2, currImgS1])
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				else:
					if valid_dir != params.VALID_S2_DIR:
						imageName = os.path.split(imgPathS2)[-1]
						if params.LABEL_FILE_EXT != params.S2_FILE_EXT:
							imageName = imageName.replace('.'+params.S2_FILE_EXT, '.'+params.LABEL_FILE_EXT)
						labelPath = os.path.join(valid_dir, imageName.replace(params.S2_FILE_STR, labelReplaceStr))
					else:
						labelPath = imgPathS2.replace(params.S2_FILE_STR, labelReplaceStr)
					currImgS2 = load_img(imgPathS2)
					currImgS1 = load_img(imgPathS1)
					if params.S1_IN_DB:
						currImgS1=dBConvert(currImgS1)
					currImgS1 = np.stack((currImgS1[:,:,0],currImgS1[:,:,3],currImgS1[:,:,4:]), axis=2)
					currLabel = get_label_mask(labelPath, params)
					rStart,cStart = currData[2:4]
					rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
					currImgS2 = currImgS2[rStart:rEnd, cStart:cEnd, :]
					currImgS1 = currImgS1[rStart:rEnd, cStart:cEnd, :]
					currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
					imgBatchS2[batchInd,:,:,:] = currImgS2
					imgBatchS1[batchInd,:,:,:] = currImgS1
					#imgBatch.append([currImgS2, currImgS1])
					labelBatch[batchInd,:,:,:] = currLabel[:,:,:params.NUM_CATEGORIES]
				batchInd += 1
			imgBatch=[imgBatchS2, imgBatchS1]
		else:
			raise ValueError("Incorrect DATA_USAGE definition for this Network. Correct in unetParams")

	return imgBatch,labelBatch


'''
Function to calculate the proportion of the classes in the training dataset. The inverse of this proportion is used as weights for the loss functions
params - parameters of the model
type_train - if it is a training from scratch or training in active learning (tested but didnt get good results.)
typeGT - if it is the 10 or the 30m reference data
'''
def calc_class_proportions(params, type_train, typeGT):

	train_dir=None
	if typeGT == 30:
		train_dir=params.TRAIN_LABEL30_DIR
		gt_image_path = '%s*.%s' % (params.LABEL30_FILE_STR, params.LABEL_FILE_EXT)
	else:
		if type_train == 0:
			train_dir=params.TRAIN_LABEL_DIR
		else:
			train_dir=params.TRAIN_LABEL_NEW
		gt_image_path = '%s*.%s' % (params.LABEL_FILE_STR, params.LABEL_FILE_EXT)

	glob_path = os.path.join(train_dir, gt_image_path)
	img_paths = glob(glob_path)
	class_counts = np.zeros(params.NUM_CATEGORIES)
	for img_path in img_paths:
		label_img = load_img(img_path)
		for i in range(params.NUM_CATEGORIES):
			class_counts[i]=class_counts[i]+np.sum(label_img==i)
	class_counts[0]=0
	class_proportions = class_counts / np.sum(class_counts)
	return class_proportions


'''
Function to calculate the class weights for the loss function based on their proportion and normalized by the maximum proportion.
params - parameters of the model
type_train - if it is a training from scratch or training in active learning (tested but didnt get good results.)
typeGT - if it is the 10 or the 30m reference data
scale - if the proportions will be scaled to log or not.
'''
def calc_class_weights_maxprop(params, type_train, scale = None, typeGT=10):
	class_props = calc_class_proportions(params, type_train, typeGT)
	#print(class_props)
	if scale == 'log':
		weights = np.log(1 / class_props)
	else: 
		max_prop = np.max(class_props)
		weights = [max_prop / i if i != 0 else 0 for i in class_props]
		#manter alpha entre 0 e 1: -> isso eh apenas para o focal_loss.
		#max_weight = np.max(weights)
		#weights = weights / max_weight
	return weights


'''
Function to calculate the class weights for the loss function based on their proportion and normalized by the mean proportion.
params - parameters of the model
type_train - if it is a training from scratch or training in active learning (tested but didnt get good results.)
typeGT - if it is the 10 or the 30m reference data
scale - if the proportions will be scaled to log or not.
'''
def calc_class_weights_meanprop(params, type_train, scale = None, typeGT=10):
	class_props = calc_class_proportions(params, type_train, typeGT)
	#print(class_props)
	if scale == 'log':
		weights = np.log(1 / class_props)
	else: 
		mean_prop = np.sum(class_props)/(params.NUM_CATEGORIES-1)
		weights = [mean_prop / i if i != 0 else 0 for i in class_props]
		#manter alpha entre 0 e 1: -> isso eh apenas para o focal_loss.
		#max_weight = np.max(weights)
		#weights = weights / max_weight
	return weights

'''
Function that calculates the total amount of training data
params - params - parameters of the model
'''
def get_total_train_data(params):
	train_dir = params.TRAIN_LABEL_DIR
	gt_image_path = '%s*.%s' % (params.LABEL_FILE_STR, params.LABEL_FILE_EXT)
	glob_path = os.path.join(train_dir, gt_image_path)
	img_paths = glob(glob_path)

	return len(img_paths)