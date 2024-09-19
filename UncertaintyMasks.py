import glob
import os
import numpy as np
import unetParams as params
from datafunctions import *
from osgeo import gdal



'''
Function that creates the uncertainty masks (Correct uncertain and incorrect certain masks) for a numpy array
GT - reference numpy array
CLS - inference numpy array 
S1 - S1 numpy array 
S2 - S2 numpy array
certainty - certainty numpy array
unc_threshold - threshold value used for the certainty.
'''
def UncertaintyMaskCreate (GT, CLS, S1, S2, certainty, unc_threshold):
	correct_uncertain = np.zeros_like(GT)
	incorrect_certain = np.zeros_like(GT)
	A = S1 != [0,0,0,0,0,0,0]
	A = A[:,:,0]
	B = S2 != [0,0,0,0,0,0,0,0,0,0]
	B = B[:,:,0]
	A = A*B
	GT = np.multiply(GT,A)
	CLS = np.multiply(CLS,A)
	certains = certainty>unc_threshold
	uncertains = certainty<=unc_threshold
	corrects = GT==CLS
	#take out the filtered values from no-values in S1 and S2 that were filtered in line 22 and 24 and invalid class (0)
	corrects = corrects*A
	corrects = corrects*(GT!=0)
	corrects = corrects*(CLS!=0)
	#take out invalid class (0)
	incorrects = GT!=CLS
	incorrects = incorrects*(GT!=0)
	incorrects = incorrects*(CLS!=0)
	correct_uncertain = corrects*uncertains
	incorrect_certain = incorrects*certains
	return correct_uncertain, incorrect_certain



'''
Function that creates the uncertainty masks (Correct uncertain and incorrect certain masks) for all test inference images
endGT - address where Reference images are
endCLS - address where inference images are
endS1 - address where S1 images are
endS2 - address where S2 images are
certainty - which uncertainty will be used (MCLUD (MCLU with MCdropout), MCLUS (MCLU on a deterministic inference), STD (standar deviation), DELTP (95% confidence interval) or Entropy)
threshold -  threshold value that will be used for the uncertainty
'''
def UncertaintyMask (endGT, endCLS, endS1, endS2, certainty, threshold):

	listGT = glob(os.path.join(endGT, '*%s*.%s' % (params.LABEL_FILE_STR,params.LABEL_FILE_EXT)))
	max_std = 0

	if certainty == 'STD':
		for i in listGT:
			filename = os.path.split(i)[-1]
			cert = load_img(os.path.join(endCLS+'Drop/',filename.replace(params.LABEL_FILE_STR, params.LABEL_STD_STR)))
			a = np.max(cert)
			if a>max_std:
				max_std=a

	for i in listGT:
		img_GT = load_img(i)
		base_name = os.path.split(i)[-1]
		S1_name = base_name.replace(params.LABEL_FILE_STR,params.S1_FILE_STR)
		CLS_name = base_name.replace(params.LABEL_FILE_STR,params.CLSPRED_FILE_STR)
		S2_name = base_name.replace(params.LABEL_FILE_STR,params.S2_FILE_STR)

		S1_img = load_img(endS1+S1_name)
		S2_img = load_img(endS2+S2_name)
		CLS_img = load_img(endCLS+CLS_name)

		if certainty == 'Entropy':
			certainty_img = load_img(os.path.join(endCLS+'Drop/',base_name.replace(params.LABEL_FILE_STR, params.LABEL_ENT_STR)))
			certainty_img = 1-certainty_img
		elif certainty == 'MCLUD':
			certainty_img = load_img(os.path.join(endCLS+'Drop/',base_name.replace(params.LABEL_FILE_STR, params.LABEL_MCLU_STR)))
		elif certainty == 'MCLUS':
			certainty_img = load_img(os.path.join(endCLS+'Still/',base_name.replace(params.LABEL_FILE_STR, params.LABEL_MCLU_STR)))
		elif certainty == 'STD':
			certainty_img = load_img(os.path.join(endCLS+'Drop/',base_name.replace(params.LABEL_FILE_STR, params.LABEL_STD_STR)))
			certainty_img = cert/max_std
			certainty_img = 1-certainty_img
		elif certainty == 'DELTP':
			certainty_img = load_img(os.path.join(endCLS+'Drop/',base_name.replace(params.LABEL_FILE_STR, params.LABEL_DELPE_STR)))
			certainty_img = 1-certainty_img

		correct_uncertain, incorrect_certain = UncertaintyMaskCreate(img_GT,CLS_img,S1_img,S2_img, certainty_img, threshold)
		dataset = gdal.Open(i)
		width = dataset.RasterXSize
		height = dataset.RasterYSize
		driver = gdal.GetDriverByName("GTiff")
		outdata = driver.Create(os.path.join(endCLS, base_name.replace(params.LABEL_FILE_STR,"CU_"+certainty)), width, height, 1)
		outdata.SetGeoTransform(dataset.GetGeoTransform())  ##sets same geotransform as input
		outdata.SetProjection(dataset.GetProjection())  ##sets same projection as input
		outdata.GetRasterBand(1).WriteArray(correct_uncertain)
		outdata.GetRasterBand(1).SetNoDataValue(10000)  ##if you want these values transparent
		outdata.FlushCache()  ##saves to disk!!
		outdata = None
		outdata1 = driver.Create(os.path.join(endCLS, base_name.replace(params.LABEL_FILE_STR,"IC_"+certainty)), width, height, 1)
		outdata1.SetGeoTransform(dataset.GetGeoTransform())  ##sets same geotransform as input
		outdata1.SetProjection(dataset.GetProjection())  ##sets same projection as input
		outdata1.GetRasterBand(1).WriteArray(incorrect_certain)
		outdata1.GetRasterBand(1).SetNoDataValue(10000)  ##if you want these values transparent
		outdata1.FlushCache()  ##saves to disk!!
		outdata1 = None
		band = None
		ds = None