from glob import glob
import os
import numpy as np
import random
import unetParams as params
import pandas as pd
from datafunctions import *
import matplotlib.pyplot as plt


'''
Auxiliary Metrics functions 
'''

def IoU(class_index, confusionMat):
	I = confusionMat[class_index-1,class_index-1]
	U = np.sum(confusionMat[class_index-1,:]) + np.sum(confusionMat[:,class_index-1])-I
	if(U==0):
		return 0
	else:
		return float(I/U)

def precision_perClass(chosenClass, confusionMat):

	if np.sum(confusionMat[chosenClass-1,:]) == 0:
		return 0
	else:
		return float(confusionMat[chosenClass-1,chosenClass-1]/np.sum(confusionMat[chosenClass-1,:]))

def recall_perClass(chosenClass, confusionMat):
	if np.sum(confusionMat[:,chosenClass-1])==0:
		return 0
	return float(confusionMat[chosenClass-1,chosenClass-1]/np.sum(confusionMat[:,chosenClass-1]))

def f1_perClass(chosenClass, confusionMat):
	precision = precision_perClass(chosenClass, confusionMat)
	recall = recall_perClass(chosenClass, confusionMat)
	if precision+recall>0:
		return 2*((precision*recall)/(precision+recall))
	else:
		return 0


def confusionMatrix(pred, refData):

	confusionMat = np.zeros((params.NUM_CATEGORIES-1, params.NUM_CATEGORIES-1))

	for i in range(params.NUM_CATEGORIES-1):
				for j in range(params.NUM_CATEGORIES-1):
					a = refData == i+1
					b = pred == j+1
					confusionMat[i,j] = confusionMat[i,j] + np.sum(np.logical_and(a,b))
	return confusionMat


def statistics(confusionMat):

	a = 0
	for i in range(params.NUM_CATEGORIES-1):
		a = a+confusionMat[i,i]
	global_prec = float(a/np.sum(confusionMat))

	a = []
	for i in range(params.NUM_CATEGORIES-1):
		a.append(IoU(i+1, confusionMat))
	meanIoU = np.mean(a)

	a = []
	for i in range(params.NUM_CATEGORIES-1):
		a.append(f1_perClass(i+1, confusionMat))
	meanF1Score = np.mean(a)

	a = []
	for i in range(params.NUM_CATEGORIES-1):
		a.append(recall_perClass(i+1, confusionMat))
	meanRecall = np.mean(a)

	return global_prec, meanIoU, meanF1Score, meanRecall


'''
Uncertanty statistics metrics 
'''
def Uncertstatistics(confMatUncertain, confMatCertain):

	AC = 0
	for i in range(params.NUM_CATEGORIES-1):
		AC = AC+confMatCertain[i,i]
	#print(AC)
	IC = np.sum(confMatCertain) - AC
	#print(IC)
	AU = 0
	for i in range(params.NUM_CATEGORIES-1):
		AU = AU+confMatUncertain[i,i]
	#print(AU)
	IU = np.sum(confMatUncertain) - AU
	#print(IU)

	OAUncert = (AC+IU)/(AC+IU+AU+IC)
	ratio_CC = (AC)/(AC+IC)
	ration_IU = (IU)/(IU+IC)
	sensitivity = (AC)/(AC+AU)
	uncertPrec = (IU)/(IU+AU)

	return OAUncert, ratio_CC, ration_IU, sensitivity, uncertPrec

'''
Function to calculate the Expected Calibration Error
endGT - address where the Reference images are
endCLS - address where the predicted images are
endS1 - address where the Sentinel-1 input images are
endS2 - address where the Sentinel-2 input images are
certainty - which type of uncertainty to use (Entropy, MCLU, MCLU with MC Dropout, Standard deviation or confidence interval)
'''
def Expected_Calibration_Error(endGT, endCLS, endS1, endS2, certainty):
	expected_OA = x = [((i+1)/10) for i in range(10)]
	print(expected_OA)
	listOfCLS = glob(os.path.join(endCLS,'*%s*.%s' % (params.CLSPRED_FILE_STR,params.LABEL_FILE_EXT)))
	#print(len(listOfCLS))
	#print(x)
	statusPerBin = []
	bm=[]
	conf=[]
	max_std = 0

	if certainty == 'STD':
		for i in listOfCLS:
			filename = os.path.split(i)[-1]
			cert = load_img(os.path.join(endCLS+'Drop/',filename.replace(params.CLSPRED_FILE_STR, params.LABEL_STD_STR)))
			a = np.max(cert)
			if a>max_std:
				max_std=a

	for j in range(10):
		infLim = float(j/10)
		supLim = float((j+1)/10)
		confMatPerImage=[]
		bmPerImg = []
		confsumPerimage = []
		n=0
		for i in listOfCLS:
			clsImg = load_img(i)
			filename = os.path.split(i)[-1]
			if certainty == 'Entropy':
				cert = load_img(os.path.join(endCLS+'Drop/',filename.replace(params.CLSPRED_FILE_STR, params.LABEL_ENT_STR)))
				cert = 1-cert
			elif certainty == 'MCLUD':
				cert = load_img(os.path.join(endCLS+'Drop/',filename.replace(params.CLSPRED_FILE_STR, params.LABEL_MCLU_STR)))
			elif certainty == 'MCLUS':
				cert = load_img(os.path.join(endCLS+'Still/',filename.replace(params.CLSPRED_FILE_STR, params.LABEL_MCLU_STR)))
			elif certainty == 'STD':
				cert = load_img(os.path.join(endCLS+'Drop/',filename.replace(params.CLSPRED_FILE_STR, params.LABEL_STD_STR)))
				cert = cert/max_std
				cert = 1-cert
			elif certainty == 'DELTP':
				cert = load_img(os.path.join(endCLS+'Drop/',filename.replace(params.CLSPRED_FILE_STR, params.LABEL_DELPE_STR)))
				cert = 1-cert

			gt = load_img(os.path.join(endGT,filename.replace(params.CLSPRED_FILE_STR, params.LABEL_FILE_STR)))
			S1 = load_img(os.path.join(endS1,filename.replace(params.CLSPRED_FILE_STR, params.S1_FILE_STR)))
			S2 = load_img(os.path.join(endS2,filename.replace(params.CLSPRED_FILE_STR, params.S2_FILE_STR)))
			A = S1 != [0,0,0,0,0,0,0]
			A = A[:,:,0]
			B = S2 != [0,0,0,0,0,0,0,0,0,0]
			B = B[:,:,0]
			A = A*B
			gt = np.multiply(gt,A)
			clsImg = np.multiply(clsImg,A)
			cert = np.multiply(cert,A)
			n=n+np.sum(gt!=0)
			A1 = cert>=infLim
			A2 = cert<supLim
			A=A1*A2
			clsImg=clsImg*A
			gt = gt*A
			cert = cert*(gt!=0)
			confsumPerimage.append(np.nansum(cert*A))
			bmPerImg.append(np.sum(gt!=0))
			confMatPerImage.append(confusionMatrix(clsImg,gt))
		bmPerImg = np.array(bmPerImg)
		confsumPerimage = np.array(confsumPerimage)
		bm.append(np.sum(bmPerImg))
		if np.sum(bmPerImg)>0:
			conf.append(np.sum(confsumPerimage)/np.sum(bmPerImg))
		else:
			conf.append(0)
		print(conf)
		confMatPerImage =  np.array(confMatPerImage)
		confMatPerBin = np.sum(confMatPerImage, axis=0)
		global_prec, meanIoU, meanF1Score, meanRecall = statistics(confMatPerBin)
		statusPerBin.append([global_prec, meanIoU, meanF1Score, meanRecall])
		#print(statusPerBin)

	statusPerBin = np.array(statusPerBin)
	bm=np.array(bm)
	conf = np.array(conf)
	OA = statusPerBin[:,0]
	print(n)
	print(bm)
	print(OA)
	print(conf)
	ece = np.sum((bm/n)*np.abs(OA-conf))
	return ece

'''
Function to calculate the Reliability Diagrams
endGT - address where the Reference images are
endCLS - address where the predicted images are
endS1 - address where the Sentinel-1 input images are
endS2 - address where the Sentinel-2 input images are
certainty - which type of uncertainty to use (Entropy, MCLU, MCLU with MC Dropout, Standard deviation or confidence interval)
'''
def ReliabilityDiagrams (endGT, endCLS, endS1, endS2, certainty):

	listOfCLS = glob(os.path.join(endCLS,'*%s*.%s' % (params.CLSPRED_FILE_STR,params.LABEL_FILE_EXT)))
	#print(len(listOfCLS))
	x = [(i/10) for i in range(10)]
	#print(x)
	statusPerBin = []
	max_std = 0

	if certainty == 'STD':
		for i in listOfCLS:
			filename = os.path.split(i)[-1]
			cert = load_img(os.path.join(endCLS+'Drop/',filename.replace(params.CLSPRED_FILE_STR, params.LABEL_STD_STR)))
			a = np.max(cert)
			if a>max_std:
				max_std=a



	for j in range(10):
		infLim = float(j/10)
		supLim = float((j+1)/10)
		confMatPerImage=[]

		for i in listOfCLS:
			clsImg = load_img(i)
			filename = os.path.split(i)[-1]
			if certainty == 'Entropy':
				cert = load_img(os.path.join(endCLS+'Drop/',filename.replace(params.CLSPRED_FILE_STR, params.LABEL_ENT_STR)))
				cert = 1-cert
			elif certainty == 'MCLUD':
				cert = load_img(os.path.join(endCLS+'Drop/',filename.replace(params.CLSPRED_FILE_STR, params.LABEL_MCLU_STR)))
			elif certainty == 'MCLUS':
				cert = load_img(os.path.join(endCLS+'Still/',filename.replace(params.CLSPRED_FILE_STR, params.LABEL_MCLU_STR)))
			elif certainty == 'STD':
				cert = load_img(os.path.join(endCLS+'Drop/',filename.replace(params.CLSPRED_FILE_STR, params.LABEL_STD_STR)))
				cert = cert/max_std
				cert = 1-cert
			elif certainty == 'DELTP':
				cert = load_img(os.path.join(endCLS+'Drop/',filename.replace(params.CLSPRED_FILE_STR, params.LABEL_DELPE_STR)))
				cert = 1-cert

			gt = load_img(os.path.join(endGT,filename.replace(params.CLSPRED_FILE_STR, params.LABEL_FILE_STR)))
			S1 = load_img(os.path.join(endS1,filename.replace(params.CLSPRED_FILE_STR, params.S1_FILE_STR)))
			S2 = load_img(os.path.join(endS2,filename.replace(params.CLSPRED_FILE_STR, params.S2_FILE_STR)))
			A = S1 != [0,0,0,0,0,0,0]
			A = A[:,:,0]
			B = S2 != [0,0,0,0,0,0,0,0,0,0]
			B = B[:,:,0]
			A = A*B
			gt = np.multiply(gt,A)
			clsImg = np.multiply(clsImg,A)
			A1 = cert>=infLim
			A2 = cert<supLim
			A=A1*A2
			clsImg=clsImg*A
			gt = gt*A
			confMatPerImage.append(confusionMatrix(clsImg,gt))

		confMatPerImage =  np.array(confMatPerImage)
		confMatPerBin = np.sum(confMatPerImage, axis=0)
		global_prec, meanIoU, meanF1Score, meanRecall = statistics(confMatPerBin)
		statusPerBin.append([global_prec, meanIoU, meanF1Score, meanRecall])
		#print(statusPerBin)

	statusPerBin = np.array(statusPerBin)
	plt.figure()
	plt.bar(x, height=statusPerBin[:,0], width=0.1, linewidth=0.01, edgecolor='w',align='edge')
	plt.plot([0,1],[0,1],color='r')
	plt.title('Uncertainty Evaluation')
	plt.xlabel(certainty)
	plt.ylabel('Overall Accuracy')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.savefig(endCLS+certainty+'xOA.png')

	plt.figure()
	plt.bar(x, height=statusPerBin[:,1], width=0.1 ,linewidth=0.01, edgecolor='w', align='edge')
	plt.plot([0,1],[0,1],color='r')
	plt.title('Uncertainty Evaluation')
	plt.xlabel(certainty)
	plt.ylabel('MeanIoU')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.savefig(endCLS+certainty+'xIoU.png')

	plt.figure()
	plt.bar(x, height=statusPerBin[:,2],width=0.1, linewidth=0.01, edgecolor='w', align='edge')
	plt.plot([0,1],[0,1],color='r')
	plt.title('Uncertainty Evaluation')
	plt.xlabel(certainty)
	plt.ylabel('F1Score')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.savefig(endCLS+certainty+'xF1Score.png')

	plt.figure()
	plt.bar(x, height=statusPerBin[:,3], width=0.1, linewidth=0.01, edgecolor='w', align='edge')
	plt.plot([0,1],[0,1],color='r')
	plt.title('Uncertainty Evaluation')
	plt.xlabel(certainty)
	plt.ylabel('Recall')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.savefig(endCLS+certainty+'xRecall.png')


'''
Function to calculate the Confidence Threshold Graphs
endGT - address where the Reference images are
endCLS - address where the predicted images are
endS1 - address where the Sentinel-1 input images are
endS2 - address where the Sentinel-2 input images are
certainty - which type of uncertainty to use (Entropy, MCLU, MCLU with MC Dropout, Standard deviation or confidence interval)
'''
def ConfidenceThresholdGraph(endGT, endCLS, endS1, endS2, certainty):

	listOfCLS = glob(os.path.join(endCLS,'*%s*.%s' % (params.CLSPRED_FILE_STR,params.LABEL_FILE_EXT)))
	x = [((i+1)/10) for i in range(10)]
	statusPerBin = []
	max_std = 0

	if certainty == 'STD':
		for i in listOfCLS:
			filename = os.path.split(i)[-1]
			cert = load_img(os.path.join(endCLS+'Drop/',filename.replace(params.CLSPRED_FILE_STR, params.LABEL_STD_STR)))
			a = np.max(cert)
			if a>max_std:
				max_std=a

	for j in range(10):
		supLim = float((j+1)/10)
		confMatPerImageUncertain = []
		confMatPerImageCertain = []

		for i in listOfCLS:
			clsImg = load_img(i)
			filename = os.path.split(i)[-1]

			if certainty == 'Entropy':
				cert = load_img(os.path.join(endCLS+'Drop/',filename.replace(params.CLSPRED_FILE_STR, params.LABEL_ENT_STR)))
				cert = 1-cert
			elif certainty == 'MCLUD':
				cert = load_img(os.path.join(endCLS+'Drop/',filename.replace(params.CLSPRED_FILE_STR, params.LABEL_MCLU_STR)))
			elif certainty == 'MCLUS':
				cert = load_img(os.path.join(endCLS+'Still/',filename.replace(params.CLSPRED_FILE_STR, params.LABEL_MCLU_STR)))
			elif certainty == 'STD':
				cert = load_img(os.path.join(endCLS+'Drop/',filename.replace(params.CLSPRED_FILE_STR, params.LABEL_STD_STR)))
				cert = cert/max_std
				cert = 1-cert
			elif certainty == 'DELTP':
				cert = load_img(os.path.join(endCLS+'Drop/',filename.replace(params.CLSPRED_FILE_STR, params.LABEL_DELPE_STR)))
				cert = 1-cert
			gt = load_img(os.path.join(endGT,filename.replace(params.CLSPRED_FILE_STR, params.LABEL_FILE_STR)))
			S1= load_img(os.path.join(endS1,filename.replace(params.CLSPRED_FILE_STR, params.S1_FILE_STR)))
			S2= load_img(os.path.join(endS2,filename.replace(params.CLSPRED_FILE_STR, params.S2_FILE_STR)))
			A = S1 != [0,0,0,0,0,0,0]
			A = A[:,:,0]
			B = S2 != [0,0,0,0,0,0,0,0,0,0]
			B = B[:,:,0]
			A = A*B
			gt = np.multiply(gt,A)
			clsImg = np.multiply(clsImg,A)

			A1 = cert<supLim
			A2 = cert>=supLim
			
			clsImg1=clsImg*A1
			gt1 = gt*A1
			confMatPerImageUncertain.append(confusionMatrix(clsImg1,gt1))
			clsImg2=clsImg*A2
			gt2 = gt*A2
			confMatPerImageCertain.append(confusionMatrix(clsImg2,gt2))

		confMatPerImageUncertain =  np.array(confMatPerImageUncertain)
		confMatPerImageCertain =  np.array(confMatPerImageCertain)
		confMatPerBinUncertain = np.sum(confMatPerImageUncertain, axis=0)
		confMatPerBinCertain = np.sum(confMatPerImageCertain, axis=0)
		OAUncert, ratio_CC, ration_IU, sensitivity, uncertPrec = Uncertstatistics(confMatPerBinUncertain, confMatPerBinCertain)
		statusPerBin.append([OAUncert, ratio_CC, ration_IU, sensitivity, uncertPrec])
	
	statusPerBin = np.array(statusPerBin)
	print(statusPerBin)
	
	plt.figure()
	plt.plot(x, statusPerBin[:,0], color='r')
	plt.title('Confidence Threshold evaluation')
	plt.xlabel('Certainty Measurement')
	plt.ylabel('Overall Accuracy Uncertainty')
	plt.savefig(endCLS+certainty+'xUOA.png')

	plt.figure()
	plt.plot(x, statusPerBin[:,1], color='r')
	plt.title('Confidence Threshold evaluation')
	plt.xlabel('Certainty Measurement')
	plt.ylabel('Ratio Correct-Certain')
	plt.savefig(endCLS+certainty+'xRCC.png')

	plt.figure()
	plt.plot(x, statusPerBin[:,2], color='r')
	plt.title('Confidence Threshold evaluation')
	plt.xlabel('Certainty Measurement')
	plt.ylabel('Ratio Incorrect-Uncertain')
	plt.savefig(endCLS+certainty+'xRIU.png')
	
	plt.figure()
	plt.plot(x, statusPerBin[:,3], color='r')
	plt.title('Confidence Threshold evaluation')
	plt.xlabel('Certainty Measurement')
	plt.ylabel('Sensitivity')
	plt.savefig(endCLS+certainty+'Sensiti.png')

	plt.figure()
	plt.plot(x, statusPerBin[:,4], color='r')
	plt.title('Confidence Threshold evaluation')
	plt.xlabel('Certainty Measurement')
	plt.ylabel('Uncertainty Precision')
	plt.savefig(endCLS+certainty+'xPrecision.png')
	

'''
Function to calculate the Uncertainty distribution Graph
endGT - address where the Reference images are
endCLS - address where the predicted images are
endS1 - address where the Sentinel-1 input images are
endS2 - address where the Sentinel-2 input images are
certainty - which type of uncertainty to use (Entropy, MCLU, MCLU with MC Dropout, Standard deviation or confidence interval)
'''
def distributionGraph(endGT, endCLS, endS1, endS2, certainty):

	listOfCLS = glob(os.path.join(endCLS,'*%s*.%s' % (params.CLSPRED_FILE_STR,params.LABEL_FILE_EXT)))
	#print(len(listOfCLS))
	x = [((i+1)/10) for i in range(10)]

	#print(x)
	y_corretos = np.zeros_like(x)
	y_errados = np.zeros_like(x)

	max_std = 0

	if certainty == 'STD':
		for i in listOfCLS:
			filename = os.path.split(i)[-1]
			cert = load_img(os.path.join(endCLS+'Drop/',filename.replace(params.CLSPRED_FILE_STR, params.LABEL_STD_STR)))
			a = np.max(cert)
			if a>max_std:
				max_std=a

	for j in range(10):
		infLim = float(j/10)
		supLim = float((j+1)/10)
		confMatPerImage=[]

		for i in listOfCLS:
			clsImg = load_img(i)
			filename = os.path.split(i)[-1]
			if certainty == 'Entropy':
				cert = load_img(os.path.join(endCLS+'Drop/',filename.replace(params.CLSPRED_FILE_STR, params.LABEL_ENT_STR)))
				cert = 1-cert
			elif certainty == 'MCLUD':
				cert = load_img(os.path.join(endCLS+'Drop/',filename.replace(params.CLSPRED_FILE_STR, params.LABEL_MCLU_STR)))
			elif certainty == 'MCLUS':
				cert = load_img(os.path.join(endCLS+'Still/',filename.replace(params.CLSPRED_FILE_STR, params.LABEL_MCLU_STR)))
			elif certainty == 'STD':
				cert = load_img(os.path.join(endCLS+'Drop/',filename.replace(params.CLSPRED_FILE_STR, params.LABEL_STD_STR)))
				cert = cert/max_std
				cert = 1-cert
			elif certainty == 'DELTP':
				cert = load_img(os.path.join(endCLS+'Drop/',filename.replace(params.CLSPRED_FILE_STR, params.LABEL_DELPE_STR)))
				cert = 1-cert

			gt = load_img(os.path.join(endGT,filename.replace(params.CLSPRED_FILE_STR, params.LABEL_FILE_STR)))
			S1= load_img(os.path.join(endS1,filename.replace(params.CLSPRED_FILE_STR, params.S1_FILE_STR)))
			S2= load_img(os.path.join(endS2,filename.replace(params.CLSPRED_FILE_STR, params.S2_FILE_STR)))
			A = S1 != [0,0,0,0,0,0,0]
			A = A[:,:,0]
			B = S2 != [0,0,0,0,0,0,0,0,0,0]
			B = B[:,:,0]
			A = A*B
			X = gt == clsImg
			X = A*X
			Y = gt != clsImg
			Y = A*Y

			A1 = cert>=infLim
			A2 = cert<supLim
			A=A1*A2
			X = A*X
			Y = A*Y
			y_corretos[j]=y_corretos[j]+np.sum(X)
			y_errados[j]=y_errados[j]+np.sum(Y)

	plt.figure()
	plt.plot(x, y_corretos, color='r', label='Correct Predictions')
	plt.plot(x, y_errados, color='b', label='Wrong Predictions')
	plt.title('Distribution of Uncertainty')
	plt.xlabel('Uncertainty Level')
	plt.ylabel('Number of points')
	plt.legend()
	plt.savefig(endCLS+certainty+'_Distribution.png')