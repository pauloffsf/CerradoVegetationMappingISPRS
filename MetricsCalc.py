import glob
import os
import numpy as np
import unetParams as params
from datafunctions import *



'''
Class created to calculate all the metrics for a specific test image
'''
class MetricsPerImg:

	'''
	Constructor for the class
	GT - Reference image filemane
	CLS - Inference  image filemane
	S1 - Sentinel-1 image  filename
	S2 - Sentinel-2  image filename
	n_cat - number of classes (cathegories)
	'''
	def __init__(self, GT, CLS, S1, S2, n_cat):
		self.CLS = CLS
		self.GT = GT
		self.S1 = S1
		self.S2 = S2
		self.confMatrix = None
		self.n_cat=n_cat

	'''
	method that calculates the confusion matrix for the image in context.
	'''
	def confusionMatrix(self):
		self.confMatrix = np.zeros((self.n_cat-1, self.n_cat-1))
		img_CLS = load_img(self.CLS)
		img_GT = load_img(self.GT)
		#Filtrar GT e CLS pelos pixels com dados de S1 e S2 (n"ao contar quando algum dos dois esta total zerado) para poder verificar a precisao de forma efetiva.
		S1_img = load_img(self.S1)
		S2_img = load_img(self.S2)
		A = S1_img != [0,0,0,0,0,0,0]
		#A = S1_img != [0,0,0,0,0,0]
		A = A[:,:,0]
		B = S2_img != [0,0,0,0,0,0,0,0,0,0]
		B = B[:,:,0]
		A = A*B
		img_GT = np.multiply(img_GT,A)
		img_CLS = np.multiply(img_CLS,A)
		for i in range(self.n_cat-1):
			for j in range(self.n_cat-1):
				a = img_GT == i+1
				b = img_CLS == j+1
				self.confMatrix[i,j] = self.confMatrix[i,j] + np.sum(np.logical_and(a,b))


	'''
	function that calculates the recall for a given class
	chosenClass - integer number representing the class
	'''
	def recall_perClass(self, chosenClass):
		if self.confMatrix is None:
			self.confusionMatrix()
		if np.sum(self.confMatrix[chosenClass-1,:])==0:
			return 1
		return float(self.confMatrix[chosenClass-1,chosenClass-1]/np.sum(self.confMatrix[chosenClass-1,:]))

	'''
	function that calculates the precision for a given class
	chosenClass - integer number representing the class
	'''
	def precision_perClass(self, chosenClass):
		if self.confMatrix is None:
			self.confusionMatrix()
		if np.sum(self.confMatrix[:,chosenClass-1])==0:
			return 0
		return float(self.confMatrix[chosenClass-1,chosenClass-1]/np.sum(self.confMatrix[:,chosenClass-1]))

    '''
	function that calculates the f1-score for a given class
	chosenClass - integer number representing the class
	'''
	def f1_perClass(self, chosenClass):
		precision = self.precision_perClass(chosenClass)
		recall = self.recall_perClass(chosenClass)
		if precision+recall>0:
			return 2*((precision*recall)/(precision+recall))
		else:
			return 0
	'''
	function that calculates the Overall accuracy
	'''
	def global_Precision(self):
		if self.confMatrix is None:
			self.confusionMatrix()
		a = 0
		for i in range(self.n_cat-1):
			a = a+self.confMatrix[i,i]
		return float(a/np.sum(self.confMatrix))

	'''
	function that obtains the mean Intersection Over Union
	'''
	def meanIoU(self):
		a = []
		for i in range(self.n_cat-1):
			a.append(self.IoU(i+1))
		return np.mean(a)

	'''
	function that obtain the mean recall
	'''
	def meanRecall(self):
		a = []
		for i in range(self.n_cat-1):
			a.append(self.recall_perClass(i+1))
		return np.mean(a)

	'''
	function that calculates the mean F1-Score
	'''
	def meanF1(self):
		a = []
		for i in range(self.n_cat-1):
			a.append(self.f1_perClass(i+1))
		return np.mean(a)

	'''
	function that calculates the intersection over union for a given class
	chosenClass - integer number representing the class
	'''
	def IoU(self, chosenClass):
		if self.confMatrix is None:
			self.confusionMatrix()
		I = self.confMatrix[chosenClass-1,chosenClass-1]
		U = np.sum(self.confMatrix[chosenClass-1,:]) + np.sum(self.confMatrix[:,chosenClass-1])-I
		if U==0:
			return 1
		return float(I/U)

	'''
	function that prints all the status on a *.txt file
	file - file object where the status will be printed on.
	'''
	def printStatus(self, file):
		file.write('Global Precision: ')
		file.write(str(self.global_Precision()))
		file.write('\n')
		for i in range(1, self.n_cat):
			file.write('Precision Class'+str(i)+': ')
			file.write(str(self.precision_perClass(i)))
			file.write('\n')
		file.write('mean IoU: ')
		file.write(str(self.meanIoU()))
		file.write('\n')
		for i in range(1, self.n_cat):
			file.write('IoU per Class'+str(i)+': ')
			file.write(str(self.IoU(i)))
			file.write('\n')
		file.write('mean F1-Score: ')
		file.write(str(self.meanF1()))
		file.write('\n')
		for i in range(1, self.n_cat):
			file.write('F1-Score per Class'+str(i)+': ')
			file.write(str(self.f1_perClass(i)))
			file.write('\n')
		file.write('mean Recall: ')
		file.write(str(self.meanRecall()))
		file.write('\n')
		for i in range(1, self.n_cat):
			file.write('Recall per Class'+str(i)+': ')
			file.write(str(self.recall_perClass(i)))
			file.write('\n')

'''
Class that calculates the metrics for all test images, separating per image. This class creates objects of the previous class (MetricsPerImg) to calculate the metrics to each image in the folder
and print in one only *.txt file the information about metrics for each file.
'''
class MetricsGroupImg:

	'''
	Constructor of the class.
	add_GT - address were the reference images are
	add_CLS - address where the inference images are
	add_S1 - address where the S1 images are
	add_S2 - address where the S2 images are
	n_cat - number of classes (cathegories)
	'''
	def __init__ (self, add_GT, add_CLS, add_S1, add_S2, n_cat):
		self.add_CLS = add_CLS
		self.add_GT = add_GT
		self.add_S1 = add_S1
		self.add_S2 = add_S2
		self.listCLS = glob(os.path.join(add_CLS, '*%s*.%s' % (params.CLSPRED_FILE_STR,params.LABEL_FILE_EXT)))
		self.confMatrix = None
		self.n_cat = n_cat

	'''
	method to save all the metrics for each image in a *.txt file
	filename - name of the *.txt file
	'''
	def StatsOnFile(self, filename):
		f = open(self.add_CLS+filename, 'w', encoding='utf-8')

		for img_add in self.listCLS:
			f.write('Data for'+ img_add)
			f.write('\n')
			GT_img_name = os.path.split(img_add)[-1]
			S1_img_name = self.add_S1+GT_img_name.replace(params.CLSPRED_FILE_STR,params.S1_FILE_STR)
			S2_img_name = self.add_S2+GT_img_name.replace(params.CLSPRED_FILE_STR,params.S2_FILE_STR)
			GT_img_name = self.add_GT+GT_img_name.replace(params.CLSPRED_FILE_STR,params.LABEL_FILE_STR)
			img = MetricsPerImg(GT_img_name, img_add, S1_img_name, S2_img_name, self.n_cat)
			img.printStatus(f)
		f.close()


'''
Class that calculates all the metrics for the whole test set.
'''
class MetricsCalc:


	'''
	Constructor of the class.
	add_GT - address were the reference images are
	add_CLS - address where the inference images are
	add_S1 - address where the S1 images are
	add_S2 - address where the S2 images are
	n_cat - number of classes (cathegories)
	'''
	def __init__ (self, add_GT, add_CLS, add_S1, add_S2, n_cat):

		
		self.add_CLS = add_CLS
		self.add_GT = add_GT
		self.add_S1 = add_S1
		self.add_S2 = add_S2
		#change back: CLSPRED_FILE_STR vs LABEL30_FILE_STR
		self.listGT = glob(os.path.join(add_GT, '*%s*.%s' % (params.LABEL_FILE_STR,params.LABEL_FILE_EXT)))
		self.listCLS = glob(os.path.join(add_CLS, '*%s*.%s' % (params.CLSPRED_FILE_STR,params.LABEL_FILE_EXT)))
		self.confMatrix = None
		self.n_cat = n_cat
		if len(self.listGT) != len(self.listCLS):
			raise ValueError("Different number of Predicted and Ground Truth Tiles")
	
	'''
	method that calculates the confusion matrix for the image in context.
	'''
	def confusionMatrix(self):

		self.confMatrix = np.zeros((self.n_cat-1, self.n_cat-1))
		for img_add in self.listCLS:
			img_CLS = load_img(img_add)
			GT_img_name = os.path.split(img_add)[-1]
			S1_img_name = GT_img_name.replace(params.CLSPRED_FILE_STR,params.S1_FILE_STR)
			S2_img_name = GT_img_name.replace(params.CLSPRED_FILE_STR,params.S2_FILE_STR)
			GT_img_name = GT_img_name.replace(params.CLSPRED_FILE_STR,params.LABEL_FILE_STR)
			img_GT = load_img(self.add_GT+GT_img_name)
			#Filtrar GT e CLS pelos pixels com dados de S1 e S2 (n"ao contar quando algum dos dois esta total zerado) para poder verificar a precisao de forma efetiva.
			S1_img = load_img(self.add_S1+S1_img_name)
			S2_img = load_img(self.add_S2+S2_img_name)
			#A = S1_img != [0,0,0,0,0,0]
			A = S1_img != [0,0,0,0,0,0,0]
			A = A[:,:,0]
			B = S2_img != [0,0,0,0,0,0,0,0,0,0]
			B = B[:,:,0]
			A = A*B
			print(A.shape)
			#change back, by deleting
			A = cv2.resize(A.astype('uint8'), (img_GT.shape[1],img_GT.shape[0]), interpolation=cv2.INTER_AREA)
			print(A.shape)
			print(img_GT.shape)
			img_GT = np.multiply(img_GT,A)
			img_CLS = np.multiply(img_CLS,A)
			for i in range(self.n_cat-1):
				for j in range(self.n_cat-1):
					a = img_GT == i+1
					b = img_CLS == j+1
					self.confMatrix[i,j] = self.confMatrix[i,j] + np.sum(np.logical_and(a,b))


	'''
	function that calculates the precision for a given class
	chosenClass - integer number representing the class
	'''
	def precision_perClass(self, chosenClass):
		if self.confMatrix is None:
			self.confusionMatrix()
		if np.sum(self.confMatrix[:,chosenClass-1])==0:
			return 1
		return float(self.confMatrix[chosenClass-1,chosenClass-1]/np.sum(self.confMatrix[:,chosenClass-1]))

	'''
	function that calculates the recall for a given class
	chosenClass - integer number representing the class
	'''
	def recall_perClass(self, chosenClass):
		if self.confMatrix is None:
			self.confusionMatrix()
		if np.sum(self.confMatrix[chosenClass-1,:])==0:
			return 0
		return float(self.confMatrix[chosenClass-1,chosenClass-1]/np.sum(self.confMatrix[chosenClass-1,:]))
    
    '''
	function that calculates the f1-score for a given class
	chosenClass - integer number representing the class
	'''
	def f1_perClass(self, chosenClass):
		precision = self.precision_perClass(chosenClass)
		recall = self.recall_perClass(chosenClass)
		if precision+recall>0:
			return 2*((precision*recall)/(precision+recall))
		else:
			return 0

	'''
	function that calculates the Overall accuracy
	'''
	def global_Precision(self):
		if self.confMatrix is None:
			self.confusionMatrix()
		a = 0
		for i in range(self.n_cat-1):
			a = a+self.confMatrix[i,i]
		return float(a/np.sum(self.confMatrix))

	'''
	function that obtains the mean Intersection Over Union
	'''
	def meanIoU(self):
		a = []
		for i in range(self.n_cat-1):
			a.append(self.IoU(i+1))
		return np.mean(a)

	'''
	function that obtain the mean recall
	'''
	def meanRecall(self):
		a = []
		for i in range(self.n_cat-1):
			a.append(self.recall_perClass(i+1))
		return np.mean(a)

	'''
	function that calculates the mean F1-Score
	'''
	def meanF1(self):
		a = []
		for i in range(self.n_cat-1):
			a.append(self.f1_perClass(i+1))
		return np.mean(a)

	'''
	function that calculates the intersection over union for a given class
	chosenClass - integer number representing the class
	'''
	def IoU(self, chosenClass):
		if self.confMatrix is None:
			self.confusionMatrix()
		I = self.confMatrix[chosenClass-1,chosenClass-1]
		U = np.sum(self.confMatrix[chosenClass-1,:]) + np.sum(self.confMatrix[:,chosenClass-1])-I
		if U==0:
			return 1
		return float(I/U)

	'''
	function that prints all the status on a *.txt file
	filename - name of the *.txt file
	'''
	def StatsOnFile(self, filename):
		with open(self.add_CLS+filename, 'w', encoding='utf-8') as f:
			f.write('Global Precision: ')
			f.write(str(self.global_Precision()))
			f.write('\n')
			for i in range(1, self.n_cat):
				f.write('Precision Class'+str(i)+': ')
				f.write(str(self.precision_perClass(i)))
				f.write('\n')
			f.write('mean IoU: ')
			f.write(str(self.meanIoU()))
			f.write('\n')
			for i in range(1, self.n_cat):
				f.write('IoU per Class'+str(i)+': ')
				f.write(str(self.IoU(i)))
				f.write('\n')
			f.write('mean F1-Score: ')
			f.write(str(self.meanF1()))
			f.write('\n')
			for i in range(1, self.n_cat):
				f.write('F1-Score per Class'+str(i)+': ')
				f.write(str(self.f1_perClass(i)))
				f.write('\n')
			f.write('mean Recall: ')
			f.write(str(self.meanRecall()))
			f.write('\n')
			for i in range(1, self.n_cat):
				f.write('Recall per Class'+str(i)+': ')
				f.write(str(self.recall_perClass(i)))
				f.write('\n')