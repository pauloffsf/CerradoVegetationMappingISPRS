
import os
import numpy as np
import pandas as pd


'''
Class Checkpoint Selector:
This class creates an object that selects the best checkpoint (weights) of a previously trained model
either for inference (test) or for refining the training.

'''


class CheckpointSel:
	def __init__ (self, loss_file):

		self.data = pd.read_csv(loss_file, delimiter=',')


	'''
	findBest: Only function of the Class
	Uses the parameters to find the best model based on the best validation accuracy.
	params - list of models parameters
	limiter - delimiter of training epochs. 0 for a single training run.
	train - if want to find the best model to continue training, or for an inference.
	'''

	def findBest(self, params, limiter=0, train=True):
		if train:
			if limiter==0:
				indx=self.data.idxmax(0)
				if (params.NETWORK=='labelUnet' or params.NETWORK=='calibration'):
					indx=indx['val_output10_fn']
					print(indx)
				else:
					indx=indx['val_categorical_accuracy']
				result = self.data.epoch[indx]
			else:
				size = self.data.shape[0]
				if limiter<=size:
					lower_data = self.data.tail(limiter)
					indx=lower_data.idxmax(0)
					if (params.NETWORK=='labelUnet' or params.NETWORK=='calibration'):
						indx=indx['val_output10_fn']
					else:
						indx=indx['val_categorical_accuracy']
					result = lower_data.epoch[indx]
				else:
					raise ValueError("Limiter higher than the number of epochs")
		else:
			if limiter==0:
				indx=self.data.idxmax(0)
				if (params.NETWORK=='labelUnet'):
					indx=indx['val_output10_fn']
					print(indx)
				elif (params.NETWORK=='calibration'):
					indx=self.data.idxmin(0)
					indx=indx['loss']
					print(indx)
				else:
					indx=indx['val_categorical_accuracy']
				result = self.data.epoch[indx]
			else:
				size = self.data.shape[0]
				if limiter<=size:
					lower_data = self.data.tail(limiter)
					indx=lower_data.idxmax(0)
					if (params.NETWORK=='labelUnet'):
						indx=indx['val_output10_fn']
					elif (params.NETWORK=='calibration'):
						indx=lower_data.idxmin(0)
						indx=indx['loss']
						print(indx)
					else:
						indx=indx['val_categorical_accuracy']
					result = lower_data.epoch[indx]
				else:
					raise ValueError("Limiter higher than the number of epochs")
		return 1+result
