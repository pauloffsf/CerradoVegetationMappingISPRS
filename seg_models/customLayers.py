import tensorflow as tf
from tensorflow.keras import layers
import math
from seg_models.wavetf import *


'''
Layer created based on the work in "https://www.mdpi.com/2072-4292/11/3/282".
'''

class quadLayer (layers.Layer):
	def __init__ (self, featuremaps, stride=2, kernelsize=2):
		super(quadLayer, self).__init__()
		self.stride = stride
		self.kernelsize = kernelsize
		self.featuremaps = featuremaps

	def get_config(self):
		config = super().get_config()
		config.update({
			"featuremaps": self.featuremaps,
			"stride": self.stride,
			"kernelsize": self.kernelsize,
		})
		return config

	def build(self, inputs):
		in_shape = inputs.as_list()
		self.channels = in_shape[3]
		kernel_initializer = tf.keras.initializers.VarianceScaling(2.0, distribution='untruncated_normal')
		self.weights1 = self.add_weight(shape=[self.channels, self.kernelsize*self.kernelsize, self.featuremaps], initializer=kernel_initializer, trainable=True,name = 'W1')
		self.weights2 = self.add_weight(shape=[self.channels, self.kernelsize*self.kernelsize*self.kernelsize*self.kernelsize, self.featuremaps], initializer=kernel_initializer, trainable=True,name = 'W2')
		super(quadLayer, self).build(inputs)

	@tf.function
	def call(self, inputs, *args, **kwargs):
		return tf.vectorized_map(fn = self.functionbatch, elems=inputs)

	def functionbatch(self, tensor):
		shape = tensor.get_shape().as_list()
		tensor=tf.expand_dims(tensor, axis=0)
		patches = tf.image.extract_patches(tensor, sizes=[1, self.kernelsize, self.kernelsize, 1], strides = [1, self.stride, self.stride, 1], rates=[1, 1, 1, 1], padding='VALID')
		patches_shape = patches.get_shape().as_list()
		patches = tf.reshape(patches,(-1,shape[2], self.kernelsize*self.kernelsize))
		patches = tf.vectorized_map(fn=self.functionElement,elems=patches)
		return tf.reshape(patches,(patches_shape[1], patches_shape[2],-1))

	def functionElement(self, tensor):
		z1 = tf.linalg.matmul(tensor,self.weights1)
		tensor = tf.expand_dims(tensor, axis=-1)
		secondOrder = tf.linalg.matmul(tensor,tensor,transpose_b=True)
		shape = secondOrder.get_shape().as_list()
		secondOrder = tf.reshape(secondOrder,[shape[0], shape[1]*shape[2]])
		z2=tf.linalg.matmul(secondOrder,self.weights2)
		return tf.math.add(z1,z2)


'''
Layer created based on Wavelet Pooling, according to https://openreview.net/pdf?id=rkhlb8lCZ
Created based on WaveTF.
'''

def waveletPooling (input_layer, kernel='haar'):
	'''
		input_layer = layer
		tipo = 'db2' ou 'haar'
	'''
	w = WaveTFFactory().build(kernel, dim=2) 
	t = w.call(input_layer)
	in_shape = t.get_shape().as_list()
	a = tf.split(t,in_shape[3],axis=-1)
	t  = layers.concatenate(a[0:in_shape[3]], axis=-1)
	t = w.call(t)
	w_i = WaveTFFactory().build(kernel, dim=2, inverse=True)
	return w_i.call(t)


'''
Layer for the Temperature Scaling method for Calibration.
'''

class Temperature(layers.Layer):
  def __init__(self):
    super(Temperature, self).__init__()

    #your variable goes here
    self.variable = tf.Variable(1., trainable=True, dtype=tf.float32)

  def call(self, inputs, **kwargs):

    # your mul operation goes here
    x = tf.math.divide(inputs, self.variable)

    return x