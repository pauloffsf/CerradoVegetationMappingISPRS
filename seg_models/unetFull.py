import tensorflow as tf
from tensorflow.keras import layers
from seg_models import customLayers
from seg_models import CustomModel
import tensorflow_probability as tfp
import cvnn.layers as complex_layers
import cvnn.activations as complex_activations

'''
	Documentation on Encode Blocks
		inputs:
		input_layer - the previous layer of the network, 
		num_filters - list with the number of channels that each conv must have. The lenght of the list must be equal to num_deps.
		num_deps - how many conv layers in this stage of the Unet
		outputs:
		out_layer - output layer.
		c - the layer that must be concatenated in the Decodeblocks of the Unet.
'''
def FPB_Block(input_layer, num_filters, dropout_rate, layers_name):

	last_filters = input_layer.get_shape().as_list()[-1]
	c1 = layers.Conv2D(num_filters, kernel_size=1, padding="same", name = layers_name+"_conv1")(input_layer)
	c1 = layers.Dropout(dropout_rate)(c1)
	c1 = layers.BatchNormalization(name=layers_name+"_BN1")(c1)
	c1 = layers.Activation('relu', name=layers_name+"_Activation1")(c1)
	c2 =layers.Conv2D(num_filters, kernel_size=3, padding="same", name = layers_name+"_conv3")(input_layer)
	c2 = layers.Dropout(dropout_rate)(c2)
	c2 = layers.BatchNormalization(name=layers_name+"_BN2")(c2)
	c2 = layers.Activation('relu', name=layers_name+"_Activation2")(c2)
	c3 = layers.Conv2D(num_filters, kernel_size=5, padding="same", name = layers_name+"_conv5")(input_layer)
	c3 = layers.Dropout(dropout_rate)(c3)
	c3 = layers.BatchNormalization(name=layers_name+"_BN3")(c3)
	c3 = layers.Activation('relu', name=layers_name+"_Activation3")(c3)
	c = layers.concatenate([c1, c2, c3], axis=-1)
	c = layers.Conv2D(last_filters, kernel_size=1, padding="same", name = layers_name+"_convOut")(c)
	c = layers.Dropout(dropout_rate)(c)
	c = layers.BatchNormalization(name=layers_name+"_BNOut")(c)
	c = layers.Activation('relu', name=layers_name+"_ActivationOut")(c)

	return layers.Add()([c,input_layer])


def Complex_FPB_Block(input_layer, num_filters, dropout_rate, layers_name):

	last_filters = input_layer.get_shape().as_list()[-1]
	c1 = complex_layers.ComplexConv2D(num_filters, kernel_size=1, padding="same", name = layers_name+"_conv1")(input_layer)
	c1 = complex_layers.ComplexDropout(dropout_rate)(c1)
	c1 = layers.Activation(complex_activations.cart_relu, name=layers_name+"_Activation1")(c1)
	c2 = complex_layers.ComplexConv2D(num_filters, kernel_size=3, padding="same", name = layers_name+"_conv3")(input_layer)
	c2 = complex_layers.ComplexDropout(dropout_rate)(c2)
	c2 = layers.Activation(complex_activations.cart_relu, name=layers_name+"_Activation2")(c2)
	c3 = complex_layers.ComplexConv2D(num_filters, kernel_size=5, padding="same", name = layers_name+"_conv5")(input_layer)
	c3 = complex_layers.ComplexDropout(dropout_rate)(c3)
	c3 = layers.Activation(complex_activations.cart_relu, name=layers_name+"_Activation3")(c3)
	c = layers.concatenate([c1, c2, c3], axis=-1)
	c = complex_layers.ComplexConv2D(last_filters, kernel_size=1, padding="same", name = layers_name+"_convOut")(c)
	c = complex_layers.ComplexDropout(dropout_rate)(c)
	c = layers.Activation(complex_activations.cart_relu, name=layers_name+"_ActivationOut")(c)

	return layers.Add()([c,input_layer])


def EncodeComplexFPB_Block(input_layer, num_filters, num_deps, layers_name, pool_size=(2, 2), strides=(2, 2)):
	dropout_rate = 0.3
	c = None
	out_layer = None
	if len(num_filters) != num_deps:
		raise ValueError('Error on the definition of the number of filters for the layer. The lenth of num_filters must be equal to num_deps')
	else:
		c = Complex_FPB_Block(input_layer, num_filters[0], dropout_rate, layers_name+"_FPB_Block_0")
		for i in range(1, num_deps):
			c = Complex_FPB_Block(c, num_filters[i], dropout_rate, layers_name+"_FPB_Block_"+str(i))
		out_layer = complex_layers.ComplexMaxPooling2D(pool_size=pool_size, strides=strides, padding='valid', name = layers_name+"_pool")(c)
	return out_layer, c


def EncodeFPB_Block(input_layer, num_filters, num_deps, layers_name, pool_size=(2, 2), strides=(2, 2)):
	dropout_rate = 0.3
	c = None
	out_layer = None
	if len(num_filters) != num_deps:
		raise ValueError('Error on the definition of the number of filters for the layer. The lenth of num_filters must be equal to num_deps')
	else:
		c = FPB_Block(input_layer, num_filters[0], dropout_rate, layers_name+"_FPB_Block_0")
		for i in range(1, num_deps):
			c = FPB_Block(c, num_filters[i], dropout_rate, layers_name+"_FPB_Block_"+str(i))
		out_layer = layers.MaxPool2D(pool_size=pool_size, strides=strides, padding='valid', name = layers_name+"_pool")(c)
	return out_layer, c


def EncodeVGGBlocks(input_layer, num_filters, num_deps, layers_name, pool_size=(2, 2), strides=(2, 2)):
	dropout_rate = 0.3
	c = None
	out_layer = None
	if len(num_filters) != num_deps:
		raise ValueError('Error on the definition of the number of filters for the layer. The lenth of num_filters must be equal to num_deps')
	else:
		c = layers.Conv2D(num_filters[0], kernel_size=3, padding="same", name = layers_name+"_conv0")(input_layer)
		c = layers.Dropout(dropout_rate)(c)
		for i in range(1,num_deps):
			c = layers.BatchNormalization(name=layers_name+"_BN"+str(i-1))(c)
			c = layers.Activation('relu', name=layers_name+"_Activation"+str(i-1))(c)
			c = layers.Conv2D(num_filters[i], kernel_size=3, padding="same", name = layers_name+"_conv"+str(i))(c)
			c = layers.Dropout(dropout_rate)(c)
		c = layers.BatchNormalization(name=layers_name+"_BN"+str(num_deps-1))(c)
		c = layers.Activation('relu', name=layers_name+"_Activation"+str(num_deps-1))(c)
		out_layer = layers.MaxPool2D(pool_size=pool_size, strides=strides, padding='valid', name = layers_name+"_pool")(c)
	return out_layer, c

def EncodeWaveVGGBlocks(input_layer, num_filters, num_deps, layers_name):
	dropout_rate = 0.3
	c = None
	out_layer = None
	if len(num_filters) != num_deps:
		raise ValueError('Error on the definition of the number of filters for the layer. The lenth of num_filters must be equal to num_deps')
	else:
		c = layers.Conv2D(num_filters[0], kernel_size=3, padding="same", name = layers_name+"_conv0")(input_layer)
		c = layers.Dropout(dropout_rate)(c)
		for i in range(1,num_deps):
			c = layers.BatchNormalization(name=layers_name+"_BN"+str(i-1))(c)
			c = layers.Activation('relu', name=layers_name+"_Activation"+str(i-1))(c)
			c = layers.Conv2D(num_filters[i], kernel_size=3, padding="same", name = layers_name+"_conv"+str(i))(c)
			c = layers.Dropout(dropout_rate)(c)
		c = layers.BatchNormalization(name=layers_name+"_BN"+str(num_deps-1))(c)
		c = layers.Activation('relu', name=layers_name+"_Activation"+str(num_deps-1))(c)
		out_layer = customLayers.waveletPooling(c)
	return out_layer, c

def EncodeResNetBlocks(input_layer, num_filters, num_deps, layers_name, pool_size=(2, 2), strides=(2, 2)):
	dropout_rate = 0.3
	c = None
	out_layer = None
	if len(num_filters) != num_deps:
		raise ValueError('Error on the definition of the number of filters for the layer. The lenth of num_filters must be equal to num_deps')
	else:
		c = layers.Conv2D(num_filters[0], kernel_size=3, padding="same", name = layers_name+"_conv0")(input_layer)
		c = layers.Dropout(dropout_rate)(c)
		c0=c
		for i in range(1,num_deps):
			c = layers.BatchNormalization(name=layers_name+"_BN"+str(i-1))(c)
			c = layers.Activation('relu', name=layers_name+"_Activation"+str(i-1))(c)
			c = layers.Conv2D(num_filters[i], kernel_size=3, padding="same", name = layers_name+"_conv"+str(i))(c)
			c = layers.Dropout(dropout_rate)(c)
		c = layers.BatchNormalization(name=layers_name+"_BN"+str(num_deps-1))(c)
		c = layers.Activation('relu', name=layers_name+"_Activation"+str(num_deps-1))(c)
		c = layers.concatenate([c, c0], axis=-1)
		out_layer = layers.MaxPool2D(pool_size=pool_size, strides=strides, padding='valid', name = layers_name+"_pool")(c)
	return out_layer, c


def EncodeWaveResNetBlocks(input_layer, num_filters, num_deps, layers_name):
	dropout_rate = 0.3
	c = None
	out_layer = None
	if len(num_filters) != num_deps:
		raise ValueError('Error on the definition of the number of filters for the layer. The lenth of num_filters must be equal to num_deps')
	else:
		c = layers.Conv2D(num_filters[0], kernel_size=3, padding="same", name = layers_name+"_conv0")(input_layer)
		c = layers.Dropout(dropout_rate)(c)
		c0=c
		for i in range(1,num_deps):
			c = layers.BatchNormalization(name=layers_name+"_BN"+str(i-1))(c)
			c = layers.Activation('relu', name=layers_name+"_Activation"+str(i-1))(c)
			c = layers.Conv2D(num_filters[i], kernel_size=3, padding="same", name = layers_name+"_conv"+str(i))(c)
			c = layers.Dropout(dropout_rate)(c)
		c = layers.concatenate([c, c0], axis=-1)
		c = layers.BatchNormalization(name=layers_name+"_BN"+str(num_deps-1))(c)
		c = layers.Activation('relu', name=layers_name+"_Activation"+str(num_deps-1))(c)
		out_layer = customLayers.waveletPooling(c)
	return out_layer, c

def EncodeDenseNetBlocks(input_layer, num_filters, num_deps, layers_name, pool_size=(2, 2), strides=(2, 2)):
	dropout_rate = 0.3
	c = None
	out_layer = None
	layers_before = []
	if len(num_filters) != num_deps:
		raise ValueError('Error on the definition of the number of filters for the layer. The lenth of num_filters must be equal to num_deps')
	else:
		c = layers.Conv2D(num_filters[0], kernel_size=3, padding="same", name = layers_name+"_conv0")(input_layer)
		c = layers.Dropout(dropout_rate)(c)
		layers_before.append(c)
		for i in range(1,num_deps):
			c = layers.BatchNormalization(name=layers_name+"_BN"+str(i-1))(c)
			c = layers.Activation('relu', name=layers_name+"_Activation"+str(i-1))(c)
			c = layers.Conv2D(num_filters[i], kernel_size=3, padding="same", name = layers_name+"_conv"+str(i))(c)
			c = layers.Dropout(dropout_rate)(c)
			layers_before.append(c)
			c = layers.concatenate(layers_before, axis=-1)
		c = layers.BatchNormalization(name=layers_name+"_BN"+str(num_deps-1))(c)
		c = layers.Activation('relu', name=layers_name+"_Activation"+str(num_deps-1))(c)
		out_layer = layers.MaxPool2D(pool_size=pool_size, strides=strides, padding='valid', name = layers_name+"_pool")(c)
	return out_layer, c


def EncodeWaveDenseNetBlocks(input_layer, num_filters, num_deps, layers_name):
	dropout_rate = 0.3
	c = None
	out_layer = None
	layers_before = []
	if len(num_filters) != num_deps:
		raise ValueError('Error on the definition of the number of filters for the layer. The lenth of num_filters must be equal to num_deps')
	else:
		c = layers.Conv2D(num_filters[0], kernel_size=3, padding="same", name = layers_name+"_conv0")(input_layer)
		c = layers.Dropout(dropout_rate)(c)
		layers_before.append(c)
		for i in range(1,num_deps):
			c = layers.BatchNormalization(name=layers_name+"_BN"+str(i-1))(c)
			c = layers.Activation('relu', name=layers_name+"_Activation"+str(i-1))(c)
			c = layers.Conv2D(num_filters[i], kernel_size=3, padding="same", name = layers_name+"_conv"+str(i))(c)
			c = layers.Dropout(dropout_rate)(c)
			layers_before.append(c)
			c = layers.concatenate(layers_before, axis=-1)
		c = layers.BatchNormalization(name=layers_name+"_BN"+str(num_deps-1))(c)
		c = layers.Activation('relu', name=layers_name+"_Activation"+str(num_deps-1))(c)
		out_layer = customLayers.waveletPooling(c)
	return out_layer, c



def EncodeComplexResNetBlocks(input_layer, num_filters, num_deps, layers_name, pool_size=(2, 2), strides=(2, 2)):
	dropout_rate = 0.3
	c = None
	out_layer = None
	if len(num_filters) != num_deps:
		raise ValueError('Error on the definition of the number of filters for the layer. The lenth of num_filters must be equal to num_deps')
	else:
		c = complex_layers.ComplexConv2D(num_filters[0], kernel_size=3, padding="same", name = layers_name+"_conv0")(input_layer)
		c = complex_layers.ComplexDropout(dropout_rate)(c)
		c0=c
		for i in range(1,num_deps):
			#c = complex_layers.ComplexBatchNormalization(name=layers_name+"_BN"+str(i-1))(c)
			c = layers.Activation(complex_activations.cart_relu, name=layers_name+"_Activation"+str(i-1))(c)
			c = complex_layers.ComplexConv2D(num_filters[i], kernel_size=3, padding="same", name = layers_name+"_conv"+str(i))(c)
			c = complex_layers.ComplexDropout(dropout_rate)(c)
		c = layers.concatenate([c, c0], axis=-1)
		#c = complex_layers.ComplexBatchNormalization(name=layers_name+"_BN"+str(num_deps-1))(c)
		c = layers.Activation(complex_activations.cart_relu, name=layers_name+"_Activation"+str(num_deps-1))(c)
		out_layer = complex_layers.ComplexMaxPooling2D(pool_size=pool_size, strides=strides, padding='valid', name = layers_name+"_pool")(c)
	return out_layer, c


##########################################################################################################################################################

'''
Documentation on DecodeBlocks
		inputs:
		input_layer - the previous layer of the network, 
		concat_layer - layer to be concatenated through the skip-connections of the Unet
		num_filters - list with the number of channels that each conv must have. The lenght of the list must be equal to num_deps.
		num_deps - how many conv layers in this stage of the Unet
		outputs:
		c - output layer.
'''
def DecodeFPB_Block(input_layer, concat_layer, num_filters, num_deps, layers_name, kernel_size=2, strides=(2,2)):
	dropout_rate = 0.3
	c = None
	if len(num_filters) != num_deps:
		raise ValueError('Error on the definition of the number of filters for the layer. The lenth of num_filters must be equal to num_deps')
	else:
		c = layers.Conv2DTranspose(num_filters[0], kernel_size=kernel_size, strides=strides, name = layers_name+"_transp")(input_layer)
		c = layers.Dropout(dropout_rate)(c)
		c = layers.BatchNormalization(name=layers_name+"_BN")(c)
		c = layers.Activation('relu', name=layers_name+"_Activation")(c)
		if concat_layer is not None:
			c = layers.concatenate([concat_layer, c], axis=-1)
		for i in range(num_deps):
			c = FPB_Block(c, num_filters[i], dropout_rate, layers_name+"_FPB_Block_"+str(i))
	return c


def DecodeComplexFPB_Block(input_layer, concat_layer, num_filters, num_deps, layers_name, kernel_size=2, strides=(2,2)):
	dropout_rate = 0.3
	c = None
	if len(num_filters) != num_deps:
		raise ValueError('Error on the definition of the number of filters for the layer. The lenth of num_filters must be equal to num_deps')
	else:
		c = complex_layers.ComplexConv2DTranspose(num_filters[0], kernel_size=kernel_size, strides=strides, name = layers_name+"_transp")(input_layer)
		c = complex_layers.ComplexDropout(dropout_rate)(c)
		c = layers.Activation(complex_activations.cart_relu, name=layers_name+"_Activation")(c)
		if concat_layer is not None:
			c = layers.concatenate([concat_layer, c], axis=-1)
		for i in range(num_deps):
			c = Complex_FPB_Block(c, num_filters[i], dropout_rate, layers_name+"_FPB_Block_"+str(i))
	return c

def DecodeVGGBlocks(input_layer, concat_layer, num_filters, num_deps, layers_name, kernel_size=2, strides=(2,2)):
	dropout_rate = 0.3
	c = None
	if len(num_filters) != num_deps:
		raise ValueError('Error on the definition of the number of filters for the layer. The lenth of num_filters must be equal to num_deps')
	else:
		c = layers.Conv2DTranspose(num_filters[0], kernel_size=kernel_size, strides=strides, name = layers_name+"_transp")(input_layer)
		c = layers.Dropout(dropout_rate)(c)
		c = layers.BatchNormalization(name=layers_name+"_BN")(c)
		c = layers.Activation('relu', name=layers_name+"_Activation")(c)
		if concat_layer is not None:
			c = layers.concatenate([concat_layer, c], axis=-1)
		for i in range(num_deps):
			c = layers.Conv2D(num_filters[i], kernel_size=3, padding="same", name = layers_name+"_conv"+str(i))(c)
			c = layers.Dropout(dropout_rate)(c)
			c = layers.BatchNormalization(name=layers_name+"_BN"+str(i))(c)
			c = layers.Activation('relu', name=layers_name+"_Activation"+str(i))(c)
	return c


def DecodeResNetBlocks(input_layer, concat_layer, num_filters, num_deps, layers_name, kernel_size=2, strides=(2,2)):
	dropout_rate = 0.3
	c = None
	if len(num_filters) != num_deps:
		raise ValueError('Error on the definition of the number of filters for the layer. The lenth of num_filters must be equal to num_deps')
	else:
		print('entering')
		print(tf.shape(input_layer))
		c = layers.Conv2DTranspose(num_filters[0], kernel_size=kernel_size, strides=strides, name = layers_name+"_transp")(input_layer)
		print('left')
		print(tf.shape(c))
		c = layers.Dropout(dropout_rate)(c)
		c = layers.BatchNormalization(name=layers_name+"_BN")(c)
		c = layers.Activation('relu', name=layers_name+"_Activation")(c)
		if concat_layer is not None:
			c = layers.concatenate([concat_layer, c], axis=-1)
		c0=c
		for i in range(num_deps):
			c = layers.Conv2D(num_filters[i], kernel_size=3, padding="same", name = layers_name+"_conv"+str(i))(c)
			c = layers.Dropout(dropout_rate)(c)
			c = layers.BatchNormalization(name=layers_name+"_BN"+str(i))(c)
			c = layers.Activation('relu', name=layers_name+"_Activation"+str(i))(c)
		c = layers.concatenate([c, c0], axis=-1)
	return c

def DecodeDenseNetBlocks(input_layer, concat_layer, num_filters, num_deps, layers_name, kernel_size=2, strides=(2,2)):
	dropout_rate = 0.3
	c = None
	layers_before = []
	if len(num_filters) != num_deps:
		raise ValueError('Error on the definition of the number of filters for the layer. The lenth of num_filters must be equal to num_deps')
	else:
		c = layers.Conv2DTranspose(num_filters[0], kernel_size=kernel_size, strides=strides, name = layers_name+"_transp")(input_layer)
		c = layers.Dropout(dropout_rate)(c)
		c = layers.BatchNormalization(name=layers_name+"_BN")(c)
		c = layers.Activation('relu', name=layers_name+"_Activation")(c)
		layers_before.append(c)
		if concat_layer is not None:
			c = layers.concatenate([concat_layer, c], axis=-1)
		for i in range(num_deps):
			c = layers.Conv2D(num_filters[i], kernel_size=3, padding="same", name = layers_name+"_conv"+str(i))(c)
			c = layers.Dropout(dropout_rate)(c)
			layers_before.append(c)
			c = layers.concatenate(layers_before, axis=-1)
			c = layers.BatchNormalization(name=layers_name+"_BN"+str(i))(c)
			c = layers.Activation('relu', name=layers_name+"_Activation"+str(i))(c)
	return c


def DecodeComplexResNetBlocks(input_layer, concat_layer, num_filters, num_deps, layers_name, kernel_size=2, strides=(2,2)):
	dropout_rate = 0.3
	c = None
	if len(num_filters) != num_deps:
		raise ValueError('Error on the definition of the number of filters for the layer. The lenth of num_filters must be equal to num_deps')
	else:
		c = complex_layers.ComplexConv2DTranspose(num_filters[0], kernel_size=kernel_size, strides=strides, name = layers_name+"_transp")(input_layer)
		c = complex_layers.ComplexDropout(dropout_rate)(c)
		#c = complex_layers.ComplexBatchNormalization(name=layers_name+"_BN")(c)
		c = layers.Activation(complex_activations.cart_relu, name=layers_name+"_Activation")(c)
		if concat_layer is not None:
			c = layers.concatenate([concat_layer, c], axis=-1)
		c0=c
		for i in range(num_deps):
			c = complex_layers.ComplexConv2D(num_filters[i], kernel_size=3, padding="same", name = layers_name+"_conv"+str(i))(c)
			c = complex_layers.ComplexDropout(dropout_rate)(c)
			#c = complex_layers.ComplexBatchNormalization(name=layers_name+"_BN"+str(i))(c)
			c = layers.Activation(complex_activations.cart_relu, name=layers_name+"_Activation"+str(i))(c)
		c = layers.concatenate([c, c0], axis=-1)
	return c


####################################################################################################################################################3
'''
Models created based on encode and decode blocks created before:
'''
#Unet close to the original implementation. The difference is the use of zero-padding to better control the resolution changes.
def unetFull(input_shape = None, input_tensor = None, classes=None):
	internal_depths = [2,2,2,2,2]
	num_filters = [64, 128, 256, 512, 1024]
	depth = 5
	filters = []

	for i in range(depth-1):
		a = []
		for j in range(internal_depths[i]):
			a.append(num_filters[i])
		filters.append(a)

	concat_layers = []

	#Encode Part:
	c,conc = EncodeVGGBlocks(input_tensor, filters[0] , internal_depths[0], "EncodeBlock_lvl0")
	concat_layers.append(conc)
	for i in range(1, depth-1):
		c, conc = EncodeVGGBlocks(c, filters[i] , internal_depths[i], "EncodeBlock_lvl"+str(i))
		concat_layers.append(conc)

	for i in range(internal_depths[depth-1]):
		c = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c)
		c = layers.BatchNormalization(name="EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c)
		c = layers.Activation('relu', name="EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c)

	for i in range(depth-2,-1,-1):
		c = DecodeVGGBlocks(c, concat_layers[i], filters[i], internal_depths[i], "DecodeBlock_lvl"+str(i))

	outputs = layers.Conv2D(classes, kernel_size=1, name="final_conv")(c)
	outputs = layers.Activation(activation="sigmoid", name="output")(outputs)
	model = tf.keras.Model(inputs=input_tensor, outputs=outputs, name="u-net")

	return model



#Unet-like middle-fusion.
def UnetDoubleFull(input_shape = None, input_tensor = None, classes=None):
	internal_depths = [2,2,2,2,2]
	num_filters = [64, 128, 256, 512, 1024]
	depth = 5
	filters = []

	for i in range(depth-1):
		a = []
		for j in range(internal_depths[i]):
			a.append(num_filters[i])
		filters.append(a)

	concat_layers1 = []
	concat_layers2 = []
	#Encode Part 1:
	c1,conc = EncodeVGGBlocks(input_tensor[0], filters[0] , internal_depths[0], "W1EncodeBlock_lvl0")
	concat_layers1.append(conc)
	for i in range(1, depth-1):
		c1, conc = EncodeVGGBlocks(c1, filters[i] , internal_depths[i], "W1EncodeBlock_lvl"+str(i))
		concat_layers1.append(conc)

	for i in range(internal_depths[depth-1]):
		c1 = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "W1EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c1)
		c1 = layers.BatchNormalization(name="W1EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c1)
		c1 = layers.Activation('relu', name="W1EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c1)

	#Encode Part 2:
	c2,conc = EncodeVGGBlocks(input_tensor[1], filters[0] , internal_depths[0], "W2EncodeBlock_lvl0")
	concat_layers2.append(conc)
	for i in range(1, depth-1):
		c2, conc = EncodeVGGBlocks(c2, filters[i] , internal_depths[i], "W2EncodeBlock_lvl"+str(i))
		concat_layers2.append(conc)

	for i in range(internal_depths[depth-1]):
		c2 = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c2)
		c2 = layers.BatchNormalization(name="W2EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c2)
		c2 = layers.Activation('relu', name="W2EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c2)

	#Decode with feature fusion
	c = layers.concatenate([c1, c2], axis=-1)

	for i in range(depth-2,-1,-1):
		d = layers.concatenate([concat_layers1[i], concat_layers2[i]], axis=-1)
		c = DecodeVGGBlocks(c, d, filters[i], internal_depths[i], "DecodeBlock_lvl"+str(i))

	outputs = layers.Conv2D(classes, kernel_size=1, name="final_conv")(c)
	outputs = layers.Activation(activation="sigmoid", name="output")(outputs)
	model = tf.keras.Model(inputs=input_tensor, outputs=outputs, name="u-net_double")

	return model



#Unet-like middle-fusion. using wavelet pooling instead of max pooling.
def UnetDoubleFullWave(input_shape = None, input_tensor = None, classes=None):
	internal_depths = [2,2,2,2,2]
	num_filters = [64, 128, 256, 512, 1024]
	depth = 5
	filters = []

	for i in range(depth-1):
		a = []
		for j in range(internal_depths[i]):
			a.append(num_filters[i])
		filters.append(a)

	concat_layers1 = []
	concat_layers2 = []
	#Encode Part 1:
	c1,conc = EncodeVGGBlocks(input_tensor[0], filters[0] , internal_depths[0], "W1EncodeBlock_lvl0")
	concat_layers1.append(conc)
	for i in range(1, depth-1):
		c1, conc = EncodeVGGBlocks(c1, filters[i] , internal_depths[i], "W1EncodeBlock_lvl"+str(i))
		concat_layers1.append(conc)

	for i in range(internal_depths[depth-1]):
		c1 = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "W1EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c1)
		c1 = layers.BatchNormalization(name="W1EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c1)
		c1 = layers.Activation('relu', name="W1EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c1)

	#Encode Part 2:
	c2,conc = EncodeWaveVGGBlocks(input_tensor[1], filters[0] , internal_depths[0], "W2EncodeBlock_lvl0")
	concat_layers2.append(conc)
	for i in range(1, depth-1):
		c2, conc = EncodeWaveVGGBlocks(c2, filters[i] , internal_depths[i], "W2EncodeBlock_lvl"+str(i))
		concat_layers2.append(conc)

	for i in range(internal_depths[depth-1]):
		c2 = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c2)
		c2 = layers.BatchNormalization(name="W2EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c2)
		c2 = layers.Activation('relu', name="W2EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c2)

	#Decode with feature fusion
	c = layers.concatenate([c1, c2], axis=-1)

	for i in range(depth-2,-1,-1):
		d = layers.concatenate([concat_layers1[i], concat_layers2[i]], axis=-1)
		c = DecodeVGGBlocks(c, d, filters[i], internal_depths[i], "DecodeBlock_lvl"+str(i))

	outputs = layers.Conv2D(classes, kernel_size=1, name="final_conv")(c)
	outputs = layers.Activation(activation="sigmoid", name="output")(outputs)
	model = tf.keras.Model(inputs=input_tensor, outputs=outputs, name="u-net_double")

	return model


#Encoder-decoder architecture unet-like, with fewer weights and less resolution reduction.
def unetFree(input_shape = None, input_tensor = None, classes=None):

	internal_depths = [2,2,2,2]
	num_filters = [32, 64, 128, 256]
	depth = 4
	filters = []

	for i in range(depth-1):
		a = []
		for j in range(internal_depths[i]):
			a.append(num_filters[i])
		filters.append(a)

	concat_layers = []

	#Encode Part:
	c,conc = EncodeResNetBlocks(input_tensor, filters[0] , internal_depths[0], "EncodeBlock_lvl0")
	concat_layers.append(conc)
	for i in range(1, depth-1):
		c, conc = EncodeResNetBlocks(c, filters[i] , internal_depths[i], "EncodeBlock_lvl"+str(i))
		concat_layers.append(conc)

	for i in range(internal_depths[depth-1]):
		c = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c)
		c = layers.BatchNormalization(name="EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c)
		c = layers.Activation('relu', name="EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c)

	for i in range(depth-2,-1,-1):
		c = DecodeResNetBlocks(c, concat_layers[i], filters[i], internal_depths[i], "DecodeBlock_lvl"+str(i))

	outputs = layers.Conv2D(classes, kernel_size=1, name="final_conv")(c)
	outputs = layers.Activation(activation="sigmoid", name="output")(outputs)
	model = tf.keras.Model(inputs=input_tensor, outputs=outputs, name="u-net_Free")

	return model


#Encoder-decoder architecture unet-like middle fusion, with fewer weights and less resolution reduction.
def UnetDoubleFree(input_shape = None, input_tensor = None, classes=None):
	internal_depths = [2,2,2,2]
	num_filters = [32, 64, 128, 256]
	depth = 4
	filters = []

	for i in range(depth-1):
		a = []
		for j in range(internal_depths[i]):
			a.append(num_filters[i])
		filters.append(a)

	concat_layers1 = []
	concat_layers2 = []
	#Encode Part 1:
	c1,conc = EncodeResNetBlocks(input_tensor[0], filters[0] , internal_depths[0], "W1EncodeBlock_lvl0")
	concat_layers1.append(conc)
	for i in range(1, depth-1):
		c1, conc = EncodeResNetBlocks(c1, filters[i] , internal_depths[i], "W1EncodeBlock_lvl"+str(i))
		concat_layers1.append(conc)

	for i in range(internal_depths[depth-1]):
		c1 = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "W1EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c1)
		c1 = layers.BatchNormalization(name="W1EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c1)
		c1 = layers.Activation('relu', name="W1EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c1)

	#Encode Part 2:
	c2,conc = EncodeResNetBlocks(input_tensor[1], filters[0] , internal_depths[0], "W2EncodeBlock_lvl0")
	concat_layers2.append(conc)
	for i in range(1, depth-1):
		c2, conc = EncodeResNetBlocks(c2, filters[i] , internal_depths[i], "W2EncodeBlock_lvl"+str(i))
		concat_layers2.append(conc)

	for i in range(internal_depths[depth-1]):
		c2 = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c2)
		c2 = layers.BatchNormalization(name="W2EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c2)
		c2 = layers.Activation('relu', name="W2EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c2)

	#Decode with feature fusion
	c = layers.concatenate([c1, c2], axis=-1)

	for i in range(depth-2,-1,-1):
		d = layers.concatenate([concat_layers1[i], concat_layers2[i]], axis=-1)
		c = DecodeResNetBlocks(c, d, filters[i], internal_depths[i], "DecodeBlock_lvl"+str(i))

	outputs = layers.Conv2D(classes, kernel_size=1, name="final_conv")(c)
	outputs = layers.Activation(activation="sigmoid", name="output")(outputs)
	model = tf.keras.Model(inputs=input_tensor, outputs=outputs, name="u-net_double_Free")

	return model


#Encoder-decoder architecture unet-like for SAR with the statistics learning network adapted to semantic segmentation. Use of the quadratic module layer.
def quadUnet(input_shape = None, input_tensor = None, classes=None):

	internal_depths = [2,2,2,2,2]
	num_filters = [3, 64, 128, 256, 512]
	depth = 5
	filters = []

	for i in range(depth-1):
		a = []
		for j in range(internal_depths[i]):
			a.append(num_filters[i])
		filters.append(a)

	concat_layers = []

	c = customLayers.quadLayer(num_filters[0])
	c = c(input_tensor)
	concat_layers.append(c)
	for i in range(1, depth-1):
		c, conc = EncodeVGGBlocks(c, filters[i] , internal_depths[i], "WEncodeBlock_lvl"+str(i))
		concat_layers.append(conc)

	for i in range(internal_depths[depth-1]):
		c = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c)
		c = layers.BatchNormalization(name="EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c)
		c = layers.Activation('relu', name="EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c)

	for i in range(depth-2,1,-1):
		c = DecodeVGGBlocks(c, concat_layers[i], filters[i], internal_depths[i], "DecodeBlock_lvl"+str(i))

	c = DecodeVGGBlocks(c,layers.concatenate([concat_layers[1], concat_layers[0]],axis=-1), filters[1], internal_depths[1], "DecodeBlock_lvl"+str(1))
	c = DecodeVGGBlocks(c, None, filters[1], internal_depths[0], "DecodeBlock_lvl"+str(0))

	outputs = layers.Conv2D(classes, kernel_size=1, name="final_conv")(c)
	outputs = layers.Activation(activation="sigmoid", name="output")(outputs)
	model = tf.keras.Model(inputs=input_tensor, outputs=outputs, name="Quad_Unet")
	return model

#Encoder-decoder architecture unet-like using Densenet as encoder for SAR with the statistics learning network adapted to semantic segmentation. Use of the quadratic module layer.
def quadUnetDense(input_shape = None, input_tensor = None, classes=None):

	internal_depths = [2,2,2,2,2]
	num_filters = [8, 64, 128, 256, 512]
	depth = 5
	filters = []

	for i in range(depth-1):
		a = []
		for j in range(internal_depths[i]):
			a.append(num_filters[i])
		filters.append(a)

	concat_layers = []

	c = customLayers.quadLayer(num_filters[0])
	c = c(input_tensor)
	concat_layers.append(c)
	for i in range(1, depth-1):
		c, conc = EncodeDenseNetBlocks(c, filters[i] , internal_depths[i], "WEncodeBlock_lvl"+str(i))
		concat_layers.append(conc)

	for i in range(internal_depths[depth-1]):
		c = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c)
		c = layers.BatchNormalization(name="EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c)
		c = layers.Activation('relu', name="EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c)

	for i in range(depth-2,1,-1):
		c = DecodeDenseNetBlocks(c, concat_layers[i], filters[i], internal_depths[i], "DecodeBlock_lvl"+str(i))

	c = DecodeDenseNetBlocks(c,layers.concatenate([concat_layers[1], concat_layers[0]],axis=-1), filters[1], internal_depths[1], "DecodeBlock_lvl"+str(1))
	c = DecodeDenseNetBlocks(c, None, filters[1], internal_depths[0], "DecodeBlock_lvl"+str(0))

	c = layers.Conv2D(128, kernel_size=3, padding="same", name="pre_final_conv")(c)
	outputs = layers.Conv2D(classes, kernel_size=1, name="final_conv")(c)
	outputs = layers.Activation(activation="sigmoid", name="output")(outputs)
	model = tf.keras.Model(inputs=input_tensor, outputs=outputs, name="Quad_Unet")
	return model

#Encoder-decoder architecture unet-like using resnet as encoder for data fusion in middle fusion. Use of statistics learning network adapted to semantic segmentation. Use of the quadratic module layer.
def QuadDoubleUnet(input_shape = None, input_tensor = None, classes=None):

	internal_depths_S1 = [2,2,2,2,2]
	internal_depths_S2 = [2,2,2,2,2]
	num_filters_S1 = [20, 64, 88, 128, 256]
	num_filters_S2 = [32, 64, 88, 128, 256]
	depth = 5
	filters_S1 = []
	filters_S2 = []


	for i in range(depth-1):
		a = []
		for j in range(internal_depths_S1[i]):
			a.append(num_filters_S1[i])
		filters_S1.append(a)

	for i in range(depth-1):
		a = []
		for j in range(internal_depths_S2[i]):
			a.append(num_filters_S2[i])
		filters_S2.append(a)

	concat_layers1 = []
	#Encode for SAR (S1):
	c1 = customLayers.quadLayer(num_filters_S1[0], stride=3, kernelsize=3)
	c1 = c1(input_tensor[1])
	c1 = layers.Activation('relu', name="W1Quadlayer_lvl0_Activation0")(c1)
	for i in range(1, depth-1):
		c1, conc = EncodeResNetBlocks(c1, filters_S1[i] , internal_depths_S1[i], "W1EncodeBlock_lvl"+str(i))
		concat_layers1.append(conc)

	for i in range(internal_depths_S1[depth-1]):
		c1 = layers.Conv2D(num_filters_S1[depth-1], kernel_size=3, padding="same", name = "W1EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c1)
		c1 = layers.BatchNormalization(name="W1EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c1)
		c1 = layers.Activation('relu', name="W1EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c1)

	#Encode for Optical (S2):
	concat_layers2 = []
	c2,conc = EncodeResNetBlocks(input_tensor[0], filters_S2[0] , internal_depths_S2[0], "W2EncodeBlock_lvl0", pool_size=(3, 3), strides=(3, 3))
	concat_layers2.append(conc)
	for i in range(1, depth-1):
		c2, conc = EncodeResNetBlocks(c2, filters_S2[i] , internal_depths_S2[i], "W2EncodeBlock_lvl"+str(i))
		concat_layers2.append(conc)

	for i in range(internal_depths_S2[depth-1]):
		c2 = layers.Conv2D(num_filters_S2[depth-1], kernel_size=3, padding="same", name = "W2EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c2)
		c2 = layers.BatchNormalization(name="W2EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c2)
		c2 = layers.Activation('relu', name="W2EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c2)

	#Decode with feature fusion
	c = layers.concatenate([c1, c2], axis=-1)

	for i in range(depth-2,1,-1):
		d = layers.concatenate([concat_layers1[i-1], concat_layers2[i]], axis=-1)
		c = DecodeResNetBlocks(c, d, filters_S2[i], internal_depths_S2[i], "DecodeBlock_lvl"+str(i))

	c = DecodeResNetBlocks(c,layers.concatenate([concat_layers1[0],concat_layers2[1]],axis=-1), filters_S2[1], internal_depths_S2[1], "DecodeBlock_lvl"+str(1))
	c = DecodeResNetBlocks(c, concat_layers2[0], filters_S2[0], internal_depths_S2[0], "DecodeBlock_lvl"+str(0), kernel_size=3, strides=(3,3))

	outputs = layers.Conv2D(classes, kernel_size=1, name="final_conv")(c)
	outputs = layers.Activation(activation="softmax", name="output")(outputs)
	model = tf.keras.Model(inputs=input_tensor, outputs=outputs, name="Double-QuadUnet")
	return model


#Resolution Preserving network (RPN) - semantic segmentation model based on RPB (here called FPB) that does not use pooling layers.
def FPB_Network(input_shape = None, input_tensor = None, classes=None):

	filters = [16,32,64,64,64]
	dropout_rate = 0.3

	#Inicial SAR
	c1 =layers.Conv2D(16, kernel_size=3, padding="same", name = "FPB_Network_convInit_SAR")(input_tensor[1])
	c1 = layers.Dropout(dropout_rate)(c1)
	c1 = layers.BatchNormalization(name="FPB_Network_BN_SAR")(c1)
	c1 = layers.Activation('relu', name="FPB_Network_Activation_SAR")(c1)

	#Inicial MSI
	c2 =layers.Conv2D(16, kernel_size=3, padding="same", name = "FPB_Network_convInit_MSI")(input_tensor[0])
	c2 = layers.Dropout(dropout_rate)(c2)
	c2 = layers.BatchNormalization(name="FPB_Network_BN_MSI")(c2)
	c2 = layers.Activation('relu', name="FPB_Network_Activation_MSI")(c2)

	c = layers.concatenate([c1, c2], axis=-1)

	for i in range(len(filters)):
		c = FPB_Block(c, filters[i], dropout_rate, layers_name="FPB_Block_"+str(i))

	outputs = layers.Conv2D(classes, kernel_size=1, name="final_conv")(c)
	outputs = layers.Activation(activation="softmax", name="output")(outputs)
	model = tf.keras.Model(inputs = input_tensor, outputs=outputs, name="FPB_Network")
	return model


#encode-decode network with two outputs and resunet as backbone. Re-entrance of the output from the 30m resolution result as a feature for the 10m output.
def labelsuperResUnet (input_shape = None, input_tensor = None, classes=None):

	internal_depths = [2,2,2,2]
	num_filters = [32, 64, 64, 128]
	depth = 4
	filters = []

	for i in range(depth-1):
		a = []
		for j in range(internal_depths[i]):
			a.append(num_filters[i])
		filters.append(a)

	concat_layers1 = []
	concat_layers2 = []
	#Encode Part 1:
	c1,conc = EncodeResNetBlocks(input_tensor[0], filters[0] , internal_depths[0], "W1EncodeBlock_lvl0", pool_size=(3, 3), strides=(3, 3))
	concat_layers1.append(conc)
	for i in range(1, depth-1):
		c1, conc = EncodeResNetBlocks(c1, filters[i] , internal_depths[i], "W1EncodeBlock_lvl"+str(i))
		concat_layers1.append(conc)

	for i in range(internal_depths[depth-1]):
		c1 = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "W1EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c1)
		c1 = layers.BatchNormalization(name="W1EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c1)
		c1 = layers.Activation('relu', name="W1EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c1)

	#Encode Part 2:
	c2,conc = EncodeResNetBlocks(input_tensor[1], filters[0] , internal_depths[0], "W2EncodeBlock_lvl0",pool_size=(3, 3), strides=(3, 3))
	concat_layers2.append(conc)
	for i in range(1, depth-1):
		c2, conc = EncodeResNetBlocks(c2, filters[i] , internal_depths[i], "W2EncodeBlock_lvl"+str(i))
		concat_layers2.append(conc)

	for i in range(internal_depths[depth-1]):
		c2 = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c2)
		c2 = layers.BatchNormalization(name="W2EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c2)
		c2 = layers.Activation('relu', name="W2EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c2)

	#Decode with feature fusion
	c = layers.concatenate([c1, c2], axis=-1)

	for i in range(depth-2,0,-1):
		d = layers.concatenate([concat_layers1[i], concat_layers2[i]], axis=-1)
		c = DecodeResNetBlocks(c, d, filters[i], internal_depths[i], "DecodeBlock_lvl"+str(i))

	outputs1 = layers.Conv2D(classes, kernel_size=1, name="final_conv_30m")(c)
	c_int = layers.Activation(activation="sigmoid", name="output_sigmoid")(outputs1)
	outputs1 = layers.Activation(activation="softmax", name="output30")(outputs1)
	upOut1 = layers.UpSampling2D(size=(3,3), name = "Output1_Up")(c_int)

	d = layers.concatenate([concat_layers1[0], concat_layers2[0]], axis=-1)
	c = DecodeResNetBlocks(c, d, filters[0], internal_depths[0], "DecodeBlock_lvl"+str(0), kernel_size=3, strides=(3,3))

	c = layers.concatenate([c, upOut1], axis=-1) #change in the re-entering of the class info from 30m
	outputs2 = layers.Conv2D(classes, kernel_size=1, name="final_conv_10m")(c)
	outputs2 = layers.Activation(activation="softmax", name="output10")(outputs2)
	model = tf.keras.Model(inputs=input_tensor, outputs=(outputs1, outputs2), name="u-net_double")

	return model


#encode-decode network with two outputs and resunet as backbone. No re-entrance.
def labelsuperResUnet_pre (input_shape = None, input_tensor = None, classes=None):
	internal_depths = [2,2,2,2]
	num_filters = [32, 64, 64, 128]
	depth = 4
	filters = []

	for i in range(depth-1):
		a = []
		for j in range(internal_depths[i]):
			a.append(num_filters[i])
		filters.append(a)

	concat_layers1 = []
	concat_layers2 = []
	#Encode Part 1:
	c1,conc = EncodeResNetBlocks(input_tensor[0], filters[0] , internal_depths[0], "W1EncodeBlock_lvl0", pool_size=(3, 3), strides=(3, 3))
	concat_layers1.append(conc)
	for i in range(1, depth-1):
		c1, conc = EncodeResNetBlocks(c1, filters[i] , internal_depths[i], "W1EncodeBlock_lvl"+str(i))
		concat_layers1.append(conc)

	for i in range(internal_depths[depth-1]):
		c1 = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "W1EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c1)
		c1 = layers.BatchNormalization(name="W1EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c1)
		c1 = layers.Activation('relu', name="W1EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c1)

	#Encode Part 2:
	c2,conc = EncodeResNetBlocks(input_tensor[1], filters[0] , internal_depths[0], "W2EncodeBlock_lvl0",pool_size=(3, 3), strides=(3, 3))
	concat_layers2.append(conc)
	for i in range(1, depth-1):
		c2, conc = EncodeResNetBlocks(c2, filters[i] , internal_depths[i], "W2EncodeBlock_lvl"+str(i))
		concat_layers2.append(conc)

	for i in range(internal_depths[depth-1]):
		c2 = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c2)
		c2 = layers.BatchNormalization(name="W2EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c2)
		c2 = layers.Activation('relu', name="W2EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c2)

	#Decode with feature fusion
	c = layers.concatenate([c1, c2], axis=-1)

	for i in range(depth-2,0,-1):
		d = layers.concatenate([concat_layers1[i], concat_layers2[i]], axis=-1)
		c = DecodeResNetBlocks(c, d, filters[i], internal_depths[i], "DecodeBlock_lvl"+str(i))

	outputs1 = layers.Conv2D(classes, kernel_size=1, name="final_conv_30m")(c)
	outputs1 = layers.Activation(activation="softmax", name="output30")(outputs1)

	d = layers.concatenate([concat_layers1[0], concat_layers2[0]], axis=-1)
	c = DecodeResNetBlocks(c, d, filters[0], internal_depths[0], "DecodeBlock_lvl"+str(0), kernel_size=3, strides=(3,3))

	outputs2 = layers.Conv2D(classes, kernel_size=1, name="final_conv_10m")(c)
	outputs2 = layers.Activation(activation="softmax", name="output10")(outputs2)
	model = tf.keras.Model(inputs=input_tensor, outputs=(outputs1, outputs2), name="u-net_double")

	return model


#encode-decode network with two outputs and resunet as backbone. Re-entrance of the output from the 30m resolution result as a feature for the 10m output. Middle fusion data fusion applied.
def labelsuperResUnet_middle_mod (input_shape = None, input_tensor = None, classes=None):
	internal_depths = [2,2,2,2]
	num_filters = [32, 64, 64, 128]
	depth = 4
	filters = []

	for i in range(depth-1):
		a = []
		for j in range(internal_depths[i]):
			a.append(num_filters[i])
		filters.append(a)

	concat_layers1 = []
	concat_layers2 = []
	#Encode Part 1:
	c1,conc = EncodeResNetBlocks(input_tensor[0], filters[0] , internal_depths[0], "W1EncodeBlock_lvl0", pool_size=(3, 3), strides=(3, 3))
	concat_layers1.append(conc)
	for i in range(1, depth-1):
		c1, conc = EncodeResNetBlocks(c1, filters[i] , internal_depths[i], "W1EncodeBlock_lvl"+str(i))
		concat_layers1.append(conc)

	for i in range(internal_depths[depth-1]):
		c1 = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "W1EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c1)
		c1 = layers.BatchNormalization(name="W1EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c1)
		c1 = layers.Activation('relu', name="W1EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c1)

	#Encode Part 2:
	c2,conc = EncodeResNetBlocks(input_tensor[1], filters[0] , internal_depths[0], "W2EncodeBlock_lvl0",pool_size=(3, 3), strides=(3, 3))
	concat_layers2.append(conc)
	for i in range(1, depth-1):
		c2, conc = EncodeResNetBlocks(c2, filters[i] , internal_depths[i], "W2EncodeBlock_lvl"+str(i))
		concat_layers2.append(conc)

	for i in range(internal_depths[depth-1]):
		c2 = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c2)
		c2 = layers.BatchNormalization(name="W2EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c2)
		c2 = layers.Activation('relu', name="W2EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c2)

	#Decode with feature fusion
	c = layers.concatenate([c1, c2], axis=-1)

	for i in range(depth-2,0,-1):
		d = layers.concatenate([concat_layers1[i], concat_layers2[i]], axis=-1)
		c = DecodeResNetBlocks(c, d, filters[i], internal_depths[i], "DecodeBlock_lvl"+str(i))

	outputs1 = layers.Conv2D(classes, kernel_size=1, name="final_conv_30m")(c)
	x = layers.Activation(activation="sigmoid", name="output30_sig")(outputs1)
	outputs1 = layers.Activation(activation="softmax", name="output30")(outputs1)

	d = layers.concatenate([concat_layers1[0], concat_layers2[0]], axis=-1)
	c = DecodeResNetBlocks(c, d, filters[0], internal_depths[0], "DecodeBlock_lvl"+str(0), kernel_size=3, strides=(3,3))
	x = layers.UpSampling2D(size=(3,3), name = "Output1_Up")(x)
	c = layers.concatenate([c, x], axis=-1)
	outputs2 = layers.Conv2D(classes, kernel_size=1, name="final_conv_10m")(c)
	outputs2 = layers.Activation(activation="softmax", name="output10")(outputs2)
	model = tf.keras.Model(inputs=input_tensor, outputs=(outputs1, outputs2), name="u-net_double")

	return model


#encode-decode network with two outputs and resunet as backbone. Early fusion without feature adaptation module data fusion architecture.
def labelsuperResUnet_early1(input_shape = None, input_tensor = None, classes=None):
	internal_depths = [2,2,2,2]
	num_filters = [32, 64, 64, 128]
	depth = 4
	filters = []


	c = layers.concatenate([input_tensor[0], input_tensor[1]], axis=-1)
	c = layers.Conv2D(64, kernel_size=1, name="data_fusion")(c)
	c = layers.BatchNormalization(name='data_fusion_BN')(c)
	c = layers.Activation('relu', name='data_fusion_Activation')(c)

	for i in range(depth-1):
		a = []
		for j in range(internal_depths[i]):
			a.append(num_filters[i])
		filters.append(a)

	concat_layers = []

	#Encode Part:
	c,conc = EncodeResNetBlocks(c, filters[0] , internal_depths[0], "EncodeBlock_lvl0", pool_size=(3, 3), strides=(3, 3))
	concat_layers.append(conc)
	for i in range(1, depth-1):
		c, conc = EncodeResNetBlocks(c, filters[i] , internal_depths[i], "EncodeBlock_lvl"+str(i))
		concat_layers.append(conc)

	for i in range(internal_depths[depth-1]):
		c = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c)
		c = layers.BatchNormalization(name="EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c)
		c = layers.Activation('relu', name="EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c)

	for i in range(depth-2,0,-1):
		c = DecodeResNetBlocks(c, concat_layers[i], filters[i], internal_depths[i], "DecodeBlock_lvl"+str(i))

	outputs1 = layers.Conv2D(classes, kernel_size=1, name="final_conv_30m")(c)
	outputs1 = layers.Activation(activation="softmax", name="output30")(outputs1)
	c = DecodeResNetBlocks(c, concat_layers[0], filters[0], internal_depths[0], "DecodeBlock_lvl"+str(0), kernel_size=3, strides=(3,3))
	outputs2 = layers.Conv2D(classes, kernel_size=1, name="final_conv")(c)
	outputs2 = layers.Activation(activation="softmax", name="output10")(outputs2)
	model = tf.keras.Model(inputs=input_tensor, outputs=[outputs1, outputs2], name="u-net")

	return model

#encode-decode network with two outputs and resunet as backbone. Re-entrance of the output from the 30m resolution result as a feature for the 10m output. Early fusion without feature adaptation module data fusion architecture.
def labelsuperResUnet_early1_mod(input_shape = None, input_tensor = None, classes=None):
	internal_depths = [2,2,2,2]
	num_filters = [32, 64, 64, 128]
	depth = 4
	filters = []


	c = layers.concatenate([input_tensor[0], input_tensor[1]], axis=-1)
	c = layers.Conv2D(64, kernel_size=1, name="data_fusion")(c)
	c = layers.BatchNormalization(name='data_fusion_BN')(c)
	c = layers.Activation('relu', name='data_fusion_Activation')(c)

	for i in range(depth-1):
		a = []
		for j in range(internal_depths[i]):
			a.append(num_filters[i])
		filters.append(a)

	concat_layers = []

	#Encode Part:
	c,conc = EncodeResNetBlocks(c, filters[0] , internal_depths[0], "EncodeBlock_lvl0", pool_size=(3, 3), strides=(3, 3))
	concat_layers.append(conc)
	for i in range(1, depth-1):
		c, conc = EncodeResNetBlocks(c, filters[i] , internal_depths[i], "EncodeBlock_lvl"+str(i))
		concat_layers.append(conc)

	for i in range(internal_depths[depth-1]):
		c = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c)
		c = layers.BatchNormalization(name="EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c)
		c = layers.Activation('relu', name="EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c)

	for i in range(depth-2,0,-1):
		c = DecodeResNetBlocks(c, concat_layers[i], filters[i], internal_depths[i], "DecodeBlock_lvl"+str(i))

	outputs1 = layers.Conv2D(classes, kernel_size=1, name="final_conv_30m")(c)
	x = layers.Activation(activation="sigmoid", name="Sig_output30")(outputs1)
	outputs1 = layers.Activation(activation="softmax", name="output30")(outputs1)
	c = DecodeResNetBlocks(c, concat_layers[0], filters[0], internal_depths[0], "DecodeBlock_lvl"+str(0), kernel_size=3, strides=(3,3))
	x = layers.UpSampling2D(size=(3,3), name = "Output1_Up")(x)
	c = layers.concatenate([c, x], axis=-1)
	outputs2 = layers.Conv2D(classes, kernel_size=1, name="final_conv")(c)
	outputs2 = layers.Activation(activation="softmax", name="output10")(outputs2)
	model = tf.keras.Model(inputs=input_tensor, outputs=[outputs1, outputs2], name="u-net")

	return model

#encode-decode network with two outputs and resunet as backbone. Early fusion with feature adaptation module data fusion architecture.
def labelsuperResUnet_early2(input_shape = None, input_tensor = None, classes=None):
	internal_depths = [2,2,2,2]
	num_filters = [32, 64, 64, 128]
	depth = 4
	filters = []

	c1 = layers.Conv2D(64, kernel_size=3, padding="same", name = "Input1_conv0")(input_tensor[0])
	c1 = layers.BatchNormalization(name='Input1_BN')(c1)
	c1 = layers.Activation('relu', name='Input1_Activation')(c1)
	c2 = layers.Conv2D(64, kernel_size=3, padding="same", name = "Input2_conv0")(input_tensor[1])
	c2 = layers.BatchNormalization(name='Input2_BN')(c2)
	c2 = layers.Activation('relu', name='Input2_Activation')(c2)

	c = layers.concatenate([c1, c2], axis=-1)
	c = layers.Conv2D(64, kernel_size=1, name="data_fusion")(c)
	c = layers.BatchNormalization(name='data_fusion_BN')(c)
	c = layers.Activation('relu', name='data_fusion_Activation')(c)

	for i in range(depth-1):
		a = []
		for j in range(internal_depths[i]):
			a.append(num_filters[i])
		filters.append(a)

	concat_layers = []

	#Encode Part:
	c,conc = EncodeResNetBlocks(c, filters[0] , internal_depths[0], "EncodeBlock_lvl0", pool_size=(3, 3), strides=(3, 3))
	concat_layers.append(conc)
	for i in range(1, depth-1):
		c, conc = EncodeResNetBlocks(c, filters[i] , internal_depths[i], "EncodeBlock_lvl"+str(i))
		concat_layers.append(conc)

	for i in range(internal_depths[depth-1]):
		c = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c)
		c = layers.BatchNormalization(name="EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c)
		c = layers.Activation('relu', name="EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c)

	for i in range(depth-2,0,-1):
		c = DecodeResNetBlocks(c, concat_layers[i], filters[i], internal_depths[i], "DecodeBlock_lvl"+str(i))

	outputs1 = layers.Conv2D(classes, kernel_size=1, name="final_conv_30m")(c)
	outputs1 = layers.Activation(activation="softmax", name="output30")(outputs1)
	c = DecodeResNetBlocks(c, concat_layers[0], filters[0], internal_depths[0], "DecodeBlock_lvl"+str(0), kernel_size=3, strides=(3,3))
	outputs2 = layers.Conv2D(classes, kernel_size=1, name="final_conv")(c)
	outputs2 = layers.Activation(activation="softmax", name="output10")(outputs2)
	model = tf.keras.Model(inputs=input_tensor, outputs=[outputs1, outputs2], name="u-net")

	return model


#encode-decode network with one output and resunet as backbone. Early fusion with feature adaptation module data fusion architecture.
def ResUnet_early2_1out(input_shape = None, input_tensor = None, classes=None):
	internal_depths = [2,2,2,2]
	num_filters = [32, 64, 64, 128]
	depth = 4
	filters = []

	c1 = layers.Conv2D(64, kernel_size=3, padding="same", name = "Input1_conv0")(input_tensor[0])
	c1 = layers.BatchNormalization(name='Input1_BN')(c1)
	c1 = layers.Activation('relu', name='Input1_Activation')(c1)
	c2 = layers.Conv2D(64, kernel_size=3, padding="same", name = "Input2_conv0")(input_tensor[1])
	c2 = layers.BatchNormalization(name='Input2_BN')(c2)
	c2 = layers.Activation('relu', name='Input2_Activation')(c2)

	c = layers.concatenate([c1, c2], axis=-1)
	c = layers.Conv2D(64, kernel_size=1, name="data_fusion")(c)
	c = layers.BatchNormalization(name='data_fusion_BN')(c)
	c = layers.Activation('relu', name='data_fusion_Activation')(c)

	for i in range(depth-1):
		a = []
		for j in range(internal_depths[i]):
			a.append(num_filters[i])
		filters.append(a)

	concat_layers = []

	#Encode Part:
	c,conc = EncodeResNetBlocks(c, filters[0] , internal_depths[0], "EncodeBlock_lvl0", pool_size=(3, 3), strides=(3, 3))
	concat_layers.append(conc)
	for i in range(1, depth-1):
		c, conc = EncodeResNetBlocks(c, filters[i] , internal_depths[i], "EncodeBlock_lvl"+str(i))
		concat_layers.append(conc)

	for i in range(internal_depths[depth-1]):
		c = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c)
		c = layers.Dropout(0.3)(c)
		c = layers.BatchNormalization(name="EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c)
		c = layers.Activation('relu', name="EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c)

	for i in range(depth-2,0,-1):
		c = DecodeResNetBlocks(c, concat_layers[i], filters[i], internal_depths[i], "DecodeBlock_lvl"+str(i))

	c = DecodeResNetBlocks(c, concat_layers[0], filters[0], internal_depths[0], "DecodeBlock_lvl"+str(0), kernel_size=3, strides=(3,3))

	outputs2 = layers.Conv2D(classes, kernel_size=1, name="final_conv")(c)
	outputs2 = layers.Activation(activation="softmax", name="output10")(outputs2)
	model = tf.keras.Model(inputs=input_tensor, outputs=outputs2, name="Resu-net-1out")
	return model

#encode-decode network with two outputs and resunet as backbone. Re-entrance of the output from the 30m resolution result as a feature for the 10m output. Early fusion with feature adaptation module data fusion architecture.
def labelsuperResUnet_early2_mod(input_shape = None, input_tensor = None, classes=None):
	internal_depths = [2,2,2,2]
	num_filters = [32, 64, 64, 128]
	depth = 4
	filters = []

	c1 = layers.Conv2D(64, kernel_size=3, padding="same", name = "Input1_conv0")(input_tensor[0])
	c1 = layers.BatchNormalization(name='Input1_BN')(c1)
	c1 = layers.Activation('relu', name='Input1_Activation')(c1)
	c2 = layers.Conv2D(64, kernel_size=3, padding="same", name = "Input2_conv0")(input_tensor[1])
	c2 = layers.BatchNormalization(name='Input2_BN')(c2)
	c2 = layers.Activation('relu', name='Input2_Activation')(c2)

	c = layers.concatenate([c1, c2], axis=-1)
	c = layers.Conv2D(64, kernel_size=1, name="data_fusion")(c)
	c = layers.BatchNormalization(name='data_fusion_BN')(c)
	c = layers.Activation('relu', name='data_fusion_Activation')(c)

	for i in range(depth-1):
		a = []
		for j in range(internal_depths[i]):
			a.append(num_filters[i])
		filters.append(a)

	concat_layers = []

	#Encode Part:
	c,conc = EncodeResNetBlocks(c, filters[0] , internal_depths[0], "EncodeBlock_lvl0", pool_size=(3, 3), strides=(3, 3))
	concat_layers.append(conc)
	for i in range(1, depth-1):
		c, conc = EncodeResNetBlocks(c, filters[i] , internal_depths[i], "EncodeBlock_lvl"+str(i))
		concat_layers.append(conc)

	for i in range(internal_depths[depth-1]):
		c = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c)
		c = layers.Dropout(0.3)(c)
		c = layers.BatchNormalization(name="EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c)
		c = layers.Activation('relu', name="EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c)

	for i in range(depth-2,0,-1):
		c = DecodeResNetBlocks(c, concat_layers[i], filters[i], internal_depths[i], "DecodeBlock_lvl"+str(i))

	outputs1 = layers.Conv2D(classes, kernel_size=1, name="final_conv_30m")(c)

	c_int = layers.Activation(activation="sigmoid", name="output_sigmoid")(outputs1)
	upOut1 = layers.UpSampling2D(size=(3,3), name = "Output1_Up")(c_int)
	outputs1 = layers.Activation(activation="softmax", name="output30")(outputs1)

	c = DecodeResNetBlocks(c, concat_layers[0], filters[0], internal_depths[0], "DecodeBlock_lvl"+str(0), kernel_size=3, strides=(3,3))

	c = layers.concatenate([c, upOut1], axis=-1) #change in the re-entering of the class info from 30m
	outputs2 = layers.Conv2D(classes, kernel_size=1, name="final_conv")(c)
	outputs2 = layers.Activation(activation="softmax", name="output10")(outputs2)
	model = tf.keras.Model(inputs=input_tensor, outputs=[outputs1, outputs2], name="u-net")
	return model


#encode-decode network with two outputs and VGG as backbone. Re-entrance of the output from the 30m resolution result as a feature for the 10m output. Early fusion with feature adaptation module data fusion architecture.
def labelsuperVGGUnet_early2_mod(input_shape = None, input_tensor = None, classes=None):
	internal_depths = [2,2,2,2]
	num_filters = [32, 64, 64, 128]
	depth = 4
	filters = []

	c1 = layers.Conv2D(64, kernel_size=3, padding="same", name = "Input1_conv0")(input_tensor[0])
	c1 = layers.BatchNormalization(name='Input1_BN')(c1)
	c1 = layers.Activation('relu', name='Input1_Activation')(c1)
	c2 = layers.Conv2D(64, kernel_size=3, padding="same", name = "Input2_conv0")(input_tensor[1])
	c2 = layers.BatchNormalization(name='Input2_BN')(c2)
	c2 = layers.Activation('relu', name='Input2_Activation')(c2)

	c = layers.concatenate([c1, c2], axis=-1)
	c = layers.Conv2D(64, kernel_size=1, name="data_fusion")(c)
	c = layers.BatchNormalization(name='data_fusion_BN')(c)
	c = layers.Activation('relu', name='data_fusion_Activation')(c)

	for i in range(depth-1):
		a = []
		for j in range(internal_depths[i]):
			a.append(num_filters[i])
		filters.append(a)

	concat_layers = []

	#Encode Part:
	c,conc = EncodeVGGBlocks(c, filters[0] , internal_depths[0], "EncodeBlock_lvl0", pool_size=(3, 3), strides=(3, 3))
	concat_layers.append(conc)
	for i in range(1, depth-1):
		c, conc = EncodeVGGBlocks(c, filters[i] , internal_depths[i], "EncodeBlock_lvl"+str(i))
		concat_layers.append(conc)

	for i in range(internal_depths[depth-1]):
		c = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c)
		c = layers.Dropout(0.3)(c)
		c = layers.BatchNormalization(name="EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c)
		c = layers.Activation('relu', name="EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c)

	for i in range(depth-2,0,-1):
		c = DecodeVGGBlocks(c, concat_layers[i], filters[i], internal_depths[i], "DecodeBlock_lvl"+str(i))

	outputs1 = layers.Conv2D(classes, kernel_size=1, name="final_conv_30m")(c)

	c_int = layers.Activation(activation="sigmoid", name="output_sigmoid")(outputs1)
	upOut1 = layers.UpSampling2D(size=(3,3), name = "Output1_Up")(c_int)
	outputs1 = layers.Activation(activation="softmax", name="output30")(outputs1)

	c = DecodeVGGBlocks(c, concat_layers[0], filters[0], internal_depths[0], "DecodeBlock_lvl"+str(0), kernel_size=3, strides=(3,3))

	c = layers.concatenate([c, upOut1], axis=-1) #change in the re-entering of the class info from 30m
	outputs2 = layers.Conv2D(classes, kernel_size=1, name="final_conv")(c)
	outputs2 = layers.Activation(activation="softmax", name="output10")(outputs2)
	model = tf.keras.Model(inputs=input_tensor, outputs=[outputs1, outputs2], name="u-net")

	return model

#encode-decode network with two outputs and RPB as backbone. Re-entrance of the output from the 30m resolution result as a feature for the 10m output. Late fusion data fusion architecture.
def labelsuperFPB_late_mod(input_shape = None, input_tensor = None, classes=None):

	internal_depths = [2,2,2,2]
	num_filters = [32, 64, 64, 128]
	depth = 4
	filters = []


	for i in range(depth-1):
		a = []
		for j in range(internal_depths[i]):
			a.append(num_filters[i])
		filters.append(a)

	#S2 Branch
	concat_layers1 = []

	#Encode Part:
	c1,conc = EncodeFPB_Block(input_tensor[0], filters[0] , internal_depths[0], "W1EncodeBlock_lvl0", pool_size=(3, 3), strides=(3, 3))
	concat_layers1.append(conc)
	for i in range(1, depth-1):
		c1, conc = EncodeFPB_Block(c1, filters[i] , internal_depths[i], "W1EncodeBlock_lvl"+str(i))
		concat_layers1.append(conc)

	for i in range(internal_depths[depth-1]):
		c1 = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "W1EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c1)
		c1 = layers.BatchNormalization(name="W1EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c1)
		c1 = layers.Activation('relu', name="W1EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c1)

	for i in range(depth-2,0,-1):
		c1 = DecodeFPB_Block(c1, concat_layers1[i], filters[i], internal_depths[i], "W1DecodeBlock_lvl"+str(i))

	#S1 Branch
	concat_layers2 = []

	#Encode Part:
	c2,conc = EncodeFPB_Block(input_tensor[1], filters[0] , internal_depths[0], "W2EncodeBlock_lvl0", pool_size=(3, 3), strides=(3, 3))
	concat_layers2.append(conc)
	for i in range(1, depth-1):
		c2, conc = EncodeFPB_Block(c2, filters[i] , internal_depths[i], "W2EncodeBlock_lvl"+str(i))
		concat_layers2.append(conc)

	for i in range(internal_depths[depth-1]):
		c2 = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "W2EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c2)
		c2 = layers.BatchNormalization(name="W2EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c2)
		c2 = layers.Activation('relu', name="W2EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c2)

	for i in range(depth-2,0,-1):
		c2 = DecodeFPB_Block(c2, concat_layers2[i], filters[i], internal_depths[i], "W2DecodeBlock_lvl"+str(i))

	c = layers.concatenate([c1, c2], axis=-1)
	outputs1 = layers.Conv2D(classes, kernel_size=1, name="final_conv_30m")(c)
	x = layers.Activation(activation="sigmoid", name="output30_sigmoid")(outputs1)
	x = layers.UpSampling2D(size=(3,3), name = "Output1_Up")(x)
	outputs1 = layers.Activation(activation="softmax", name="output30")(outputs1)
	c1 = DecodeFPB_Block(c1, concat_layers1[0], filters[0], internal_depths[0], "W1DecodeBlock_lvl"+str(0), kernel_size=3, strides=(3,3))
	c2 = DecodeFPB_Block(c2, concat_layers2[0], filters[0], internal_depths[0], "W2DecodeBlock_lvl"+str(0), kernel_size=3, strides=(3,3))
	c = layers.concatenate([c1, c2, x], axis=-1)
	outputs2 = layers.Conv2D(classes, kernel_size=1, name="final_conv")(c)
	outputs2 = layers.Activation(activation="softmax", name="output10")(outputs2)
	model = tf.keras.Model(inputs=input_tensor, outputs=[outputs1, outputs2], name="u-net")

	return model


#encode-decode network with two outputs and RPB as backbone. Re-entrance of the output from the 30m resolution result as a feature for the 10m output. Middle fusion data fusion architecture.
def labelsuperFPB_middle_mod (input_shape = None, input_tensor = None, classes=None):

	internal_depths = [2,2,2,2]
	num_filters = [32, 64, 64, 128]
	depth = 4
	filters = []

	for i in range(depth-1):
		a = []
		for j in range(internal_depths[i]):
			a.append(num_filters[i])
		filters.append(a)

	concat_layers1 = []
	concat_layers2 = []
	#Encode Part 1:
	c1,conc = EncodeFPB_Block(input_tensor[0], filters[0] , internal_depths[0], "W1EncodeBlock_lvl0", pool_size=(3, 3), strides=(3, 3))
	concat_layers1.append(conc)
	for i in range(1, depth-1):
		c1, conc = EncodeFPB_Block(c1, filters[i] , internal_depths[i], "W1EncodeBlock_lvl"+str(i))
		concat_layers1.append(conc)

	for i in range(internal_depths[depth-1]):
		c1 = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "W1EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c1)
		c1 = layers.BatchNormalization(name="W1EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c1)
		c1 = layers.Activation('relu', name="W1EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c1)

	#Encode Part 2:
	c2,conc = EncodeFPB_Block(input_tensor[1], filters[0] , internal_depths[0], "W2EncodeBlock_lvl0",pool_size=(3, 3), strides=(3, 3))
	concat_layers2.append(conc)
	for i in range(1, depth-1):
		c2, conc = EncodeFPB_Block(c2, filters[i] , internal_depths[i], "W2EncodeBlock_lvl"+str(i))
		concat_layers2.append(conc)

	for i in range(internal_depths[depth-1]):
		c2 = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c2)
		c2 = layers.BatchNormalization(name="W2EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c2)
		c2 = layers.Activation('relu', name="W2EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c2)

	#Decode with feature fusion
	c = layers.concatenate([c1, c2], axis=-1)
	c = FPB_Block(c, 128, 0.3, 'FPB_Block_Lastlevel')

	for i in range(depth-2,0,-1):
		d = layers.concatenate([concat_layers1[i], concat_layers2[i]], axis=-1)
		c = DecodeFPB_Block(c, d, filters[i], internal_depths[i], "DecodeBlock_lvl"+str(i))

	outputs1 = layers.Conv2D(classes, kernel_size=1, name="final_conv_30m")(c)
	x = layers.Activation(activation="sigmoid", name="output30_sigmoid")(outputs1)
	outputs1 = layers.Activation(activation="softmax", name="output30")(outputs1)

	d = layers.concatenate([concat_layers1[0], concat_layers2[0]], axis=-1)
	c = DecodeFPB_Block(c, d, filters[0], internal_depths[0], "DecodeBlock_lvl"+str(0), kernel_size=3, strides=(3,3))
	x = layers.UpSampling2D(size=(3,3), name = "Output1_Up")(x)
	c = layers.concatenate([c, x], axis=-1)
	outputs2 = layers.Conv2D(classes, kernel_size=1, name="final_conv_10m")(c)
	outputs2 = layers.Activation(activation="softmax", name="output10")(outputs2)
	model = tf.keras.Model(inputs=input_tensor, outputs=(outputs1, outputs2), name="u-net_double")

	return model


#encode-decode network with two outputs and RPB as backbone. Re-entrance of the output from the 30m resolution result as a feature for the 10m output. Early fusion without feature adaptation module data fusion architecture.
def labelsuperFPB_early1_mod(input_shape = None, input_tensor = None, classes=None):
	internal_depths = [2,2,2,2]
	num_filters = [32, 64, 64, 128]
	depth = 4
	filters = []

	c = layers.concatenate([input_tensor[0], input_tensor[1]], axis=-1)
	c = layers.Conv2D(64, kernel_size=1, name="data_fusion")(c)
	c = layers.BatchNormalization(name='data_fusion_BN')(c)
	c = layers.Activation('relu', name='data_fusion_Activation')(c)

	for i in range(depth-1):
		a = []
		for j in range(internal_depths[i]):
			a.append(num_filters[i])
		filters.append(a)

	concat_layers = []

	#Encode Part:
	c,conc = EncodeFPB_Block(c, filters[0] , internal_depths[0], "EncodeBlock_lvl0", pool_size=(3, 3), strides=(3, 3))
	concat_layers.append(conc)
	for i in range(1, depth-1):
		c, conc = EncodeFPB_Block(c, filters[i] , internal_depths[i], "EncodeBlock_lvl"+str(i))
		concat_layers.append(conc)

	for i in range(internal_depths[depth-1]):
		c = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c)
		c = layers.Dropout(0.3)(c)
		c = layers.BatchNormalization(name="EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c)
		c = layers.Activation('relu', name="EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c)

	for i in range(depth-2,0,-1):
		c = DecodeFPB_Block(c, concat_layers[i], filters[i], internal_depths[i], "DecodeBlock_lvl"+str(i))

	outputs1 = layers.Conv2D(classes, kernel_size=1, name="final_conv_30m")(c)

	c_int = layers.Activation(activation="sigmoid", name="output_sigmoid")(outputs1)
	upOut1 = layers.UpSampling2D(size=(3,3), name = "Output1_Up")(c_int)
	outputs1 = layers.Activation(activation="softmax", name="output30")(outputs1)

	c = DecodeFPB_Block(c, concat_layers[0], filters[0], internal_depths[0], "DecodeBlock_lvl"+str(0), kernel_size=3, strides=(3,3))

	c = layers.concatenate([c, upOut1], axis=-1) #change in the re-entering of the class info from 30m
	outputs2 = layers.Conv2D(classes, kernel_size=1, name="final_conv")(c)
	outputs2 = layers.Activation(activation="softmax", name="output10")(outputs2)
	model = tf.keras.Model(inputs=input_tensor, outputs=[outputs1, outputs2], name="u-net")
	return model

#encode-decode network with two outputs and RPB as backbone. Re-entrance of the output from the 30m resolution result as a feature for the 10m output. Early fusion with feature adaptation module data fusion architecture.
#THIS IS MY FINAL ARCHITECTURE USED!!
def labelsuperFPB_early2_mod(input_shape = None, input_tensor = None, classes=None):
	internal_depths = [2,2,2,2]
	num_filters = [32, 64, 64, 128]
	depth = 4
	filters = []

	c1 = layers.Conv2D(32, kernel_size=3, padding="same", name = "Input1_conv0")(input_tensor[0])
	c1 = layers.BatchNormalization(name='Input1_BN')(c1)
	c1 = layers.Activation('relu', name='Input1_Activation')(c1)
	c2 = layers.Conv2D(32, kernel_size=3, padding="same", name = "Input2_conv0")(input_tensor[1])
	c2 = layers.BatchNormalization(name='Input2_BN')(c2)
	c2 = layers.Activation('relu', name='Input2_Activation')(c2)

	c = layers.concatenate([c1, c2], axis=-1)
	c = layers.Conv2D(64, kernel_size=1, name="data_fusion")(c)
	c = layers.BatchNormalization(name='data_fusion_BN')(c)
	c = layers.Activation('relu', name='data_fusion_Activation')(c)

	for i in range(depth-1):
		a = []
		for j in range(internal_depths[i]):
			a.append(num_filters[i])
		filters.append(a)

	concat_layers = []

	#Encode Part:
	c,conc = EncodeFPB_Block(c, filters[0] , internal_depths[0], "EncodeBlock_lvl0", pool_size=(3, 3), strides=(3, 3))
	concat_layers.append(conc)
	for i in range(1, depth-1):
		c, conc = EncodeFPB_Block(c, filters[i] , internal_depths[i], "EncodeBlock_lvl"+str(i))
		concat_layers.append(conc)

	for i in range(internal_depths[depth-1]):
		c = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c)
		c = layers.Dropout(0.3)(c)
		c = layers.BatchNormalization(name="EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c)
		c = layers.Activation('relu', name="EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c)

	for i in range(depth-2,0,-1):
		c = DecodeFPB_Block(c, concat_layers[i], filters[i], internal_depths[i], "DecodeBlock_lvl"+str(i))

	outputs1 = layers.Conv2D(classes, kernel_size=1, name="final_conv_30m")(c)

	c_int = layers.Activation(activation="sigmoid", name="output_sigmoid")(outputs1)
	upOut1 = layers.UpSampling2D(size=(3,3), name = "Output1_Up")(c_int)
	outputs1 = layers.Activation(activation="softmax", name="output30")(outputs1)

	c = DecodeFPB_Block(c, concat_layers[0], filters[0], internal_depths[0], "DecodeBlock_lvl"+str(0), kernel_size=3, strides=(3,3))

	c = layers.concatenate([c, upOut1], axis=-1) #change in the re-entering of the class info from 30m
	outputs2 = layers.Conv2D(classes, kernel_size=1, name="final_conv")(c)
	outputs2 = layers.Activation(activation="softmax", name="output10")(outputs2)
	model = tf.keras.Model(inputs=input_tensor, outputs=[outputs1, outputs2], name="u-net")
	return model

#encode-decode network with two outputs and RPB as backbone. Re-entrance of the output from the 30m resolution result as a feature for the 10m output. Not a fusion model.
def labelsuperFPB_1input_mod(input_shape = None, input_tensor = None, classes=None):
	internal_depths = [2,2,2,2]
	num_filters = [32, 64, 64, 128]
	depth = 4
	filters = []

	for i in range(depth-1):
		a = []
		for j in range(internal_depths[i]):
			a.append(num_filters[i])
		filters.append(a)

	concat_layers = []

	#Encode Part:
	c,conc = EncodeFPB_Block(input_tensor, filters[0] , internal_depths[0], "EncodeBlock_lvl0", pool_size=(3, 3), strides=(3, 3))
	concat_layers.append(conc)
	for i in range(1, depth-1):
		c, conc = EncodeFPB_Block(c, filters[i] , internal_depths[i], "EncodeBlock_lvl"+str(i))
		concat_layers.append(conc)

	for i in range(internal_depths[depth-1]):
		c = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c)
		c = layers.Dropout(0.3)(c)
		c = layers.BatchNormalization(name="EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c)
		c = layers.Activation('relu', name="EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c)

	for i in range(depth-2,0,-1):
		c = DecodeFPB_Block(c, concat_layers[i], filters[i], internal_depths[i], "DecodeBlock_lvl"+str(i))

	outputs1 = layers.Conv2D(classes, kernel_size=1, name="final_conv_30m")(c)

	c_int = layers.Activation(activation="sigmoid", name="output_sigmoid")(outputs1)
	upOut1 = layers.UpSampling2D(size=(3,3), name = "Output1_Up")(c_int)
	outputs1 = layers.Activation(activation="softmax", name="output30")(outputs1)

	c = DecodeFPB_Block(c, concat_layers[0], filters[0], internal_depths[0], "DecodeBlock_lvl"+str(0), kernel_size=3, strides=(3,3))

	c = layers.concatenate([c, upOut1], axis=-1) #change in the re-entering of the class info from 30m
	outputs2 = layers.Conv2D(classes, kernel_size=1, name="final_conv")(c)
	outputs2 = layers.Activation(activation="softmax", name="output10")(outputs2)
	model = tf.keras.Model(inputs=input_tensor, outputs=[outputs1, outputs2], name="u-net")
	return model


#encode-decode network with one output and RPB as backbone. Early fusion with feature adaptation module data fusion architecture.
def FPB_early2_mod(input_shape = None, input_tensor = None, classes=None):
	internal_depths = [2,2,2,2]
	num_filters = [32, 64, 64, 128]
	depth = 4
	filters = []

	c1 = layers.Conv2D(32, kernel_size=3, padding="same", name = "Input1_conv0")(input_tensor[0])
	c1 = layers.BatchNormalization(name='Input1_BN')(c1)
	c1 = layers.Activation('relu', name='Input1_Activation')(c1)
	c2 = layers.Conv2D(32, kernel_size=3, padding="same", name = "Input2_conv0")(input_tensor[1])
	c2 = layers.BatchNormalization(name='Input2_BN')(c2)
	c2 = layers.Activation('relu', name='Input2_Activation')(c2)

	c = layers.concatenate([c1, c2], axis=-1)
	c = layers.Conv2D(64, kernel_size=1, name="data_fusion")(c)
	c = layers.BatchNormalization(name='data_fusion_BN')(c)
	c = layers.Activation('relu', name='data_fusion_Activation')(c)

	for i in range(depth-1):
		a = []
		for j in range(internal_depths[i]):
			a.append(num_filters[i])
		filters.append(a)

	concat_layers = []

	#Encode Part:
	c,conc = EncodeFPB_Block(c, filters[0] , internal_depths[0], "EncodeBlock_lvl0", pool_size=(3, 3), strides=(3, 3))
	concat_layers.append(conc)
	for i in range(1, depth-1):
		c, conc = EncodeFPB_Block(c, filters[i] , internal_depths[i], "EncodeBlock_lvl"+str(i))
		concat_layers.append(conc)

	for i in range(internal_depths[depth-1]):
		c = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c)
		c = layers.Dropout(0.3)(c)
		c = layers.BatchNormalization(name="EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c)
		c = layers.Activation('relu', name="EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c)

	for i in range(depth-2,0,-1):
		c = DecodeFPB_Block(c, concat_layers[i], filters[i], internal_depths[i], "DecodeBlock_lvl"+str(i))

	c = DecodeFPB_Block(c, concat_layers[0], filters[0], internal_depths[0], "DecodeBlock_lvl"+str(0), kernel_size=3, strides=(3,3))

	outputs = layers.Conv2D(classes, kernel_size=1, name="final_conv")(c)
	outputs = layers.Activation(activation="softmax", name="output10")(outputs)
	model = tf.keras.Model(inputs=input_tensor, outputs=outputs, name="u-net")
	return model


#encode-decode network with one output and RPB as backbone. Not a fusion module.
def FPB_1input_1out_mod(input_shape = None, input_tensor = None, classes=None):
	internal_depths = [2,2,2,2]
	num_filters = [32, 64, 64, 128]
	depth = 4
	filters = []

	for i in range(depth-1):
		a = []
		for j in range(internal_depths[i]):
			a.append(num_filters[i])
		filters.append(a)

	concat_layers = []

	#Encode Part:
	c,conc = EncodeFPB_Block(input_tensor, filters[0] , internal_depths[0], "EncodeBlock_lvl0", pool_size=(3, 3), strides=(3, 3))
	concat_layers.append(conc)
	for i in range(1, depth-1):
		c, conc = EncodeFPB_Block(c, filters[i] , internal_depths[i], "EncodeBlock_lvl"+str(i))
		concat_layers.append(conc)

	for i in range(internal_depths[depth-1]):
		c = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c)
		c = layers.Dropout(0.3)(c)
		c = layers.BatchNormalization(name="EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c)
		c = layers.Activation('relu', name="EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c)

	for i in range(depth-2,0,-1):
		c = DecodeFPB_Block(c, concat_layers[i], filters[i], internal_depths[i], "DecodeBlock_lvl"+str(i))

	c = DecodeFPB_Block(c, concat_layers[0], filters[0], internal_depths[0], "DecodeBlock_lvl"+str(0), kernel_size=3, strides=(3,3))

	outputs = layers.Conv2D(classes, kernel_size=1, name="final_conv")(c)
	outputs = layers.Activation(activation="softmax", name="output10")(outputs)
	model = tf.keras.Model(inputs=input_tensor, outputs=outputs, name="u-net")
	return model

#encode-decode network with two outputs and resnet as backbone. Re-entrance of the output from the 30m resolution result as a feature for the 10m output. Addition of the features before the fusion as features for the classification as well. Early fusion with feature adaptation module data fusion architecture.
def labelsuperResUnet_early2_mod2(input_shape = None, input_tensor = None, classes=None):
	internal_depths = [2,2,2,2]
	num_filters = [32, 64, 64, 128]
	depth = 4
	filters = []

	c1 = layers.Conv2D(64, kernel_size=3, padding="same", name = "Input1_conv0")(input_tensor[0])
	c1 = layers.BatchNormalization(name='Input1_BN')(c1)
	c1 = layers.Activation('relu', name='Input1_Activation')(c1)
	#b = input_tensor[0]
	c2 = layers.Conv2D(64, kernel_size=3, padding="same", name = "Input2_conv0")(input_tensor[1])
	c2 = layers.BatchNormalization(name='Input2_BN')(c2)
	c2 = layers.Activation('relu', name='Input2_Activation')(c2)

	c = layers.concatenate([c1, c2], axis=-1)
	c = layers.Conv2D(64, kernel_size=1, name="data_fusion")(c)
	c = layers.BatchNormalization(name='data_fusion_BN')(c)
	c = layers.Activation('relu', name='data_fusion_Activation')(c)

	for i in range(depth-1):
		a = []
		for j in range(internal_depths[i]):
			a.append(num_filters[i])
		filters.append(a)

	concat_layers = []

	#Encode Part:
	c,conc = EncodeResNetBlocks(c, filters[0] , internal_depths[0], "EncodeBlock_lvl0", pool_size=(3, 3), strides=(3, 3))
	concat_layers.append(conc)
	for i in range(1, depth-1):
		c, conc = EncodeResNetBlocks(c, filters[i] , internal_depths[i], "EncodeBlock_lvl"+str(i))
		concat_layers.append(conc)

	for i in range(internal_depths[depth-1]):
		c = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c)
		c = layers.Dropout(0.3)(c)
		c = layers.BatchNormalization(name="EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c)
		c = layers.Activation('relu', name="EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c)

	for i in range(depth-2,0,-1):
		c = DecodeResNetBlocks(c, concat_layers[i], filters[i], internal_depths[i], "DecodeBlock_lvl"+str(i))

	outputs1 = layers.Conv2D(classes, kernel_size=1, name="final_conv_30m")(c)

	c_int = layers.Activation(activation="sigmoid", name="output_sigmoid")(outputs1)
	upOut1 = layers.UpSampling2D(size=(3,3), name = "Output1_Up")(c_int)
	outputs1 = layers.Activation(activation="softmax", name="output30")(outputs1)

	c = DecodeResNetBlocks(c, concat_layers[0], filters[0], internal_depths[0], "DecodeBlock_lvl"+str(0), kernel_size=3, strides=(3,3))

	c = layers.concatenate([c, c1, c2, upOut1], axis=-1) #change in the re-entering of the class info from 30m + the first conv features from before the fusion
	outputs2 = layers.Conv2D(classes, kernel_size=1, name="final_conv")(c)
	outputs2 = layers.Activation(activation="softmax", name="output10")(outputs2)
	model = tf.keras.Model(inputs=input_tensor, outputs=[outputs1, outputs2], name="u-net")
	return model

#encode-decode network with two outputs and DenseNet as backbone. Re-entrance of the output from the 30m resolution result as a feature for the 10m output. Early fusion with feature adaptation module data fusion architecture.
def labelsuperDenseUnet_early2_mod(input_shape = None, input_tensor = None, classes=None):
	internal_depths = [2,2,2,2]
	num_filters = [32, 64, 64, 128]
	depth = 4
	filters = []

	c1 = layers.Conv2D(64, kernel_size=3, padding="same", name = "Input1_conv0")(input_tensor[0])
	c1 = layers.BatchNormalization(name='Input1_BN')(c1)
	c1 = layers.Activation('relu', name='Input1_Activation')(c1)
	c2 = layers.Conv2D(64, kernel_size=3, padding="same", name = "Input2_conv0")(input_tensor[1])
	c2 = layers.BatchNormalization(name='Input2_BN')(c2)
	c2 = layers.Activation('relu', name='Input2_Activation')(c2)

	c = layers.concatenate([c1, c2], axis=-1)
	c = layers.Conv2D(64, kernel_size=1, name="data_fusion")(c)
	c = layers.BatchNormalization(name='data_fusion_BN')(c)
	c = layers.Activation('relu', name='data_fusion_Activation')(c)

	for i in range(depth-1):
		a = []
		for j in range(internal_depths[i]):
			a.append(num_filters[i])
		filters.append(a)

	concat_layers = []

	#Encode Part:
	c,conc = EncodeDenseNetBlocks(c, filters[0] , internal_depths[0], "EncodeBlock_lvl0", pool_size=(3, 3), strides=(3, 3))
	concat_layers.append(conc)
	for i in range(1, depth-1):
		c, conc = EncodeDenseNetBlocks(c, filters[i] , internal_depths[i], "EncodeBlock_lvl"+str(i))
		concat_layers.append(conc)

	for i in range(internal_depths[depth-1]):
		c = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c)
		c = layers.BatchNormalization(name="EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c)
		c = layers.Activation('relu', name="EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c)

	for i in range(depth-2,0,-1):
		c = DecodeDenseNetBlocks(c, concat_layers[i], filters[i], internal_depths[i], "DecodeBlock_lvl"+str(i))

	outputs1 = layers.Conv2D(classes, kernel_size=1, name="final_conv_30m")(c)

	c_int = layers.Activation(activation="sigmoid", name="output_sigmoid")(outputs1)
	upOut1 = layers.UpSampling2D(size=(3,3), name = "Output1_Up")(c_int)
	outputs1 = layers.Activation(activation="softmax", name="output30")(outputs1)

	c = DecodeDenseNetBlocks(c, concat_layers[0], filters[0], internal_depths[0], "DecodeBlock_lvl"+str(0), kernel_size=3, strides=(3,3))

	c = layers.concatenate([c, upOut1], axis=-1) #change in the re-entering of the class info from 30m
	outputs2 = layers.Conv2D(classes, kernel_size=1, name="final_conv")(c)
	outputs2 = layers.Activation(activation="softmax", name="output10")(outputs2)
	model = tf.keras.Model(inputs=input_tensor, outputs=[outputs1, outputs2], name="u-net")

	return model


#encode-decode network with two outputs and Resnet with Statistical learning network (quadratic module) as backbone. Re-entrance of the output from the 30m resolution result as a feature for the 10m output. Early fusion with feature adaptation module data fusion architecture.
def labelsuperResUnet_early2_sln(input_shape = None, input_tensor = None, classes=None):
	internal_depths = [2,2,2,2]
	num_filters = [32, 64, 64, 128]
	depth = 4
	filters = []

	c1 = layers.Conv2D(64, kernel_size=3, padding="same", name = "Input1_conv0")(input_tensor[0])
	c1 = layers.BatchNormalization(name='Input1_BN')(c1)
	c1 = layers.Activation('relu', name='Input1_Activation')(c1)
	c2 = layers.Conv2D(64, kernel_size=3, padding="same", name = "Input2_conv0")(input_tensor[1])
	c2 = layers.BatchNormalization(name='Input2_BN')(c2)
	c2 = layers.Activation('relu', name='Input2_Activation')(c2)

	c = layers.concatenate([c1, c2], axis=-1)
	c = layers.Conv2D(64, kernel_size=1, name="data_fusion")(c)
	c = layers.BatchNormalization(name='data_fusion_BN')(c)
	c = layers.Activation('relu', name='data_fusion_Activation')(c)

	quadlayer1 = customLayers.quadLayer(num_filters[0]*internal_depths[0],stride=3, kernelsize=3)
	b = quadlayer1(input_tensor[1])
	b = layers.BatchNormalization(name='quad1_1_BN')(b)
	b = layers.Activation('relu', name='quad1_1_Activation')(b)
	#print(tf.shape(b))

	for i in range(depth-1):
		a = []
		for j in range(internal_depths[i]):
			a.append(num_filters[i])
		filters.append(a)

	concat_layers = []

	#Encode Part:
	c,conc = EncodeResNetBlocks(c, filters[0] , internal_depths[0], "EncodeBlock_lvl0", pool_size=(3, 3), strides=(3, 3))
	#print(tf.shape(c))
	c = layers.concatenate([c, b], axis=-1)
	#print('passei')
	concat_layers.append(conc)
	for i in range(1, depth-1):
		c, conc = EncodeResNetBlocks(c, filters[i] , internal_depths[i], "EncodeBlock_lvl"+str(i))
		concat_layers.append(conc)

	for i in range(internal_depths[depth-1]):
		c = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c)
		c = layers.Dropout(0.3)(c)
		c = layers.BatchNormalization(name="EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c)
		c = layers.Activation('relu', name="EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c)

	for i in range(depth-2,0,-1):
		c = DecodeResNetBlocks(c, concat_layers[i], filters[i], internal_depths[i], "DecodeBlock_lvl"+str(i))

	outputs1 = layers.Conv2D(classes, kernel_size=1, name="final_conv_30m")(c)

	c_int = layers.Activation(activation="sigmoid", name="output_sigmoid")(outputs1)
	upOut1 = layers.UpSampling2D(size=(3,3), name = "Output1_Up")(c_int)
	outputs1 = layers.Activation(activation="softmax", name="output30")(outputs1)

	c = DecodeResNetBlocks(c, concat_layers[0], filters[0], internal_depths[0], "DecodeBlock_lvl"+str(0), kernel_size=3, strides=(3,3))

	c = layers.concatenate([c, upOut1], axis=-1) #change in the re-entering of the class info from 30m
	outputs2 = layers.Conv2D(classes, kernel_size=1, name="final_conv")(c)
	outputs2 = layers.Activation(activation="softmax", name="output10")(outputs2)
	model = tf.keras.Model(inputs=input_tensor, outputs=[outputs1, outputs2], name="sln")
	return model


#encode-decode network with two outputs and Resnet as backbone. Re-entrance of the output from the 30m resolution result as a feature for the 10m output. Late fusion data fusion architecture.
def labelsuperResUnet_late_mod(input_shape = None, input_tensor = None, classes=None):
	internal_depths = [2,2,2,2]
	num_filters = [32, 64, 64, 128]
	depth = 4
	filters = []

	
	for i in range(depth-1):
		a = []
		for j in range(internal_depths[i]):
			a.append(num_filters[i])
		filters.append(a)

	#S2 Branch
	concat_layers1 = []

	#Encode Part:
	c1,conc = EncodeResNetBlocks(input_tensor[0], filters[0] , internal_depths[0], "W1EncodeBlock_lvl0", pool_size=(3, 3), strides=(3, 3))
	concat_layers1.append(conc)
	for i in range(1, depth-1):
		c1, conc = EncodeResNetBlocks(c1, filters[i] , internal_depths[i], "W1EncodeBlock_lvl"+str(i))
		concat_layers1.append(conc)

	for i in range(internal_depths[depth-1]):
		c1 = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "W1EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c1)
		c1 = layers.BatchNormalization(name="W1EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c1)
		c1 = layers.Activation('relu', name="W1EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c1)

	for i in range(depth-2,0,-1):
		c1 = DecodeResNetBlocks(c1, concat_layers1[i], filters[i], internal_depths[i], "W1DecodeBlock_lvl"+str(i))

	#S1 Branch
	concat_layers2 = []

	#Encode Part:
	c2,conc = EncodeResNetBlocks(input_tensor[1], filters[0] , internal_depths[0], "W2EncodeBlock_lvl0", pool_size=(3, 3), strides=(3, 3))
	concat_layers2.append(conc)
	for i in range(1, depth-1):
		c2, conc = EncodeResNetBlocks(c2, filters[i] , internal_depths[i], "W2EncodeBlock_lvl"+str(i))
		concat_layers2.append(conc)

	for i in range(internal_depths[depth-1]):
		c2 = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "W2EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c2)
		c2 = layers.BatchNormalization(name="W2EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c2)
		c2 = layers.Activation('relu', name="W2EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c2)

	for i in range(depth-2,0,-1):
		c2 = DecodeResNetBlocks(c2, concat_layers2[i], filters[i], internal_depths[i], "W2DecodeBlock_lvl"+str(i))

	c = layers.concatenate([c1, c2], axis=-1)
	outputs1 = layers.Conv2D(classes, kernel_size=1, name="final_conv_30m")(c)
	x = layers.Activation(activation="sigmoid", name="output30_sig")(outputs1)
	x = layers.UpSampling2D(size=(3,3), name = "Output1_Up")(x)
	outputs1 = layers.Activation(activation="softmax", name="output30")(outputs1)
	c1 = DecodeResNetBlocks(c1, concat_layers1[0], filters[0], internal_depths[0], "W1DecodeBlock_lvl"+str(0), kernel_size=3, strides=(3,3))
	c2 = DecodeResNetBlocks(c2, concat_layers2[0], filters[0], internal_depths[0], "W2DecodeBlock_lvl"+str(0), kernel_size=3, strides=(3,3))
	c = layers.concatenate([c1, c2, x], axis=-1)
	outputs2 = layers.Conv2D(classes, kernel_size=1, name="final_conv")(c)
	outputs2 = layers.Activation(activation="softmax", name="output10")(outputs2)
	model = tf.keras.Model(inputs=input_tensor, outputs=[outputs1, outputs2], name="u-net")

	return model

#encode-decode network with two outputs and resnet as backbone. Late fusion data fusion architecture.
def labelsuperResUnet_late(input_shape = None, input_tensor = None, classes=None):

	internal_depths = [2,2,2,2]
	num_filters = [32, 64, 64, 128]
	depth = 4
	filters = []

	for i in range(depth-1):
		a = []
		for j in range(internal_depths[i]):
			a.append(num_filters[i])
		filters.append(a)

	#S2 Branch
	concat_layers1 = []

	#Encode Part:
	c1,conc = EncodeResNetBlocks(input_tensor[0], filters[0] , internal_depths[0], "W1EncodeBlock_lvl0", pool_size=(3, 3), strides=(3, 3))
	concat_layers1.append(conc)
	for i in range(1, depth-1):
		c1, conc = EncodeResNetBlocks(c1, filters[i] , internal_depths[i], "W1EncodeBlock_lvl"+str(i))
		concat_layers1.append(conc)

	for i in range(internal_depths[depth-1]):
		c1 = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "W1EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c1)
		c1 = layers.BatchNormalization(name="W1EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c1)
		c1 = layers.Activation('relu', name="W1EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c1)

	for i in range(depth-2,0,-1):
		c1 = DecodeResNetBlocks(c1, concat_layers1[i], filters[i], internal_depths[i], "W1DecodeBlock_lvl"+str(i))

	#S1 Branch
	concat_layers2 = []

	#Encode Part:
	c2,conc = EncodeResNetBlocks(input_tensor[1], filters[0] , internal_depths[0], "W2EncodeBlock_lvl0", pool_size=(3, 3), strides=(3, 3))
	concat_layers2.append(conc)
	for i in range(1, depth-1):
		c2, conc = EncodeResNetBlocks(c2, filters[i] , internal_depths[i], "W2EncodeBlock_lvl"+str(i))
		concat_layers2.append(conc)

	for i in range(internal_depths[depth-1]):
		c2 = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "W2EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c2)
		c2 = layers.BatchNormalization(name="W2EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c2)
		c2 = layers.Activation('relu', name="W2EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c2)

	for i in range(depth-2,0,-1):
		c2 = DecodeResNetBlocks(c2, concat_layers2[i], filters[i], internal_depths[i], "W2DecodeBlock_lvl"+str(i))

	c = layers.concatenate([c1, c2], axis=-1)
	outputs1 = layers.Conv2D(classes, kernel_size=1, name="final_conv_30m")(c)
	outputs1 = layers.Activation(activation="softmax", name="output30")(outputs1)
	c1 = DecodeResNetBlocks(c1, concat_layers1[0], filters[0], internal_depths[0], "W1DecodeBlock_lvl"+str(0), kernel_size=3, strides=(3,3))
	c2 = DecodeResNetBlocks(c2, concat_layers2[0], filters[0], internal_depths[0], "W2DecodeBlock_lvl"+str(0), kernel_size=3, strides=(3,3))
	c = layers.concatenate([c1, c2], axis=-1)
	outputs2 = layers.Conv2D(classes, kernel_size=1, name="final_conv")(c)
	outputs2 = layers.Activation(activation="softmax", name="output10")(outputs2)
	model = tf.keras.Model(inputs=input_tensor, outputs=[outputs1, outputs2], name="u-net")

	return model

#encode-decode network with one output and ResNet as backbone. Middle fusion data fusion architecture. Downsample of 3 instead of 2 in the first level of the encoder-decoder.
def ResUnet3_double_middle (input_shape = None, input_tensor = None, classes=None):
	internal_depths = [2,2,2,2]
	num_filters = [32, 64, 64, 128]
	depth = 4
	filters = []

	for i in range(depth-1):
		a = []
		for j in range(internal_depths[i]):
			a.append(num_filters[i])
		filters.append(a)

	concat_layers1 = []
	concat_layers2 = []
	#Encode Part 1:
	c1,conc = EncodeResNetBlocks(input_tensor[0], filters[0] , internal_depths[0], "W1EncodeBlock_lvl0", pool_size=(3, 3), strides=(3, 3))
	concat_layers1.append(conc)
	for i in range(1, depth-1):
		c1, conc = EncodeResNetBlocks(c1, filters[i] , internal_depths[i], "W1EncodeBlock_lvl"+str(i))
		concat_layers1.append(conc)

	for i in range(internal_depths[depth-1]):
		c1 = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "W1EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c1)
		c1 = layers.BatchNormalization(name="W1EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c1)
		c1 = layers.Activation('relu', name="W1EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c1)

	#Encode Part 2:
	c2,conc = EncodeResNetBlocks(input_tensor[1], filters[0] , internal_depths[0], "W2EncodeBlock_lvl0",pool_size=(3, 3), strides=(3, 3))
	concat_layers2.append(conc)
	for i in range(1, depth-1):
		c2, conc = EncodeResNetBlocks(c2, filters[i] , internal_depths[i], "W2EncodeBlock_lvl"+str(i))
		concat_layers2.append(conc)

	for i in range(internal_depths[depth-1]):
		c2 = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c2)
		c2 = layers.BatchNormalization(name="W2EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c2)
		c2 = layers.Activation('relu', name="W2EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c2)

	#Decode with feature fusion
	c = layers.concatenate([c1, c2], axis=-1)

	for i in range(depth-2,0,-1):
		d = layers.concatenate([concat_layers1[i], concat_layers2[i]], axis=-1)
		c = DecodeResNetBlocks(c, d, filters[i], internal_depths[i], "DecodeBlock_lvl"+str(i))

	d = layers.concatenate([concat_layers1[0], concat_layers2[0]], axis=-1)
	c = DecodeResNetBlocks(c, d, filters[0], internal_depths[0], "DecodeBlock_lvl"+str(0), kernel_size=3, strides=(3,3))

	outputs2 = layers.Conv2D(classes, kernel_size=1, name="final_conv_10m")(c)
	outputs2 = layers.Activation(activation="softmax", name="output10")(outputs2)
	model = tf.keras.Model(inputs=input_tensor, outputs=(outputs2), name="ResUnet3_double_middle")

	return model


#encode-decode network with one output and VGG as backbone. Not a fusion network. Downsample of 3 instead of 2 in the first level of the encoder-decoder.
def unet3Depth0(input_shape = None, input_tensor = None, classes=None):
	internal_depths = [2,2,2,2]
	num_filters = [32, 64, 64, 128]
	depth = 4
	filters = []

	for i in range(depth-1):
		a = []
		for j in range(internal_depths[i]):
			a.append(num_filters[i])
		filters.append(a)

	concat_layers = []

	#Encode Part:
	c,conc = EncodeVGGBlocks(input_tensor, filters[0] , internal_depths[0], "EncodeBlock_lvl0", pool_size=(3, 3), strides=(3, 3))
	concat_layers.append(conc)
	for i in range(1, depth-1):
		c, conc = EncodeVGGBlocks(c, filters[i] , internal_depths[i], "EncodeBlock_lvl"+str(i))
		concat_layers.append(conc)

	for i in range(internal_depths[depth-1]):
		c = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c)
		c = layers.BatchNormalization(name="EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c)
		c = layers.Activation('relu', name="EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c)

	for i in range(depth-2,0,-1):
		c = DecodeVGGBlocks(c, concat_layers[i], filters[i], internal_depths[i], "DecodeBlock_lvl"+str(i))

	c = DecodeVGGBlocks(c, concat_layers[0], filters[0], internal_depths[0], "DecodeBlock_lvl"+str(0), kernel_size=3, strides=(3,3))
	outputs = layers.Conv2D(classes, kernel_size=1, name="final_conv")(c)
	outputs = layers.Activation(activation="softmax", name="output")(outputs)
	model = tf.keras.Model(inputs=input_tensor, outputs=outputs, name="u-net")

	return model

#encode-decode network with one output and ResNet as backbone. Not a fusion network.
def unet3Resnet(input_shape = None, input_tensor = None, classes=None):
	internal_depths = [2,2,2,2]
	num_filters = [32, 64, 64, 128]
	depth = 4
	filters = []

	for i in range(depth-1):
		a = []
		for j in range(internal_depths[i]):
			a.append(num_filters[i])
		filters.append(a)

	concat_layers = []

	#Encode Part:
	c,conc = EncodeResNetBlocks(input_tensor, filters[0] , internal_depths[0], "EncodeBlock_lvl0", pool_size=(3, 3), strides=(3, 3))
	concat_layers.append(conc)
	for i in range(1, depth-1):
		c, conc = EncodeResNetBlocks(c, filters[i] , internal_depths[i], "EncodeBlock_lvl"+str(i))
		concat_layers.append(conc)

	for i in range(internal_depths[depth-1]):
		c = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c)
		c = layers.BatchNormalization(name="EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c)
		c = layers.Activation('relu', name="EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c)

	for i in range(depth-2,0,-1):
		c = DecodeResNetBlocks(c, concat_layers[i], filters[i], internal_depths[i], "DecodeBlock_lvl"+str(i))

	c = DecodeResNetBlocks(c, concat_layers[0], filters[0], internal_depths[0], "DecodeBlock_lvl"+str(0), kernel_size=3, strides=(3,3))
	outputs = layers.Conv2D(classes, kernel_size=1, name="final_conv")(c)
	outputs = layers.Activation(activation="softmax", name="output")(outputs)
	model = tf.keras.Model(inputs=input_tensor, outputs=outputs, name="u-net")

	return model

#encode-decode network with two outputs and VGG as backbone. Not a fusion network.
def unet3Depth1(input_shape = None, input_tensor = None, classes=None):
	internal_depths = [2,2,2,2]
	num_filters = [32, 64, 64, 128]
	depth = 4
	filters = []

	for i in range(depth-1):
		a = []
		for j in range(internal_depths[i]):
			a.append(num_filters[i])
		filters.append(a)

	concat_layers = []

	#Encode Part:
	c,conc = EncodeVGGBlocks(input_tensor, filters[0] , internal_depths[0], "EncodeBlock_lvl0", pool_size=(3, 3), strides=(3, 3))
	concat_layers.append(conc)
	for i in range(1, depth-1):
		c, conc = EncodeVGGBlocks(c, filters[i] , internal_depths[i], "EncodeBlock_lvl"+str(i))
		concat_layers.append(conc)

	for i in range(internal_depths[depth-1]):
		c = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c)
		c = layers.BatchNormalization(name="EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c)
		c = layers.Activation('relu', name="EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c)

	for i in range(depth-2,0,-1):
		c = DecodeVGGBlocks(c, concat_layers[i], filters[i], internal_depths[i], "DecodeBlock_lvl"+str(i))

	outputs1 = layers.Conv2D(classes, kernel_size=1, name="final_conv_30m")(c)
	outputs1 = layers.Activation(activation="softmax", name="output30")(outputs1)
	c = DecodeVGGBlocks(c, concat_layers[0], filters[0], internal_depths[0], "DecodeBlock_lvl"+str(0), kernel_size=3, strides=(3,3))
	outputs2 = layers.Conv2D(classes, kernel_size=1, name="final_conv")(c)
	outputs2 = layers.Activation(activation="softmax", name="output10")(outputs2)
	model = tf.keras.Model(inputs=input_tensor, outputs=[outputs1, outputs2], name="u-net")

	return model

#encode-decode network with two outputs and ResNet as backbone. Re-entrance of the output from the 30m resolution result (sigmoid) as a feature for the 10m output. Not a fusion network.
def Resunet3Depth1(input_shape = None, input_tensor = None, classes=None):
	internal_depths = [2,2,2,2]
	num_filters = [32, 64, 64, 128]
	depth = 4
	filters = []

	for i in range(depth-1):
		a = []
		for j in range(internal_depths[i]):
			a.append(num_filters[i])
		filters.append(a)

	concat_layers = []

	#Encode Part:
	c,conc = EncodeResNetBlocks(input_tensor, filters[0] , internal_depths[0], "EncodeBlock_lvl0", pool_size=(3, 3), strides=(3, 3))
	concat_layers.append(conc)
	for i in range(1, depth-1):
		c, conc = EncodeResNetBlocks(c, filters[i] , internal_depths[i], "EncodeBlock_lvl"+str(i))
		concat_layers.append(conc)

	for i in range(internal_depths[depth-1]):
		c = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c)
		c = layers.Dropout(0.3)(c)
		c = layers.BatchNormalization(name="EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c)
		c = layers.Activation('relu', name="EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c)

	for i in range(depth-2,0,-1):
		c = DecodeResNetBlocks(c, concat_layers[i], filters[i], internal_depths[i], "DecodeBlock_lvl"+str(i))

	outputs1 = layers.Conv2D(classes, kernel_size=1, name="final_conv_30m")(c)
	c_int = layers.Activation(activation="sigmoid", name="output_sigmoid")(outputs1)
	upOut1 = layers.UpSampling2D(size=(3,3), name = "Output1_Up")(c_int)
	outputs1 = layers.Activation(activation="softmax", name="output30")(outputs1)
	c = DecodeResNetBlocks(c, concat_layers[0], filters[0], internal_depths[0], "DecodeBlock_lvl"+str(0), kernel_size=3, strides=(3,3))
	c = layers.concatenate([c, upOut1], axis=-1)
	outputs2 = layers.Conv2D(classes, kernel_size=1, name="final_conv")(c)
	outputs2 = layers.Activation(activation="softmax", name="output10")(outputs2)
	model = tf.keras.Model(inputs=input_tensor, outputs=[outputs1, outputs2], name="u-net")

	return model

#encode-decode network with two outputs and VGG as backbone. Re-entrance of the output from the 30m resolution result (softmax) as a feature for the 10m output. Not a fusion network.
def unet3Depth2(input_shape = None, input_tensor = None, classes=None):
	internal_depths = [2,2,2,2]
	num_filters = [32, 64, 64, 128]
	depth = 4
	filters = []

	for i in range(depth-1):
		a = []
		for j in range(internal_depths[i]):
			a.append(num_filters[i])
		filters.append(a)

	concat_layers = []

	#Encode Part:
	c,conc = EncodeVGGBlocks(input_tensor, filters[0] , internal_depths[0], "EncodeBlock_lvl0", pool_size=(3, 3), strides=(3, 3))
	concat_layers.append(conc)
	for i in range(1, depth-1):
		c, conc = EncodeVGGBlocks(c, filters[i] , internal_depths[i], "EncodeBlock_lvl"+str(i))
		concat_layers.append(conc)

	for i in range(internal_depths[depth-1]):
		c = layers.Conv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c)
		c = layers.BatchNormalization(name="EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c)
		c = layers.Activation('relu', name="EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c)

	for i in range(depth-2,0,-1):
		c = DecodeVGGBlocks(c, concat_layers[i], filters[i], internal_depths[i], "DecodeBlock_lvl"+str(i))

	outputs1 = layers.Conv2D(classes, kernel_size=1, name="final_conv_30m")(c)
	outputs1 = layers.Activation(activation="softmax", name="output30")(outputs1)
	upOut1 = layers.UpSampling2D(size=(3,3), name = "Output1_Up")(outputs1)
	d = layers.concatenate([concat_layers[0], upOut1], axis=-1)
	c = DecodeVGGBlocks(c, d, filters[0], internal_depths[0], "DecodeBlock_lvl"+str(0), kernel_size=3, strides=(3,3))
	outputs2 = layers.Conv2D(classes, kernel_size=1, name="final_conv")(c)
	outputs2 = layers.Activation(activation="softmax", name="output10")(outputs2)
	model = tf.keras.Model(inputs=input_tensor, outputs=[outputs1, outputs2], name="u-net")

	return model


#encode-decode network with two outputs and Complex-valued Resnet as backbone. Re-entrance of the output from the 30m resolution result as a feature for the 10m output. Not a fusion network. Applied to SAR data
def labelsuperResComplex(input_shape = None, input_tensor = None, classes=None):
	internal_depths = [2,2,2,2]
	num_filters = [32, 64, 64, 128]
	depth = 4
	filters = []

	for i in range(depth-1):
		a = []
		for j in range(internal_depths[i]):
			a.append(num_filters[i])
		filters.append(a)

	concat_layers = []

	#Encode Part:
	c,conc = EncodeComplexResNetBlocks(input_tensor, filters[0] , internal_depths[0], "EncodeBlock_lvl0", pool_size=(3, 3), strides=(3, 3))
	concat_layers.append(conc)
	for i in range(1, depth-1):
		c, conc = EncodeComplexResNetBlocks(c, filters[i] , internal_depths[i], "EncodeBlock_lvl"+str(i))
		concat_layers.append(conc)

	for i in range(internal_depths[depth-1]):
		c = complex_layers.ComplexConv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c)
		c = complex_layers.ComplexDropout(0.3)(c)
		#c = complex_layers.ComplexBatchNormalization(name="EncodeBlock_lvl"+str(depth-1)+"_BN"+str(i))(c)
		c = layers.Activation(complex_activations.cart_relu, name="EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c)

	for i in range(depth-2,0,-1):
		c = DecodeComplexResNetBlocks(c, concat_layers[i], filters[i], internal_depths[i], "DecodeBlock_lvl"+str(i))

	outputs1 = complex_layers.ComplexConv2D(classes, kernel_size=1, name="final_conv_30m")(c)
	c_int = layers.Activation(activation=complex_activations.sigmoid_real, name="output_sigmoid")(outputs1)
	upOut1 = complex_layers.ComplexUpSampling2D(size=(3,3), name = "Output1_Up")(c_int)
	upOut1 = tf.cast(upOut1, dtype=tf.complex64)
	outputs1 = layers.Activation(activation=complex_activations.softmax_real_with_abs, name="output30")(outputs1)
	c = DecodeComplexResNetBlocks(c, concat_layers[0], filters[0], internal_depths[0], "DecodeBlock_lvl"+str(0), kernel_size=3, strides=(3,3))
	c = layers.concatenate([c, upOut1], axis=-1)
	outputs2 = complex_layers.ComplexConv2D(classes, kernel_size=1, name="final_conv")(c)
	outputs2 = layers.Activation(activation=complex_activations.softmax_real_with_abs, name="output10")(outputs2)
	model = tf.keras.Model(inputs=input_tensor, outputs=[outputs1, outputs2], name="u-net")

	return model


#encode-decode network with two outputs and Complex-valued ResNet as backbone. Re-entrance of the output from the 30m resolution result as a feature for the 10m output. Early fusion with feature adaptation module data fusion architecture.
def labelsuperResComplexFusion_early(input_shape = None, input_tensor = None, classes=None):
	internal_depths = [2,2,2,2]
	num_filters = [32, 64, 64, 128]
	depth = 4
	filters = []

	c1 = layers.Conv2D(32, kernel_size=3, padding="same", name = "Input1_conv0")(input_tensor[0])
	c1 = layers.BatchNormalization(name='Input1_BN')(c1)
	c1 = layers.Activation('relu', name='Input1_Activation')(c1)
	#b = input_tensor[0]
	c2 = complex_layers.ComplexConv2D(32, kernel_size=3, padding="same", name = "Input2_conv0")(input_tensor[1])
	c2 = layers.Activation(complex_activations.cart_relu, name='Input2_Activation')(c2)
	c1 = tf.cast(c1, dtype=tf.complex64)
	c = layers.concatenate([c1, c2], axis=-1)
	c = complex_layers.ComplexConv2D(64, kernel_size=1, name="data_fusion")(c)
	c = layers.Activation(complex_activations.cart_relu, name='data_fusion_Activation')(c)

	for i in range(depth-1):
		a = []
		for j in range(internal_depths[i]):
			a.append(num_filters[i])
		filters.append(a)

	concat_layers = []

	#Encode Part:
	c,conc = EncodeComplexResNetBlocks(c, filters[0] , internal_depths[0], "EncodeBlock_lvl0", pool_size=(3, 3), strides=(3, 3))
	concat_layers.append(conc)
	for i in range(1, depth-1):
		c, conc = EncodeComplexResNetBlocks(c, filters[i] , internal_depths[i], "EncodeBlock_lvl"+str(i))
		concat_layers.append(conc)

	for i in range(internal_depths[depth-1]):
		c = complex_layers.ComplexConv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c)
		c = complex_layers.ComplexDropout(0.3)(c)
		c = layers.Activation(complex_activations.cart_relu, name="EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c)

	for i in range(depth-2,0,-1):
		c = DecodeComplexResNetBlocks(c, concat_layers[i], filters[i], internal_depths[i], "DecodeBlock_lvl"+str(i))

	outputs1 = complex_layers.ComplexConv2D(classes, kernel_size=1, name="final_conv_30m")(c)

	c_int = layers.Activation(activation=complex_activations.sigmoid_real, name="output_sigmoid")(outputs1)
	upOut1 = complex_layers.ComplexUpSampling2D(size=(3,3), name = "Output1_Up")(c_int)
	upOut1 = tf.cast(upOut1, dtype=tf.complex64)
	outputs1 = layers.Activation(activation=complex_activations.softmax_real_with_abs, name="output30")(outputs1)

	c = DecodeComplexResNetBlocks(c, concat_layers[0], filters[0], internal_depths[0], "DecodeBlock_lvl"+str(0), kernel_size=3, strides=(3,3))

	c = layers.concatenate([c, upOut1], axis=-1) #change in the re-entering of the class info from 30m + the first conv features from before the fusion
	outputs2 = complex_layers.ComplexConv2D(classes, kernel_size=1, name="final_conv")(c)
	outputs2 = layers.Activation(activation=complex_activations.softmax_real_with_abs, name="output10")(outputs2)
	model = tf.keras.Model(inputs=input_tensor, outputs=[outputs1, outputs2], name="u-net")
	return model


#encode-decode network with two outputs and Complex-valued RPB as backbone. Re-entrance of the output from the 30m resolution result as a feature for the 10m output. Early fusion with feature adaptation module data fusion architecture.
def labelsuperFPBComplexFusion_early(input_shape = None, input_tensor = None, classes=None):
	internal_depths = [2,2,2,2]
	num_filters = [32, 64, 64, 128]
	depth = 4
	filters = []

	c1 = layers.Conv2D(20, kernel_size=3, padding="same", name = "Input1_conv0")(input_tensor[0])
	c1 = layers.BatchNormalization(name='Input1_BN')(c1)
	c1 = layers.Activation('relu', name='Input1_Activation')(c1)
	#b = input_tensor[0]
	c2 = complex_layers.ComplexConv2D(20, kernel_size=3, padding="same", name = "Input2_conv0")(input_tensor[1])
	c2 = layers.Activation(complex_activations.cart_relu, name='Input2_Activation')(c2)
	c1 = tf.cast(c1, dtype=tf.complex64)
	c = layers.concatenate([c1, c2], axis=-1)
	c = complex_layers.ComplexConv2D(32, kernel_size=1, name="data_fusion")(c)
	c = layers.Activation(complex_activations.cart_relu, name='data_fusion_Activation')(c)

	for i in range(depth-1):
		a = []
		for j in range(internal_depths[i]):
			a.append(num_filters[i])
		filters.append(a)

	concat_layers = []

	#Encode Part:
	c,conc = EncodeComplexFPB_Block(c, filters[0] , internal_depths[0], "EncodeBlock_lvl0", pool_size=(3, 3), strides=(3, 3))
	concat_layers.append(conc)
	for i in range(1, depth-1):
		c, conc = EncodeComplexFPB_Block(c, filters[i] , internal_depths[i], "EncodeBlock_lvl"+str(i))
		concat_layers.append(conc)

	for i in range(internal_depths[depth-1]):
		c = complex_layers.ComplexConv2D(num_filters[depth-1], kernel_size=3, padding="same", name = "EncodeBlock_lvl"+str(depth-1)+"_conv"+str(i))(c)
		c = complex_layers.ComplexDropout(0.3)(c)
		c = layers.Activation(complex_activations.cart_relu, name="EncodeBlock_lvl"+str(depth-1)+"_Activation"+str(i))(c)

	for i in range(depth-2,0,-1):
		c = DecodeComplexFPB_Block(c, concat_layers[i], filters[i], internal_depths[i], "DecodeBlock_lvl"+str(i))

	outputs1 = complex_layers.ComplexConv2D(classes, kernel_size=1, name="final_conv_30m")(c)

	c_int = layers.Activation(activation=complex_activations.sigmoid_real, name="output_sigmoid")(outputs1)
	upOut1 = complex_layers.ComplexUpSampling2D(size=(3,3), name = "Output1_Up")(c_int)
	upOut1 = tf.cast(upOut1, dtype=tf.complex64)
	outputs1 = layers.Activation(activation=complex_activations.softmax_real_with_abs, name="output30")(outputs1)

	c = DecodeComplexFPB_Block(c, concat_layers[0], filters[0], internal_depths[0], "DecodeBlock_lvl"+str(0), kernel_size=3, strides=(3,3))

	c = layers.concatenate([c, upOut1], axis=-1) #change in the re-entering of the class info from 30m + the first conv features from before the fusion
	outputs2 = complex_layers.ComplexConv2D(classes, kernel_size=1, name="final_conv")(c)
	outputs2 = layers.Activation(activation=complex_activations.softmax_real_with_abs, name="output10")(outputs2)
	model = tf.keras.Model(inputs=input_tensor, outputs=[outputs1, outputs2], name="u-net")
	return model

#Model to apply temperature scaling on the model labelsuperFPB_early2_mod
def uncertaintyTemperatureCalibration(weights, input_shape = None, input_tensor = None, classes=None):

	base_model = labelsuperFPB_early2_mod(input_shape = input_shape, input_tensor = input_tensor, classes = classes)
	base_model.trainable = False
	base_model.load_weights(weights, skip_mismatch=True, by_name=True)
	x = base_model.get_layer('final_conv').output

	c = customLayers.Temperature()

	x = c(x)

	output = layers.Activation(activation="softmax", name="output10")(x)

	model = tf.keras.Model(inputs=input_tensor, outputs=output, name="TemperatureScaling")
	return model