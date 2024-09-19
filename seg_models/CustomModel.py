import numpy as np
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow_probability as tfp



'''
Implementaton of the FPB_Block as a class layer. This is not used in the final model
'''
class FPB_Block (layers.Layer):
	def __init__(self, num_filters, dropout_rate, name):
		super(FPB_Block, self).__init__(name=name)
		self.num_filters=num_filters
		self.dropout_rate=dropout_rate

	def get_config(self):
		config = super().get_config()
		config.update({
			"num_filters": self.num_filters,
			"dropout_rate": self.dropout_rate,
		})
		return config

	@classmethod
	def from_config(cls, config):
		return cls(**config)

	def build(self, input_shape):

		last_filters = input_shape[-1]
		self.path1 = tf.keras.Sequential([layers.Conv2D(self.num_filters, kernel_size=1, padding="same", name = self.name+"_conv1"),layers.BatchNormalization(name=self.name+"_BN1")])
		self.path2 = tf.keras.Sequential([layers.Conv2D(self.num_filters, kernel_size=3, padding="same", name = self.name+"_conv3"),layers.BatchNormalization(name=self.name+"_BN2")])
		self.path3 = tf.keras.Sequential([layers.Conv2D(self.num_filters, kernel_size=5, padding="same", name = self.name+"_conv5"),layers.BatchNormalization(name=self.name+"_BN3")])
		self.convConcat = tf.keras.Sequential([layers.Conv2D(last_filters, kernel_size=1, padding="same", name = self.name+"_convOut"), layers.BatchNormalization(name=self.name+"_BNOut")])

	@tf.function
	def call(self, inputs):

		c1 = self.path1(inputs)
		c1 = layers.Activation('relu', name=self.name+"_Activation1")(c1)
		c1 = layers.Dropout(self.dropout_rate)(c1)
		c2 = self.path2(inputs)
		c2 = layers.Activation('relu', name=self.name+"_Activation2")(c2)
		c2 = layers.Dropout(self.dropout_rate)(c2)
		c3 = self.path3(inputs)
		c3 = layers.Activation('relu', name=self.name+"_Activation3")(c3)
		c3 = layers.Dropout(self.dropout_rate)(c3)
		c = layers.concatenate([c1, c2, c3], axis=-1)
		c = self.convConcat(c)
		c = layers.Activation('relu', name=self.name+"_ActivationOut")(c)
		c = layers.Dropout(self.dropout_rate)(c)

		return layers.Add()([c,inputs])



'''
Attempt to create a model that is trained and validated on MCDropOut. The idea was that the feedforward during training would already be on multiple executions and the result averaged on it to create the prediction for the loss function comparison.
This idea was not developed further, but had basis on a work in the literature applied to medical images.
'''
class MCDropOutModel(Model):

	def __init__(self, num_classes, num_blocks=5, filters = [8,8,8,8,8,16], num_iter=5, dropout_rate=0.3, name='FPB_Network'):
		super(MCDropOutModel, self).__init__(name=name)
		self.num_iter=num_iter
		self.num_classes=num_classes
		self.num_blocks = num_blocks
		self.filters = filters
		self.dropout_rate = dropout_rate


	def get_config(self):
		config = super().get_config()
		config.update({
			"num_classes": self.num_filters,
			"dropout_rate": self.dropout_rate,
			"num_iter": self.num_iter,
			"num_blocks": self.num_blocks,
			"filters": self.filters,
		})
		return config

	@classmethod
	def from_config(cls, config):
		return cls(**config)

	def build(self, inputs_shape):

		self.SARInput = tf.keras.Sequential([layers.Conv2D(self.filters[0], kernel_size=3, padding="same", name = self.name+"_convInit_SAR"),layers.BatchNormalization(name=self.name+"_BN_SAR")])
		self.MSIInput = tf.keras.Sequential([layers.Conv2D(self.filters[0], kernel_size=3, padding="same", name = self.name+"_convInit_MSI"),layers.BatchNormalization(name=self.name+"_BN_MSI")])

		blocks = []
		for i in range(self.num_blocks):
			blocks.append(FPB_Block(num_filters=self.filters[i+1], dropout_rate=self.dropout_rate, name=self.name+'_FPB_Block'+str(i+1)))

		self.FPB_Blocks = tf.keras.Sequential(blocks)
		self.classification = tf.keras.Sequential([layers.Conv2D(self.num_classes, kernel_size=1, name=self.name+"_final_conv"),layers.Activation(activation="softmax", name=self.name+"_output")])

	def call(self, inputs, training=False):


		c1 = self.SARInput(inputs[1])
		c1 = layers.Activation('relu', name=self.name+"_Activation_SAR")(c1)
		c1 = layers.Dropout(self.dropout_rate)(c1)
		c2 = self.MSIInput(inputs[0])
		c2 = layers.Activation('relu', name=self.name+"_Activation_MSI")(c2)
		c2 = layers.Dropout(self.dropout_rate)(c2)
		c = layers.concatenate([c1, c2], axis=-1)
		c = self.FPB_Blocks(c)

		return self.classification(c)


	@tf.function
	def loop_data(self, x):
		return self(x,training=True)

	@tf.function
	def train_step(self,data):
		x,y=data

		#forward pass
		with tf.GradientTape() as tape:
			
			sar = x[1]
			msi = x[0]
			sar = tf.expand_dims(sar, axis=0)
			msi = tf.expand_dims(msi, axis=0)
			multiplyer = tf.constant([self.num_iter,1,1,1,1])
			sar = tf.tile(sar,multiplyer)
			msi = tf.tile(msi,multiplyer)
			x = [msi,sar]
			y_pred_d = tf.vectorized_map(fn=self.loop_data,elems=x)

			y_pred_mean = tf.math.reduce_mean(y_pred_d,axis=0)
			y_pred_std = tf.math.reduce_std(y_pred_d,axis=0)

			loss = self.compiled_loss(y, y_pred_mean, regularization_losses=self.losses)
	
		#compute gradients:
		trainable_vars = self.trainable_variables
		gradients = tape.gradient(loss, trainable_vars)

		#update weights:
		self.optimizer.apply_gradients(zip(gradients, trainable_vars))

		#update metrics:
		self.compiled_metrics.update_state(y, y_pred_mean)

		#return a dict mapping metric names to current value:
		return {m.name: m.result() for m in self.metrics}


'''
Model Iterative Trimmed Loss,based on filtering training data that is understood as noisy based on the loss during training.
Based on: http://proceedings.mlr.press/v97/shen19e/shen19e.pdf
This model was trained and produced good results, but needs to be further inspected and optimized.
'''
class ModelITLM(Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        _len = y[0].get_shape()[0]
        _alpha = 0.95
        _idx = 5 #mudar depois
		# forward for get loss from every samples
        y_pred = self(x, training=False)  # Forward pass
        loss_ITLM = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        loss1 = loss_ITLM(y[0], y_pred[0])
        loss1 = tf.math.reduce_sum(loss1, axis=-1)
        loss1 = tf.math.reduce_sum(loss1, axis=-1)
        loss2 = loss_ITLM(y[1], y_pred[1])
        loss2 = tf.math.reduce_sum(loss2, axis=-1)
        loss2 = tf.math.reduce_sum(loss2, axis=-1)
        loss = loss1+loss2
        argsort = tf.argsort(loss)
        # get new training data
        a = tf.gather(x[0], argsort[:_idx])
        b = tf.gather(x[1], argsort[:_idx])
        x = [a,b]
        a = tf.gather(y[0], argsort[:_idx])
        b = tf.gather(y[1], argsort[:_idx])
        y = [a,b]
        with tf.GradientTape() as tape:
           y_pred = self(x, training=True)  # Forward pass
           # Compute the loss value
           # (the loss function is configured in `compile()`)
           loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


'''
Model Iterative Trimmed Loss,based on filtering training data that is understood as noisy based on the loss during training.
This is an evolution of the previous model for Segmentation and based on percentile to define the threshold to filter data.
Based on: http://proceedings.mlr.press/v97/shen19e/shen19e.pdf
This model was trained and produced good results, but needs to be further inspected and optimized.
'''
class ModelITLMSeg(Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        self.t = 0
        T=400
        _alpha = 0.907
        # forward for get loss from every samples
        if self.t>T:
	        y_pred = self(x, training=False)  # Forward pass
	        loss_ITLM = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
	        loss1 = loss_ITLM(y[0], y_pred[0])
	        loss2 = loss_ITLM(y[1], y_pred[1])
	        percentil1 = tfp.stats.percentile(loss1, 100*_alpha, interpolation='midpoint')
	        percentil2 = tfp.stats.percentile(loss2, 100*_alpha, interpolation='midpoint')
	        # get new training data
	        a = loss1<percentil1
	        a = tf.cast(a, tf.float32)
	        a = tf.expand_dims(a,axis=-1)
	        #x[0] = x[0]*a
	        a = layers.multiply([y[0],a])
	        b = loss2<percentil2
	        b = tf.cast(b, tf.float32)
	        b = tf.expand_dims(b,axis=-1)
	        #x[1] = x[1]*b
	        b = layers.multiply([b, y[1]])
	        y = [a, b]
        self.t=self.t+1
        with tf.GradientTape() as tape:
           y_pred = self(x, training=True)  # Forward pass
           # Compute the loss value
           # (the loss function is configured in `compile()`)
           loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}