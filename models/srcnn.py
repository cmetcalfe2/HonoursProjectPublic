import tensorflow as tf
from tensorflow import keras
import customlayers

def SRCNNLayers(inputLayer):
	layers = inputLayer

	# Patch extraction
	layers = (keras.layers.Conv1D(
		filters=64,
		kernel_size=9,
		strides=1,
		activation=None,
		padding='same',
		kernel_initializer='random_normal',
		bias_initializer='zeros',
		use_bias=True,
		name='patch_extraction'
	))(layers)
	layers = (keras.layers.ReLU())(layers)

	# Non linear mapping
	layers = (keras.layers.Conv1D(
		filters=32,
		kernel_size=1,
		strides=1,
		activation=None,
		padding='same',
		kernel_initializer='random_normal',
		bias_initializer='zeros',
		use_bias=True,
		name='non_linear_extraction'
	))(layers)
	layers = (keras.layers.ReLU())(layers)

	# Reconstruction
	layers = (keras.layers.Conv1D(
		filters=1,
		kernel_size=5,
		strides=1,
		activation=None,
		padding='same',
		kernel_initializer='random_normal',
		bias_initializer='zeros',
		use_bias=True,
		name='reconstruction'
	))(layers)

	return layers

#
# inputLayer = keras.Input(shape=(None, 1))
# output = SRCNNLayers(inputLayer)
# model = keras.models.Model(inputs=inputLayer, outputs=output)
# model.compile(optimizer=tf.train.AdamOptimizer(0.003),
# 				  loss='mean_squared_error',
# 				  metrics=['accuracy'])
#
# randTensor = tf.random.uniform([200, 1, 1], seed=125357134)
# prediction = model.predict(randTensor, steps=1)
# print(len(prediction))
# print(prediction[0])