import tensorflow as tf
from tensorflow import keras
import customlayers

def AudioUNetLayers(inputLayer, halfFilters=False):
	layers = inputLayer

	print('Input shape: ', inputLayer.get_shape())

	n_filters = [128, 256, 512, 512, 512, 512, 512, 512]

	if halfFilters:
		n_filters = [64, 128, 256, 256, 256, 256, 256, 256]

	n_filtersizes = [65, 33, 17, 9, 9, 9, 9, 9, 9]

	numBlocks = 4

	for l, nf, fs in zip(range(numBlocks), n_filters, n_filtersizes):
		print('Layer: ', l, ' NFilters: ', nf, ' FSize: ', fs)

	for l, nf, fs in reversed(list(zip(range(numBlocks), n_filters, n_filtersizes))):
		print('Layer: ', l, ' NFilters: ', nf, ' FSize: ', fs)

	downsamplingLayers = []

	# Downsampling blocks
	# for i in range(numBlocks):
	for i, nf, fs in zip(range(numBlocks), n_filters, n_filtersizes):
		# numFilters = max(2**(6 + i), 512)
		# filterSize = min(2**(7 - i) + 1, 9)
		# print('DS Block ', i, ' Numfilters: ', numFilters, ' Filtersize: ', filterSize)

		layers = (keras.layers.Conv1D(
			filters=nf,
			kernel_size=fs,
			strides=2,
			activation=None,
			padding='same',
			kernel_initializer='orthogonal'
			#name='ds_conv_' + str(i)
		))(layers)

		layers = keras.layers.LeakyReLU(0.2)(layers)
		print('Downsampling Block: ', layers.get_shape())
		downsamplingLayers.append(layers)

	# Bottleneck block
	layers = (keras.layers.Conv1D(
		filters=512,
		kernel_size=9,
		strides=2,
		activation=None,
		padding='same',
		kernel_initializer='orthogonal'
		#name='bn_conv'
	))(layers)
	layers = keras.layers.Dropout(0.5)(layers)
	layers = keras.layers.LeakyReLU(0.2)(layers)

	# Upsampling layers
	# for i in range(numBlocks):
	for i, nf, fs, ds_l in reversed(list(zip(range(numBlocks), n_filters, n_filtersizes, downsamplingLayers))):
		# numFilters = max(2 ** (7 + (numBlocks - i)), 512)
		# filterSize = min(2 ** (7 - (numBlocks - i)) + 1, 9)
		# print('US Block ', i, ' Numfilters: ', numFilters, ' Filtersize: ', filterSize)

		layers = (keras.layers.Conv1D(
			filters=nf * 2,
			kernel_size=fs,
			activation=None,
			padding='same',
			kernel_initializer='orthogonal'
			#name='us_conv_' + str(i)
		))(layers)
		layers = keras.layers.Dropout(0.5)(layers)
		layers = keras.layers.Activation('relu')(layers)
		layers = customlayers.SubPixel1D(r=2)(layers)
		layers = keras.layers.concatenate([layers, ds_l], -1)
		print('Upsampling Block: ', layers.get_shape())

	# Final convolution layer
	layers = keras.layers.Conv1D(
		filters=2,
		kernel_size=9,
		padding='same',
		kernel_initializer='orthogonal'
		#name='final_conv'
	)(layers)
	layers = customlayers.SubPixel1D(r=2)(layers)
	print(layers.get_shape())

	outputs = keras.layers.add([layers, inputLayer])

	return outputs
