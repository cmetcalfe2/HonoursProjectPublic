import tensorflow as tf
from tensorflow import keras
import customlayers

def DenseNetLayers(inputLayer, batchSize, inputSampleLength):
	layers = inputLayer

	# Squeeze to 2 dimensions
	layers = customlayers.SqueezeLayer(batchSize)(layers)

	numBlocks = 3

	for block in range(numBlocks):
		layers = keras.layers.Dense(inputSampleLength)(layers)
		layers = keras.layers.ReLU()(layers)
		layers = keras.layers.BatchNormalization()(layers)

	# Output layer
	layers = customlayers.ExpansionLayer(batchSize)(layers)

	return layers

