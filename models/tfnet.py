import tensorflow as tf
from tensorflow import keras
import customlayers
from models.audiounet import AudioUNetLayers

def TFNetLayers(inputLayer, batchSize, sampleInputLength):
	realFFT = customlayers.RealFFTMagnitudeLayer(batchSize)(inputLayer)
	frequencyBranch = customlayers.SpectralReplicatorLayer(batchSize, sampleInputLength)(realFFT[1])
	frequencyBranch = AudioUNetLayers(frequencyBranch)
	frequencyBranch = keras.layers.concatenate([realFFT[0], frequencyBranch], -2)

	timeBranch = AudioUNetLayers(inputLayer)

	fusionLayer = customlayers.SpectralFusionLayer(batchSize, sampleInputLength)([frequencyBranch, timeBranch])

	return fusionLayer
