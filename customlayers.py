import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys


class SubPixel1D(keras.layers.Layer):

	def __init__(self, r):
		super(SubPixel1D, self).__init__()

		self.r = r

	def call(self, input):
		x = tf.transpose(input, [2, 1, 0])  # (r, w, b)
		x = tf.batch_to_space_nd(x, [self.r], [[0, 0]])  # (1, r*w, b)
		x = tf.transpose(x, [2, 1, 0])
		return x


class SpectralReplicatorLayer(keras.layers.Layer):

	def __init__(self, batchSize, inputSampleLength):
		super(SpectralReplicatorLayer, self).__init__(name='SpectralReplicator')
		self.batchSize = batchSize
		self.inputSampleLength = inputSampleLength

	def call(self, input):
		# 2x upsampling, copy first half to last half
		firstHalf = tf.slice(input, [0, 0, 0], [self.batchSize, int(self.inputSampleLength / 4), 1])
		output = tf.tile(firstHalf, [1, 2, 1])
		print('Spectral Replicator Shape: ', output.get_shape())
		return output


class SpectralFusionLayer(keras.layers.Layer):

	def __init__(self, batchSize, sampleInputLength):
		super(SpectralFusionLayer, self).__init__(name='SpectralFusion')
		self.batchSize = batchSize
		self.sampleInputLength = sampleInputLength

	def build(self, input_shape):
		assert isinstance(input_shape, list)
		#print(input_shape)
		self.trainable_parameter = self.add_variable("w", shape=[1, int(self.sampleInputLength / 2) + 1], dtype=tf.float32)
		self.ones = tf.ones([1, int(self.sampleInputLength / 2) + 1])

		super(SpectralFusionLayer, self).build(input_shape)

	# def angle(self, z):
	# 	x = tf.real(z)
	# 	y = tf.imag(z)
	# 	x_neg = tf.cast(x < 0.0, tf.complex64)
	# 	y_neg = tf.cast(y < 0.0, tf.complex64)
	# 	y_pos = tf.cast(y >= 0.0, tf.complex64)
	# 	offset = x_neg * (y_pos - y_neg) * np.pi
	# 	return tf.atan(y / x) + offset

	def call(self, input):
		assert isinstance(input, list)

		# m = fft
		# z = raw samples
		m, z = input

		# m and z will be of shape (numSamples, 1, 1), needs to be (1, 1, numSamples) for rfft
		mReshaped = tf.reshape(m, [self.batchSize, 1, -1])
		zReshaped = tf.reshape(z, [self.batchSize, 1, -1])
		zFourierTransform = tf.spectral.rfft(zReshaped)

		M = tf.math.multiply(self.trainable_parameter, tf.math.abs(zFourierTransform))
		M = M + (tf.math.multiply((self.ones - self.trainable_parameter), mReshaped))

		#highResSignal = tf.spectral.irfft(M * (np.e ** (self.angle(zFourierTransform))))

		# Convert polar complex numbers to rectangular
		angles = tf.math.angle(zFourierTransform)
		realComponents = tf.math.multiply(M, tf.math.cos(angles))
		imaginaryComponents = tf.math.multiply(M, tf.math.sin(angles))

		complexNumbers = tf.complex(realComponents, imaginaryComponents)

		highResSignal = tf.spectral.irfft(complexNumbers)

		# Reshape highResSignal to same as input layer
		highResSignalReshaped = tf.reshape(highResSignal, [self.batchSize, -1, 1])

		return highResSignalReshaped


class RealFFTMagnitudeLayer(keras.layers.Layer):

	def __init__(self, batchSize):
		super(RealFFTMagnitudeLayer, self).__init__(name='RealFFTMag')
		self.batchSize = batchSize

	def call(self, input):
		# Input will be in shape (batchSize, numSamples, 1), needs to be (batchSize, 1, numSamples) (RFFT works on inner dimension)
		inputReshaped = tf.reshape(input, [self.batchSize, 1, -1])  # -1 in shape keeps total size the same

		# Calculate real-valued fourier transform
		ft = tf.spectral.rfft(inputReshaped)

		# Calculate magnitude
		mag = tf.math.abs(ft)

		# Convert output back to (batchsize, numsamples, 1)
		magReshaped = tf.reshape(mag, [self.batchSize, -1, 1])

		# Slice off 0th component
		dcComponent = tf.slice(magReshaped, [0, 0, 0], [self.batchSize, 1, 1])
		otherComponents = tf.slice(magReshaped, [0, 1, 0], [self.batchSize, -1, 1])

		print('DC shape: ', magReshaped.get_shape())
		print('Other shape: ', otherComponents.get_shape())

		return dcComponent, otherComponents


class SqueezeLayer(keras.layers.Layer):

	def __init__(self, batchSize):
		super(SqueezeLayer, self).__init__(name='Squeeze')
		self.batchSize = batchSize

	def call(self, input):
		output = tf.squeeze(input)
		return tf.reshape(output, [self.batchSize, -1])

class ExpansionLayer(keras.layers.Layer):

	def __init__(self, batchSize):
		super(ExpansionLayer, self).__init__(name='Expand')
		self.batchSize = batchSize

	def call(self, input):
		return tf.reshape(input, [self.batchSize, -1, 1])


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
