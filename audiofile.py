import wave
import filehelper as fh
import numpy as np
import math
import csv


# Number of samples to input and output at once to neural network
inputRowSize = 8192
stride = 2048


# Class for storing audio samples as integers
class AudioSamples:
	def __init__(self, numChannels, sampleWidth, sampleRate, numSamplesPerChannel):
		self.numChannels = numChannels
		self.sampleWidth = sampleWidth
		self.sampleRate = sampleRate
		self.numSamplesPerChannel = numSamplesPerChannel

		intType = ''
		if sampleWidth == 1:
			intType = 'Int8'
		elif sampleWidth == 2:
			intType = 'Int16'
		else:
			RuntimeError('Creating audio sample object with unsupported bit depth (' + sampleWidth + ')')

		self.samples = np.empty((numChannels, numSamplesPerChannel), dtype=intType)


# Verify all .wav files in directory are using same sample rate
def verifySampleRate(wavFilesPath):
	wavFilePaths = fh.findFilesWithExtensionRecursive(wavFilesPath, ".wav")
	firstSampleRate = None

	for path in wavFilePaths:
		with wave.open(path, 'rb') as file:
			sampleRate = file.getframerate()
			if firstSampleRate is None:
				firstSampleRate = sampleRate
			elif sampleRate != firstSampleRate:
				return False

	return True


# Get all samples from a wav file
def getSamples(path):

	with wave.open(path, 'rb') as file:

		rawSamples = file.readframes(-1)
		sampleWidth = file.getsampwidth()
		numChannels = file.getnchannels()
		sampleRate = file.getframerate()

		intType = ''
		if sampleWidth == 1:
			intType = 'Int8'
		elif sampleWidth == 2:
			intType = 'Int16'
		else:
			RuntimeError('Reading unsupported bit depth (' + sampleWidth + ')')

		rawSamples = np.fromstring(rawSamples, dtype=intType)
		numSamplesPerChannel = int(len(rawSamples) / numChannels)

		audioSamples = AudioSamples(numChannels, sampleWidth, sampleRate, numSamplesPerChannel)

		'''
		channelSampleIndex = 0
		for index, sample in enumerate(rawSamples):
			audioSamples.samples[index % numChannels][channelSampleIndex] = sample
			channelSampleIndex += index % numChannels
		'''
		audioSamples.samples = rawSamples.reshape((numChannels, numSamplesPerChannel), order='F')

		return audioSamples


def downsample(inputSamples, downsamplingRate):
	downsampledSamples = AudioSamples(inputSamples.numChannels, inputSamples.sampleWidth, inputSamples.sampleRate / 2, math.ceil(inputSamples.numSamplesPerChannel / 2))
	for channelIndex in range(len(inputSamples.samples)):
		sampleIndex = 0
		while sampleIndex < len(inputSamples.samples[channelIndex]):
			downsampledSamples.samples[channelIndex][int(sampleIndex / downsamplingRate)] = inputSamples.samples[channelIndex][sampleIndex]
			sampleIndex += downsamplingRate

	return downsampledSamples


# Write samples to .wav file
def writeWavFile(outPath, samples):
	with wave.open(outPath, 'wb') as file:
		compType = "NONE"
		compName = "not compressed"
		file.setparams((samples.numChannels, samples.sampleWidth, samples.sampleRate, samples.numSamplesPerChannel * samples.numChannels, compType, compName))

		# Write samples into file after converting array bytes, F specifies column-major flattening of multi-dimensional arrays
		file.writeframes(samples.samples.tostring('F'))


# Split sample data into rows
def splitSamples(inputSamples, lineSplit, sameSize=True):
	numLines = int(math.floor(inputSamples.numSamplesPerChannel / lineSplit))
	rows = []
	for channelIndex in range(inputSamples.numChannels):
		for lineIndex in range(numLines):
			row = []
			row.extend(inputSamples.samples[channelIndex][lineIndex * lineSplit:(lineIndex * lineSplit) + lineSplit])
			rows.append(row)

	if not sameSize:
		for channelIndex in range(inputSamples.numChannels):
			numRows = len(rows)
			lastRow = []
			lastRow.extend(inputSamples.samples[channelIndex][numRows * lineSplit:])
			if len(lastRow) > 0:
				rows.append(lastRow)

	return rows


# Convert samples to CSV text
def appendSamplesToCSV(filePath, inputSamples, outputSamples):
	with open(filePath, 'a') as csvFile:
		csvwriter = csv.writer(csvFile, delimiter=',', lineterminator='\n')

		for channelIndex in range(inputSamples.numChannels):

			curStride = 0
			while (curStride + inputRowSize) < len(inputSamples.samples[channelIndex]):
				csvRow = []
				csvRow.extend(inputSamples.samples[channelIndex][curStride : curStride + inputRowSize])
				csvRow.extend(outputSamples.samples[channelIndex][int(curStride / 2) : int(curStride / 2) + int(inputRowSize / 2)])
				csvwriter.writerow(csvRow)
				curStride += stride

