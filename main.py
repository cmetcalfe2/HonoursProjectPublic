import tensorflow as tf
from tensorflow import keras
import audiofile as af
import argparse
import csv
import numpy as np
import math
import os
import time
from models import audiounet
from models import tfnet
from models import srcnn
from models import densenet
import customlayers
import samplerate as sr
from scipy import interpolate


batchSize = 64

def saveModel(model, filepath):
	NotImplementedError()


# Expands array by factor of 2 and interpolates missing values with linear interpolation
def expandAndInterpolateArray(inputArray):
	upsampleRate = 2

	lowRes = inputArray
	highResLength = len(lowRes) * 2
	highRes = np.zeros(highResLength)

	i_lr = np.arange(highResLength, step=2)
	i_hr = np.arange(highResLength)

	spline = interpolate.splrep(i_lr, lowRes)
	highRes = interpolate.splev(i_hr, spline)

	return highRes


# Convert CSV dataset to TFRecord for easier parsing
def convertCSVDataset(inputCSVPath, outputTFPath, trainValSplit):
	os.makedirs(outputTFPath, exist_ok=True)
	numTrainingRows = 0
	numValidationRows = 0

	with open(inputCSVPath, 'r') as csvFile:

		csvReader = csv.reader(csvFile, delimiter=',')
		csvRowCount = sum(1 for row in csvReader)
		numTrainingRows = int(math.floor(csvRowCount * trainValSplit))
		numValidationRows = csvRowCount - numTrainingRows

	with open(os.path.join(outputTFPath, 'datasetinfo.txt'), 'w') as infoFile:
		infoFile.write(str(numTrainingRows) + '\n' + str(numValidationRows))

	with open(inputCSVPath, 'r') as csvFile:

		csvReader = csv.reader(csvFile, delimiter=',')

		trainingTFWriter = tf.python_io.TFRecordWriter(os.path.join(outputTFPath, 'trainingdataset.tf'))
		validationTFWriter = tf.python_io.TFRecordWriter(os.path.join(outputTFPath, 'validationdataset.tf'))

		rowIndex = 0
		for row in csvReader:

			originalSamplesArray = np.array([ int(sample) for sample in row[0:af.inputRowSize] ])
			originalSamples = originalSamplesArray.astype(np.float32).tostring()

			downsampledSamplesArray = np.array([ int(sample) for sample in row[af.inputRowSize:af.inputRowSize + int(af.inputRowSize / 2) ] ])
			downsampledSamplesArray = expandAndInterpolateArray(downsampledSamplesArray.astype(np.float32))
			downsampledSamples = downsampledSamplesArray.astype(np.float32).tostring()

			example = tf.train.Example()
			example.features.feature['original_samples'].bytes_list.value.append(originalSamples)
			example.features.feature['downsampled_samples'].bytes_list.value.append(downsampledSamples)

			if rowIndex < numTrainingRows:
				trainingTFWriter.write(example.SerializeToString())
			else:
				validationTFWriter.write(example.SerializeToString())

			rowIndex += 1

		trainingTFWriter.close()
		validationTFWriter.close()


def parseDatasetRecord(record):

	features = {
		'original_samples': tf.FixedLenFeature([], tf.string),
		'downsampled_samples': tf.FixedLenFeature([], tf.string)
	}

	parsed = tf.parse_single_example(record, features)

	originalSamples = tf.convert_to_tensor(tf.decode_raw(parsed['original_samples'], tf.float32))
	downsampledSamples = tf.convert_to_tensor(tf.decode_raw(parsed['downsampled_samples'], tf.float32))

	originalSamples = tf.reshape(originalSamples, [af.inputRowSize, 1])
	downsampledSamples = tf.reshape(downsampledSamples, [af.inputRowSize, 1])

	return downsampledSamples, originalSamples


def loadTFDataset(path):
	dataset = tf.data.TFRecordDataset(path).map(parseDatasetRecord)
	dataset = dataset.shuffle(1000)
	dataset = dataset.repeat()
	dataset = dataset.batch(batchSize)
	print('Dataset output types: ', dataset.output_types)
	print('Dataset output shapes: ', dataset.output_shapes)
	return dataset.repeat()


def getModelH5Path(modelName):
	return 'trainedmodels/' + modelName + '.h5'


def compileModel(modelNumber):

	inputLayer = keras.Input(shape=(8192, 1), batch_size=batchSize)
	outputs = None

	if modelNumber == 0:
		outputs = audiounet.AudioUNetLayers(inputLayer, halfFilters=False)
	elif modelNumber == 1:
		outputs = tfnet.TFNetLayers(inputLayer, batchSize, 8192)
	elif modelNumber == 2:
		outputs = srcnn.SRCNNLayers(inputLayer)
	elif modelNumber == 3:
		outputs = densenet.DenseNetLayers(inputLayer, batchSize, 8192)

	model = keras.Model(inputs=inputLayer, outputs=outputs)

	model.compile(optimizer=tf.train.AdamOptimizer(0.001),
				  loss='mean_squared_error',
				  metrics=['accuracy'])

	model.summary()

	return model


def trainModel(modelIndex, inputTFPath, baseCheckpointDir, trainingTask, numEpochs):

	model = compileModel(modelIndex)

	trainData = loadTFDataset(os.path.join(inputTFPath, 'trainingdataset.tf'))
	validationData = loadTFDataset(os.path.join(inputTFPath, 'validationdataset.tf'))

	numTrainSteps = 0
	numValidationSteps = 0

	with open(os.path.join(inputTFPath, 'datasetinfo.txt'), 'r') as file:
		contents = file.read().splitlines()
		numTrainSteps = int(int(contents[0]) / batchSize)
		numValidationSteps = int(int(contents[1]) / batchSize)

	os.makedirs(baseCheckpointDir + str(modelIndex) + trainingTask, exist_ok=True)

	checkpointDir = baseCheckpointDir + str(modelIndex) + trainingTask + '/cp-{epoch:04d}.ckpt'
	checkpointCallback = tf.keras.callbacks.ModelCheckpoint(
		checkpointDir,
		verbose=1,
		save_weights_only=True
	)

	initialEpoch = 0
	if args['resume'] is not None:
		initialEpoch = args['resume']
		model.load_weights(checkpointDir.format(epoch=initialEpoch))

	model.fit(trainData, epochs=numEpochs, verbose=1, validation_data=validationData, steps_per_epoch=numTrainSteps, validation_steps=numValidationSteps, callbacks=[checkpointCallback], initial_epoch=initialEpoch)


def upsampleWAV(modelIndex, baseCheckpointDir, trainingTask, epoch, inputWAVPath):

	model = compileModel(modelIndex)

	checkpointDir = baseCheckpointDir + str(modelIndex) + trainingTask + '/cp-{epoch:04d}.ckpt'
	model.load_weights(checkpointDir.format(epoch=epoch))

	originalSamples = af.getSamples(inputWAVPath)

	originalSampleRate = originalSamples.sampleRate
	upsampledSampleRate = originalSampleRate * 2

	sampleWidth = originalSamples.sampleWidth

	upsampledSamples = af.AudioSamples(originalSamples.numChannels, sampleWidth, upsampledSampleRate, originalSamples.numSamplesPerChannel * 2)

	inputSampleSets = af.splitSamples(originalSamples, int(af.inputRowSize / 2), sameSize=False)
	numSampleSetsPerChannel = int(len(inputSampleSets) / originalSamples.numChannels)

	numBatches = int(math.ceil(len(inputSampleSets) / batchSize))

	print('Creating input batches')

	inputBatches = []

	curSampleSet = 0
	for batchIndex in range(numBatches):
		batch = np.zeros([batchSize, af.inputRowSize])
		for batchSampleSetIndex in range(batchSize):
			if curSampleSet < len(inputSampleSets):
				sampleSet = np.array([int(sample) for sample in inputSampleSets[curSampleSet]]).astype(np.float32)
				sampleSet = expandAndInterpolateArray(sampleSet)
				sampleSet = np.pad(sampleSet, (0, af.inputRowSize - len(sampleSet)), 'constant')
				batch[batchSampleSetIndex] = sampleSet

			curSampleSet = curSampleSet + 1

		print(len(batch))
		batch = tf.convert_to_tensor(batch, dtype=tf.float32)
		batch = tf.reshape(batch, [batchSize, af.inputRowSize, 1])

		inputBatches.append(batch)

	outputBatches = []

	sampleDataType = None
	if sampleWidth == 1:
		sampleDataType = np.int8
	elif sampleWidth == 2:
		sampleDataType = np.int16
	else:
		RuntimeError('Upsampling audio with unsupported bit depth (' + sampleWidth + ')')

	startPredictionTime = time.time()

	print('Predicting')

	numPredictions = 0
	for batch in inputBatches:
		prediction = model.predict(batch, steps=1)
		numPredictions = numPredictions + 1
		print('Predicted: ', numPredictions)
		print(len(prediction))

		outputBatches.append(prediction.astype(sampleDataType))

	#numSteps = int(math.ceil(len(inputBatches)/batchSize))

	#predictions = model.predict(inputBatches, steps=numSteps, batch_size=batchSize)

	endPredictionTime = time.time()

	print('Upsampling prediction took ', endPredictionTime - startPredictionTime, ' seconds')
	print('Writing WAV file...')

	# Flatten output
	outputSamples = []
	curSampleSet = 0
	for batchIndex in range(len(outputBatches)):
		for sampleSetIndex in range(batchSize):
			if curSampleSet < len(inputSampleSets):
				outputSamples.extend(outputBatches[batchIndex][sampleSetIndex])
				curSampleSet = curSampleSet + 1
			else:
				break

	outputSamples = [x for y in outputSamples for x in y]

	# Create upsampled wav
	for channelIndex in range(originalSamples.numChannels):
		sampleStartIndex = channelIndex * int(len(outputSamples) / upsampledSamples.numChannels)
		channelSamples = outputSamples[sampleStartIndex : sampleStartIndex + upsampledSamples.numSamplesPerChannel]
		#print(channelSamples)
		upsampledSamples.samples[channelIndex] = channelSamples


	splitInputPath = os.path.split(inputWAVPath)
	newFileName = splitInputPath[1].replace('.wav', '-upsampled2x_' + str(modelIndex) + trainingTask + '.wav')
	outputPath = os.path.join(splitInputPath[0], newFileName)

	af.writeWavFile(outputPath, upsampledSamples)


def evaluate(modelName, inputWAVPath):
	return NotImplementedError()


# Create required directories
os.makedirs('modelconfigs', exist_ok=True)
os.makedirs('trainedmodels', exist_ok=True)

argParser = argparse.ArgumentParser(description='Neural net audio upsampler')

argParser.add_argument('-csv', '--csvdata', help='Load CSV file to convert to TFRecord dataset', metavar='CSVPATH', type=str)
argParser.add_argument('-d', '--datainput', help='The directory containing the .tf datasets to use as input for training, or output when converting from CSV', metavar='TFPATH', type=str)
argParser.add_argument('-tvs', '--trainvalsplit', help='Ratio of training data to validation data to use when converting dataset, default is 0.8', type=float, metavar='RATIO', default=0.8)
argParser.add_argument('-c', '--compile', help='Compiles and saves the currently defined model as JSON file', metavar='MODELNAME')
argParser.add_argument('-t', '--train', help='Start training the selected model using the created dataset', metavar='MODELINDEX', type=int)
argParser.add_argument('-r', '--resume', help='Whether to resume training from an existing checkpoint', metavar='EPOCHTORESUMEFROM', type=int)
argParser.add_argument('-cd', '--checkpointdir', help='Base directory for checkpoint files', metavar='PATH', type=str, default='checkpoints/')
argParser.add_argument('-tt', '--traintask', help='The name of this training task', metavar='TRAINTASK', type=str)
argParser.add_argument('-te', '--trainepochs', help='Number of epochs to train for', metavar='NUMEPOCHS', type=int, default=5)
argParser.add_argument('-u', '--upsample', help='Upsample a .wav file with the selected model', metavar='MODELINDEX', type=int)
argParser.add_argument('-wf', '--wavfile', help='Upsample a .wav file with the selected model', metavar='WAVPATH',)

args = vars(argParser.parse_args())

conf = tf.ConfigProto()
conf.gpu_options.allow_growth = True

#sess = tf.Session(config=conf)
#keras.backend.set_session(sess)

if args['csvdata'] is not None:
	convertCSVDataset(args['csvdata'], args['datainput'], args['trainvalsplit'])

if args['compile'] is not None:
	compileModel(args['compile'])

if args['train'] is not None:
	trainModel(args['train'], args['datainput'], args['checkpointdir'], args['traintask'], args['trainepochs'])

if args['upsample'] is not None:
	upsampleWAV(args['upsample'], args['checkpointdir'], args['traintask'], args['trainepochs'], args['wavfile'])

