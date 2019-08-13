import argparse
import os
import audiofile as af
import filehelper as fh

argParser = argparse.ArgumentParser(description='Downsample .wav files and create dataset for training and evaluation')

argParser.add_argument('-i', '--inputpath', help='Path where .wav files are located', required=True, type=str)
argParser.add_argument('-o', '--outputcsvpath', help='Path of output downsampled audio samples CSV file', required=False, type=str)

argParser.add_argument('-v', '--verify', action='store_true', help='Verify all .wav files in the path use the same sample rate')
argParser.add_argument('-s', '--save', action='store_true', help='Save downsampled audio files')
argParser.add_argument('-dsr', '--downsamplingrate', help='Rate of downsampling, default is 2', type=int, default=2)
#argParser.add_argument('-rs', '--rowsplit', help='Number of samples per row, downsampled rows will be this size divided by downsampling rate, default is 1000', type=int, default=1000)

args = vars(argParser.parse_args())

if args["verify"]:
	if not af.verifySampleRate(args["inputpath"]):
		RuntimeError('Sample rate is not consistent!')

# Create downsampled files directory
if args['save']:
	inputPath = os.path.split(args['inputpath'])
	downsampledPath = os.path.join(inputPath[0], inputPath[1] + 'downsampled' + str(args["downsamplingrate"]) + 'x')
	os.makedirs(downsampledPath, exist_ok=True)

inputFiles = fh.findFilesWithExtensionRecursive(args['inputpath'], '.wav')

for fileIndex in range(len(inputFiles)):
	wavFilePath = inputFiles[fileIndex]
	inputSamples = af.getSamples(wavFilePath)
	outputSamples = af.downsample(inputSamples, int(args['downsamplingrate']))

	af.appendSamplesToCSV(args['outputcsvpath'], inputSamples, outputSamples)

	if args['save']:
		af.writeWavFile(wavFilePath.replace(args['inputpath'], downsampledPath, 1), outputSamples)

	print('Downsampled file ' + str(fileIndex + 1) + '/' + str(len(inputFiles)))
