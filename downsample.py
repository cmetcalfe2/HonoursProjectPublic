# Downsamples a single .wav file

import argparse
import os
import audiofile as af
import filehelper as fh

argParser = argparse.ArgumentParser(description='Downsample a single .wav file')

argParser.add_argument('-f', '--inputpath', help='Path to .wav file', required=True, type=str)
argParser.add_argument('-dsr', '--downsamplingrate', help='Rate of downsampling, default is 2', type=int, default=2)

args = vars(argParser.parse_args())

inputSamples = af.getSamples(args['inputpath'])
outputSamples = af.downsample(inputSamples, int(args['downsamplingrate']))

inputPath = os.path.split(args['inputpath'])
outputPath = os.path.join(inputPath[0], inputPath[1].replace('.wav', 'downsampled' + str(args["downsamplingrate"]) + 'x.wav', 1))

af.writeWavFile(outputPath, outputSamples)