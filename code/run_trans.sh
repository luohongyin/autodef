#!/bin/sh
set -ex

if [ $# -ne 6 ];
then
	echo "usage: $0 [source input] [output] [model_state] [model file] [beam size] [cpu|gpu0|gpu1|..]"
	exit 1
fi

SOURCE=$1
OUTPUT=$2
MODEL_STATE=$3
MODEL=$4
BEAM_SIZE=$5
GPU=$6
CODE=/home/halidan/rnn/rnnBaseline/trunk

PYTHONPATH=$CODE THEANO_FLAGS=floatX=float32,device=$GPU python $CODE/experiments/nmt/sample.py --state $MODEL_STATE --beam-search --beam-size $BEAM_SIZE --source $SOURCE --trans $OUTPUT $MODEL
