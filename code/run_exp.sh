#!/bin/sh
set -ex

if [ $# -ne 2 ];
then
	echo "usage: $0 [state] [cpu|gpu0|gpu1|...]"
	exit 1
fi

STATE=$1
GPU=$2

PYTHONPATH=`pwd` THEANO_FLAGS=floatX=float32,device=$GPU python experiments/nmt/train.py --proto prototype_search_state --state $STATE
