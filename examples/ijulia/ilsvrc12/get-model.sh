#!/bin/bash

MEAN_FILE=model/ilsvrc12_mean.hdf5
MODEL_FILE=model/bvlc_reference_caffenet.hdf5

mkdir -p model

if [ ! -f $MEAN_FILE ]; then
  wget 'https://www.dropbox.com/s/036smr69qvwz23b/ilsvrc12_mean.hdf5?dl=0' -O $MEAN_FILE
fi
if [ ! -f $MODEL_FILE ]; then
  wget 'https://www.dropbox.com/s/cgrxcge38z8yb5i/bvlc_reference_caffenet.hdf5?dl=0' -O $MODEL_FILE
fi
