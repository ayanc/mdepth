# Depth from a Single Image by Harmonizing Overcomplete Local Network Predictions
Copyright (C) 2016, Authors.

This directory contains code and instructions for training the local prediction
network using the [Caffe](https://github.com/BVLC/caffe) framework.

The primary network definition is in the file `train.prototxt` in this directory.
In addition to the prediction network, we also use existing Caffe layers to
compute depth derivatives, and generate classification targets for each depth
map on the fly.

## Custom Layers

Our network employs two custom layers, included in the `layers/` sub-directory.

1. The first is simply a python data layer in `layers/NYUdata.py`, and handles
   loading training data from the NYUv2 dataset (details on how to prepare the
   data are in the next section). Make sure you compile Caffe with python layers
   enabled, and place the above file in the current directory or somewhere
   in your `PYTHONPATH`.

2. The second layer is the SoftMax + KL-Divergence loss layer. You will need to
   compile this into Caffe. Copy the header file `softmax_kld_loss_layer.hpp`
   into the `include/caffe/layers/` directory of your caffe distribution, and
   `softmax_kld_loss_layer.c*` files into the `src/caffe/layers/` directory.
   Then run `make` to compile / update caffe.

## Preparing NYUv2 Data

Download the RAW distribution and toolbox from the [NYUv2 depth dataset
page](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html). Read
the documentation to figure out how to process the RAW data to
create aligned RGB-depth image pairs, and to *fill-in* missing depth
values. Also, make sure you only use scenes corresponding to the training
set in the official train-test split.

For each scene, generate a pair of PNG files to store the RGB and depth data
respectively. These should be named with a common base name and different
suffixes: `_i.png`, for the 8-bit 3 channel PNG corresponding to the
RGB image, and `_f.png` for a 16-bit 1 channel PNG image corresponding
to depth---the depth png should be scaled so that the max UINT16 value
(2^16-1) corresponds to a depth of 10 meters.

All images should be of size 561x427, corresponding to the valid projection
area (you can use the `crop_image` function in the NYU toolbox). If you
decide to train on a different dataset, you might need to edit the data layer
and the network architecture to work with different resolution images.

Place all pairs you want to use in the same directory, and prior to calling
affe, set the environment variable `NYU_DATA_DIR` to its path, e.g. as
`export NYU_DATA_DIR=/pathto/nyu_data_dir`. Then, create a text file called
`train.txt` (and place it in the same directory from which you are calling caffe).
Each line in this file should correspond to the common prefix for each scene. So,
if you have a line with `scene1_frame005`, then the data layer will read the
files:

```
/pathto/nyu_data_dir/scene1_frame005_i.png
/pathto/nyu_data_dir/scene1_frame005_f.png
```

for the image and depth data respectively.
   

## Training

Use the provided `train.prototxt` file for the network definition, and create a
solver prototxt file based on the description in the paper (momentum of 0.9, no
weight decay, and learning rate schedule described in the paper).

When you begin training, you should provide as an option to caffe:

```
-weights filters_init.caffemodel.h5,/path/to/vgg19.caffemodel
```

where `vgg19.caffemodel` is the pre-trained VGG-19 model from the caffe model
zoo. `filters_init.caffemodel.h5` is provided in this directory, and initializes
the weights of various layers in `train.prototxt` that compute depth-derivatives,
mixture weights with respect to various bins, perform bilinear up-sampling
of the scene features, etc. These layers have a learning rate factor of 0, and
will not change through training. However, they will be saved with model
snapshots, so you will need to provide the above option only the first time you
start training.

Please see the paper for more details, and contact <ayanc@ttic.edu> if you
still have any questions.
