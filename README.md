# Depth from a Single Image by Harmonizing Overcomplete Local Network Predictions
Copyright (C) 2016, Authors.

This is a reference implementation of the algorithm described in the
paper, ["**Depth from a Single Image by Harmonizing Overcomplete Local Network Predictions**"
*arXiv:1605.07081 [cs.CV]*](https://arxiv.org/abs/1605.07081). It is
being made available for non-commercial research use only. If you find
this code useful in your research, please consider citing the paper.

Contact <ayanc@ttic.edu> with any questions.

### Requirements

1. You can download our trained neural network model weights,
   available as a .caffemodel.h5 file [here][model.h5].
   
2. This implementation requires a modern CUDA-capable high-memory GPU
   (it has been tested on an NVIDIA Titan X), and a recent version of
   MATLAB's Parallel Computing Toolbox that supports the `GPUArray`
   class.
   
3. You will also need to compile the mex function postMAP. With modern
   versions of Matlab, this can be done by running `mexcuda
   postMAP.cu`. Requires the CUDA toolkit with `nvcc` to be installed.

[model.h5]: http://www.ttic.edu/chakrabarti/mdepth/wts.caffemodel.h5

### Usage for Inference

First, you will need to load the network weights from the model file as:

```>>> net = load('/path/to/wts.caffemodel.h5');```

Then given a floating-point RGB image `img`, normalized to `[0,1]`, estimate the corresponding depth map as:

```>>> Z = mdepth(img,net);```

Note that we expect `img` to be of size `561x427`, which corresponds to the axis aligned crops in the NYU dataset where there is a valid depth map projection. You can recover these as: `img = imgOrig(45:471, 41:601, :)`.

### Training with Caffe

Training code will be released soon.
