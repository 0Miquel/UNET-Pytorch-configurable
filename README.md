# UNET-Pytorch-configurable
Configurable UNET model from scratch using PyTorch.

## Requirements
- PyTorch

## Configuration
- in_channels: 

  Input channels
- out_channels: 

  Output channels
- features: 

  initial features extracted from first layer
- n_blocks: 
  
  number of encoder/decoder blocks
- kernel_size: 
  
  kernel size of every convolution, padding will be always equal to kernel_size//2 in order to preserve the size of the image
