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

  features extracted from the first convolution, in the rest of convolutions the number of features will get multipled by 2.
- n_blocks: 
  
  number of encoder/decoder blocks
- kernel_size: 
  
  kernel size of every convolution, padding will be always equal to kernel_size//2 in order to preserve the size of the image
