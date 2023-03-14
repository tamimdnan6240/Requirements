import torch 

## enable cuda:
  
device = torch.device('cuda' if torch.cuda.is_available else 'cpu') 
torch.cuda.is_available()

import torch


# check if cuda is enabled
if device.type == 'cuda':
    print('CUDA is enabled!')
else:
    print('CUDA is not enabled!')

# use the device
# check the version of PyTorch
print("PyTorch version:", torch.__version__)

# check the CUDA version
print("CUDA version:", torch.version.cuda)

# check the GPU model
if torch.cuda.is_available():
    print("GPU model:", torch.cuda.get_device_name())
else:
    print("No GPU available.")

# check if the GPU is being used
#print("Is the GPU being used?", torch.cuda.current_device())

# test if CUDA is working by running a simple computation on the GPU
if torch.cuda.is_available():
    x = torch.randn(1000, 1000).to("cuda")
    y = torch.randn(1000, 1000).to("cuda")
    z = torch.matmul(x, y)
    print("CUDA is working!")
else:
    print("CUDA is not working.")
