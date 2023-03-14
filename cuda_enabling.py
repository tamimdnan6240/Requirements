## enable cuda: 

import torch 
if torch.cuda.is_available():
  device = torch.device("cuda")
  torch.cuda.set_device(0)   ## set device id to use

## check
if torch.cuda.is_available():
  print("cuda is enabled")
else:
  print("cuda is not enabled")