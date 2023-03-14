import torch 

## enable cuda: 
torch.device('cuda' if torch.cuda.is_available else 'cpu') 
torch.cuda.set_device(0)   ## set device id to use

## check
if torch.cuda.is_available():
  print("cuda is enabled")
else:
  print("cuda is not enabled")
  
device = torch.device('cuda' if torch.cuda.is_available else 'cpu') 
torch.cuda.is_available()
