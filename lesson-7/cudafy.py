import torch

class cudafy:
    
    def __init__(self, device=None):
        if torch.cuda.is_available() and device:
            self.device = device
        else:
            self.device = 0
    
    def name(self):
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(self.device)
        return 'Cuda is not available.'
    
    def put(self, x):
        """Put x on the default cuda device."""
        if torch.cuda.is_available():
            return x.to(device=self.device)
        return x

    def __call__(self, x):
        return self.put(x)
    
    def get(self,x):
        """Get from cpu."""
        if x.is_cuda:
            return x.to(device='cpu')
        return x
    
def cpu(x):
    if x.is_cuda:
        return x.to(device='cpu')
    return x
