import torch
from fl_platform.src.models.model import DropoffLSTM, SimpleLSTM

def check_model_gpu_memory(model):
    """Check the GPU memory usage of a PyTorch model"""
    if torch.cuda.is_available():
        # Get model memory usage
        model_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        model_memory_mb = model_memory / (1024 ** 2)
        
        # Get current GPU memory usage
        allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2)
        cached_memory = torch.cuda.memory_reserved() / (1024 ** 2)
        
        print(f"Model memory: {model_memory_mb:.2f} MB")
        print(f"GPU allocated memory: {allocated_memory:.2f} MB")
        print(f"GPU cached memory: {cached_memory:.2f} MB")
        
        return model_memory_mb
    else:
        print("CUDA not available")
        return None

model = DropoffLSTM().cuda()
check_model_gpu_memory(model)

print("-----------------")


param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print('model size: {:.3f}MB'.format(size_all_mb))