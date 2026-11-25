import torch
assert torch.backends.mps.is_available(), 'Apple MPS unavailable'
x=torch.randn(1,4,2,8,8,device='mps',dtype=torch.float16)
try:
 y=torch.nn.Conv3d(4,4,3,padding=1).to(device='mps',dtype=torch.float16)(x)
 print('Conv3d:',y.shape, y.device)
except Exception as e:
 print('Conv3d probe failed:',repr(e));raise
print('PyTorch',torch.__version__,'MPS ready')
