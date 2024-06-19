import torch
x = torch.rand(5, 3)
print(x)

print(torch.cuda.is_available())
print(torch.__version__)

# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118