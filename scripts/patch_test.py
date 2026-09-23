import torch
import torch.nn as nn
import torch.nn.functional as F

class DummyLinear4bit(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2))
    def forward(self, x):
        return F.linear(x, self.weight)

class Gemma4ClippableLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = DummyLinear4bit()
    def forward(self, x):
        # Emulate the buggy behavior
        return F.linear(x, self.linear.weight).clamp(-1, 1)

model = Gemma4ClippableLinear()

# Let's say PEFT replaces self.linear
class LoraWrapper(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base_layer = base
        self.lora_A = nn.Parameter(torch.ones(2, 2))
    @property
    def weight(self):
        return self.base_layer.weight
    def forward(self, x):
        return self.base_layer(x) + F.linear(x, self.lora_A)

model.linear = LoraWrapper(model.linear)

# Forward pass
x = torch.ones(2, 2)
x.requires_grad = True
y = model(x)
y.sum().backward()

print("lora_A grad:", model.linear.lora_A.grad) # Will be None if bypassed!
