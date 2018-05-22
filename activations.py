import torch
from torch import nn

class LeakyTanh(nn.Module):
	def __init__(self):
		super(LeakyTanh, self).__init__()

	def forward(self, x):
		return torch.sign(x) * torch.log(1 + torch.abs(x))

class SimSqrt(nn.Module):
	def __init__(self):
		super(SimSqrt, self).__init__()

	def forward(self, x):
		return torch.sign(x) * torch.sqrt(torch.abs(x))

class ReSU(nn.Module):
	def __init__(self):
		super(ReSU, self).__init__()

	def forward(self, x):
		x = torch.sign(x) * torch.sqrt(torch.abs(x))
		return torch.max(torch.zeros_like(x), x)

class ReLogU(nn.Module):
	def __init__(self):
		super(ReLogU, self).__init__()

	def forward(self, x):
		x = torch.sign(x) * torch.log(1 + torch.abs(x))
		return torch.max(torch.zeros_like(x), x)

class Cos(nn.Module):
	def __init__(self):
		super(Cos, self).__init__()

	def forward(self, x):
		return torch.cos(x)

class Sin(nn.Module):
	def __init__(self):
		super(Sin, self).__init__()

	def forward(self, x):
		return torch.sin(x)
