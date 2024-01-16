from torch import nn

class DotDict(dict):
	"""dot.notation access to dictionary attributes"""
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__


ACTIVATION_FUNCTIONS = {
	"swish": nn.SiLU(),
	"silu": nn.SiLU(),
	"mish": nn.Mish(),
	"gelu": nn.GELU(),
	"relu": nn.ReLU(),
}

def get_activation(act_fn: str) -> nn.Module:
	act_fn = act_fn.lower()
	if act_fn in ACTIVATION_FUNCTIONS:
		return ACTIVATION_FUNCTIONS[act_fn]
	else:
		raise ValueError(f"Unsupported activation function: {act_fn}")
