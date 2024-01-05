class DictDotNotation:
	def __init__(self, **kwargs):
		"""Initialize scheduler with configuration dictionary."""
		for key, value in kwargs.items():
			setattr(self, key, value)