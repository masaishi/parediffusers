from setuptools import setup, find_packages

install_requires = [
	'torch',
	'torchvision',
	'transformers'
]

setup(
	name='PareDiffusion',
	author='Masamune Ishihara',
	author_email='mwishiha@ucsc.edu',
	description="The library `pared` down the features of `diffusers` implemented the minimum function to generate images without using huggingface/diffusers to understand the inner workings of the library.",
	long_description=open('README.md').read(),
	long_description_content_type='text/markdown',
	license='MIT License',
	url='https://github.com/masaishi/MBTiles_Segmentation',
	version='0.1.0',
	python_requires='>=3.6',
	install_requires=install_requires,
	package_dir={"": "src"},
	packages=find_packages("src")
)