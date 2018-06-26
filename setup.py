from setuptools import setup

setup(
    name='sentiment',
    version='0.0.0',
    url='http://www.codycoleman.com',
    author='Cody Austun Coleman',
    author_email='cody.coleman@cs.stanford.edu',
    packages=['sentiment'],
    install_requires=[
        'pandas',
        'numpy'
        'torch',
        'torchvision',
    ]
)
