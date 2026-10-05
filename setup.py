from setuptools import find_packages, setup

setup(
    name="ui-bug-detection",
    version="0.1.0",
    description="UI software bug detection using deep learning",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pandas>=1.5.0",
        "pillow>=9.5.0",
        "opencv-python>=4.7.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
    ],
)
