from setuptools import find_packages, setup
 
setup(
    name="Swin-litemedsam",
    version="0.0.1",
    author="Ruochen Gao, Donhang Lyu",
    python_requires=">=3.9",
    install_requires=["monai", "einops", "omegaconf", "transformers", "connected-components-3d", "timm", "matplotlib", "scikit-image", "SimpleITK>=2.2.1", "pyarrow", "pandas", "nibabel", "tqdm", "scipy", "ipympl", "opencv-python", "jupyterlab", "ipywidgets"],
    packages=find_packages(exclude="notebooks"),
    extras_require={
        "all": ["pycocotools", "opencv-python"],
        "dev": ["flake8", "isort", "black", "mypy"],
    },
)