from setuptools import setup, find_packages

setup(
    name="pesto",
    version="0.1.0",
    description="PeSTo: Protein Ensemble Structural Topology tools",
    packages=find_packages(where="src"),
    package_dir={"": "src"},    # root package lives in /src
    python_requires=">=3.8",
    install_requires=[
        "tensorboard",
        "gemmi",
        "h5py",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "jupyterlab",
        "numpy",
        "scipy",
        "torch",
        "tqdm",
    ],
    entry_points={
        'console_scripts': [
            'pesto = pesto.__main__:main'
        ]
    }
)