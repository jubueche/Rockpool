from distutils.core import setup

setup(
    name="Rockpool",
    version="0.2",
    packages=[
        "Rockpool",
        "Rockpool.weights",
        "Rockpool.weights.gpl",
        "Rockpool.weights.internal",
        "Rockpool.utilities",
        "Rockpool.utilities.gpl",
        "Rockpool.utilities.internal",
        "Rockpool.networks",
        "Rockpool.networks.gpl",
        "Rockpool.networks.internal",
        "Rockpool.layers",
        "Rockpool.layers.gpl",
        "Rockpool.layers.internal",
        "Rockpool.layers.internal.pytorch",
        "Rockpool.layers.internal.devices",
        "Rockpool.layers.training",
        "Rockpool.layers.training.gpl",
        "Rockpool.layers.training.internal",
    ],
    license="All rights reserved aiCTX AG",
    install_requires=["numba", "numpy", "scipy"],
    extras_require={
        "all": [
            "tqdm",
            "brian2",
            "pytest",
            "torch",
            "torchvision",
            "rpyc",
            "jax",
            "jaxlib",
            "sphinx",
            "nbsphinx",
            "sphinx-autobuild",
        ]
    },
    long_description=open("README.md").read(),
)
