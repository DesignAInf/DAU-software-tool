from setuptools import setup, find_packages

setup(
    name             = "dmbd_joule",
    version          = "5.0.0",
    description      = "Active Inference Markov blanket for Joule heat reduction in conductors",
    author           = "Luca Maria Possati",
    packages         = find_packages(),
    python_requires  = ">=3.10",
    install_requires = [
        "numpy>=1.24",
        "scipy>=1.10",
        "matplotlib>=3.7",
    ],
    entry_points     = {
        "console_scripts": [
            "dmbd_joule=dmbd_joule.__main__:main",
        ],
    },
)
