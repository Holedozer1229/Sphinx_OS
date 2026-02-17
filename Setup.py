from setuptools import setup, find_packages

# Read the README file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Define the setup configuration
setup(
    name="SphinxOS",
    version="0.1.0",
    author="Travis Jones, Grok (xAI)",
    author_email="holedozer@icloud.com",
    description="The Scalar Waze - A unified quantum-spacetime operating system kernel integrating a 6D Theory of Everything simulation with a universal quantum circuit simulator.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Holedozer1229/Sphinx_OS",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=0.24.0",
        "cryptography>=41.0.0",  # Replaced ecdsa due to Minerva timing attack vulnerability
        "base58>=2.1.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: SphinxOS License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    keywords="quantum computing, 6D spacetime, theory of everything, temporal vector lattice entanglement, TVLE, Rydberg gates",
    project_urls={
        "Bug Tracker": "https://github.com/Holedozer1229/Sphinx_OS/issues",
        "Documentation": "https://github.com/Holedozer1229/Sphinx_OS/blob/main/README.md",
        "Source Code": "https://github.com/Holedozer1229/Sphinx_OS",
    },
)
