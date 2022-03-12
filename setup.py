from setuptools import setup, find_packages
import os

VERSION = '0.0.1'
DESCRIPTION = 'Regression model with logical interactions'

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# Setting up
setup(
    name="lestpy",
    version=VERSION,
    author="Pyonysos (Paul-Hubert Baudelet)",
    author_email="",
    description=DESCRIPTION,
    
    long_description_content_type="text/markdown",
    long_description=README,
    packages=find_packages(),
    install_requires=['pandas', 'sklearn', 'scipy', 'matplotlib'],
    keywords=['python', 'modeling'],
    url = 'https://github.com/Pyonysos/lestpy',
    classifiers=[
        "Development Status :: 2 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows"
    ]
)