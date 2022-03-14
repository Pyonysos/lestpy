from setuptools import setup, find_packages
import os
import codecs

VERSION = '0.0.1'
DESCRIPTION = 'Regression model with logical interactions'


here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    README = "\n" + f.read()


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
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows"
    ]
)