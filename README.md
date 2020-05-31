# Installation

You need python, psychopy and jupyter to run the tutorials. It's easiest if you install everything through `conda`. If you know what you're doing or you already have a working python and jupyter environment you might also be able to do it differently than described here. However, as psychopy requires python 3.6 and you don't want to mess with your existing python installation it's probably better to follow these instructions. If you don't already have anaconda installed, install anaconda from

https://www.anaconda.com/products/individual

Next download the source code from this page (there's a download button in the top right corner) and put it into a directory called `signal_detection` or just clone the repository if you are familiar with `git`. In this directory run

```
conda env create -n signal_detection -f signal_detection_env.yml
```

to install all the packages necessary to run the tutorial. Then (still in the same directory) activate the conda environment

```
conda activate signal_detection
```

and start jupyter notebook with the tutorial

```
jupyter notebook signal_detection_tutorial.ipynb
```
