# Signal Detection Tutorial

Signal detection theory is one of the most successful theories in all of cognitive science and psychology. This is a tutorial notebook for learning the basics by running a little signal detection experiment on yourself and analyzing the data.

# Installation

You need python, psychopy and jupyter to run the tutorials. It's easiest if you install everything through `conda`. If you know what you're doing or you already have a working python and jupyter environment you might also be able to do it differently than described here. However, as psychopy requires python 3.6 and you don't want to mess with your existing python installation it's probably better to follow these instructions. If you don't already have anaconda installed, install anaconda from

https://www.anaconda.com/products/individual

Next download the source code from this page (there's a download button in the top right corner next to the `clone` button) and put it into a directory called `signal_detection`. Or just clone the repository if you are familiar with `git`. In the new directory run

```
conda env create -n signal_detection -f signal_detection_env.yml
```

to install all the required packages. Then (still in the same directory) activate the conda environment

```
conda activate signal_detection
```

and start jupyter notebook with the tutorial

```
jupyter notebook signal_detection_tutorial.ipynb
```

Now you're ready to go.

# Additional Reading

A good place to start is

Wickens, T.D. (2002). *Elementary Signal Detection Theory*. Oxford University Press.  

For this tutorial the most relevant sections are 1.1-1.3 (p. 3-15), 2.1-2.3 (p. 17-26) and sections 3.1 (p. 39-42) and 3.3 (45-48). TU Darmstadt has an [online copy](https://hds.hebis.de/ulbda/Record/HEB379323249) where you can read and copy the respective sections. Please don't borrow the whole e-book because then nobody else in class can read it while you borrow it.

An excellent paper to get an overview of why you should care about signal detection theory beyond the simple experiment in this tutorial is

Swets, J.A., Dawes, R.M., & Monahan, J. (2000). Psychological Science Can Improve Diagnostic Decisions. *Psychological Science in the Public Interest*, 1(1):1-26, DOI: [10.1111/1529-1006.001](https://doi.org/10.1111/1529-1006.001).

# Working with jupyter notebooks and git
Just in case you're working with git: To keep version control on git clean, it is better not to commit the output of the cells and the count how often they have been run and so forth.
For a clean, well readable diff it helps to strip all this information before commiting.
With the following configuration it will be done automatically, leaving your local copy intact.

To your global `~/.gitconfig` you can add the following filter (you can edit it by typing `git config --edit --global`):

```
[filter "clearoutput"]
        clean = "jupyter nbconvert --to notebook --ClearOutputPreprocessor.enabled=True --stdin --stdout"
```

In the project-level `.gitattributes` file this option turns on the filter: `*.ipynb filter=clearoutput`
