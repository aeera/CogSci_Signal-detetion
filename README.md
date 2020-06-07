# Signal Detection Theory Homework

Homework for *Cognitive Science I: Perception* at [TU Darmstadt](https://www.tu-darmstadt.de/cogsci/studying_cogsci/index.en.jsp). This homework replaces the lab work during the 2020 corona pandemic. 

Signal detection theory is one of the most successful theories in all of cognitive science and psychology. We assume you've already covered the basics of signal detection theory elsewhere. In the course we use the following text:

Wickens, T.D. (2002). *Elementary Signal Detection Theory*. Oxford University Press.  

For this homework the most relevant sections are 1.1-1.3 (p. 3-15), 2.1-2.3 (p. 17-26) and sections 3.1 (p. 39-42) and 3.3 (45-48). Here, you will consolidate the basics of signal detection theory by running a little signal detection experiment on yourself and analyzing the data. To collect data follow the instructions in `signal_detection_data_collection.ipynb`. After you've collected your data start on the actual homework excercises in `signal_detection_homework.ipynb`.

## Installation

You need python, psychopy and jupyter to run the tutorials. It's easiest if you install everything through `conda`. If you know what you're doing or you already have a working python and jupyter environment you might also be able to do it differently than described here. However, as psychopy requires python 3.6 and you don't want to mess with your existing python installation it's probably better to follow these instructions. If you don't already have anaconda installed, install anaconda from

https://www.anaconda.com/products/individual

Next download the source code from this page (there's a download button in the top right corner next to the `clone` button) and put it into a directory called `signal_detection`. Or just clone the repository if you are familiar with `git`. In the new directory run

```
conda env create -n signal_detection -f ENVIRONMENT.yml
```

to install all the required packages. Then (still in the same directory) activate the conda environment

```
conda activate signal_detection
```

start jupyter notebook

```
jupyter notebook
```

and click on `signal_detection_data_collection.ipynb` in the browser window that will open. Now you're ready to go.


## Working with jupyter notebooks and git

Just in case you're working with git and you want to fork your own repository based on this one or in case we invite you to push to this one: To keep version control on git clean, it is better not to commit the output of the cells and the count how often they have been run and so forth. For a clean, well readable diff it helps to strip all this information before commiting. With the following configuration it will be done automatically, leaving your local copy intact.

To your global `~/.gitconfig` you can add the following filter (you can edit it by typing `git config --edit --global`):

```
[filter "clearoutput"]
        clean = "jupyter nbconvert --to notebook --ClearOutputPreprocessor.enabled=True --stdin --stdout"
```

In the project-level `.gitattributes` file this option turns on the filter: `*.ipynb filter=clearoutput`
