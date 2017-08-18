# Machine Learning Engineer Nanodegree
# Capstone
## Project: Image Classification with Convolutional Neural Networks

### Install

This project requires **Python 2.7** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has most of the above packages and more included. Make sure that you select the Python 2.7 installer and not the Python 3.x installer.

### Code

### Run

### Data

The CIFAR-10 dataset consists of 60,000 images: 50,000 training images and 10,000 test images. This dataset and more can be found here [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).

I will describe the layout of the Python 2 version of the dataset. 

The archive contains the files data_batch_1, data_batch_2, ..., data_batch_5, as well as test_batch. Each of these files is a Python "pickled" object produced with `cPickle`. Here is a python2 routine which will open such a file and return a dictionary:

```bash
def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict   
```

Loaded in this way, each of the batch files contains a dictionary with the following elements:

**data** -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.

**labels** -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.

The dataset contains another file, called batches.meta. It too contains a Python dictionary object. It has the following entries:

**label_names** -- a 10-element list which gives meaningful names to the numeric labels in the labels array described above. For example, label_names[0] == "airplane", label_names[1] == "automobile", etc.