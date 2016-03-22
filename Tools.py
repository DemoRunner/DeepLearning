import cPickle
import numpy
from PIL import Image
from numpy import *
import gzip
import os
import theano
import theano.tensor as T
"""
this tools is help for testing your deeplearning mould ;it can get the load_data, accuray,save the error image ,etc.
I sincerely hope someone else can some tools in it,which are help for the beginner.


It can be use in the following mouldes,which are all in the theano tutorial.
- logisti_sgd.py
- mlp.py

References:
- tanpan:
  github:https://github.com/DemoRunner/
  email:DemoRunner@163.com
"""


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is an numpy.ndarray of 2 dimensions (a matrix)
    # witch row's correspond to an example. target is a
    # numpy.ndarray of 1 dimensions (vector)) that have the same length as
    # the number of rows in the input. It should give the target
    # target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def accuracy(test_set_y, predicted_values):
    """ computer the accuracy of our moudle
    :type test_set_y: numpy.ndarray
    :param test_set_y: the true number(lable) of original test image

    :type predicted_values: numpy.ndarray
    :param predicted_values: the predicted number of original test image
    """
    print(
        'the accuracy rate is : %f %%' %
        ((test_set_y == predicted_values).sum() / float(len(test_set_y)) * 100)
    )


def saveimage(test_set_x, test_set_y, predicted_values):
    """ save the predicted errror image
    :type test_set_x: numpy.ndarray
    :param test_set_x: the original data of test image

    :type test_set_y: numpy.ndarray
    :param test_set_y: the true number(lable) of original test image

    :type predicted_values: numpy.ndarray
    :param predicted_values: the predicted number of original test image
    """
    for num in range(0, len(test_set_y)):
        # select the predicted error image
        if test_set_y[num] != predicted_values[num]:
            image_data = test_set_x[num]
            image_lable = test_set_y[num]
            # the original image is 28*28,you can know
            # that:http://yann.lecun.com/exdb/mnist/
            image = image_data.reshape((28, 28))
            # a numpy array convert to PIL image
            PIL_image = Image.fromarray(image)
            # get the final image name
            image_error = './ErrorImage/Image' + \
                str(num) + '-' + str(image_lable) + '.gif'
            PIL_image.save(image_error)


def imagetodata(image_path, image_lable):
    """ make the orignal image to data, which accord with data format of "minist.pkl.gz"
    so all of input are Numerical image
    :type image_path: string
    :param image_path: the location of input image

    :type image_lable: int
    :param image_lable: the value of image
    """
    # Segmentation path and file name
    if not os.path.exists(image_path):
        return
    else:
        image_dir, image_f = os.path.split(image_path)
    # take the orignal image to PIL image,and get the gray map.
    image = Image.open(image_path).convert('L')
    # take the PIL image to numpy arrary(the format is accord with
    # "minist.pkl.gz" )
    image_ndarray = numpy.asarray(image, dtype='float32') / 256
    # the next few lines is also mean to  accord with "minist.pkl.gz"'s format.
    image_data = image_ndarray.reshape((1, 784))
    image_lable = array([image_lable])
    # this line you can change,because they are same with each other.
    # but it will not run in theano tutorials code.
    image = ((image_data, image_lable), (image_data,
                                         image_lable), (image_data, image_lable))
    # use cPickle to dump image
    write_file = open(image_dir + '/' + image_f.split(".")[0] + '.pkl', 'wb')
    cPickle.dump(image, write_file, 0)

    write_file.close()

    # compress the front pkl file
    with open(image_dir + '/' + image_f.split(".")[0] + '.pkl', 'rb') as plain_file:
        with gzip.open(image_dir + '/' + image_f.split(".")[0] + '.pkl.gz', 'wb') as zip_file:
            zip_file.writelines(plain_file)
