import matplotlib.pyplot as plt
import cPickle
import numpy
from PIL import Image
from numpy import *
import gzip
import os
"""
this tools is help for testing your deeplearning mould ;it can get the accuray,save the error image ,etc.
I sincerely hope someone else can some tools in it,which are help for the beginner.


It can be use in the following mouldes,which are all in the theano tutorial.
- logisti_sgd.py
- mlp.py

References:
- tanpan:
  github:https://github.com/DemoRunner/
  email:DemoRunner@163.com
"""


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
            image_error = './ErrorImage/Image' + \
                str(num) + '-' + str(image_lable) + '.png'
            plt.imsave(image_error, image, cmap=plt.cm.gray)


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
