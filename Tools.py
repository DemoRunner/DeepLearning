import matplotlib.pyplot as plt
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
            # the original image is 28*28,you can know that:http://yann.lecun.com/exdb/mnist/
            image = image_data.reshape((28, 28))
            image_error = '/home/prafly/desktop/python/DeepLearning/ErrorImage/Image' + \
                str(num) + '-' + str(image_lable) + '.png'
            plt.imsave(image_error, image, cmap=plt.cm.gray)
