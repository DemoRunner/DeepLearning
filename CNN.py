# -*- coding: utf-8 -*-
"""
Created on Mon May 12 10:57:02 2014

@author: rachel

CNN in all
"""
from theano.tensor.nnet import conv
import theano.tensor as T
import numpy
import theano
import time
import cPickle
import gzip
from theano.tensor.signal import downsample
from LR_MNIST import LogisticRegression, load_data
from MLP import HiddenLayer


class LeNetConvPoolLayer (object):

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        assert filter_shape[1] == image_shape[1]
        self.input = input  # (batches, feature, Ih, Iw)

        # number of connections for each filter
        fan_in = numpy.prod(filter_shape[1:])
        W_value = numpy.asarray(
            rng.uniform(
                low=-
                numpy.sqrt(
                    3. /
                    fan_in),
                high=numpy.sqrt(
                    3. /
                    fan_in),
                size=filter_shape),
            dtype=theano.config.floatX)
        self.W = theano.shared(W_value, name='W')  # (filters, feature, Fh, Fw)

        b_value = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(b_value, name='b')

        # (batches, filters, Ih-Fh+1, Iw-Fw+1)
        conv_res = conv.conv2d(
            input, self.W, image_shape=image_shape, filter_shape=filter_shape)
        pooled = downsample.max_pool_2d(conv_res, poolsize)
        self.output = T.tanh(pooled + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.W, self.b]


def test_CNN(learning_rate=0.01, n_epochs=1000, batch_size=20, n_hidden=500):
    dataset = load_data()
    # tt = train_set_x.get_value(); tt.shape ---(50000, 784)
    train_set_x, train_set_y = dataset[0]
    valid_set_x, valid_set_y = dataset[1]
    test_set_x, test_set_y = dataset[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    print ('training set has %i batches' % n_train_batches)
    print ('validate set has %i batches' % n_valid_batches)
    print ('testing set has %i batches' % n_test_batches)

    # symbolic variables
    x = T.matrix()
    y = T.ivector()  # lvector: [long int] labels; ivector:[int] labels
    minibatch_index = T.lscalar()

    print 'build the model...'
    rng = numpy.random.RandomState(23455)

    # transfrom x from (batchsize, 28*28) to (batchsize,feature,28,28))
    # I_shape = (28,28),F_shape = (5,5),
    N_filters_0 = 20
    D_features_0 = 1
    layer0_input = x.reshape((batch_size, D_features_0, 28, 28))
    layer0 = LeNetConvPoolLayer(
        rng, input=layer0_input, filter_shape=(
            N_filters_0, D_features_0, 5, 5), image_shape=(
            batch_size, 1, 28, 28))
    # layer0.output: (batch_size, N_filters_0, (28-5+1)/2, (28-5+1)/2) ->
    # 20*20*12*12

    N_filters_1 = 50
    D_features_1 = N_filters_0
    layer1 = LeNetConvPoolLayer(
        rng, input=layer0.output, filter_shape=(
            N_filters_1, D_features_1, 5, 5), image_shape=(
            batch_size, N_filters_0, 12, 12))
    # layer1.output: (20,50,4,4)

    layer2_input = layer1.output.flatten(2)  # (20,50,4,4)->(20,(50*4*4))
    layer2 = HiddenLayer(
        rng, layer2_input, n_in=50 * 4 * 4, n_out=500, activation=T.tanh)

    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)

    ##########################
    cost = layer3.negative_log_likelihood(y)
    test_model = theano.function(
        inputs=[minibatch_index],
        outputs=layer3.errors(y),
        givens={
            x: test_set_x[
                minibatch_index *
                batch_size: (
                    minibatch_index +
                    1) *
                batch_size],
            y: test_set_y[
                minibatch_index *
                batch_size: (
                    minibatch_index +
                    1) *
                batch_size]})

    valid_model = theano.function(
        inputs=[minibatch_index],
        outputs=layer3.errors(y),
        givens={
            x: valid_set_x[
                minibatch_index *
                batch_size: (
                    minibatch_index +
                    1) *
                batch_size],
            y: valid_set_y[
                minibatch_index *
                batch_size: (
                    minibatch_index +
                    1) *
                batch_size]})

    params = layer3.params + layer2.params + layer1.params + layer0.params
    gparams = T.grad(cost, params)

    updates = []
    for par, gpar in zip(params, gparams):
        updates.append((par, par - learning_rate * gpar))

    train_model = theano.function(
        inputs=[minibatch_index],
        outputs=[cost],
        updates=updates,
        givens={
            x: train_set_x[
                minibatch_index *
                batch_size: (
                    minibatch_index +
                    1) *
                batch_size],
            y: train_set_y[
                minibatch_index *
                batch_size: (
                    minibatch_index +
                    1) *
                batch_size]})

    #---------------------Train-----------------------#
    print 'training...'

    epoch = 0
    patience = 10000
    patience_increase = 2
    validation_frequency = min(n_train_batches, patience / 2)
    improvement_threshold = 0.995

    best_parameters = None
    min_validation_error = numpy.inf
    done_looping = False

    start_time = time.clock()
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):
            #cur_batch_train_error,cur_params = train_model(minibatch_index)
            cur_batch_train_error = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                #validation_error = numpy.mean([valid_model(idx) for idx in xrange(n_valid_batches)])
                validation_losses = [valid_model(i) for i
                                     in xrange(n_valid_batches)]
                validation_error = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       validation_error * 100.))

                if validation_error < min_validation_error:
                    if validation_error < min_validation_error * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    min_validation_error = validation_error
                    #best_parameters = cur_params
                    best_iter = iter

                    # test
                    test_error = numpy.mean(
                        [test_model(idx) for idx in xrange(n_test_batches)])
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_error * 100.))

            if iter >= patience:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (min_validation_error * 100., best_iter + 1, test_error * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    test_CNN()
