# -*- coding: utf-8 -*-
"""
Created on Sun May  4 12:52:01 2014

@author: Rachel Zhang @ CCNT
"""


import theano
import theano.tensor as T
import cPickle, gzip, numpy
import time


class LogisticRegression(object):
    
    def __init__(self,input,n_in,n_out):
        self.W = theano.shared(value = numpy.zeros((n_in,n_out),
                                                   dtype = theano.config.floatX),
                               name = 'W', 
                               borrow = True)
                               
        self.b = theano.shared(value = numpy.zeros((n_out,),
                                                   dtype = theano.config.floatX),
                               name = 'b', 
                               borrow= True)
                               
        self.p_y_given_x = T.nnet.softmax(T.dot(input,self.W)+self.b)
        
        self.y_pred = T.argmax(self.p_y_given_x,axis = 1)
        
        self.params = [self.W,self.b]
        
        
    
    
    
    
    def negative_log_likelihood(self,y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])
    
    #def errors(self,y):
        
    
    
    
    def errors(self,y):
        #returns the total error of samples in a minibatch
        #type y:theano.tensor.TensorType
        
        if y.ndim!=self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y',y.type,'y_pred',self.y_pred.type))
    
        if y.dtype.startswith('int'):
            return T.mean(T.neq(y,self.y_pred))
        else:
            raise NotImplementedError()
    










def load_data():
    print 'loading data...'
    f = gzip.open('..//Dataset//mnist.pkl.gz','rb')
    train_s,valid_s,test_s = cPickle.load(f)
    f.close()
    
    def shared_dataset(data_xy, borrow=True):#loads the dataset into shared variables
        """ Function that loads the dataset into shared variables
    
        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX),borrow = borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX),borrow = borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets us get around this issue
        return shared_x, T.cast(shared_y, 'int32')
    # Minibatch Partition
    test_set_x, test_set_y = shared_dataset(test_s)
    valid_set_x, valid_set_y = shared_dataset(valid_s)
    train_set_x, train_set_y = shared_dataset(train_s)
    rval = [(train_set_x,train_set_y),(valid_set_x,valid_set_y),(test_set_x, test_set_y)]
    return rval
    
    
        

    
    
def sgd(learning_rate = 0.13, n_epoch = 1000, batch_size = 10):
    
    #load data
    dataset = load_data();
    train_set_x, train_set_y = dataset[0] # array([50000,   784])
    valid_set_x, valid_set_y = dataset[1] # array([10000,   784])
    test_set_x, test_set_y = dataset[2] #array([10000,   784])
    
    #compute number of minibatches
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
    
    print ('training set has %i batches' %n_train_batches)
    print ('validate set has %i batches' %n_valid_batches)
    print ('testing set has %i batches' %n_test_batches)
    
    
    
    #---------------------BUILD MODEL-----------------------#
    print 'Build Model...'
    
    minibatch_index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    
    #construct the logistic regression class
    classifier = LogisticRegression(input = x, n_in = 28*28, n_out = 10)
    cost = classifier.negative_log_likelihood(y)
    
    #model on a minibatch
    test_model = theano.function(inputs=[minibatch_index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[minibatch_index * batch_size: (minibatch_index + 1) * batch_size],
                y: test_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]})
               
    validate_model = theano.function(inputs = [minibatch_index],
                                     outputs = classifier.errors(y),
                                     givens = {x:valid_set_x[minibatch_index*batch_size : (minibatch_index+1)*batch_size],
                                               y:valid_set_y[minibatch_index*batch_size : (minibatch_index+1)*batch_size]})
    
    #compute gradient
    g_W = T.grad(cost = cost, wrt = classifier.W)
    g_b = T.grad(cost = cost, wrt = classifier.b)
    updates = [(classifier.W , classifier.W - learning_rate*g_W),
               (classifier.b, classifier.b - learning_rate*g_b)]
    #updates should be defined as a list of pairs of (shared-variable, new expression)
               
    train_model = theano.function(inputs = [minibatch_index],
                                  outputs = cost,
                                  updates = updates,
                                  givens = {x: train_set_x[minibatch_index*batch_size : (minibatch_index+1)*batch_size],
                                            y: train_set_y[minibatch_index*batch_size : (minibatch_index+1)*batch_size]})
    
    
    
    
    
    #---------------------Train-----------------------#
    print 'Training the model...'
    
    #early stop parameters
    patience = 5000
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_thres = 0.995
    validation_frequency = min(n_train_batches,patience/2)
    best_validation_loss = numpy.inf
    test_score = 0
    start_time = time.clock()
    done_looping = False
    epoch = 0
    
    while (epoch<n_epoch) and (not done_looping):
        epoch = epoch+1
        for minibatch_index_train in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index_train)
            # number of iterations (updates)
            iter = (epoch-1) * n_train_batches + minibatch_index_train
            
            if (iter+1)%validation_frequency == 0:
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                cur_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, training minibatch %i/%i, validation error %f %%' %\
                (epoch, minibatch_index_train+1, n_train_batches, cur_validation_loss*100.))
                
                #compare with best validation loss
                if  cur_validation_loss< best_validation_loss:
                    if cur_validation_loss < best_validation_loss*improvement_thres:
                        #define patience: if best_validation_loss not changes over 'patience' iterations
                        patience = max(patience, iter*patience_increase)
                    best_validation_loss = cur_validation_loss
                    test_loss = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_loss)
                    print (('     epoch %i, minibatch %i/%i, test error of best'
                       ' model %f %%') %
                       (epoch, minibatch_index_train + 1, n_train_batches,
                         test_score * 100.))
            if patience<=iter:
                done_looping = True
                break
    
    end_time = time.clock()
    print(('Optimization completed with best validation score of %f %%,'
            'with test performance %f %%')%
            (best_validation_loss*100., test_score*100.))
    
    print 'The code run for %d epochs, with %f epochs/sec' %(
            epoch, 1.*epoch/(end_time-start_time))
    

if __name__ == '__main__':
    sgd()
    


    
    
    
    
    
