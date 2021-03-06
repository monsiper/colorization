"""
Source Code for Homework 3 of ECBM E4040, Fall 2016, Columbia University

This code contains implementation of some basic components in neural network.

Instructor: Prof. Zoran Kostic

This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/lenet.html
"""

from __future__ import print_function

import timeit
import inspect
import sys
import numpy
from theano.tensor.nnet import conv
import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d, bn
from theano.tensor.nnet.abstract_conv import conv2d_grad_wrt_inputs

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )


        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).
        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]



def train_nn(train_model, validate_model, test_model,
            n_train_batches, n_valid_batches, n_test_batches, n_epochs,
            verbose = True):
    """
    Wrapper function for training and test THEANO model

    :type train_model: Theano.function
    :param train_model:

    :type validate_model: Theano.function
    :param validate_model:

    :type test_model: Theano.function
    :param test_model:

    :type n_train_batches: int
    :param n_train_batches: number of training batches

    :type n_valid_batches: int
    :param n_valid_batches: number of validation batches

    :type n_test_batches: int
    :param n_test_batches: number of testing batches

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to

    """

    # early-stopping parameters
    patience = 100000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter % 100 == 0) and verbose:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                if verbose:
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                        (epoch,
                         minibatch_index + 1,
                         n_train_batches,
                         this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)

                    if verbose:
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1,
                               n_train_batches,
                               test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()

    # Retrieve the name of function who invokes train_nn() (caller's name)
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)

    # Print out summary
    print('Optimization complete.')
    print('Best validation error of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The training process for function ' +
           calframe[1][3] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
    
    


##########  from mehmet #############
class ConvSubSample(object):
    def __init__(self,
                 input,
                 filter_shape,
                 image_shape,
                 border_mode='full',
                 conv_stride=None):

        self.input = input
        self.W = theano.shared(numpy.ones(shape=filter_shape, dtype=theano.config.floatX))

        conv_out_first = conv2d(
            input=input[:,0:1],
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape,
            border_mode=border_mode,
            subsample=conv_stride,
        )

        conv_out_second = conv2d(
            input=input[:, 1:2],
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape,
            border_mode=border_mode,
            subsample=conv_stride,
        )

        self.output = T.concatenate([conv_out_first, conv_out_second], axis=1)

def dec_net_out_to_rgb(image_net_out, data_l, temp=0.4, model_type='prob'):
    if model_type=='prob':

        #image_net_out is a tensor of shape (313,256,256) and data_l is a tensor of shape (1,256,256)
        ref_Qcolor_bins = numpy.load('pts_in_hull.npy')
        image_net_out_matrix = numpy.exp(numpy.log(image_net_out.transpose(1,2,0).reshape(256*256, 313))/temp)
        image_net_out_matrix = image_net_out_matrix/image_net_out_matrix.sum(axis=1, keepdims=True)
        data_ab = numpy.dot(image_net_out_matrix, ref_Qcolor_bins)
        data_a_chan = (data_ab[:,0]).reshape(1,256,256)
        data_b_chan = (data_ab[:,1]).reshape(1,256,256)
        img_construct = numpy.concatenate((data_a_chan, data_b_chan), axis=0)
        img_construct = numpy.concatenate((data_l, img_construct), axis=0)

        #reconstructed image is returned in 256x256x3 format
        return img_construct.transpose(1,2,0)

    if model_type=='regression':

        img_construct = numpy.concatenate((data_l, image_net_out), axis=0)
        return img_construct.transpose(1, 2, 0)



##########  mehmet #############

class Conv(object):
    def __init__(self, 
                 rng, 
                 input, 
                 filter_shape, 
                 image_shape,
                 border_mode='full',
                 conv_stride=None,
                 alpha=0,
                 conv_dilation=None,
                 loaded_params=None):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        if conv_stride==None:
            conv_stride=(1,1)
        if conv_dilation==None:
            conv_dilation=(1,1)
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]))
        if loaded_params==None:
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
            # the bias is a 1D tensor -- one bias per output feature map
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)
        else:
            self.W = theano.shared(
                loaded_params[0].get_value(),
                borrow=True
            )
            self.b = theano.shared(
                loaded_params[1].get_value(),
                borrow=True
            )

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape,
            border_mode =border_mode,
            subsample = conv_stride,
            filter_dilation = conv_dilation
        )
        self.output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input


class ConvReLU(object):
    def __init__(self, 
                 rng, 
                 input, 
                 filter_shape, 
                 image_shape,
                 border_mode='full',
                 conv_stride=None,
                 alpha=0,
                 conv_dilation=None,
                 loaded_params=None):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        if conv_stride==None:
            conv_stride=(1,1)
        if conv_dilation==None:
            conv_dilation=(1,1)
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]))
        if loaded_params==None:
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
            # the bias is a 1D tensor -- one bias per output feature map
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)
        else:
            self.W = theano.shared(
                loaded_params[0].get_value(),
                borrow=True
            )
            self.b = theano.shared(
                loaded_params[1].get_value(),
                borrow=True
            )
            
        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape,
            border_mode =border_mode,
            subsample = conv_stride,
            filter_dilation = conv_dilation
        )
        self.output = T.nnet.relu(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'),alpha=alpha)

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input
        

class DeConvReLU(object):
    def __init__(self, 
                 rng, 
                 input, 
                 filter_shape, 
                 image_shape,
                 border_mode='full',
                 conv_stride=None,
                 alpha=0,
                 conv_dilation=None,
                 loaded_params=None):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        #assert image_shape[1] == filter_shape[0]
        self.input = input

        if conv_stride==None:
            conv_stride=(1,1)
        if conv_dilation==None:
            conv_dilation=(1,1)
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]))
        if loaded_params==None:
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
            # the bias is a 1D tensor -- one bias per output feature map
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)
        else:
            self.W = theano.shared(
                loaded_params[0].get_value(),
                borrow=True
            )
            self.b = theano.shared(
                loaded_params[1].get_value(),
                borrow=True
            )
            
        deconv_out = conv2d_grad_wrt_inputs(
            output_grad=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape,
            border_mode=border_mode,
            subsample=conv_stride)
        # convolve input feature maps with filters
        """
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape,
            border_mode =border_mode,
            subsample = conv_stride,
            filter_dilation = conv_dilation
        )
        """
        self.output = T.nnet.relu(deconv_out + self.b.dimshuffle('x', 0, 'x', 'x'),alpha=alpha)

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input
        
class BatchNorm(object):
    def __init__(self,
                 rng, 
                 input,
                 image_shape,
                 loaded_params=None):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        self.input = input
        if loaded_params==None:
            self.gamma = theano.shared(
                 numpy.asarray(
                    rng.uniform(low=-0.1, high=0.1, size=image_shape),
                    dtype=theano.config.floatX
                ),
                borrow = True
            )
            self.beta = theano.shared(
                 numpy.asarray(
                    rng.uniform(low=-0.1, high=0.1, size=image_shape),
                    dtype=theano.config.floatX
                ),
                borrow = True
            )
        else:
            self.gamma = theano.shared(
                 loaded_params[0].get_value(),
                borrow = True
            )
            self.beta = theano.shared(
                 loaded_params[1].get_value(),
                borrow = True
            )
        #self.beta = theano.shared(value = numpy.zeros(image_shape, dtype=theano.config.floatX), name='beta')


        # bn_output = lin_output
        self.output = bn.batch_normalization(inputs = self.input,
            gamma = self.gamma, beta = self.beta, mean = self.input.mean((0,), keepdims=True),
            std = T.ones_like(self.input.var((0,), keepdims = True)), mode='high_mem')
        
        # store parameters of this layer
        self.params = [self.gamma, self.beta]
        # keep track of model input
        self.input = input
        

class Colorization_Softmax(object):
    def __init__(self, 
                 rng, 
                 input, 
                 filter_shape, 
                 image_shape,
                 border_mode='full',
                 conv_stride=None,
                 alpha=0,
                 conv_dilation=None,
                 loaded_params=None
                ):
        """
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        if conv_stride==None:
            conv_stride=(1,1)
        if conv_dilation==None:
            conv_dilation=(1,1)
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]))
        if loaded_params==None:
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
            # the bias is a 1D tensor -- one bias per output feature map
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)
        else:
            self.W = theano.shared(
                loaded_params[0].get_value(),
                borrow=True
            )
            self.b = theano.shared(
                loaded_params[1].get_value(),
                borrow=True
            )

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape,
            border_mode =border_mode,
            subsample = conv_stride,
            filter_dilation = conv_dilation
        )
        self.preout = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        e_x = T.exp(self.preout - self.preout.max(axis=1, keepdims=True))
        self.output = e_x / e_x.sum(axis=1, keepdims=True)
        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

        
class Colorization_Decoding(object):
    def __init__(self, 
                 rng, 
                 input, 
                 filter_shape, 
                 image_shape,
                 border_mode='full',
                 conv_stride=None,
                 alpha=0,
                 conv_dilation=None,
                 loaded_params=None
                ):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]

        if conv_stride==None:
            conv_stride=(1,1)
        if conv_dilation==None:
            conv_dilation=(1,1)
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]))
        if loaded_params==None:
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
            # the bias is a 1D tensor -- one bias per output feature map
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)
        else:
            self.W = theano.shared(
                loaded_params[0].get_value(),
                borrow=True
            )
            self.b = theano.shared(
                loaded_params[1].get_value(),
                borrow=True
            )
        self.params = [self.W, self.b]

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape,
            border_mode =border_mode,
            subsample = conv_stride,
            filter_dilation = conv_dilation
        )
        self.output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')

        # store parameters of this layer

        # keep track of model input
        self.input = input
        
        
        
class PriorFactor():
    def __init__(self, 
                 rng, 
                 input,  
                 alpha,
                 batch_size,
                 gamma=0,
                 verbose=False,
                 priorFile='prior_probs.npy'
                ):
        
        self.input = input
        # settings
        self.alpha = alpha
        self.gamma = gamma
        self.verbose = verbose

        # empirical prior probability
        self.prior_probs = numpy.load(priorFile)

        # define uniform probability
        self.uni_probs = numpy.zeros_like(self.prior_probs)
        self.uni_probs[self.prior_probs!=0] = 1.
        self.uni_probs = self.uni_probs/numpy.sum(self.uni_probs)

        # convex combination of empirical prior and uniform distribution       
        self.prior_mix = (1-self.gamma)*self.prior_probs + self.gamma*self.uni_probs

        # set prior factor
        self.prior_factor = self.prior_mix**-self.alpha
        self.prior_factor = theano.shared(
            numpy.asarray(
            self.prior_factor/numpy.sum(self.prior_probs*self.prior_factor),
                dtype=theano.config.floatX)
        ) # re-normalize

        # implied empirical prior
        #self.implied_prior = self.prior_probs*self.prior_factor
        #self.implied_prior = self.implied_prior/numpy.sum(self.implied_prior) # re-normalize

        if(self.verbose):
            self.print_correction_stats()
        self.output = self.forward(self.input,axis=1).reshape((batch_size,1,64,64))

    def forward(self,data_ab_quant,axis=1):
        data_ab_maxind = T.argmax(data_ab_quant,axis=axis)
        corr_factor = self.prior_factor[data_ab_maxind]
        if(axis==0):
            return corr_factor[na(),:]
        elif(axis==1):
            #return corr_factor[:,na(),:]
            return corr_factor
        elif(axis==2):
            return corr_factor[:,:,na(),:]
        elif(axis==3):
            return corr_factor[:,:,:,na()]
        
class Colorization_PriorBoost(object):
    def __init__(self,
                 rng,
                 input,
                 batch_size=5,
                 image_shape=None,
                 gamma=0,
                 verbose=True
                ):
        self.input = input
        self.gamma = .5
        self.alpha = 1.
        self.batch_size=batch_size
        self.ENC_DIR = './'
        self.pc = PriorFactor(rng,
                              self.input,
                              self.alpha,
                              batch_size=self.batch_size,
                              gamma=self.gamma,
                              priorFile='./prior_probs.npy'
                             )
        #self.N = image_shape[0]
        #self.Q = image_shape[1]
        #self.X = image_shape[2]
        #self.Y = image_shape[3]
        self.output = self.pc.output
