
import numpy
import timeit
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
import scipy

import sys

from hw3_utils import shared_dataset, load_data
from hw3_nn import LogisticRegression, HiddenLayer, LeNetConvPoolLayer, train_nn, drop, LeNetConvLayer

def MY_lenet(learning_rate=0.1, n_epochs=200,
                    ds_rate=None,
                    nkerns=[20, 50], batch_size=500,num_augment=80000):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)


    train_set, valid_set, test_set = load_data(ds_rate=ds_rate,theano_shared=False)
    aug_x = numpy.zeros((num_augment,3*32*32))
    aug_y = numpy.zeros(num_augment)
    randinds = numpy.random.randint(numpy.shape(train_set[0])[0],size=num_augment)
    for i in range(num_augment):
        aug_x[i,:] = translate_image(train_set[0][randinds[i]],1,numpy.random.randint(1,4))
        aug_y[i] = train_set[1][randinds[i]]        
    train_set[1] = numpy.append(train_set[1],aug_y)
    train_set[0] = numpy.vstack((train_set[0],aug_x))
    shuffle_inds = numpy.random.permutation(numpy.shape(train_set[1])[0])
    
    train_set[0][:,:] = train_set[0][shuffle_inds,:]
    train_set[1][:] = train_set[1][shuffle_inds]
    
    # Convert raw dataset to Theano shared variables.
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    print('Current training data size is %i'%train_set_x.get_value(borrow=True).shape[0])
    print('Current validation data size is %i'%valid_set_x.get_value(borrow=True).shape[0])
    print('Current test data size is %i'%test_set_x.get_value(borrow=True).shape[0])
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 3, 32, 32))
    layer0_input_drop = drop(layer0_input,0.8)
    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvLayer(
        rng,
        input=layer0_input_drop,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(128, 3, 3, 3),
        border_mode=1,alpha=0.33
    )
    layer1 = LeNetConvLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, 128, 32, 32),
        filter_shape=(128, 128, 3, 3),
        border_mode=1,alpha=0.33
    )
    stride0 = LeNetConvLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, 128, 32, 32),
        filter_shape=(128, 128, 3, 3),
        border_mode=1,
        conv_stride=(2,2)
    )
    drop0 = drop(stride0.output,0.5)
    
    layer2 = LeNetConvLayer(
        rng,
        input=drop0,
        image_shape=(batch_size, 128, 16, 16),
        filter_shape=(384, 128, 3, 3),
        border_mode=1,alpha=0.33
    )
    layer3 = LeNetConvLayer(
        rng,
        input=layer2.output,
        image_shape=(batch_size, 384, 16, 16),
        filter_shape=(384, 384, 3, 3),
        border_mode=1
    )
    stride1 = LeNetConvLayer(
        rng,
        input=layer3.output,
        image_shape=(batch_size, 384, 16, 16),
        filter_shape=(384, 384, 3, 3),
        border_mode=1,
        conv_stride=(2,2)
    )
    drop1 = drop(stride1.output,0.5)
    
    layer4 = LeNetConvLayer(
        rng,
        input=drop1,
        image_shape=(batch_size, 384, 8, 8),
        filter_shape=(768, 384, 3, 3),
        border_mode=1,alpha=0.33
    )
    layer5 = LeNetConvLayer(
        rng,
        input=layer4.output,
        image_shape=(batch_size, 768, 8, 8),
        filter_shape=(768, 768, 1, 1),
        border_mode='valid'
    )
    layer6 = LeNetConvLayer(
        rng,
        input=layer5.output,
        image_shape=(batch_size, 768, 8, 8),
        filter_shape=(1280, 768, 1, 1),
        border_mode='valid'
    )
    
    pool0 = pool.pool_2d(
            input=layer6.output,
            ds=(8,8),
            ignore_border=True,
            mode='average_exc_pad'
            )
    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer7_input = pool0.flatten(2)

    
    # classify the values of the fully-connected sigmoidal layer
    layer7 = LogisticRegression(input=layer7_input, n_in=1280, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer7.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer7.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer7.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = stride0.params + stride1.params + layer7.params + layer6.params + layer5.params + layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)
    def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for p, g in zip(params, grads):
            acc = theano.shared(p.get_value() * 0.)
            acc_new = rho * acc + (1 - rho) * g ** 2
            gradient_scaling = T.sqrt(acc_new + epsilon)
            g = g / gradient_scaling
            updates.append((acc, acc_new))
            updates.append((p, p - lr * g))
        return updates
    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = RMSprop(cost,params)
    #updates = [
    #    (param_i, param_i - learning_rate * grad_i)
    #    for param_i, grad_i in zip(params, grads)
    #]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    ###############
    # TRAIN MODEL #
    ###############
    
    print('... training')
    train_nn(train_model, validate_model, test_model,
            n_train_batches, n_valid_batches, n_test_batches, n_epochs,
            verbose = True)
