
import numpy
import timeit
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d, bn
import scipy

import sys

from hw3_utils import shared_dataset, load_data
from hw3_nn import LogisticRegression, HiddenLayer, LeNetConvPoolLayer, train_nn, drop, LeNetConvLayer

def colorization(learning_rate=0.1, n_epochs=200,
                    ds_rate=None,
                    nkerns=[20, 50], batch_size=500,num_augment=80000,dim_in=256):
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


    train_set, valid_set, test_set = load_data(ds_rate=None,theano_shared=False)
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
    #layer0_input = x.reshape((batch_size, 3, 32, 32))
    #layer0_input_drop = drop(layer0_input,0.8)
    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    bw_input = T.mean(x.reshape((batch_size,3,dim_in,dim_in)),axis=1)
    
    #######################
    #####   conv_1   ######
    #######################
    dim_in = 256
    convrelu1_1 = ConvReLU(
        rng,
        input=bw_input,
        image_shape=(batch_size, 1, dim_in, dim_in),
        filter_shape=(64, 1, 3, 3),
        border_mode=1
    )
    convrelu1_2 = ConvReLU(
        rng,
        input=convrelu1_1.output,
        image_shape=(batch_size, 64, dim_in, dim_in),
        filter_shape=(64, 64, 3, 3),
        border_mode=1,
        conv_stride=(2,2)
    )
    bn_1_output = bn.batch_normalization(convrelu1_2.output,
                               g, 
                               b, 
                               convrelu1_2.output.mean(axis=0, keepdims=True), 
                               convrelu1_2.output.std(axis=0, keepdims=True))#, mode=mode)
    
    #######################
    #####   conv_2   ######
    #######################
    
    convrelu2_1 = ConvReLU(
        rng,
        input=bn_1_output,
        image_shape=(batch_size, 64,  dim_in/2,  dim_in/2),
        filter_shape=(128, 64, 3, 3),
        border_mode=1
    )
    convrelu2_2 = ConvReLU(
        rng,
        input=convrelu2_1.output,
        image_shape=(batch_size, 128,  dim_in/2, dim_in/2),
        filter_shape=(128, 128, 3, 3),
        border_mode=1,
        conv_stride=(2,2)
    )
    bn_2_output = bn.batch_normalization(convrelu2_2.output,
                               g, 
                               b, 
                               convrelu2_2.output.mean(axis=0, keepdims=True), 
                               convrelu2_2.output.std(axis=0, keepdims=True))#, mode=mode)
    
    #######################
    #####   conv_3   ######
    #######################
    
    convrelu3_1 = ConvReLU(
        rng,
        input=bn_2_output,
        image_shape=(batch_size, 128, dim_in/2/2, dim_in/2/2),
        filter_shape=(256, 128, 3, 3),
        border_mode=1
    )
    convrelu3_2 = ConvReLU(
        rng,
        input=convrelu3_1.output,
        image_shape=(batch_size, 256, dim_in/2/2, dim_in/2/2),
        filter_shape=(256, 256, 3, 3),
        border_mode=1
    )
    convrelu3_3 = ConvReLU(
        rng,
        input=convrelu3_2.output,
        image_shape=(batch_size, 256, dim_in/2/2, dim_in/2/2),
        filter_shape=(256, 256, 3, 3),
        border_mode=1,
        conv_stride=(2,2)
    )
    bn_3_output = bn.batch_normalization(convrelu3_3.output,
                               g, 
                               b, 
                               convrelu3_3.output.mean(axis=0, keepdims=True), 
                               convrelu3_3.output.std(axis=0, keepdims=True))#, mode=mode)
    
    #######################
    #####   conv_4   ######
    #######################
    
    convrelu4_1 = ConvReLU(
        rng,
        input=bn_3_output,
        image_shape=(batch_size, 256, dim_in/2/2/2, dim_in/2/2/2),
        filter_shape=(512, 256, 3, 3),
        border_mode=1
    )
    convrelu4_2 = ConvReLU(
        rng,
        input=convrelu4_1.output,
        image_shape=(batch_size, 512, dim_in/2/2/2, dim_in/2/2/2),
        filter_shape=(512, 512, 3, 3),
        border_mode=1
    )
    convrelu4_3 = ConvReLU(
        rng,
        input=convrelu4_2.output,
        image_shape=(batch_size, 512, dim_in/2/2/2, dim_in/2/2/2),
        filter_shape=(512, 512, 3, 3),
        border_mode=1
    )
    bn_4_output = bn.batch_normalization(convrelu4_3.output,
                               g, 
                               b, 
                               convrelu4_3.output.mean(axis=0, keepdims=True), 
                               convrelu4_3.output.std(axis=0, keepdims=True))#, mode=mode)
    
    #######################
    #####   conv_5   ######
    #######################
    
    convrelu5_1 = ConvReLU(
        rng,
        input=bn_4_output,
        image_shape=(batch_size, 512, dim_in/2/2/2, dim_in/2/2/2),
        filter_shape=(512, 512, 3, 3),
        border_mode=2,
        conv_dilation=(2,2)
    )
    convrelu5_2 = ConvReLU(
        rng,
        input=convrelu5_1.output,
        image_shape=(batch_size, 512, dim_in/2/2/2, dim_in/2/2/2),
        filter_shape=(512, 512, 3, 3),
        border_mode=2,
        conv_dilation=(2,2)
    )
    convrelu5_3 = ConvReLU(
        rng,
        input=convrelu5_2.output,
        image_shape=(batch_size, 512, dim_in/2/2/2, dim_in/2/2/2),
        filter_shape=(512, 512, 3, 3),
        border_mode=2,
        conv_dilation=(2,2)
    )
    bn_5_output = bn.batch_normalization(convrelu5_3.output,
                               g, 
                               b, 
                               convrelu5_3.output.mean(axis=0, keepdims=True), 
                               convrelu5_3.output.std(axis=0, keepdims=True))#, mode=mode)
    
    #######################
    #####   conv_6   ######
    #######################
    
    convrelu6_1 = ConvReLU(
        rng,
        input=bn_5_output,
        image_shape=(batch_size, 512, dim_in/2/2/2, dim_in/2/2/2),
        filter_shape=(512, 512, 3, 3),
        border_mode=2,
        conv_dilation=(2,2)
    )
    convrelu6_2 = ConvReLU(
        rng,
        input=convrelu6_1.output,
        image_shape=(batch_size, 512, dim_in/2/2/2, dim_in/2/2/2),
        filter_shape=(512, 512, 3, 3),
        border_mode=2,
        conv_dilation=(2,2)
    )
    convrelu6_3 = ConvReLU(
        rng,
        input=convrelu6_2.output,
        image_shape=(batch_size, 512, dim_in/2/2/2, dim_in/2/2/2),
        filter_shape=(512, 512, 3, 3),
        border_mode=2,
        conv_dilation=(2,2)
    )
    bn_6_output = bn.batch_normalization(convrelu6_3.output,
                               g, 
                               b, 
                               convrelu6_3.output.mean(axis=0, keepdims=True), 
                               convrelu6_3.output.std(axis=0, keepdims=True))#, mode=mode)
    
    
    #######################
    #####   conv_7   ######
    #######################
    
    convrelu7_1 = ConvReLU(
        rng,
        input=bn_6_output,
        image_shape=(batch_size, 512, dim_in/2/2/2, dim_in/2/2/2),
        filter_shape=(512, 512, 3, 3),
        border_mode=1
    )
    convrelu7_2 = ConvReLU(
        rng,
        input=convrelu7_1.output,
        image_shape=(batch_size, 512, dim_in/2/2/2, dim_in/2/2/2),
        filter_shape=(512, 512, 3, 3),
        border_mode=1
    )
    convrelu7_3 = ConvReLU(
        rng,
        input=convrelu7_2.output,
        image_shape=(batch_size, 512, dim_in/2/2/2, dim_in/2/2/2),
        filter_shape=(512, 512, 3, 3),
        border_mode=1
    )
    bn_7_output = bn.batch_normalization(convrelu7_3.output,
                               g, 
                               b, 
                               convrelu7_3.output.mean(axis=0, keepdims=True), 
                               convrelu7_3.output.std(axis=0, keepdims=True))#, mode=mode)
    
    #######################
    #####   conv_8   ######
    #######################
    
    convrelu8_1 = DeConvReLU(
        rng,
        input=bn_7_output,
        image_shape=(batch_size, 512, dim_in/2/2, dim_in/2/2),
        filter_shape=(512, 256, 4, 4),
        border_mode=1
    )
    convrelu8_2 = ConvReLU(
        rng,
        input=convrelu8_1.output,
        image_shape=(batch_size, 256, dim_in/2/2, dim_in/2/2),
        filter_shape=(256, 256, 3, 3),
        border_mode=1
    )
    convrelu8_3 = ConvReLU(
        rng,
        input=convrelu8_2.output,
        image_shape=(batch_size, 256, dim_in/2/2, dim_in/2/2),
        filter_shape=(256, 256, 3, 3),
        border_mode=1
    )
    
    test_conv = ConvReLU(
        rng,
        input=convrelu8_3.output,
        image_shape=(batch_size, 1, dim_in/2/2, dim_in/2/2),
        filter_shape=(1, 256, 3, 3),
        border_mode=1
    )
    
    test_out_pre = (test_conv.output).repeat(2, axis=2).repeat(2, axis=3)
    test_out = (test_out_pre).repeat(2, axis=2).repeat(2, axis=3)
    

    cost = T.sqrt(T.mean(T.square(T.flatten(bw_input-test_out))))

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        cost,#T.mean(T.neq(input_x, final.output)),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )
    validate_model = theano.function(
        [index],
        cost,#T.mean(T.neq(input_x, final.output)),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )
    output_model = theano.function(
        [index],
        [input_x,corrupt_input,final.output],#T.mean(T.neq(input_x, final.output)),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )
    # create a list of all model parameters to be fit by gradient descent
    params = conv1_1.params + conv1_2.params + conv2_1.params + conv2_2.params + conv3_1.params + conv3_2.params + conv3_3.params + conv4_1.params + conv4_2.params + conv4_3.params + conv5_1.params + conv5_2.params + conv5_3.params + conv6_1.params + conv6_2.params + conv6_3.params + conv7_1.params + conv7_2.params + conv7_3.params + conv8_1.params + conv8_2.params + conv8_3.params
    #params = box10.params + box1.params + final.params
    # create a list of gradients for all model parameters
    #grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    #updates = [
    #    (param_i, param_i - learning_rate * grad_i)
    #    for param_i, grad_i in zip(params, grads)
    #]
    
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

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    ###############
    # TRAIN MODEL #
    ###############
    
    print('... training')
    # early-stopping parameters
    patience = 100000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
    
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
    #curframe = inspect.currentframe()
    #calframe = inspect.getouterframes(curframe, 2)

    # Print out summary
    print('Optimization complete.')
    print('Best validation error of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    return output_model(0)

"""
    
    # classify the values of the fully-connected sigmoidal layer
    #layer7 = LogisticRegression(input=layer7_input, n_in=1280, n_out=10)

    # the cost we minimize during training is the NLL of the model
    #cost = layer7.negative_log_likelihood(y)

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
    """
    """
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
    """
    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    
