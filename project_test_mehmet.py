import numpy
import timeit
import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d, bn, abstract_conv
from project_util import download_images, prepare_image_sets
import scipy
import os

import sys

from project_util import shared_dataset, load_data
from project_nn import LogisticRegression, HiddenLayer, ConvReLU, DeConvReLU, train_nn, drop, \
    BatchNorm, Colorization_Softmax, Colorization_Decoding, ConvSubSample


def colorization(learning_rate=0.1, n_epochs=200,
                 ds_rate=None,
                 nkerns=[20, 50], batch_size=500, num_augment=80000, dim_in=256, verbose=True, dir_name='data'):
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

    new_path = os.path.join(
        os.path.split(__file__)[0], dir_name)
    if not os.path.isdir(new_path):
        download_images(dir_name, 3)
        prepare_image_sets(dir_name, batch_size=200)
    # train_set, valid_set, test_set =
    train_set_x, train_set_y = load_data(dir_name, theano_shared=True, ds=ds_rate)
    # Convert raw dataset to Theano shared variables.
    test_set_x = train_set_x
    test_set_y = train_set_y
    valid_set_x = train_set_x
    # test_set_x = shared_dataset(train_set_l_mat)
    # test_set_y = shared_dataset(train_set_ab_mat)
    # test_set_x, test_set_y = shared_dataset(test_set)
    # valid_set_x, valid_set_y = shared_dataset(valid_set)
    # train_set_x, train_set_y = shared_dataset(train_set)

    print('Current training data size is %i' % train_set_x.get_value(borrow=True).shape[0])
    print('Current validation data size is %i' % valid_set_x.get_value(borrow=True).shape[0])
    print('Current test data size is %i' % test_set_x.get_value(borrow=True).shape[0])

    # print('Current training data size is %i' % train_set_x.shape[0])
    # print('Current validation data size is %i' % valid_set_x.shape[0])
    # print('Current test data size is %i' % test_set_x.shape[0])

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]

    # n_train_batches = train_set_x.shape[0]
    # n_valid_batches = valid_set_x.shape[0]
    # n_test_batches = test_set_x.shape[0]

    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.matrix('y')  # the labels are presented as 1D vector of
    # [int] labels
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')
    bw_input = x.reshape((batch_size, 1, dim_in, dim_in)) - 50
    color_output = y.reshape((batch_size, 2, dim_in, dim_in))

    #######################
    #####   subsample ab space  ######
    #######################

    convsubsample = ConvSubSample(
        input=color_output,
        filter_shape=(1, 1, 1, 1),
        image_shape=(batch_size, 1, dim_in, dim_in),
        border_mode=0,
        conv_stride=(4, 4)
    )

    #######################
    #####   conv_1   ######
    #######################
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
        conv_stride=(2, 2)
    )
    bn_1 = BatchNorm(rng,
                     convrelu1_2.output,
                     image_shape=(batch_size, 64, dim_in / 2, dim_in / 2)
                     )
    #######################
    #####   conv_2   ######
    #######################

    convrelu2_1 = ConvReLU(
        rng,
        input=bn_1.output,
        image_shape=(batch_size, 64, dim_in / 2, dim_in / 2),
        filter_shape=(128, 64, 3, 3),
        border_mode=1
    )
    convrelu2_2 = ConvReLU(
        rng,
        input=convrelu2_1.output,
        image_shape=(batch_size, 128, dim_in / 2, dim_in / 2),
        filter_shape=(128, 128, 3, 3),
        border_mode=1,
        conv_stride=(2, 2)
    )
    bn_2 = BatchNorm(rng,
                     convrelu2_2.output,
                     image_shape=(batch_size, 128, dim_in / 2 / 2, dim_in / 2 / 2)
                     )
    #######################
    #####   conv_3   ######
    #######################

    convrelu3_1 = ConvReLU(
        rng,
        input=bn_2.output,
        image_shape=(batch_size, 128, dim_in / 2 / 2, dim_in / 2 / 2),
        filter_shape=(256, 128, 3, 3),
        border_mode=1
    )
    convrelu3_2 = ConvReLU(
        rng,
        input=convrelu3_1.output,
        image_shape=(batch_size, 256, dim_in / 2 / 2, dim_in / 2 / 2),
        filter_shape=(256, 256, 3, 3),
        border_mode=1
    )
    convrelu3_3 = ConvReLU(
        rng,
        input=convrelu3_2.output,
        image_shape=(batch_size, 256, dim_in / 2 / 2, dim_in / 2 / 2),
        filter_shape=(256, 256, 3, 3),
        border_mode=1,
        conv_stride=(2, 2)
    )
    bn_3 = BatchNorm(rng,
                     convrelu3_3.output,
                     image_shape=(batch_size, 256, dim_in / 2 / 2 / 2, dim_in / 2 / 2 / 2)
                     )

    #######################
    #####   conv_4   ######
    #######################

    convrelu4_1 = ConvReLU(
        rng,
        input=bn_3.output,
        image_shape=(batch_size, 256, dim_in / 2 / 2 / 2, dim_in / 2 / 2 / 2),
        filter_shape=(512, 256, 3, 3),
        border_mode=1
    )
    convrelu4_2 = ConvReLU(
        rng,
        input=convrelu4_1.output,
        image_shape=(batch_size, 512, dim_in / 2 / 2 / 2, dim_in / 2 / 2 / 2),
        filter_shape=(512, 512, 3, 3),
        border_mode=1
    )
    convrelu4_3 = ConvReLU(
        rng,
        input=convrelu4_2.output,
        image_shape=(batch_size, 512, dim_in / 2 / 2 / 2, dim_in / 2 / 2 / 2),
        filter_shape=(512, 512, 3, 3),
        border_mode=1
    )
    bn_4 = BatchNorm(rng,
                     convrelu4_3.output,
                     image_shape=(batch_size, 512, dim_in / 2 / 2 / 2, dim_in / 2 / 2 / 2)
                     )

    #######################
    #####   conv_5   ######
    #######################

    convrelu5_1 = ConvReLU(
        rng,
        input=bn_4.output,
        image_shape=(batch_size, 512, dim_in / 2 / 2 / 2, dim_in / 2 / 2 / 2),
        filter_shape=(512, 512, 3, 3),
        border_mode=2,
        conv_dilation=(2, 2)
    )
    convrelu5_2 = ConvReLU(
        rng,
        input=convrelu5_1.output,
        image_shape=(batch_size, 512, dim_in / 2 / 2 / 2, dim_in / 2 / 2 / 2),
        filter_shape=(512, 512, 3, 3),
        border_mode=2,
        conv_dilation=(2, 2)
    )
    convrelu5_3 = ConvReLU(
        rng,
        input=convrelu5_2.output,
        image_shape=(batch_size, 512, dim_in / 2 / 2 / 2, dim_in / 2 / 2 / 2),
        filter_shape=(512, 512, 3, 3),
        border_mode=2,
        conv_dilation=(2, 2)
    )
    bn_5 = BatchNorm(rng,
                     convrelu5_3.output,
                     image_shape=(batch_size, 512, dim_in / 2 / 2 / 2, dim_in / 2 / 2 / 2)
                     )

    #######################
    #####   conv_6   ######
    #######################

    convrelu6_1 = ConvReLU(
        rng,
        input=bn_5.output,
        image_shape=(batch_size, 512, dim_in / 2 / 2 / 2, dim_in / 2 / 2 / 2),
        filter_shape=(512, 512, 3, 3),
        border_mode=2,
        conv_dilation=(2, 2)
    )
    convrelu6_2 = ConvReLU(
        rng,
        input=convrelu6_1.output,
        image_shape=(batch_size, 512, dim_in / 2 / 2 / 2, dim_in / 2 / 2 / 2),
        filter_shape=(512, 512, 3, 3),
        border_mode=2,
        conv_dilation=(2, 2)
    )
    convrelu6_3 = ConvReLU(
        rng,
        input=convrelu6_2.output,
        image_shape=(batch_size, 512, dim_in / 2 / 2 / 2, dim_in / 2 / 2 / 2),
        filter_shape=(512, 512, 3, 3),
        border_mode=2,
        conv_dilation=(2, 2)
    )
    bn_6 = BatchNorm(rng,
                     convrelu6_3.output,
                     image_shape=(batch_size, 512, dim_in / 2 / 2 / 2, dim_in / 2 / 2 / 2)
                     )

    #######################
    #####   conv_7   ######
    #######################

    convrelu7_1 = ConvReLU(
        rng,
        input=bn_6.output,
        image_shape=(batch_size, 512, dim_in / 2 / 2 / 2, dim_in / 2 / 2 / 2),
        filter_shape=(512, 512, 3, 3),
        border_mode=1
    )
    convrelu7_2 = ConvReLU(
        rng,
        input=convrelu7_1.output,
        image_shape=(batch_size, 512, dim_in / 2 / 2 / 2, dim_in / 2 / 2 / 2),
        filter_shape=(512, 512, 3, 3),
        border_mode=1
    )
    convrelu7_3 = ConvReLU(
        rng,
        input=convrelu7_2.output,
        image_shape=(batch_size, 512, dim_in / 2 / 2 / 2, dim_in / 2 / 2 / 2),
        filter_shape=(256, 512, 3, 3),
        border_mode=1
    )
    bn_7 = BatchNorm(rng,
                     convrelu7_3.output,
                     image_shape=(batch_size, 256, dim_in / 2 / 2 / 2, dim_in / 2 / 2 / 2)
                     )

    #######################
    #####   conv_8   ######
    #######################

    convrelu8_1 = DeConvReLU(
        rng,
        input=bn_7.output,
        image_shape=(batch_size, 256, dim_in / 2 / 2, dim_in / 2 / 2),
        filter_shape=(256, 256, 4, 4),
        border_mode=1,
        conv_stride=(2, 2)
    )

    convrelu8_2 = ConvReLU(
        rng,
        input=convrelu8_1.output,
        image_shape=(batch_size, 256, dim_in / 2 / 2, dim_in / 2 / 2),
        filter_shape=(256, 256, 3, 3),
        border_mode=1
    )
    convrelu8_3 = ConvReLU(
        rng,
        input=convrelu8_2.output,
        image_shape=(batch_size, 256, dim_in / 2 / 2, dim_in / 2 / 2),
        filter_shape=(256, 256, 3, 3),
        border_mode=1
    )

    ########################
    #####   Softmax   ######
    ########################

    class8_313_rh = Colorization_Softmax(
        rng,
        input=convrelu8_3.output,
        image_shape=(batch_size, 256, dim_in / 2 / 2, dim_in / 2 / 2),
        filter_shape=(313, 256, 1, 1),
        border_mode=0
    )

    #########################
    #####   Decoding   ######
    #########################

    class8_ab = Colorization_Decoding(
        rng,
        input=class8_313_rh.output,
        image_shape=(batch_size, 313, dim_in / 2 / 2, dim_in / 2 / 2),
        filter_shape=(2, 313, 1, 1),
        border_mode=0
    )
    test_out = abstract_conv.bilinear_upsampling(class8_ab.output, 4)

    # test_out_pre = (class8_ab.output).repeat(2, axis=2).repeat(2, axis=3)
    # test_out = (test_out_pre).repeat(2, axis=2).repeat(2, axis=3)


    # test_out_pre = (convrelu8_3.output).repeat(2, axis=2).repeat(2, axis=3)
    # test_out = (test_out_pre).repeat(2, axis=2).repeat(2, axis=3)

    # test_conv = ConvReLU(
    #    rng,
    #    input=test_out,
    #    image_shape=(batch_size, 256, dim_in, dim_in),
    #    filter_shape=(2, 256, 3, 3),
    #    border_mode=1
    # )



    cost = T.sqrt(T.mean(T.square(T.flatten(y - test_out.flatten(2)))))
    """
    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        cost,#T.mean(T.neq(input_x, final.output)),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    validate_model = theano.function(
        [index],
        cost,#T.mean(T.neq(input_x, final.output)),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    """
    output_model = theano.function(
        [index],
        [bw_input, y, convsubsample.output, test_out.flatten(2)],  # T.mean(T.neq(input_x, final.output)),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # create a list of all model parameters to be fit by gradient descent
    params = convrelu1_1.params + convrelu1_2.params + bn_1.params + convrelu2_1.params + convrelu2_2.params + bn_2.params + convrelu3_1.params + convrelu3_2.params + convrelu3_3.params + bn_3.params + convrelu4_1.params + convrelu4_2.params + convrelu4_3.params + bn_4.params + convrelu5_1.params + convrelu5_2.params + convrelu5_3.params + bn_5.params + convrelu6_1.params + convrelu6_2.params + convrelu6_3.params + bn_6.params + convrelu7_1.params + convrelu7_2.params + convrelu7_3.params + bn_7.params + convrelu8_1.params + convrelu8_2.params + convrelu8_3.params + class8_313_rh.params + class8_ab.params

    def RMSprop(cost, params, lr=learning_rate, rho=0.9, epsilon=1e-6):
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

    updates = RMSprop(cost, params)

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
        for minibatch_index in range(1):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            cost_ij = train_model(minibatch_index)

            if (iter % 100 == 0) and verbose:
                print('training @ iter = ', iter)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch,
                       minibatch_index + 1,
                       n_train_batches,
                       cost_ij * 100.))
            """
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
            """
    end_time = timeit.default_timer()

    # Retrieve the name of function who invokes train_nn() (caller's name)
    # curframe = inspect.currentframe()
    # calframe = inspect.getouterframes(curframe, 2)

    # Print out summary
    print('Optimization complete.')
    # print('Best validation error of %f %% obtained at iteration %i, '
    #      'with test performance %f %%' %
    #      (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    return output_model(0)

