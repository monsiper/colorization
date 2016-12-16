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
from project_nn import  ConvReLU, DeConvReLU,  \
    BatchNorm, Colorization_Softmax, Colorization_Decoding, ConvSubSample,Colorization_PriorBoost, Conv

class optimizer(object):
    def __init__(self,type='ADAM'):
        self.type = type

    def adam(cost, params, learning_rate=0.0002, beta1=0.1, beta2=0.001, epsilon=1e-8, gamma=1 - 1e-7):
        updates = []
        grads = T.grad(cost, params)
        t = theano.shared(numpy.float32(1))
        beta1_decay = (1. - beta1) * gamma ** (t - 1)

        for param_i, g in zip(params, grads):
            param_val = param_i.get_value(borrow=True)
            m = theano.shared(numpy.zeros(param_val.shape, dtype=theano.config.floatX))
            v = theano.shared(numpy.zeros(param_val.shape, dtype=theano.config.floatX))

            m_biased = (beta1_decay * g) + ((1. - beta1_decay) * m)  #
            v_biased = (beta2 * g ** 2) + ((1. - beta2) * v)
            m_hat = m_biased / (1 - (1. - beta1) ** t)
            v_hat = v_biased / (1 - (1. - beta2) ** t)
            g_t = m_hat / (T.sqrt(v_hat) + epsilon)
            param_new = param_i - (learning_rate * g_t)

            updates.append((m, m_biased))
            updates.append((v, v_biased))
            updates.append((param_i, param_new))
        updates.append((t, t + 1.))
        return updates

    def RMSprop(cost, params, lr=0.0001, rho=0.9, epsilon=1e-6):
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

    def update(self,cost,params):
        if self.type=='ADAM':
            return self.adam(cost, params)
        if self.type=='RMSprop':
            return self.RMSprop(cost, params)


class colorization(object):
    def __init__(self,
                 learning_rate=0.1, 
                 n_epochs=200,
                 ds_rate=None, 
                 batch_size=500, 
                 num_augment=80000, 
                 dim_in=256, 
                 verbose=True, 
                 dir_name='data',
                 train_batches=1
                ):
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
        
        self.rng = numpy.random.RandomState(23455)
    
        
    def build_model(self,
                    batch_size=1,
                    dim_in=256
                    ):
        
        self.dim_in = dim_in
        self.batch_size = batch_size
    
        # start-snippet-1
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.matrix('y')  # the labels are presented as 1D vector of
        
        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print('... building the model')
        self.bw_input = self.x.reshape((batch_size, 1, dim_in, dim_in)) - 50
        self.data_ab_enc = self.y.reshape((batch_size, 313, 64, 64))
    
        #######################
        #####   subsample ab space  ######
        #######################
        
        self.prior_boost = Colorization_PriorBoost(
            self.rng,
            input = self.data_ab_enc,
            batch_size=batch_size,
            gamma=0,
            verbose=True
        )
        
        
        #######################
        #####   conv_1   ######
        #######################
        self.convrelu1_1 = ConvReLU(
            self.rng,
            input=self.bw_input,
            image_shape=(batch_size, 1, dim_in, dim_in),
            filter_shape=(64, 1, 3, 3),
            border_mode=1
        )
        self.convrelu1_2 = ConvReLU(
            self.rng,
            input=self.convrelu1_1.output,
            image_shape=(batch_size, 64, dim_in, dim_in),
            filter_shape=(64, 64, 3, 3),
            border_mode=1,
            conv_stride=(2, 2)
        )
        self.bn_1 = BatchNorm(
                        self.rng,
                        self.convrelu1_2.output,
                        image_shape=(batch_size, 64, dim_in / 2, dim_in / 2)
        )
        #######################
        #####   conv_2   ######
        #######################
    
        self.convrelu2_1 = ConvReLU(
            self.rng,
            input=self.bn_1.output,
            image_shape=(batch_size, 64, dim_in / 2, dim_in / 2),
            filter_shape=(128, 64, 3, 3),
            border_mode=1
        )
        self.convrelu2_2 = ConvReLU(
            self.rng,
            input=self.convrelu2_1.output,
            image_shape=(batch_size, 128, dim_in / 2, dim_in / 2),
            filter_shape=(128, 128, 3, 3),
            border_mode=1,
            conv_stride=(2, 2)
        )
        self.bn_2 = BatchNorm(
            self.rng,
            self.convrelu2_2.output,
            image_shape=(batch_size, 128, dim_in / 2 / 2, dim_in / 2 / 2)
        )
        #######################
        #####   conv_3   ######
        #######################
    
        self.convrelu3_1 = ConvReLU(
            self.rng,
            input=self.bn_2.output,
            image_shape=(batch_size, 128, dim_in / 2 / 2, dim_in / 2 / 2),
            filter_shape=(256, 128, 3, 3),
            border_mode=1
        )
        self.convrelu3_2 = ConvReLU(
            self.rng,
            input=self.convrelu3_1.output,
            image_shape=(batch_size, 256, dim_in / 2 / 2, dim_in / 2 / 2),
            filter_shape=(256, 256, 3, 3),
            border_mode=1
        )
        self.convrelu3_3 = ConvReLU(
            self.rng,
            input=self.convrelu3_2.output,
            image_shape=(batch_size, 256, dim_in / 2 / 2, dim_in / 2 / 2),
            filter_shape=(256, 256, 3, 3),
            border_mode=1,
            conv_stride=(2, 2)
        )
        self.bn_3 = BatchNorm(
            self.rng,
            self.convrelu3_3.output,
            image_shape=(batch_size, 256, dim_in / 2 / 2 / 2, dim_in / 2 / 2 / 2)
        )
    
        #######################
        #####   conv_4   ######
        #######################
    
        self.convrelu4_1 = ConvReLU(
            self.rng,
            input=self.bn_3.output,
            image_shape=(batch_size, 256, dim_in / 2 / 2 / 2, dim_in / 2 / 2 / 2),
            filter_shape=(512, 256, 3, 3),
            border_mode=1
        )
        self.convrelu4_2 = ConvReLU(
            self.rng,
            input=self.convrelu4_1.output,
            image_shape=(batch_size, 512, dim_in / 2 / 2 / 2, dim_in / 2 / 2 / 2),
            filter_shape=(512, 512, 3, 3),
            border_mode=1
        )
        self.convrelu4_3 = ConvReLU(
            self.rng,
            input=self.convrelu4_2.output,
            image_shape=(batch_size, 512, dim_in / 2 / 2 / 2, dim_in / 2 / 2 / 2),
            filter_shape=(512, 512, 3, 3),
            border_mode=1
        )
        self.bn_4 = BatchNorm(
            self.rng,
            self.convrelu4_3.output,
            image_shape=(batch_size, 512, dim_in / 2 / 2 / 2, dim_in / 2 / 2 / 2)
        )
    
        #######################
        #####   conv_5   ######
        #######################
    
        self.convrelu5_1 = ConvReLU(
            self.rng,
            input=self.bn_4.output,
            image_shape=(batch_size, 512, dim_in / 2 / 2 / 2, dim_in / 2 / 2 / 2),
            filter_shape=(512, 512, 3, 3),
            border_mode=2,
            conv_dilation=(2, 2)
        )
        self.convrelu5_2 = ConvReLU(
            self.rng,
            input=self.convrelu5_1.output,
            image_shape=(batch_size, 512, dim_in / 2 / 2 / 2, dim_in / 2 / 2 / 2),
            filter_shape=(512, 512, 3, 3),
            border_mode=2,
            conv_dilation=(2, 2)
        )
        self.convrelu5_3 = ConvReLU(
            self.rng,
            input=self.convrelu5_2.output,
            image_shape=(batch_size, 512, dim_in / 2 / 2 / 2, dim_in / 2 / 2 / 2),
            filter_shape=(512, 512, 3, 3),
            border_mode=2,
            conv_dilation=(2, 2)
        )
        self.bn_5 = BatchNorm(
            self.rng,
            self.convrelu5_3.output,
            image_shape=(batch_size, 512, dim_in / 2 / 2 / 2, dim_in / 2 / 2 / 2)
        )
    
        #######################
        #####   conv_6   ######
        #######################
    
        self.convrelu6_1 = ConvReLU(
            self.rng,
            input=self.bn_5.output,
            image_shape=(batch_size, 512, dim_in / 2 / 2 / 2, dim_in / 2 / 2 / 2),
            filter_shape=(512, 512, 3, 3),
            border_mode=2,
            conv_dilation=(2, 2)
        )
        self.convrelu6_2 = ConvReLU(
            self.rng,
            input=self.convrelu6_1.output,
            image_shape=(batch_size, 512, dim_in / 2 / 2 / 2, dim_in / 2 / 2 / 2),
            filter_shape=(512, 512, 3, 3),
            border_mode=2,
            conv_dilation=(2, 2)
        )
        self.convrelu6_3 = ConvReLU(
            self.rng,
            input=self.convrelu6_2.output,
            image_shape=(batch_size, 512, dim_in / 2 / 2 / 2, dim_in / 2 / 2 / 2),
            filter_shape=(512, 512, 3, 3),
            border_mode=2,
            conv_dilation=(2, 2)
        )
        self.bn_6 = BatchNorm(
            self.rng,
            self.convrelu6_3.output,
            image_shape=(batch_size, 512, dim_in / 2 / 2 / 2, dim_in / 2 / 2 / 2)
        )
    
        #######################
        #####   conv_7   ######
        #######################
    
        self.convrelu7_1 = ConvReLU(
            self.rng,
            input=self.bn_6.output,
            image_shape=(batch_size, 512, dim_in / 2 / 2 / 2, dim_in / 2 / 2 / 2),
            filter_shape=(512, 512, 3, 3),
            border_mode=1
        )
        self.convrelu7_2 = ConvReLU(
            self.rng,
            input=self.convrelu7_1.output,
            image_shape=(batch_size, 512, dim_in / 2 / 2 / 2, dim_in / 2 / 2 / 2),
            filter_shape=(512, 512, 3, 3),
            border_mode=1
        )
        self.convrelu7_3 = ConvReLU(
            self.rng,
            input=self.convrelu7_2.output,
            image_shape=(batch_size, 512, dim_in / 2 / 2 / 2, dim_in / 2 / 2 / 2),
            filter_shape=(256, 512, 3, 3),
            border_mode=1
        )
        self.bn_7 = BatchNorm(
            self.rng,
            self.convrelu7_3.output,
            image_shape=(batch_size, 256, dim_in / 2 / 2 / 2, dim_in / 2 / 2 / 2)
        )
    
        #######################
        #####   conv_8   ######
        #######################
    
        self.convrelu8_1 = DeConvReLU(
            self.rng,
            input=self.bn_7.output,
            image_shape=(batch_size, 256, dim_in / 2 / 2, dim_in / 2 / 2),
            filter_shape=(256, 256, 4, 4),
            border_mode=1,
            conv_stride=(2, 2)
        )
    
        self.convrelu8_2 = ConvReLU(
            self.rng,
            input=self.convrelu8_1.output,
            image_shape=(batch_size, 256, dim_in / 2 / 2, dim_in / 2 / 2),
            filter_shape=(256, 256, 3, 3),
            border_mode=1
        )
        self.convrelu8_3 = ConvReLU(
            self.rng,
            input=self.convrelu8_2.output,
            image_shape=(batch_size, 256, dim_in / 2 / 2, dim_in / 2 / 2),
            filter_shape=(256, 256, 3, 3),
            border_mode=1
        )
    
                
        ########################
        #####   Softmax   ######
        ########################
    
        self.class8_313_rh = Colorization_Softmax(
            self.rng,
            input=self.convrelu8_3.output,
            image_shape=(batch_size, 256, dim_in / 2 / 2, dim_in / 2 / 2),
            filter_shape=(313, 256, 1, 1),
            border_mode=0
        )
    
        #########################
        #####   Decoding   ######
        #########################
    
        self.network_output = self.class8_313_rh.output
    
        # create a list of all model parameters to be fit by gradient descent
        self.params = self.convrelu1_1.params + self.convrelu1_2.params + self.bn_1.params + self.convrelu2_1.params + self.convrelu2_2.params + self.bn_2.params + self.convrelu3_1.params + self.convrelu3_2.params + self.convrelu3_3.params + self.bn_3.params + self.convrelu4_1.params + self.convrelu4_2.params + self.convrelu4_3.params + self.bn_4.params + self.convrelu5_1.params + self.convrelu5_2.params + self.convrelu5_3.params + self.bn_5.params + self.convrelu6_1.params + self.convrelu6_2.params + self.convrelu6_3.params + self.bn_6.params + self.convrelu7_1.params + self.convrelu7_2.params + self.convrelu7_3.params + self.bn_7.params + self.convrelu8_1.params + self.convrelu8_2.params + self.convrelu8_3.params + self.class8_313_rh.params
    
        self.net_out_for_cost_func = T.log((self.network_output.transpose((0,2,3,1))).reshape((batch_size, 4096, 313))+1e-7)
        self.data_ab_enc_for_cost_func = self.data_ab_enc.reshape((batch_size, 4096, 313))
        self.cost = -(((self.prior_boost.output).reshape((batch_size,4096))*(self.net_out_for_cost_func*self.data_ab_enc_for_cost_func).sum(axis=2)).sum(axis=1)).sum()
    
    def train_network(
        self,
        dir_name='./data/',
        learning_rate=0.0001,
        n_epochs=200,
        ds_rate=None, 
        batch_size=1, 
        dim_in=256, 
        verbose=True, 
        train_batches=1,
<<<<<<< HEAD
        batch_ind=1,
        batch_num=4
=======
        type='ADAM'
>>>>>>> 45339573393d5938aaabe5abb101a59ac7734106
    ):
        
        if not os.path.isdir(dir_name):
            download_images(dir_name, 221)
            prepare_image_sets(dir_name, batch_size=200)
        self.train_set_x, self.train_set_y = load_data(dir_name, theano_shared=True, ds=ds_rate,batch_ind=batch_ind,batch_num=batch_num)
        # Convert raw dataset to Theano shared variables.
        self.valid_set_x = self.train_set_x
    
        print('Current training data size is %i' % self.train_set_x.get_value(borrow=True).shape[0])
    
        # compute number of minibatches for training, validation and testing
        self.n_train_batches = self.train_set_x.get_value(borrow=True).shape[0]
    
        self.n_train_batches //= self.batch_size
    
        # allocate symbolic variables for the data
        self.index = T.lscalar() 

        optimizer_engine = optimizer(type=type)

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
    
        self.updates = optimizer_engine.update(self.cost, self.params)
    
        self.train_model = theano.function(
            [self.index],
            self.cost,
            updates=self.updates,
            givens={
                self.x: self.train_set_x[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.y: self.train_set_y[self.index * self.batch_size: (self.index + 1) * self.batch_size]
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
        validation_frequency = min(self.n_train_batches, patience // 2)
    
        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.
        start_time = timeit.default_timer()
    
        epoch = 0
        done_looping = False
        
        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(self.n_train_batches):
    
                iter = (epoch - 1) * self.n_train_batches + minibatch_index
    
                cost_ij = self.train_model(minibatch_index)
    
                if (iter % self.n_train_batches == 0) and verbose:
                    print('training @ iter = ', iter)
                    print('epoch %i, minibatch %i/%i, loss of %f' %
                          (epoch,
                           minibatch_index + 1,
                           self.n_train_batches,
                           cost_ij))
            
        end_time = timeit.default_timer()
        # Print out summary
        print('Optimization complete.')
        
    def test_network(
        self,
        ind = 1,
        dir_name='./data/',
        ds_rate=None,
        batch_ind=1
        ):
        self.test_set_x, self.test_set_y = load_data(dir_name, theano_shared=True, ds=ds_rate,batch_ind=batch_ind,batch_num=1)
        self.n_test_batches = self.test_set_x.get_value(borrow=True).shape[0]
        self.n_test_batches //= self.batch_size
        print('Current test data size is %i' % self.test_set_x.get_value(borrow=True).shape[0])
        self.output_model = theano.function(
            [self.index],
            [self.network_output,self.bw_input,self.prior_boost.output,self.data_ab_enc,self.y],
            givens={
                self.x: self.test_set_x[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.y: self.test_set_y[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            }
        )
        return self.output_model(ind)
    
    