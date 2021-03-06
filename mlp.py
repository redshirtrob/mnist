#!/usr/bin/env python

import cPickle
import gzip
import numpy as np
import theano
import theano.tensor as T
import timeit

from logreg import LogisticRegression, load_data

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        self.input = input

        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None else activation(lin_output)
        )

        self.params = [self.W, self.b]

class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_hidden_2, n_out):
        
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        self.hiddenLayer2 = HiddenLayer(
            rng=rng,
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_hidden_2,
            activation=T.tanh
        )

        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer2.output,
            n_in=n_hidden_2,
            n_out=n_out
        )

        # L1 norm
        self.L1 = (
            abs(self.hiddenLayer.W).sum() +
            abs(self.hiddenLayer2.W).sum() +
            abs(self.logRegressionLayer.W).sum()
        )
        print 'self.L1={}'.format(self.L1)

        # square of L2 norm
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum() +
            (self.hiddenLayer2.W ** 2).sum() +
            (self.logRegressionLayer.W ** 2).sum()
        )
        print 'self.L2_sqr={}'.format(self.L2_sqr)

        # Negative log likelihood
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        print 'self.negative_log_likelihood={}'.format(self.negative_log_likelihood)

        self.errors = self.logRegressionLayer.errors

        self.params = (
            self.hiddenLayer.params +
            self.hiddenLayer2.params +
            self.logRegressionLayer.params
        )
        print 'self.params={}'.format(self.params)

        self.input = input

def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=500, n_hidden_2=50):
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]


    # number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    #import ipdb; ipdb.set_trace()
    
    # Build the model
    print '... building the model'

    # symbolic variables for the data
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    rng = np.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_hidden_2=n_hidden_2,
        n_out=10
    )

    # minimize negative log likelihood & regularization terms during training
    cost = (
        classifier.negative_log_likelihood(y) + 
        L1_reg * classifier.L1 +
        L2_reg * classifier.L2_sqr
    )
    print 'cost={}'.format(cost)

    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index+1) * batch_size],
            y: test_set_y[index * batch_size:(index+1) * batch_size],
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index+1) * batch_size],
            y: valid_set_y[index * batch_size:(index+1) * batch_size],
        }
    )

    # Compute the gradient of cost wrt theata
    print classifier.params
    gparams = [T.grad(cost, param) for param in classifier.params]
    print gparams

    # Specify how to update the parameters of the model
    updates = [
        (param, param - learning_rate * gparam) for param, gparam in zip(classifier.params, gparams)
    ]

    # Compile training function
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size:(index+1) * batch_size],
            y: train_set_y[index * batch_size:(index+1) * batch_size],
        }
    )

    # Train the model
    print '... training'

    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience/2)

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    print 'Number of minibatches: {}'.format(n_train_batches)
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1

        epoch_start_time = timeit.default_timer()
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print 'epoch {}, minibatch {}/{}, validation error {} %'.format(
                    epoch,
                    minibatch_index + 1,
                    n_train_batches,
                    this_validation_loss * 100.
                )

                # If this is the best validation score up until now
                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # Test against tes set
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print '    epoch {}, minibatch {}/{}, test error of best model {} %'.format(
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        test_score * 100.
                    )
                    
            if patience <= iter:
                done_looping = True
                break
            
        epoch_end_time = timeit.default_timer()
        print '    epoch {}, ran for {}s'.format(
            epoch,
            (epoch_end_time - epoch_start_time)
        )

    end_time = timeit.default_timer()
    print 'Optimization complete.  Best validation score of {} %'.format(
        best_validation_loss * 100.
    )
    print 'obtained at iteration {}, with test performance {} %'.format(
        best_iter + 1,
        test_score * 100.
    )

    
if __name__ == '__main__':
    test_mlp()
