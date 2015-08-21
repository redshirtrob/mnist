#!/usr/bin/env python

import cPickle
import gzip
import numpy as np
import theano
import theano.tensor as T
import timeit

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        # Initialize the Weight Matrix
        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )

        # Initialize the biases
        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            )
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute the prediction as class
        # whose probabilit is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # model parameters
        self.params = [self.W, self.b]

        # model input
        self.input = input

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Number of errors in the minibatch over the total number of
        examples in the minibatch."""

        # Check that y has the same dimension as y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # Check that y is the correct datatype
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

def load_data(dataset):
    def shared_dataset(data_xy):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
        return shared_x, T.cast(shared_y, 'int32')
    
    with gzip.open(dataset, 'rb') as f:
        train_set, valid_set, test_set = cPickle.load(f)
        
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000,
                           dataset='mnist.pkl.gz',
                           batch_size=600):
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute the number of minibatches
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    # build model
    print '... building the model'

    # symbolic variable for index into the minibatch
    index = T.lscalar()

    # symbolic variables for input
    x = T.matrix('x')
    y = T.ivector('y')

    # construct the logistic regression instance
    # each MNIST image has size 28x28
    classifier = LogisticRegression(input=x, n_in=28*28, n_out=10)

    # cost to minimize
    cost = classifier.negative_log_likelihood(y)

    # compile Theano function for mistakes
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # compute the gradient cost wrt theta = (W, b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # specify how to update the model
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compile training function and specify model parameter updates
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # Train the model
    print '... training the model'
    # early stopping parameters
    patience = 5000 # minimum number of examples to examine
    patience_increase = 2 # how long to wait when a new best is found
    improvement_threshold = 0.995 # relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = np.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter + 1) % validation_frequency == 0:
                # computer zero-one loss on validation set (XXX)
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                print 'epoch {}, minibatch {}/{}, validation error {} %'.format(
                    epoch,
                    minibatch_index + 1,
                    n_train_batches,
                    this_validation_loss * 100.
                )

                if this_validation_loss < best_validation_loss:
                    # improve patience if loss is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss

                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print '    epoch {}, minibatch {}/{}, test error of best model {} %'.format(
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        test_score * 100.
                    )

                    with open('best_model.pkl', 'w') as f:
                        cPickle.dump(classifier, f)
                        
            if patience <= iter:
                done_looping = True
                break
            
    end_time = timeit.default_timer()

    print 'Optimization complete with best validation score of {} %%'.format(
        best_validation_loss * 100.
    )
    print 'with test performance {} %%'.format(test_score * 100.)

    print 'The code run for {} epochs, with {} epochs/sec'.format(
        epoch, 1. * epoch / (end_time - start_time)
    )

def predict():
    classifier = cPickle.load(open('best_model.pkl'))

    predict_model = theano.function(inputs=[classifier.input], outputs=classifier.y_pred)

    dataset = 'mnist.pkl.gz'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_set_x[:10])
    print 'Predicted values for the first 10 examples in test set:'
    print predicted_values

if __name__ == '__main__':
    sgd_optimization_mnist()
