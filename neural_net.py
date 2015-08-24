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
        
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        self.input = input
        self.n_out = n_out

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
        
class MLN(object):
    def __init__(self, rng, input, n_in, n_out, hidden_layers):
        assert len(hidden_layers) > 0
        
        prev_hidden_layer = None

        # Create the hidden layers
        self.hidden_layers = list()
        for hidden_layer in hidden_layers:
            if prev_hidden_layer is None:
                hl_n_in = n_in
                hl_input = input
            else:
                hl_n_in = prev_hidden_layer.n_out
                hl_input = prev_hidden_layer.output
                
            hl = HiddenLayer(
                rng=rng,
                input=hl_input,
                n_in=hl_n_in,
                n_out=hidden_layer[0],
                activation=hidden_layer[1]
            )
            self.hidden_layers.append(hl)
            prev_hidden_layer = hl
        
        self.logRegressionLayer = LogisticRegression(
            input=prev_hidden_layer.output,
            n_in=prev_hidden_layer.n_out,
            n_out=n_out
        )

        self.L1 = reduce(lambda x,y: x+y, [abs(hl.W).sum() for hl in self.hidden_layers])
        self.L2_sqr = reduce(lambda x,y: x+y, [abs(hl.W ** 2).sum() for hl in self.hidden_layers])

        # Negative log likelihood
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )

        self.errors = self.logRegressionLayer.errors
        
        self.params = (
            reduce(lambda x,y: x+y, [hl.params for hl in self.hidden_layers]) +
            self.logRegressionLayer.params
        )

        self.input = input

def test_mln(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=20, hidden_layers=None):
    if hidden_layers is None:
        hidden_layers = [(500, T.tanh)]
        
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

    # construct the MLN class
    classifier = MLN(
        rng=rng,
        input=x,
        n_in=28 * 28,
        n_out=10,
        hidden_layers=hidden_layers
    )

    # minimize negative log likelihood & regularization terms during training
    cost = (
        classifier.negative_log_likelihood(y) +
        L1_reg * classifier.L1 +
        L2_reg * classifier.L2_sqr
    )

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
            
    end_time = timeit.default_timer()
    print 'Optimization complete.  Best validation score of {} %'.format(
        best_validation_loss * 100.
    )
    print 'obtained at iteration {}, with test performance {} %'.format(
        best_iter + 1,
        test_score * 100.
    )

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


if __name__ == '__main__':
    test_mln(hidden_layers=[(500, T.tanh), (50, T.tanh)])
