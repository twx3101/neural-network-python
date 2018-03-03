import numpy as np
import math
import random

from src.classifiers import softmax
from src.layers import (linear_forward, linear_backward, relu_forward,
                        relu_backward, dropout_forward, dropout_backward)



def random_init(n_in, n_out, weight_scale=5e-2, dtype=np.float32):
    """
    Weights should be initialized from a normal distribution with standard
    deviation equal to weight_scale and biases should be initialized to zero.

    Args:
    - n_in: The number of input nodes into each output.
    - n_out: The number of output nodes for each input.
    """
    W = None
    b = None
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################
    #reference : https://isaacchanghau.github.io/2017/05/24/Weight-Initialization-in-Artificial-Neural-Networks/
    # W = np.random.normal(0,weight_scale,(n_in, n_out))
    # b = np.zeros((n_out,)).T

    W = weight_scale * np.random.randn(n_in,n_out)
    b = np.zeros(n_out)
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return W, b



class FullyConnectedNet(object):
    """
    Implements a fully-connected neural networks with arbitrary size of
    hidden layers. For a network with N hidden layers, the architecture would
    be as follows:
    [linear - relu - (dropout)] x (N - 1) - linear - softmax
    The learnable params are stored in the self.params dictionary and are
    learned with the Solver.
    """
    def __init__(self, hidden_dims, input_dim=32*32*3, num_classes=10,
                 dropout=0, reg=0.0, weight_scale=1e-2, dtype=np.float32,
                 seed=None):
        """
        Initialise the fully-connected neural networks.
        Args:
        - hidden_dims: A list of the size of each hidden layer
        - input_dim: A list giving the size of the input
        - num_classes: Number of classes to classify.
        - dropout: A scalar between 0. and 1. determining the dropout factor.
        If dropout = 0., then dropout is not applied.
        - reg: Regularisation factor.

        """
        self.dtype = dtype
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.use_dropout = True if dropout > 0.0 else False
        if seed:
            np.random.seed(seed)
        self.params = dict()
        """
        TODO: Initialise the weights and bias for all layers and store all in
        self.params. Store the weights and bias of the first layer in keys
        W1 and b1, the weights and bias of the second layer in W2 and b2, etc.
        Weights and bias are to be initialised according to the Xavier
        initialisation (see manual).
        """
        #######################################################################
        #                           BEGIN OF YOUR CODE                        #
        #######################################################################
        # dims = np.hstack((input_dim, hidden_dims, num_classes))
        # weight_scale = 5e-2
        # for i in range(self.num_layers):
        #     self.params['W%d' % (i + 1)] = weight_scale * np.random.randn(dims[i], dims[i+1])
        #     self.params['b%d' % (i + 1)] = np.zeros(dims[i+1])
        dims = np.hstack((input_dim, hidden_dims, num_classes))
        for i in range(self.num_layers):
            self.params['W%d' % (i + 1)],self.params['b%d' % (i + 1)] = random_init(dims[i], dims[i+1])

        # W = weight_scale * np.random.rand(n_in,n_out)
        # b = np.zeros(n_out)
        #
        # for i in range(self.num_layers):
        #     if i == 0:
        #         self.params['W'+str(i+1)], self.params['b'+str(i+1)] = random_init(input_dim, hidden_dims[i], weight_scale,dtype)
        #     elif i == self.num_layers - 1:
        #         self.params['W'+str(i+1)], self.params['b'+str(i+1)] = random_init(hidden_dims[i-1], num_classes, weight_scale,dtype)
        #     else:
        #         self.params['W'+str(i+1)], self.params['b'+str(i+1)] = random_init(hidden_dims[i-1], hidden_dims[i], weight_scale,dtype)
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################
        # When using dropout we need to pass a dropout_param dictionary to
        # each dropout layer so that the layer knows the dropout probability
        # and the mode (train / test). You can pass the same dropout_param to
        # each dropout layer.
        self.dropout_params = dict()
        if self.use_dropout:
            self.dropout_params = {"train": True, "p": dropout}
            if seed is not None:
                self.dropout_params["seed"] = seed
        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.
        Args:        print()

        - X: Input data, numpy array of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].
        Returns:
        If y is None, then run a test-time forward pass of the model and
        return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.
        If y is not None, then run a training-time forward and backward pass
        and return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.param s, mapping
        parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        X = X.astype(self.dtype)
        linear_cache = dict()
        relu_cache = dict()
        dropout_cache = dict()
        """
        TODO: Implement the forward pass for the fully-connected neural
        network, compute the scores and store them in the scores variable.
        """
        #######################################################################
        #                           BEGIN OF YOUR CODE                        #
        #######################################################################
            # [linear - relu - (dropout)] x (N - 1) - linear - softmax
        # -------------------------------------if else dropout--------------------------------------
        # output_cache = {}
        # if self.use_dropout:
        #     for i in range(self.num_layers-1):
        #         if i == 0:
        #             linear_cache[i] = linear_forward(X,self.params['W'+str(i+1)],self.params['b'+str(i+1)])
        #         else:
        #             linear_cache[i] = linear_forward(output_cache[i-1],self.params['W'+str(i+1)],self.params['b'+str(i+1)])
        #         relu_cache[i]  =  relu_forward(linear_cache[i])
        #         dropout_cache['dropout'+str(i)],dropout_cache['dropout_mask'+str(i)] =\
        #                         dropout_forward(relu_cache[i],self.params['p'],self.params['train'],self.params['seed'])
        #         output_cache[i] = dropout_cache['dropout'+str(i)]
        #
        #     linear_cache[self.num_layers-1] = linear_forward(output_cache[self.num_layers-2],\
        #                             self.params['W'+str(self.num_layers)],self.params['b'+str(self.num_layers)])
        #     scores = linear_cache[self.num_layers-1]
        # else:
        #     for i in range(self.num_layers-1):
        #         if i==0:
        #             linear_cache[i] = linear_forward(X,self.params['W'+str(i+1)],self.params['b'+str(i+1)])
        #         else:
        #             linear_cache[i] = linear_forward(relu_cache[i-1],self.params['W'+str(i+1)],self.params['b'+str(i+1)])
        #         relu_cache[i]= relu_forward(linear_cache[i])
        #     linear_cache[self.num_layers-1] = linear_forward(relu_cache[self.num_layers-2],\
        #                             self.params['W'+str(self.num_layers)],self.params['b'+str(self.num_layers)])
        #     scores = linear_cache[self.num_layers-1]


#         for i in range(1,self.num_layers):
#                 if i == 1:
#                     input = X
#                 else:
#                     if self.use_dropout:
#                         input = dropout_cache['dropout'+str(i-1)]
#                     else:
#                         input = relu_cache[i-1]
#
# """
#                 linear_cache[i] = linear_forward(input,self.params["W"+str(i)],self.params["b"+str(i)])
#                 relu_cache[i] = relu_forward(linear_cache[i])
#
#                 if self.use_dropout:
#                     dropout_cache['dropout'+str(i)],dropout_cache['dropout_mask'+str(i)] = dropout_forward\
#                         (relu_cache[i],self.dropout_params["p"],self.dropout_params["train"],self.dropout_params["seed"])
#                         # (relu_cache[i],self.dropout_params["p"],self.dropout_params["train"])
#
#         if self.use_dropout:
#             scores = linear_forward(dropout_cache['dropout'+str(i)],self.params["W"+str(i+1)],self.params["b"+str(i+1)])
#         else:
#             scores = linear_forward(relu_cache[i] ,self.params["W"+str(i+1)],self.params["b"+str(i+1)])
# """
        # 231
        hidden_num = self.num_layers - 1
        scores = X
        cache_history = []
        L2reg = 0
        cache_history.append(scores)
        #
        for i in range(hidden_num):
            scores = linear_forward(scores, self.params['W%d' % (i + 1)],
                                                            self.params['b%d' % (i + 1)])
            cache_history.append(scores)
            scores =  relu_forward(scores)
            cache_history.append(scores)
            if self.use_dropout:

                scores, cache = dropout_forward(scores, self.dropout_params["p"],self.dropout_params["train"],self.dropout_params["seed"])
                # scores, cache = dropout_forward(scores, self.dropout_params["p"],self.dropout_params["train"])

                cache_history.append(cache)
            L2reg += np.sum(self.params['W%d' % (i + 1)] ** 2)

        i += 1
        scores = linear_forward(scores, self.params['W%d' % (i + 1)],\
                                                self.params['b%d' % (i + 1)])
        #cache_history.append(scores)
        #
        L2reg += np.sum(self.params['W%d' % (i + 1)] ** 2)
        L2reg *= 0.5 * self.reg
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################
        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores
        loss, grads = 0, dict()

        """
        TODO: Implement the backward pass for the fully-connected net. Store
        the loss in the loss variable and all gradients in the grads
        dictionary. Compute the loss with softmax. grads[k] has the gradients
        for self.params[k]. Add L2 regularisation to the loss function.
        NOTE: To ensure that your implementation matches ours and you pass the
        automated tests, make sure that your L2 regularization includes a
        factor of 0.5 to simplify the expression for the gradient.
        """
        #######################################################################
        #                           BEGIN OF YOUR CODE                        #
        #######################################################################

        # """""""""""""""""""""""""""" if else dropout """""""""""""""""""""""""""""""""""""
        # if self.use_dropout:
        #     for i in range(self.num_layers-1, 0,-1):
        #
        #         if(i== self.num_layers-1):
        #             dX, dW, db = linear_backward(dout,dropout_cache['dropout'+str(i-1)],\
        #                                             self.params['W' + str(i+1)],self.params['b' + str(i+1)])
        #         else:
        #             dX, dW, db = linear_backward(relu_back,dropout_cache['dropout'+str(i-1)],\
        #                         self.params['W' + str(i+1)],self.params['b' + str(i+1)])
        #
        #         dX_dropout_back = dropout_backward(dX,dropout_cache['dropout_mask' + str(i-1)],self.params['p'],self.params['train'])
        #         relu_back = relu_backward(dX_dropout_back,relu_cache[i-1])
        #         dW +=  (self.reg * self.params['W' + str(i+1)])
        #         grads['W' + str(i+1)] = dW
        #         grads['b' + str(i+1)] = db
        #     dX, dW, db = linear_backward(relu_back,X,\
        #                 self.params['W1'],self.params['b1'])
        #     dW +=  (self.reg * self.params['W1'])
        #     grads['W1'] = dWhttp://www.imperial.ac.uk/computing/
        #     grads['b1'] = db
        # else:
        #     for i in range(self.num_layers-1,0,-1):
        #
        #         if(i==self.num_layers-1):
        #             dX, dW,db = linear_backward(dout,relu_cache[i-1],self.params['W'+str(i+1)],\
        #                             self.params['b'+str(i+1)])
        #         else:
        #             dX, dW,db = linear_backward(relu_back, relu_cache[i-1],self.params['W'+str(i+1)],\
        #                             self.params['b'+str(i+1)])
        #         relu_back = relu_backward(dX, relu_cache[i-1])
        #         dW +=  (self.reg * self.params['W' + str(i+1)])
        #         # dW +=   self.params['W' + str(i+1)]
        #
        #         grads['W' + str(i+1)] = dW
        #         grads['b' + str(i+1)] = db
        #
        #     dX, dW, db = linear_backward(relu_back,X,\
        #                 self.params['W1'],self.params['b1'])
        #     dW +=  (self.reg * self.params['W1'])
        #     grads['W1'] = dW
        #     grads['b1'] = db
        #
        #
        # regularization_term = 0
        # for i in range(self.num_layers):
        #     regularization_term += 0.5 * self.reg * np.sum(self.params['W'+str(i+1)])**2
        #     # print(0.5 * self.reg * np.sum(self.params['W'+str(i+1)])**2)
        #     # print(self.params['W'+str(i+1)])
        # # loss +=  regularization_term/self.params['W1'].shape[0]
        # loss +=  regularization_term

# """
#         loss, dout = softmax(scores,y)
#
#         for j in range(self.num_layers,0,-1):
#
#             if j == self.num_layers:
#
#                 if self.use_dropout:
#                     input_linear = dropout_cache['dropout'+str(j-1)]
#                 else:
#                     input_linear = relu_cache[j-1]
#
#                 dX, dW,db = linear_backward(dout, input_linear, self.params["W" + str(j)],self.params["b" + str(j)])
#
#             else:
#                 if self.use_dropout:
#                     dX = dropout_backward(dX,dropout_cache['dropout_mask'+str(j)],self.dropout_params["p"],self.dropout_params["train"])
#
#                 dX = relu_backward(dX,linear_cache[j])
#
#                 if self.use_dropout and j != 1:
#                     input_linear = dropout_cache['dropout'+str(j-1)]
#                 else:
#                     if j == 1:
#                         input_linear = X
#                     else:
#                         input_linear = relu_cache[j-1]
#
#                 dX, dW,db = linear_backward(dX,input_linear,self.params["W" + str(j)],self.params["b" + str(j)])
#
#             grads["W" + str(j)] = dW
#             grads["b" + str(j)] = db
#
#         regularization_term = 0.0
#         for i in range(1,self.num_layers+1):
#             regularization_term += np.sum((self.params["W"+str(i)])**2)
#         regularization_term *= regularization_term* 0.5*self.reg
#
#         loss += regularization_term

        #--------------------- NEW LOSS #########################
        # need to uncomment function below
    #     loss += self.regularization_2()
    #
 

        ###########################231##############################################
        loss, dout = softmax(scores, y)
        loss += L2reg

        dout, grads['W%d' % (i + 1)], grads['b%d' % (i + 1)] = linear_backward(dout, cache_history.pop(),self.params['W%d' % (i + 1)],self.params['b%d' % (i + 1)])
        grads['W%d' % (i + 1)] += self.reg * self.params['W%d' % (i + 1)]

        i -= 1
        while i >= 0:
            if self.use_dropout:
                dout = dropout_backward(dout, cache_history.pop())

            dout = relu_backward(dout, cache_history.pop())
            dout, grads['W%d' % (i + 1)], grads['b%d' % (i + 1)] = linear_backward(dout, cache_history.pop(),self.params['W%d' % (i + 1)],self.params['b%d' % (i + 1)])

            grads['W%d' % (i + 1)] += self.reg * self.params['W%d' % (i + 1)]
            i -= 1


        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################
        return loss, grads
    #def regularization_2(self):
    #     regularization = 0.0
    #     for i in range(1,self.num_layers+1):
    #         W = self.params["W"+str(i)]
    #         regularization += np.sum(W**2)
    #     regularization *= 0.5 * self.reg
    #     return regularization
