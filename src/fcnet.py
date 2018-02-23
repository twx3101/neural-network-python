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

    W = weight_scale * np.random.rand(n_in,n_out)
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
        #print("Number of input node: ",input_dim)
        #print("Number of hidden nodes: ", hidden_dims, " @@@@@@@@@@@@@@@@@@@@@@")
        # could be more generalized by using loop according to number of layers
        # self.params['W1'],self.params['b1'] = random_init(input_dim, hidden_dims[0],weight_scale,dtype)
        # self.params['W2'],self.params['b2'] = random_init(hidden_dims[0], hidden_dims[1], weight_scale,dtype)
        # print("input dim: ", input_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ weight 2 hidden layers + output ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #self.params['W1'],self.params['b1'] = random_init(input_dim, hidden_dims[0],weight_scale,dtype)
        #self.params['W2'],self.params['b2'] = random_init(hidden_dims[0], hidden_dims[1], weight_scale,dtype)
        #self.params['W3'],self.params['b3'] = random_init(hidden_dims[1], num_classes, weight_scale,dtype)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ weigh 1 hidden layer + output ~~~~~~~~~~~~~~~~~~~~#
        # self.params['W1'],self.params['b1'] = random_init(input_dim, hidden_dims[0],weight_scale,dtype)
        # self.params['W2'],self.params['b2'] = random_init(hidden_dims[0], hidden_dims[1], weight_scale,dtype)

        # for i in range(self.num_layers):
        #     no = i+1
        #     if i == 0:
        #         self.params['W'+str(i)], self.params['b'+str(i)] = random_init(input_dim, hidden_dims[i], weight_scale,dtype)
        #     elif i == self.num_layers - 1:
        #         self.params['W'+str(i)], self.params['b'+str(i)] = random_init(hidden_dim[i], num_classes, weight_scale,dtype)
        #     else:
        #         self.params['W'+str(i)], self.params['b'+str(i)] = random_init(hidden_dims[i-1], hidden_dims[i], weight_scale,dtype)
        for i in range(self.num_layers):
            # no = i+1
        #    print(" i : ",i)
            if i == 0:
                self.params['W'+str(i)], self.params['b'+str(i)] = random_init(input_dim, hidden_dims[i], weight_scale,dtype)
            elif i == self.num_layers - 1:
                self.params['W'+str(i)], self.params['b'+str(i)] = random_init(hidden_dims[i-1], num_classes, weight_scale,dtype)
            else:
                self.params['W'+str(i)], self.params['b'+str(i)] = random_init(hidden_dims[i-1], hidden_dims[i], weight_scale,dtype)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


        # print("hidden_dims[0]/ input node: ",hidden_dims)
        # print("number of class: ", num_classes)
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
        # print('param[w1] shape: ', self.params['W1'].shape)
        # print('param[w2] shape: ', self.params['W2'].shape)
        # print('param[b1] shape: ', self.params['b1'].shape)
        # print('param[b2] shape: ', self.params['b2'].shape)
        #
        #
        #
        #print("X shape: ", X.shape , " @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        ######################### 2 layers ###########################################
        # linear_cache['input_to_layer1'] =  linear_forward(X,self.params['W1'],self.params['b1'])
        # relu_cache['relu_layer1']       =  relu_forward(linear_cache['input_to_layer1'])
        # dropout_cache['dropout_out_layer1'],dropout_cache['dropout_mask_layer1'] =\
        #                 dropout_forward(relu_cache['relu_layer1'])
        #
        # linear_cache['input_to_layer2'] =  linear_forward(dropout_cache['dropout_out_layer1'],self.params['W2'],self.params['b2'])
        # relu_cache['relu_layer2']       =  relu_forward(linear_cache['input_to_layer2'])
        # dropout_cache['dropout_out_layer2'],dropout_cache['dropout_mask_layer2'] =\
        #                 dropout_forward(relu_cache['relu_layer2'])
        #
        # linear_cache['layer2_to_output'] = linear_forward(dropout_cache['dropout_out_layer2'],\
        #                                                     self.params['W3'],self.params['b3'])
        # scores = linear_cache['layer2_to_output']
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #######################linear dropout relu linear #####################
        # linear_cache['input_to_layer1'] =  linear_forward(X,self.params['W1'],self.params['b1'])
        # dropout_cache['dropout_out_layer1'],dropout_cache['dropout_mask_layer1'] =\
        #             dropout_forward(linear_cache['input_to_layer1'])
        # relu_cache['relu_layer1']       =  relu_forward(dropout_cache['dropout_out_layer1'])
        # linear_cache['layer1_to_layer2'] = linear_forward(relu_cache['relu_layer1'],\
        #                                                     self.params['W2'],self.params['b2'])
        # scores = linear_cache['layer1_to_layer2']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # linear_cache['input_to_layer1'] =  linear_forward(X,self.params['W'+str(i)],self.params['b'+str(i)])
        # relu_cache['relu_layer1']       =  relu_forward(linear_cache['input_to_layer1'])
        # dropout_cache['dropout_out_layer1'],dropout_cache['dropout_mask_layer1'] =\
        #                 dropout_forward(relu_cache['relu_layer1'])
        # linear_cache['layer1_to_layer2'] = linear_forward(relu_cache['relu_layer1'],\
        #                                                     self.params['W2'],self.params['b2'])
        # scores = linear_cache['layer1_to_layer2']

        # for i in range(self.num_layers-1):
        #     linear_cache[i] =  linear_forward(X,self.params['W'+str(i)],self.params['b'+str(i)])
        #     relu_cache[i]  =  relu_forward(linear_cache[i])
        #     dropout_cache['dropout'+str(i)],dropout_cache['dropout_mask'+str(i)] =\
        #                     dropout_forward(relu_cache[i])

        # linear_cache[self.num_layers-1] = linear_forward(relu_cache[self.num_layers-2],\
        #                         self.params['W'+str(self.num_layers-1)],self.params['b'+str(self.num_layers-1)])
        # scores = linear_cache[self.num_layers-1]
        
        output_cache = {}
        for i in range(self.num_layers-1):
            if i == 0:
                linear_cache[i] = linear_forward(X,self.params['W'+str(i)],self.params['b'+str(i)])
            else:
                linear_cache[i] = linear_forward(output_cache[i-1],self.params['W'+str(i)],self.params['b'+str(i)])
            relu_cache[i]  =  relu_forward(linear_cache[i])
            dropout_cache['dropout'+str(i)],dropout_cache['dropout_mask'+str(i)] =\
                            dropout_forward(relu_cache[i])
            output_cache[i] = dropout_cache['dropout'+str(i)]
    

        linear_cache[self.num_layers-1] = linear_forward(relu_cache[self.num_layers-2],\
                                self.params['W'+str(self.num_layers-1)],self.params['b'+str(self.num_layers-1)])
        scores = linear_cache[self.num_layers-1]

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

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2 hidden layers + output ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # loss, dout = softmax(linear_cache['layer2_to_output'],y)
        # dX_layer_output_2, dW_layer_output_2, db_layer_output_2 = linear_backward(dout,dropout_cache['dropout_out_layer2'],\
        #                                     self.params['W3'],self.params['b3'])
        # dX_dropout_back2 = dropout_backward(dX_layer_output_2,dropout_cache['dropout_mask_layer2'])
        # relu_back2 = relu_backward(dX_dropout_back2,relu_cache['relu_layer2'])
        # dX_layer2_1, dW_layer2, db_layer2 = linear_backward(relu_back2,linear_cache['input_to_layer2'],\
        #                                         self.params['W2'],self.params['b2'])
        #
        # dX_dropout_back1 = dropout_backward(dX_layer2_1,dropout_cache['dropout_mask_layer1'])
        # relu_back1 = relu_backward(dX_dropout_back1,relu_cache['relu_layer2'])
        # dX_layer1_input, dW_layer1, db_layer1 = linear_backward(relu_back1,linear_cache['input_to_layer1'],\
        #                                                     self.params['W1'],self.params['b1'])
        # #updating for use next iteration
        # dW_layer1 =dW_layer1 + (self.reg * self.params['W1'])
        # dW_layer2 = dW_layer2+ (self.reg * self.params['W2'])
        # dW_layer_output_2 = dW_layer_output_2+ (self.reg * self.params['W3'])
        #
        # # store in dictionary for use next iteration
        #
        # grads['W1'] = dW_layer1
        # grads['W2'] = dW_layer2
        # grads['W3'] = dW_layer_output_2
        # grads['b1'] = db_layer1
        # grads['b2'] = db_layer2
        # grads['b3'] = db_layer_output_2
        #
        #
        # # assume just 2 layers
        # regularization_term = self.reg * np.sum(self.params['W1']**2)+ np.sum(self.params['W2'])**2\
        #                                 +np.sum(self.params['W3']**2)*0.5
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        #~~~~~~~~~~~~~~~~~~~~~~~~ 1 hidden layer + output ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
        loss, dout = softmax(linear_cache[self.num_layers-1],y)

        dout_cache = {}

        for i in range(self.num_layers-1, 0,-1):

            if(i== self.num_layers-1):
                dX, dW, db = linear_backward(dout,dropout_cache['dropout'+str(i-1)],\
                                                self.params['W' + str(i)],self.params['b' + str(i)])
            else:
                dX, dW, db = linear_backward(relu_back,dropout_cache['dropout'+str(i-1)],\
                            self.params['W' + str(i)],self.params['b' + str(i)])

            dX_dropout_back = dropout_backward(dX,dropout_cache['dropout_mask' + str(i-1)])
            relu_back = relu_backward(dX_dropout_back,relu_cache[i-1])
            dW +=  (self.reg * self.params['W' + str(i)])
            grads['W' + str(i)] = dW
            grads['b' + str(i)] = db

        dX, dW, db = linear_backward(relu_back,X,\
                    self.params['W0'],self.params['b0'])
        dW +=  (self.reg * self.params['W0'])
        grads['W0'] = dW
        grads['b0'] = db
        
        regularization_term = 0
        for i in range(self.num_layers):
            regularization_term += 0.5 * self.reg * np.sum(self.params['W'+str(i)])**2
        
        loss +=  regularization_term
        
        # need to recheck second argument
        # dX_layer2_1, dW_layer2_1, db_layer2_1 = linear_backward(dout,dropout_cache['dropout_out_layer1'],\
        #                                     self.params['W2'],self.params['b2'])
        # dX_dropout_back = dropout_backward(dX_layer2_1,dropout_cache['dropout_mask_layer1'])
        # relu_back = relu_backward(dX_dropout_back,dX_layer2_1)
        # dX_layer1_input, dW_layer1, db_layer1 = linear_backward(relu_back,X,\
        #                                                     self.params['W1'],self.params['b1'])

        # for i in range(self.num_layers-2, -1,-1):
        #     dX, dW, db = linear_backward(dout,dropout_cache['dropout'+str(i)],\
        #                                             self.params['W' + str(i)],self.params['b' + str(i)])
        #     dX_dropout_back = dropout_backward(dX,dropout_cache['dropout_mask' + str(i)])
        #     relu_back = relu_backward(dX_dropout_back,relu_cache[i])

        #     dW +=  (self.reg * self.params['W' + str(i)])
        #     grads['W' + str(i)] = dW
        #     grads['b' + str(i)] = db
        #updating for use next iteration
        # dW_layer1 =dW_layer1 + (self.reg * self.params['W1'])
        # dW_layer2_1 = dW_layer2_1+ (self.reg * self.params['W2'])
        # store in dictionary for use next iteration

        # grads['W1'] = dW_layer1
        # grads['W2'] = dW_layer2_1
        # grads['b1'] = db_layer1
        # grads['b2'] = db_layer2_1

        # assume just 2 layers
        # regularization_term = self.reg * np.sum(self.params['W1']**2)+ np.sum(self.params['W2']**2)/2.0

        # loss +=  regularization_term

        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################
        return loss, grads
