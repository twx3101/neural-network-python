import numpy as np


def softmax(logits, y):
    """
    Computes the loss and gradient for softmax classification.

    Args:
    - logits: A numpy array of shape (N, C)
    - y: A numpy array of shape (N,). y represents the labels corresponding to
    logits, where y[i] is the label of logits[i], and the value of y have a
    range of 0 <= y[i] < C

    ///////  N = number of example
    //////   C = class (emotion)

    Returns (as a tuple):
    - loss: Loss scalar
    - dlogits: Loss gradient with respect to logits
    """
    loss, dlogits = None, None
    """
    TODO: Compute the softmax loss and its gradient using no explicit loops
    Store the loss in loss and the gradient in dW. If you are not careful
    here, it is easy to run into numeric instability. Don't forget the
    regularization!
    """
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################
#  unclean matrix X = [ [matrix ex1], [matrix ex2] ,.., [matrix exn]] : 1xN
#  clean matrix X = [[vector ex1], [vector ex2] ,.... [vector exn]] : NxK [K = column of vector(feature)]
#  y = (cleanX)W + b
#  weight  W = [[vector emotion], [vector emotion],.. [vector emotion]] : KxM [M = column of weight to ex]
#  yout =  NxK KXM = NxM (number of example x number of class) = N x C
#     ex N0 = [0.23 0.12 0.32 0.43 0.12 0.53 ..... 0.343] (example0 has 10 emotions )
#
#    before softmax   expected class : logits = [ [0], [1], [1] , [5], .....]
#                     true label :        y  = [ [0], [3], [2], ....]
#
    #print(logits.shape, " @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    # print('label shape: ', y.shape)
    # print(y)
    # print("logits shape: ", logits.shape)
    nom_first_term = logits
    nom_second_term = np.max(logits, axis=1, keepdims= True)
    nominator = np.exp(nom_first_term - nom_second_term)

    denominator = np.sum(nominator,axis=1, keepdims = True)

    probability_vector = nominator/denominator
    # print(probability_vector.shape , " prob vector#####")
    # print(probability_vector)
    example_size = logits.shape[0]
    # compute in vector form
    # loss_first_term  = (-1)*(y)
    # print(loss_first_term)
    # print(loss_first_term.shape, " loss_first_term:::::::::::::::::::;")
    #
    # loss_second_term = np.log(np.sum(np.exp(probability_vector),axis = 1))
    # print(loss_second_term, " ~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print("loss_second_term shape: ", loss_second_term.shape)
    # loss = (loss_first_term + loss_second_term)/example_size
    # loss = np.sum(loss)
    # print("loss  shape: ", loss.shape)
    # print(loss)
    # print(np.sum(loss))
    loss = -np.sum(np.log(probability_vector[np.arange(example_size),y]))/example_size
    # print(probability_vector)
    # print(y)
    # print('\n prob vec row SUM : ',np.sum(probability_vector[0]))
    # print('\n\n')
    # print(probability_vector[0]," : prob vec ,     y[]: ",y[0])
    # print(probability_vector[0,y[0]])
    # print('\n')
    # print(probability_vector[1]," : prob vec,  y[]: ",y[1])
    # print('\n')
    # print(probability_vector[2]," : prob vec,  y[]: ",y[2])
    # print('\n')
    # print(probability_vector[np.arange(example_size),y[0]])




    # loss = np.log((np.sum(y))/np.sum(np.exp(probability_vector)))
    # loss = loss*(-1/example_size)


    dlogits = probability_vector.copy()
    dlogits[np.arange(example_size), y] -= 1
    dlogits /= example_size

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return loss, dlogits
