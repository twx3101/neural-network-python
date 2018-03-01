import numpy as np

from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_CIFAR10_data
from src.utils.plot import plotGraphs
from src.data_utils_fer2013 import get_FER2013_data
import src.utils.analysis as an


"""
TODO: Use a Solver instance to train a TwoLayerNet that achieves at least 50%
accuracy on the validation set.
"""
###########################################################################
#                           BEGIN OF YOUR CODE                            #
###########################################################################
data = get_CIFAR10_data()
#
INPUT_DIMS = np.prod(data["X_train"].shape[1:])
HIDDEN_DIMS = np.asarray([80,80])
NUM_CLASS = 10
# net = FullyConnectedNet(HIDDEN_DIMS,INPUT_DIMS,NUM_CLASS,dropout = 0.6,seed = 39)
net = FullyConnectedNet(HIDDEN_DIMS,INPUT_DIMS,NUM_CLASS)

solver = Solver(net, data,update_rule='sgd',optim_config={\
                'learning_rate': 1e-3},\
            num_epochs=30,\
            batch_size = 10,\
            print_every=10000)
# solver.train()


# plotGraphs(net, solver)

#INPUT_NODE = [80,80]
# INPUT_NODE = [150,150,150,150,15,15, 400]
#
#
# #data = get_FER2013_data(14355,14354 ,3588)
# #data = get_FER2013_data(49,1 ,0)
#
# INPUT_DIMS = np.prod(data["X_train"].shape[1:])
# HIDDEN_DIMS = np.asarray(INPUT_NODE)
# NUM_CLASS = 7
# # net = FullyConnectedNet(HIDDEN_DIMS,INPUT_DIMS,num_classes=NUM_CLASS,dropout=0.6,seed =300)
# net = FullyConnectedNet(HIDDEN_DIMS,INPUT_DIMS,num_classes=NUM_CLASS)
#
# solver = Solver(net, data,update_rule='sgd_momentum',optim_config={\
#                 'learning_rate': 0.001, 'momentum': 0.0001},\
#             num_epochs=40,\
#             batch_size = 10,\
#             lr_decay=0.001,\
#             print_every =1000)
#solver.train()
plotGraphs(net, solver)

# test_solver = Solver(net, data['X_test'], ...)
targets = data['y_test']
probs = solver.model.loss(data['X_test'], None)
print(probs)
classifications = an.getClassifications(probs)
cm = an.confusionMatrix(classifications, targets, 10)
print(classifications)
print(cm)
##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
