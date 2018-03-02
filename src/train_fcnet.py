import numpy as np

from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_CIFAR10_data
from src.utils.plot import plotGraphs
from src.data_utils_fer2013 import get_FER2013_data
import src.utils.analysis as an
import pickle as pkl


"""
TODO: Use a Solver instance to train a TwoLayerNet that achieves at least 50%
accuracy on the validation set.
"""
###########################################################################
#                           BEGIN OF YOUR CODE                            #
###########################################################################
# data = get_CIFAR10_data()
# #
# INPUT_DIMS = np.prod(data["X_train"].shape[1:])
# HIDDEN_DIMS = np.asarray([80,80])
# NUM_CLASS = 10
# # net = FullyConnectedNet(HIDDEN_DIMS,INPUT_DIMS,NUM_CLASS,dropout = 0.6,seed = 39)
# net = FullyConnectedNet(HIDDEN_DIMS,INPUT_DIMS,NUM_CLASS)
#
# solver = Solver(net, data,update_rule='sgd',optim_config={\
#                 'learning_rate': 1e-3},\
#             num_epochs=30,\
#             batch_size = 10,\
#             print_every=10000)
# # solver.train()


# plotGraphs(net, solver)

INPUT_NODE = [1000,1000]
# INPUT_NODE = [150,150,150,150,15,15, 400]
#
#
data = get_FER2013_data(22967,5742,3588)
print(data['X_train'][0])
print(np.shape(data['X_train'][0]))


targets = data['y_test']
# #data = get_FER2013_data(49,1 ,0)
#
INPUT_DIMS = np.prod(data["X_train"].shape[1:])
HIDDEN_DIMS = np.asarray(INPUT_NODE)
NUM_CLASS = 7
#net = FullyConnectedNet(HIDDEN_DIMS,INPUT_DIMS,num_classes=NUM_CLASS,dropout=0.6,seed =300)
net = FullyConnectedNet(HIDDEN_DIMS,INPUT_DIMS,num_classes=NUM_CLASS)

solver = Solver(net, data,update_rule='sgd_momentum',optim_config={\
                                                                   'learning_rate': 0.001, 'momentum': 0.5},\
             num_epochs=20,\
             batch_size = 100,\
             lr_decay=0.001,\
             print_every =1000)
solver.train()
plotGraphs(net, solver)

################## OUTPUT TO PKL FILE ###########################

output = open('net.pkl', 'wb')
pkl.dump(solver.model, output, -1)
output.close()

################## READ PKL FILE ##################################

pk = open('net.pkl', 'rb')
test = pkl.load(pk)
print(test)

################## TEST NET ########################################

probs = test.loss(data['X_test'], None)
print(probs)
classifications = an.getClassifications(probs)
cm = an.confusionMatrix(classifications, targets, 7)
print(classifications)
print(cm)
recall = an.averageRecall(cm, 1)
precision = an.precisionRate(cm, 1)
f1measure = an.f1(precision, recall)
classification = an.classificationRate(cm, 7)
print("recall class 1: ", recall)
print("precision class 1: ", precision)
print("f1 class 1: ", f1measure)
print("classification rate: ", classification)



##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
