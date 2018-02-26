import numpy as np

from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_CIFAR10_data

"""
TODO: Use a Solver instance to train a TwoLayerNet that achieves at least 50% 
accuracy on the validation set.
"""
###########################################################################
#                           BEGIN OF YOUR CODE                            #
###########################################################################
net = FullyConnectedNet([10, 10, 10, 10])
data = get_CIFAR10_data(49000, 1000, 1000)
solver = Solver(net, data,
                                update_rule='sgd',
            optim_config={
                                    'learning_rate': 1e-2,
                                },
                                lr_decay=0.95,
                                num_epochs=20,
                                batch_size=50,
                                print_every=100)
solver.train()

##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
