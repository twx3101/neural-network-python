import numpy as np

from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_CIFAR10_data

"""
TODO: Overfit the network with 50 samples of CIFAR-10
"""
###########################################################################
#                           BEGIN OF YOUR CODE                            #
###########################################################################
net = FullyConnectedNet([10, 10, 10, 10])
data = get_CIFAR10_data(45, 5, 0)
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
