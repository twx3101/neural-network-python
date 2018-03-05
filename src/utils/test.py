import numpy as np

from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_CIFAR10_data
from src.utils.plot import plotGraphs
from src.data_utils_fer2013 import get_FER2013_data
import src.utils.analysis as an
import pickle as pkl

def test_fer_model(img_folder, model="/path/to/model"):
    """
    Given a folder with images, load the images (in lexico-graphical ordering
    according to the file name of the images) and your best model to predict
    the facial expression of each image.
    Args:
    - img_folder: Path to the images to be tested
    Returns:
    - preds: A numpy vector of size N with N being the number of images in
    img_folder.
    """
    preds = None
    ################## READ PKL FILE ##################################

    pk = open('net.pkl', 'rb')
    test = pkl.load(pk)
    print(test)

    ################## TEST NET ########################################

    probs = test.loss(data['X_test'], None)
    print(probs)
    preds = an.getClassifications(probs)

    return preds