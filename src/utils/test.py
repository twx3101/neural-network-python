import numpy as np
import pickle as pkl
import analysis as an

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