import cv2
import glob
import numpy as np
import pandas as pd
import os
# image_list = []
# for file_count , filename in enumerate(glob.glob("/vol/bitbucket/395ML_NN_Data/datasets/FER2013/Train/*.jpg")):
#     img = cv2.imread(filename)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img_expanded = img[:, :, np.newaxis]
#     image_list.append(img_expanded)
#     if file_count % 10== 0:
#         print()
#         print("\t %s files processed"%file_count)
#
#
# for file_count, filename in enumerate(glob.glob("/vol/bitbucket/395ML_NN_Data/datasets/FER2013/Test/*.jpg")):
#     img = cv2.imread(filename)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img_expanded = img[:, :, np.newaxis]
#     image_list.append(img_expanded)
#     if file_count % 10== 0:
#         print()
#         print("\t %s files processed"%file_count)
#
# image_list = np.array(image_list)
# print(np.shape(image_list))
#
# np.save('fer2013_data.npy', image_list)
#
# df = pd.read_csv("/vol/bitbucket/395ML_NN_Data/datasets/FER2013/labels_public.txt")
# emotion =  np.array(df["emotion"])
#
# np.save('fer2013_labels.npy', emotion)

def load_FER2013(ROOT):
    """ load all of FER2013 """
    X = np.load(os.path.join(ROOT, 'fer2013_data.npy'))
    Y = np.load(os.path.join(ROOT, 'fer2013_labels.npy'))
    X = X.transpose(0,1,2,3)
    Xtr = X[:28709]
    Ytr = Y[:28709]
    Xte = X[28709:]
    Yte = Y[28709:]
    return Xtr, Ytr, Xte, Yte


def get_FER2013_data(num_training=49000, num_validation=1000, num_test=1000,
                     subtract_mean=True):

    FER2013_dir = '/homes/wt814/machinelearning/optional/neuralnets'
    X_train, y_train, X_test, y_test = load_FER2013(FER2013_dir)

    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # if subtract_mean:
    #     mean_image = np.mean(X_train, axis=0)
    #     X_train -= mean_image
    #     X_val -= mean_image
    #     X_test -= mean_image

    # X_train = X_train.transpose(0, 3, 1, 2).copy()
    # X_val = X_val.transpose(0, 3, 1, 2).copy()
    # X_test = X_test.transpose(0, 3, 1, 2).copy()

    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }
