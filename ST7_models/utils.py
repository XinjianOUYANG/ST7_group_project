from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow.python.lib.io import file_io
from keras.preprocessing.image import ImageDataGenerator

import skimage
from skimage.transform import rescale, resize


# configure image data augmentation
datagen = ImageDataGenerator(horizontal_flip=True)

# make a prediction using test-time augmentation
def tta_prediction(datagen, model, image, n_examples):
    # convert image into dataset
    samples = np.expand_dims(image, 0)
    # prepare iterator
    it = datagen.flow(samples, batch_size=n_examples)
    # make predictions for each augmented image
    yhats = model.predict_generator(it, steps=n_examples, verbose=0)
    # sum across predictions
    summed = np.sum(yhats, axis=0)
    # argmax across classes
    return np.argmax(summed)
 
 # evaluate a model on a dataset using test-time augmentation
def tta_evaluate_model(model, testX, testY):
    # configure image data augmentation
    datagen = ImageDataGenerator(horizontal_flip=True)
    # define the number of augmented images to generate per test set image
    n_examples_per_image = 7
    yhats = list()
    for i in range(len(testX)):
        # make augmented prediction
        yhat = tta_prediction(datagen, model, testX[i], n_examples_per_image)
        # store for evaluation
        yhats.append(yhat)
    # calculate accuracy
    testY_labels = np.argmax(testY, axis=1)
    acc = accuracy_score(testY_labels, yhats)
    return acc

def get_data(dataset_path, Resize_pixelsize = 197):
    
    file_stream = file_io.FileIO(dataset_path, mode='r')
    data = pd.read_csv(file_stream)
    data['pixels'] = data['pixels'].apply(lambda x: [int(pixel) for pixel in x.split()])
    X, Y = data['pixels'].tolist(), data['emotion'].values
    X = np.array(X, dtype='float32').reshape(-1,48,48,1)
    X = X/255.0
   
    X_res = np.zeros((X.shape[0], Resize_pixelsize,Resize_pixelsize,3))
    for ind in range(X.shape[0]): 
        sample = X[ind]
        sample = sample.reshape(48, 48)
        image_resized = resize(sample, (Resize_pixelsize, Resize_pixelsize), anti_aliasing=True)
        X_res[ind,:,:,:] = image_resized.reshape(Resize_pixelsize,Resize_pixelsize,1)

    Y_res = np.zeros((Y.size, 7))
    Y_res[np.arange(Y.size),Y] = 1    
    
    return  X, X_res, Y_res