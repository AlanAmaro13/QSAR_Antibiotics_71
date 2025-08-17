import h5py 
import numpy as np
from typing import * 
import tensorflow as tf
import matplotlib.pyplot as plt 
import random
import os 
from telegram import Bot
import glob
import keras
import time 
import sys
import asyncio

def standard_callbacks(folder_name: str, patiences: list, folder_path: str = '../Models', 
                       monitor: str = 'val_metric_accuracy', flow_direction = 'max') -> tuple:
    '''
    Description:
        This functions returns the usual callbacks used during training, such as EarlyStopping, ReduceLearningRate, Checkpoint and CSV Logger. 
        It also generates the folder in which all the data will be storage.

        By default we consider the next values: 
            * EarlyStopping restores the best weights
            * The reduce on the lr is by 0.8 until we reached 1e-6
            * Checkpoint only saves the best model each epoch

    Args: 
        folder_name (str): Refers to folder's name
        patiences (list): Refers to the patience values for EarlyStopping and ReduceOnLR
        folder_path (str): Refers to path where the folder will be created
        monitor (str): Variable to be monitored, by default corresponds to 'val_metric_accuracy'
        patiences (list): Refers to the patiences values for EarlyStopping, 

    Return: tuple containing
        EarlyStopping (early)
        ReduceLearningRate (reduce_lr)
        CheckPoint (check)
        CSVlogger (csv_logger)
    '''
    
    _path = folder_path + '/' + folder_name 
    os.makedirs(_path, exist_ok = True) # Make the dir 

    return (tf.keras.callbacks.EarlyStopping(monitor = monitor, patience = patiences[0], restore_best_weights=True, mode = flow_direction),
            
            tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.8,
                              patience= patiences[1], min_lr=1e-6 ), 
           
           tf.keras.callbacks.ModelCheckpoint(
            filepath=  os.path.join(_path, '{}.keras'.format(folder_name) ),
            save_weights_only=False,
            monitor=monitor,
            mode= flow_direction,
            save_freq = 'epoch',
            save_best_only=True), 

            keras.callbacks.CSVLogger( os.path.join(_path, 'training.log') )
           )

def normalization_WL(x, m1, m2, p = False) -> np.ndarray:
    '''Normalize the array by taking a max and min value over a sample. 

    Args:
        x: A NumPy array of shape (n, m), where n is the number of samples and m is the number of features.

    Returns:
        A NumPy array of the same shape (n, m), normalized by sample.
    '''
    
    mi = np.min(x, axis = 0, keepdims=True)
    ma = np.max(x, axis = 0, keepdims=True)
    if p: print(ma)
    if p: print(mi)
 
    return ((x - ma) / (mi - ma)  * (m2-1)) + m1 

def metric_accuracy(y_true, y_pred): # Accuracy for liner prediction
    epsilon = tf.keras.backend.epsilon()
    
    return 100 - (tf.reduce_mean(tf.abs((y_true - y_pred) / (y_true + epsilon))) * 100)

def get_plot_model(model: keras.models.Model, folder_path:str ):
    '''
    Description:
        This functions does the plot of the model and save it into the give path.

    Args: 
        model (keras.models.Model): This arg corresponds to the model function
        path (str): This path corresponds to the folder where the image will be storage

    Returns
        None
    '''

    keras.utils.plot_model(model, 
                          to_file = folder_path + '/model.png', 
                          show_shapes = True,
                          show_layer_names = True)

def model_training(model: keras.models.Model, folder_path: str, batch_size: int, 
                   num_epochs: int, x_train: np.ndarray, y_train: np.ndarray, 
                   x_val: np.ndarray, y_val:np.ndarray, 
                   callbacks: tuple): 

    start_time = time.time()

    tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir= os.path.join(folder_path, 'logs')  , histogram_freq=1)


    model_trained = model.fit(x= x_train, 
                        y= y_train, 
                        epochs=num_epochs, 
                        batch_size=batch_size,
                        validation_data = (x_val, y_val),
                        shuffle=True,
                        callbacks =[
                            tensorboard_callback, 
                            callbacks[0], 
                            callbacks[1], 
                            callbacks[2], 
                            callbacks[3] 
                        ],
                        verbose=1 )

    end_time = time.time()
    minutes = (end_time - start_time)//60
    print("Time for training: {:10.4f}s".format(minutes ))

    return model_trained 

def model_training_WL(model: keras.models.Model, folder_path: str, batch_size: int, 
                   num_epochs: int, x_train: np.ndarray, y_train: np.ndarray, 
                   x_val: np.ndarray, y_val:np.ndarray, 
                   callbacks: tuple, WL: dict): 

    start_time = time.time()

    tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir= os.path.join(folder_path, 'logs')  , histogram_freq=1)


    model_trained = model.fit(x= x_train, 
                        y= y_train, 
                        epochs=num_epochs, 
                        batch_size=batch_size,
                        validation_data = (x_val, y_val),
                        shuffle=True,
                        callbacks =[
                            tensorboard_callback, 
                            callbacks[0], 
                            callbacks[1], 
                            callbacks[2], 
                            callbacks[3] 
                        ],
                        class_weight = WL,
                        verbose=1 )

    end_time = time.time()
    minutes = (end_time - start_time)//60
    print("Time for training: {:10.4f}s".format(minutes ))

    return model_trained 

def do_graphics(model_trained, title: str, ylabel: tuple, folder_path:str,  metric: str = 'metric_accuracy'):
    '''
    Description:
        This functions generates the graphics of accuracy and loss over the epochs. 

    Args:
        model_trained (keras.models.Model): Refers to the output of the model_training function
        title (str): Title of the accuracy graphic
        ylabel (tuple): Title of the y-axis of the loss and accuracy graphics. 
        folder_path (str): Folder path to saved the images
        metric (str): Metric used to evaluted the model, by default is metric_accuracy
    '''
    # Loss Graphic
    plt.figure()  # Create a new figure
    plt.plot(model_trained.history['loss'])
    plt.plot(model_trained.history['val_loss'])
    plt.title('Loss Graphic')
    plt.ylabel(ylabel[0])
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'] )
    plt.savefig(os.path.join(folder_path, 'Loss.png') )
    plt.show()
    plt.close()  

    # Accuracy plot
    plt.figure()  # Create a new figure
    plt.plot(model_trained.history[metric ])
    plt.plot(model_trained.history['val_'+metric])
    plt.title(title)
    plt.ylabel(ylabel[1])
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'] )
    plt.savefig(os.path.join(folder_path, 'Accuracy.png') )
    plt.show()
    plt.close()  

def evaluate_model_regression(model: keras.models.Model, x_test, y_test):

    # Evaluamos el modelo
    score = model.evaluate(x = x_test, y = y_test)

    print ("-> Loss = " + str(score[0]))
    print ("-> Test Accuracy = " + str(score[1]))
    print ("-> R2 Accuracy = " + str(score[2]))

    return (score[0], score[1], score[2])


def evaluate_model(model: keras.models.Model, x_test, y_test):

    # Evaluamos el modelo
    score = model.evaluate(x = x_test, y = y_test)

    print ("-> Loss = " + str(score[0]))
    print ("-> Test Accuracy = " + str(score[1]))

    return (score[0], score[1])
