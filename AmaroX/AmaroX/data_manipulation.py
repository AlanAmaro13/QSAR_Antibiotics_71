# Libraries
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

# --------------- Load Data
def load_data(folder: str, size: list = [0, 0, 0] ) ->  Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Description:
        The next function takes in the path of the folder where the train, val y test dataset are storage, the function iterates
        over the folder and select the .h5 files. In returns the x, y datasets of each train, test y val dataset. 
    
    Args
        folder (str): Get the path of the folder containing the datasets train, val y test in .h5 format
        size (list): Ints of samples to be taken of each dataset: train, test y val

    Return 
        x_train (numpy array): Train dataset of x
        y_train (numpy array): Train dataset of y

        x_test (numpy array): Test dataset of x
        y_test (numpy array): Test dataset of y

        x_val (numpy array): Validation dataset of x
        y_val (numpy array): Validation dataset of y
    ''' 

    _datapath_list = glob.glob(folder + '/*.h5') 

    for path in _datapath_list:
        if 'train' in path:
            f1 = h5py.File(path, 'r')
            
        elif 'test' in path: 
            f2 = h5py.File(path, 'r')

        elif 'val' in path:
            f3 = h5py.File(path, 'r')

    if size[0]> 0: 
        return (f1['x_total'][:size[0]], f1['y_total'][:size[0]], 
                f2['x_total'][:size[1]], f2['y_total'][:size[1]], 
                f3['x_total'][:size[2]], f3['y_total'][:size[2]])  
    else:
        return (f1['x_total'][:], f1['y_total'][:], 
                f2['x_total'][:], f2['y_total'][:], 
                f3['x_total'][:], f3['y_total'][:])

    f1.close()
    f2.close()
    f3.close()

def load_data_General(folder: str, 
                      size: list = [0, 0, 0], 
                      names: list = ['x_total', 'y_total'] ) ->  Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Description:
        The next function takes in the path of the folder where the train, val y test dataset are storage, the function iterates
        over the folder and select the .h5 files. In returns the x, y datasets of each train, test y val dataset. 
    
    Args
        folder (str): Get the path of the folder containing the datasets train, val y test in .h5 format
        size (list): Ints of samples to be taken of each dataset: train, test y val

    Return 
        x_train (numpy array): Train dataset of x
        y_train (numpy array): Train dataset of y

        x_test (numpy array): Test dataset of x
        y_test (numpy array): Test dataset of y

        x_val (numpy array): Validation dataset of x
        y_val (numpy array): Validation dataset of y
    ''' 

    _datapath_list = glob.glob(folder + '/*.h5') 

    for path in _datapath_list:
        if 'train' in path:
            f1 = h5py.File(path, 'r')
            
        elif 'test' in path: 
            f2 = h5py.File(path, 'r')

        elif 'val' in path:
            f3 = h5py.File(path, 'r')

    if size[0]> 0:
        _xtrain = f1[names[0]][:size[0]]
        _ytrain = f1[names[1]][:size[0]]

        _xtest, _ytest = f2[names[0]][:size[1]], f2[names[1]][:size[1]]

        _xval, _yval = f3[names[0]][:size[2]], f3[names[1]][:size[2]]

        return (_xtrain, _ytrain, _xtest, _ytest, _xval, _yval)
        
    else:
        
        _xtrain = f1[names[0]][:]
        _ytrain = f1[names[1]][:]

        _xtest, _ytest = f2[names[0]][:], f2[names[1]][:]

        _xval, _yval = f3[names[0]][:], f3[names[1]][:]

        return (_xtrain, _ytrain, _xtest, _ytest, _xval, _yval)
        

    f1.close()
    f2.close()
    f3.close()

# ------------ Standarization Functions 
def standarize_by_set_train(x, epsilon=1e-8, p = False) -> np.ndarray:
    '''CORRECTED: 
    Standarize the array by taking the mean and standard deviation over each sample.

    Args:
        x: A NumPy array of shape (n, m), where n is the number of samples and m is the number of features.
        epsilon: A small constant to avoid division by zero.
        p: Bool value to show the dimensions of the mean and std arrays
        

    Returns:
        A NumPy array of the same shape (n, m), normalized by sample.
    '''
    mean = np.mean(x, axis=0, keepdims=True)  # Mean over the features
    if p: print(mean.shape)
    std = np.std(x, axis=0, keepdims=True)    # Std deviation over the features
    if p: print(std.shape)

    # Add epsilon to standard deviation to avoid division by zero
    std = np.where(std == 0, epsilon, std)
    
    return ( (x - mean) / std, mean, std) # Standarized 

def standarize_by_sample(x, epsilon=1e-8, p = False) -> np.ndarray:
    '''CORRECTED: 
    Standarize the array by taking the mean and standard deviation over each sample.

    Args:
        x: A NumPy array of shape (n, m), where n is the number of samples and m is the number of features.
        epsilon: A small constant to avoid division by zero.
        p: Bool value to show the dimensions of the mean and std arrays
        

    Returns:
        A NumPy array of the same shape (n, m), normalized by sample.
    '''
    mean = np.mean(x, axis=1, keepdims=True)  # Mean over the features
    if p: print(mean.shape)
    std = np.std(x, axis=1, keepdims=True)    # Std deviation over the features
    if p: print(std.shape)

    # Add epsilon to standard deviation to avoid division by zero
    std = np.where(std == 0, epsilon, std)
    
    return ( (x - mean) / std, mean, std) # Standarized 

def standarize_by_set_other(x, mean, std) -> np.ndarray:
    '''CORRECTED: 
    Standarize the array by taking the mean and standard deviation over each sample.

    Args:
        x: A NumPy array of shape (n, m), where n is the number of samples and m is the number of features.
        epsilon: A small constant to avoid division by zero.
        p: Bool value to show the dimensions of the mean and std arrays
        

    Returns:
        A NumPy array of the same shape (n, m), normalized by sample.
    '''
    
    return (x - mean) / std # Standarized 

def unstandarize_by_set(x, mean, std) -> np.ndarray:
    '''CORRECTED: 
    Standarize the array by taking the mean and standard deviation over each sample.

    Args:
        x: A NumPy array of shape (n, m), where n is the number of samples and m is the number of features.
        epsilon: A small constant to avoid division by zero.
        p: Bool value to show the dimensions of the mean and std arrays
        

    Returns:
        A NumPy array of the same shape (n, m), normalized by sample.
    '''
    
    return (x * std) + mean # Standarized 

# -------------------- Normalization functions
def normalization_by_sample(x, p = False) -> np.ndarray:
    '''Normalize the array by taking a max and min value over a sample. 

    Args:
        x: A NumPy array of shape (n, m), where n is the number of samples and m is the number of features.

    Returns:
        A NumPy array of the same shape (n, m), normalized by sample.
    '''
    
    mi = np.min(x, axis = 1, keepdims=True)
    ma = np.max(x, axis = 1, keepdims=True)
    if p: print(ma)
    if p: print(mi)
 
    return (x - mi) / (ma - mi + 1e-6)

def normalization_by_set(x, p = False) -> np.ndarray:
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
 
    return (x - mi) / (ma - mi)

# ------------ Load Data with a function applied it
def load_data_standarized(folder: str, size: list = [0, 0, 0] ) ->  Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Description:
        The next function takes in the path of the folder where the train, val y test dataset are storage, the function iterates
        over the folder and select the .h5 files. In returns the x, y datasets of each train, test y val dataset. 
    
    Args
        folder (str): Get the path of the folder containing the datasets train, val y test in .h5 format
        size (list): Ints of samples to be taken of each dataset: train, test y val

    Return 
        x_train (numpy array): Train dataset of x
        y_train (numpy array): Train dataset of y

        x_test (numpy array): Test dataset of x
        y_test (numpy array): Test dataset of y

        x_val (numpy array): Validation dataset of x
        y_val (numpy array): Validation dataset of y
    ''' 

    _datapath_list = glob.glob(folder + '/*.h5') 

    for path in _datapath_list:
        if 'train' in path:
            f1 = h5py.File(path, 'r')
            
        elif 'test' in path: 
            f2 = h5py.File(path, 'r')

        elif 'val' in path:
            f3 = h5py.File(path, 'r')

    if size[0]> 0: 
        return (standarize_by_sample(f1['x_total'][:size[0]]), 
                f1['y_total'][:size[0]], 
                standarize_by_sample(f2['x_total'][:size[1]]), 
                f2['y_total'][:size[1]], 
                standarize_by_sample(f3['x_total'][:size[2]]), 
                f3['y_total'][:size[2]]) 
    else:
        return (standarize_by_sample(f1['x_total'][:]), 
                f1['y_total'][:], 
                standarize_by_sample(f2['x_total'][:]), 
                f2['y_total'][:], 
                standarize_by_sample(f3['x_total'][:]), 
                f3['y_total'][:])

    f1.close()
    f2.close()
    f3.close()

def load_data_standarized_set_General(folder: str, 
                                  size: list = [0, 0, 0], 
                                  names: list = ['x_total', 'y_total'] ) ->  Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Description:
        The next function takes in the path of the folder where the train, val y test dataset are storage, the function iterates
        over the folder and select the .h5 files. In returns the x, y datasets of each train, test y val dataset. 
    
    Args
        folder (str): Get the path of the folder containing the datasets train, val y test in .h5 format
        size (list): Ints of samples to be taken of each dataset: train, test y val

    Return 
        x_train (numpy array): Train dataset of x
        y_train (numpy array): Train dataset of y

        x_test (numpy array): Test dataset of x
        y_test (numpy array): Test dataset of y

        x_val (numpy array): Validation dataset of x
        y_val (numpy array): Validation dataset of y
    ''' 

    _datapath_list = glob.glob(folder + '/*.h5') 

    for path in _datapath_list:
        if 'train' in path:
            f1 = h5py.File(path, 'r')
            
        elif 'test' in path: 
            f2 = h5py.File(path, 'r')

        elif 'val' in path:
            f3 = h5py.File(path, 'r')

    if size[0]> 0:
        _xtrain, _mean, _std = standarize_by_set_train(f1[names[0]][:size[0]])
        _ytrain = f1[names[1]][:size[0]]

        _xtest, _ytest = standarize_by_set_other(f2[names[0]][:size[1]], _mean, _std), f2[names[1]][:size[1]]

        _xval, _yval = standarize_by_set_other(f3[names[0]][:size[2]], _mean, _std), f3[names[1]][:size[2]]

        return (_xtrain, _ytrain, _xtest, _ytest, _xval, _yval, _mean, _std)
        
    else:
        
        _xtrain, _mean, _std = standarize_by_set_train(f1[names[0]][:])
        _ytrain = f1[names[1]][:]

        _xtest, _ytest = standarize_by_set_other(f2[names[0]][:], _mean, _std), f2[names[1]][:]

        _xval, _yval = standarize_by_set_other(f3[names[0]][:], _mean, _std), f3[names[1]][:]

        return (_xtrain, _ytrain, _xtest, _ytest, _xval, _yval, _mean, _std)
        

    f1.close()
    f2.close()
    f3.close()

def load_standarized_sets(folder: str, size: list = [0, 0, 0] ) ->  Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Description:
        The next function takes in the path of the folder where the train, val y test dataset are storage, the function iterates
        over the folder and select the .h5 files. In returns the x, y datasets of each train, test y val dataset. 
    
    Args
        folder (str): Get the path of the folder containing the datasets train, val y test in .h5 format
        size (list): Ints of samples to be taken of each dataset: train, test y val

    Return 
        x_train (numpy array): Train dataset of x

        x_test (numpy array): Test dataset of x

        x_val (numpy array): Validation dataset of x
    ''' 

    _datapath_list = glob.glob(folder + '/*.h5') 

    for path in _datapath_list:
        if 'train' in path:
            f1 = h5py.File(path, 'r')
            
        elif 'test' in path: 
            f2 = h5py.File(path, 'r')

        elif 'val' in path:
            f3 = h5py.File(path, 'r')

    if size[0]> 0: 
        return (standarize_by_sample(f1['x_total'][:size[0]]), 
                standarize_by_sample(f1['x_total'][:size[0]]),
                standarize_by_sample(f2['x_total'][:size[1]]), 
                standarize_by_sample(f2['x_total'][:size[1]]),
                standarize_by_sample(f3['x_total'][:size[2]]),
                standarize_by_sample(f3['x_total'][:size[2]])
               ) 
    else:
        return (standarize_by_sample(f1['x_total'][:]), 
                standarize_by_sample(f1['x_total'][:]),
                standarize_by_sample(f2['x_total'][:]), 
                standarize_by_sample(f2['x_total'][:]),
                standarize_by_sample(f3['x_total'][:]), 
                standarize_by_sample(f3['x_total'][:])
               )

    f1.close()
    f2.close()
    f3.close()

def load_data_normalization_sample_General(folder: str, 
                                  size: list = [0, 0, 0], 
                                  names: list = ['x_total', 'y_total'] ) ->  Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Description:
        The next function takes in the path of the folder where the train, val y test dataset are storage, the function iterates
        over the folder and select the .h5 files. In returns the x, y datasets of each train, test y val dataset. 
    
    Args
        folder (str): Get the path of the folder containing the datasets train, val y test in .h5 format
        size (list): Ints of samples to be taken of each dataset: train, test y val

    Return 
        x_train (numpy array): Train dataset of x
        y_train (numpy array): Train dataset of y

        x_test (numpy array): Test dataset of x
        y_test (numpy array): Test dataset of y

        x_val (numpy array): Validation dataset of x
        y_val (numpy array): Validation dataset of y
    ''' 

    _datapath_list = glob.glob(folder + '/*.h5') 

    for path in _datapath_list:
        if 'train' in path:
            f1 = h5py.File(path, 'r')
            
        elif 'test' in path: 
            f2 = h5py.File(path, 'r')

        elif 'val' in path:
            f3 = h5py.File(path, 'r')

    if size[0]> 0:
        _xtrain = normalization_by_sample(f1[names[0]][:size[0]])
        _ytrain = f1[names[1]][:size[0]]

        _xtest, _ytest = normalization_by_sample(f2[names[0]][:size[1]]), f2[names[1]][:size[1]]

        _xval, _yval = normalization_by_sample(f3[names[0]][:size[2]]), f3[names[1]][:size[2]]

        return (_xtrain, _ytrain, _xtest, _ytest, _xval, _yval)
        
    else:
        
        _xtrain = normalization_by_sample(f1[names[0]][:])
        _ytrain = f1[names[1]][:]

        _xtest, _ytest = normalization_by_sample(f2[names[0]][:]), f2[names[1]][:]

        _xval, _yval = normalization_by_sample(f3[names[0]][:]), f3[names[1]][:]

        return (_xtrain, _ytrain, _xtest, _ytest, _xval, _yval)
        

    f1.close()
    f2.close()
    f3.close()

# -----------------------------------------------------------------------------------------

def get_fraction_data(data: tuple, fraction: float) -> tuple:
    '''
    Description:
        This functions split the data according to the given fraction. I.e, if fraction is 0.5, the number of returned samples corresponds 
        to the half. 

    Args: 
        data (tuple): It is a tuple like (x, y)
        fraciton (float): Fraction to be taken over the data-sets 

    Return: 
        data (tuple): Tuple containing the taken fraction of the data
    '''

    return ( data[0][ : int(data[0].shape[0] * fraction) ], 
             data[1][ : int(data[1].shape[0] * fraction) ])

def to_h5(x, y, file_name):

    if not os.path.exists(file_name): # Verifica que la ruta no exista
        f = h5py.File(file_name, 'w') # Crea la ruta
        f.create_dataset('x_total', data = x, chunks=True, maxshape=(None, x.shape[1], x.shape[2]) )
        f.create_dataset('y_total', data = y, chunks=True, maxshape=(None, y.shape[1]) )
        print(f.keys())
        f.close()

def singleh5_to_TrainTestVal( file_path: str, folder_path:str, p_train: float = 0.95, p_test: float = 0.04,  )-> None:
    '''
    Description:
        This functions converts a unique h5 file containing all the data into multiples set: train, test y val 
        according to the proportions given in the args. It save all the files in a given folder. 

    Args:
        file_path (str): Path to the h5 data
        folder_path (str): Folder path where the data will be saved
        p_train (float): Data porcentage to be used for train 
        p_test (float): Data porcentage to be used for test

    Returns: 
        None
    '''

    import h5py 

    f = h5py.File(file_path, 'r')

    x_data = f['x_total'][:]
    y_data = f['y_total'][:]

    x_train, y_train = x_data[ : int( x_data.shape[0] * p_train) ], y_data[ : int( y_data.shape[0] * p_train) ]

    x_test, y_test = x_data[int( x_data.shape[0] * p_train) :int( x_data.shape[0] * (p_train + p_test) ) ], y_data[int( y_data.shape[0] * p_train) :int( y_data.shape[0] * (p_train + p_test) ) ] 

    x_val, y_val =  x_data[int( x_data.shape[0] * (p_train + p_test) ) : ], y_data[int( y_data.shape[0] * (p_train + p_test) ) : ] 

    to_h5(x = x_train, y = y_train, file_name = folder_path + '/train.h5') 
    to_h5(x = x_test, y = y_test, file_name = folder_path + '/test.h5')
    to_h5(x = x_val, y = y_val, file_name = folder_path + '/val.h5')

    print('Done!')

# ---------------- Utilities
def show_dimensions(data: tuple) -> None:
    dims = [ dat.shape for dat in data]
    print('''
--------------------------------------------------
The dimensions of each dataset corresponds to:
--------------------------------------------------

Train: 
    x: {}
    y: {}

Test: 
    x: {}
    y: {}

Val:
    x: {}
    y: {}
    '''.format(dims[0], dims[1], dims[2], dims[3], dims[4], dims[5] ) )

def plot_xy(data: tuple) -> None:
    '''
    Description: 
        This function plots a random element in the training data and, if exists, the corresponding y element
    '''
    
    figure = plt.Figure(figsize = (12, 12)) 
    n = random.randint(0, data[0].shape[0])
    plt.plot(data[0][n], label = 'x element') 
    plt.title('A random element in the x trainining dataset')
    plt.grid(True)
    plt.legend()
    plt.show()

    if len(data)>1: print('The corresponding y element ->', data[1][n])

def get_gpu(gpu_number: int = 2, p: bool = False) -> None:
    '''
    Description: 
        This function gets, if available, a GPU. 

    Args: 
        gpu_number (int): Refering to the GPU to take
        p (bool): Do I must show the total number of GPUS?

    '''
    gpus = tf.config.list_physical_devices('GPU') #se obtiene una lista de todas las gpu's del sistema

    #si hay gpu's disponibles procedemos con lo siguiente:
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[gpu_number], 'GPU') #establece una gpu especifica como la unica gpu visible para tensorflow
        logical_gpus = tf.config.experimental.list_logical_devices('GPU') #lista los dispositivos logitos dentro de la gpu. Es util cuando la gpu se divide en varias particiones logicas
        
        if p: print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        

# -----------

