#Libraries
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

# ----------- Basics Functions for U-NET
def UNET_ConvDown_AP(inputs: tf.Tensor, filters: int, power_of_two: int, p: bool = False, 
                     kernel: int = 5, act_func: str = 'leaky_relu', pool: int = 4, stride: int = 4, 
                     average: bool = True) -> tf.Tensor:
    '''
    Description: 
        This function applies two convolutional layers, each of them followed by a BatchNormalization and Activation Function.
        The convolutional layers are specified to take, by default, a kernel_size of 5. The convolutionals preserve the dimensions.

        The activation function is set to leaky_relu

        To reduce dimensionaly we implement the AveragePooling function with a pool size of 4 and the padding = 'valid '. This functions 
        follows the equation:

        output_shape = (input_shape - 4)/ 4 + 1   - Only consider the integer

        The total information flow is:

            input -> Conv -> BN -> Act -> Conv -> BN -> Act -> AP
        
    Args: 
        inputs : layer to be convoluted 
        filtesr (int): Intial number of filters in the UNET model. 
        power_of_two (int): Scaling factor: 2**power_of_two in layer 
        p (bool): Must the dimensions be show?
        kernel (int): Kernel size of both convolutionals, by default is 5
        act_func (str): Activation function, by deafult is leaky_relu
        pool (int): Pool value, by default is set to 4
        stride (int): Stride value, by default is set to 4

    Output:
        AP -> Processed Layer with the reduction of dimension
        Act -> Convolutional layer before to AveragePooling1D
    '''

    n, f = power_of_two, filters
    
    Conv1 = keras.layers.Conv1D(filters = int((2**n)*f),
                  kernel_size = kernel, 
                  padding = "same")(inputs)
    
    if p : print(Conv1.shape)

    BN = keras.layers.BatchNormalization()(Conv1)
    Act = keras.layers.Activation(activation = act_func)(BN)

    Conv2 = keras.layers.Conv1D(filters = int((2**n)*f),
                  kernel_size = kernel, 
                  padding = "same")(Act)
    
    if p: print(Conv2.shape)

    BN = keras.layers.BatchNormalization()(Conv2)
    Act = keras.layers.Activation(activation = act_func)(BN)

    if average:
    
        AP = keras.layers.AveragePooling1D(
            pool_size = pool,
            strides = stride,
            padding = 'valid'
        )(Act)
    
        if p: print(AP.shape)
    
        return AP, Act 

    else:
        
        return Act 


# %%

def UNET_ConvUp_US(inputs: tf.Tensor, filters: int, power_of_two: int, block: tf.Tensor, p: bool = False, 
                     kernel: int = 5, act_func: str = 'leaky_relu') -> tf.Tensor:
    '''
    Description: 
        This function applies two convolutional layers, each of them followed by a BatchNormalization and Activation Function.
        The convolutional layers are specified to take, by default, a kernel_size of 5. The convolutionals preserve the dimensions.

        The activation function is set to leaky_relu

        To increase dimensionaly we implement UpSampling1D. As the dimensions among different layers may be different, the function require the 
        corresponding block to calculate the dimension and match both sizes. 

        The total information flow is:

            input -> UpSampled -> Concatenate -> Conv -> BN -> Act -> Conv -> BN -> Act
        

    Args: 
        inputs : layer to be convoluted 
        block: layer to be join with the input
        filtesr (int): Intial number of filters in the UNET model. 
        power_of_two (int): Scaling factor: 2**power_of_two in layer 
        block (tf.Tensor): Block to be join with the UpSampled block
        p (bool): Must the dimensions be show?
        kernel (int): Kernel size of both convolutionals, by default is 5
        act_func (str): Activation function, by deafult is leaky_relu

    Output:
        ProcessedLayer: 
    '''

    _f = filters
    _n = power_of_two
    
    # ---- UpSampling Section 
    dim_in, dim_out = inputs.shape[1], block.shape[1] # Get the corresponding dimensions 
    if p: print(dim_in, dim_out)
    n = dim_out//dim_in
    r = dim_out%dim_in
    m = dim_out - (n*dim_in)

    if r == 0 :
        _Up = keras.layers.UpSampling1D(size = n)(inputs)
        
    elif r!=0: 
        _Up = keras.layers.UpSampling1D(size = n)(inputs)
        _Up = keras.layers.ZeroPadding1D(padding = (0,m))(_Up)
    
    # ---- Concatenate Section
    _Up =keras.layers.Concatenate()([_Up, block]) # Match the UpSampled and the corresponding block

    # ---- Process the block 
    
    Conv1 = keras.layers.Conv1D(filters = int((2**_n)*_f),
                  kernel_size = kernel, 
                  padding = "same" )(_Up)
    
    if p : print(Conv1.shape)

    BN = keras.layers.BatchNormalization()(Conv1)
    Act = keras.layers.Activation(activation = act_func)(BN)

    Conv2 = keras.layers.Conv1D(filters = int((2**_n)*_f),
                  kernel_size = kernel, 
                  padding = "same" )(Act)
    
    if p: print(Conv2.shape)

    BN = keras.layers.BatchNormalization()(Conv2)
    Act = keras.layers.Activation(activation = act_func)(BN)

    return Act

def Flat_Dense_layers(inputs: tf.Tensor, nodes: list, DP: int, n_final: int,
                     act_func: str = 'leaky_relu', final_act_func: str = 'softmax') -> tf.Tensor:
    '''
    Description:
        This functions flatten the input tensor and does forward propagation throught a series of dense layers, returning the desired number
        of variables to characterize. 


    Args: 
        inputs (tf.Tensor): Layer to be processed
        nodes (list): Number of nodes in each dense layer
        DP (int): Dropout probability [0-100]
        n_final (int): Desired number of variables to be target
        act_func (str): Activation function to be used in dense layers, by default is leaky_relu
        final_act_func (str): Activation function to be used in the final layer, by default is softmax 

    Output: 
        tf.Tensor corresponding to the output of the last dense layer
    '''

    flat = keras.layers.Flatten()(inputs) # Flat the array
    D = keras.layers.Dense(nodes[0], activation = act_func)(flat) # Pass the information through the first dense layer
    Drop = keras.layers.Dropout(DP/100)(D)
    BN = keras.layers.BatchNormalization()(Drop)
    
    for node in nodes[1:]: 
        D = keras.layers.Dense(node, activation = act_func)(BN) # Pass the information through several dense layers
        Drop = keras.layers.Dropout(DP/100)(D)
        BN = keras.layers.BatchNormalization()(Drop)
    
    return keras.layers.Dense(n_final, activation = final_act_func)(BN) # Return the information with the shape of the target variable 


# %%

def Up_Match_USConv(inputs: tf.Tensor, block: tf.Tensor, p: bool = False):
    '''
    Description:
        This function takes into a inputs and scale it using UpSampling1D to the desire dimension (the input layer). 
        As the UpSampling conserve the number of filters, a posterior convolutional layer is required.

    Args:
        inputs (tf.Tensor): Tensor to be UpSampled 
        block (tf.Tensor): Tensor to match dimensions with
        p (bool): Must I show dimensions? 
    '''
    # ---- UpSampling Section 
    dim_in, dim_out = inputs.shape[1], block.shape[1] # Get the corresponding dimensions 
    if p: print(dim_in, dim_out)
    n = dim_out//dim_in
    r = dim_out%dim_in
    m = dim_out - (n*dim_in)

    if r == 0 :
        _Up = keras.layers.UpSampling1D(size = n)(inputs)
        
    elif r!=0: 
        _Up = keras.layers.UpSampling1D(size = n)(inputs)
        _Up = keras.layers.ZeroPadding1D(padding = (0,m))(_Up)

    if p: print(_Up.shape)

    return _Up

# ------------------------ U-NET

def UNET(inputs: tf.Tensor, unet_filters: int, unet_kernel = 5, unet_act_func = 'leaky_relu', p: bool = False, pool: int = 4 ,
        stride: int = 4):
    '''
    Description:
        This function set a U-NET architecture using the UNET_ConvDown_AP and UNET_ConvUP_UP. The dimension for the UpSampling
        is automatically done by infering the dimensions of each layer to be concatened with. 

        The information flux is as follows:

            input -> Enconder (3 Block) -> Bottleneck (1 Block) -> Decoder (4 Blocks) -> output

    Args:
        input_size (tuple): This parameters indicates the data dimension like (911,1) or (13301, 1) for usual Spectral data. 
        unet_filters (int): Refers to the initial number of filters in the U-NET. This filters will duplicated with respected of the later
                            one.

        unet_kernel (int): Refers to the kernel window in the convolutinal, by default is 5
        unet_act_func (str): Refers to the activation function to be used during the Convolutional, by default is leaky relu
        p (bool): Must I show dimensions?
        pool (int): Pool size for the AveragePooling, by default is set to 4. 
        stride (int): Stride size for the AveragePooling, by default is set to 4
        
        
    '''
    
    # Enconder Section ----------------- ConvDown_AP
    Encoder_First = UNET_ConvDown_AP(inputs, filters = unet_filters, power_of_two = 0, 
                                    kernel = unet_kernel, act_func = unet_act_func, p = p, 
                                    pool = pool, stride = stride) 

    Encoder_Second = UNET_ConvDown_AP(Encoder_First, filters = unet_filters, power_of_two = 1, 
                                    kernel = unet_kernel, act_func = unet_act_func, p = p, 
                                    pool = pool, stride = stride) 

    Encoder_Third = UNET_ConvDown_AP(Encoder_Second, filters = unet_filters, power_of_two = 2, 
                                    kernel = unet_kernel, act_func = unet_act_func, p = p, 
                                    pool = pool, stride = stride) 

    # Bottleneck -------------------------------------
    Bottleneck = UNET_ConvDown_AP(Encoder_Third, filters = unet_filters, power_of_two = 3, 
                                    kernel = unet_kernel, act_func = unet_act_func, p = p, 
                                    pool = pool, stride = stride)  

    # Decoder Section ---------------- ConvUp_US
    Decoder_First = UNET_ConvUp_US(Bottleneck, filters = unet_filters, power_of_two = 2, 
                                  block = Encoder_Third, kernel = unet_kernel, act_func = unet_act_func,
                                   p = p) 

    Decoder_Second = UNET_ConvUp_US(Decoder_First, filters = unet_filters, power_of_two = 1, 
                                  block = Encoder_Second, kernel = unet_kernel, act_func = unet_act_func,
                                   p = p)

    
    Decoder_Third = UNET_ConvUp_US(Decoder_Second, filters = unet_filters, power_of_two = 0, 
                                  block = Encoder_First, kernel = unet_kernel, act_func = unet_act_func,
                                   p = p)

    Final_Stage = keras.layers.Conv1D( filters = 1, 
                                     kernel_size = 1, 
                                     activation = 'relu')( Up_Match_USConv(Decoder_Third, inputs) )

    return Final_Stage


def G_UNET(inputs: tf.Tensor, layers: int,  unet_filters: int, unet_kernel = 3, unet_act_func = 'relu' , p: bool = False, pool: int = 12 ,
        stride: int = 3, final_func_act: str = 'relu' ):
    '''
    Description:
        This function set a General U-NET architecture using the UNET_ConvDown_AP and UNET_ConvUP_UP. The dimension for the UpSampling
        is automatically done by infering the dimensions of each layer to be concatened with. The argument layers referes to the total 
        number of blocks in each section (Encoder and Decoder)

        The information flux is as follows:

            input -> Enconder (layers Blocks) -> Bottleneck (1 Block) -> Decoder (layers Blocks) -> output

    Args:
        input_size (tuple): This parameters indicates the data dimension like (911,1) or (13301, 1) for usual Spectral data. 
        unet_filters (int): Refers to the initial number of filters in the U-NET. This filters will duplicated with respected of the later
                            one.

        unet_kernel (int): Refers to the kernel window in the convolutinal, by default is 5
        unet_act_func (str): Refers to the activation function to be used during the Convolutional, by default is leaky relu
        p (bool): Must I show dimensions?
        pool (int): Pool size for the AveragePooling, by default is set to 4. 
        stride (int): Stride size for the AveragePooling, by default is set to 4
        final func act (str): Final function activation to be employed in the UNET output, by default is relu. 
        
        
    '''
    _blocks = [] # Consider the input signal 

    # Enconder Section ----------------- ConvDown_AP
    Encoder, Concat = UNET_ConvDown_AP(inputs, filters = unet_filters, power_of_two = 0, 
                                    kernel = unet_kernel, act_func = unet_act_func, p = p, 
                                    pool = pool, stride = stride) 
    _blocks.append(Concat) # Save the first block

    for i in range(1, layers):
        Encoder, Concat = UNET_ConvDown_AP(Encoder, filters = unet_filters, power_of_two = i, 
                                    kernel = unet_kernel, act_func = unet_act_func, p = p, 
                                    pool = pool, stride = stride) 
        _blocks.append(Concat) # Save each block
    
    # Bottleneck -------------------------------------
    Bottleneck = UNET_ConvDown_AP(Encoder, filters = unet_filters, power_of_two = layers, 
                                    kernel = unet_kernel, act_func = unet_act_func, p = p, 
                                    pool = pool, stride = stride, average = False )  
    
    
    # Decoder Section ---------------- ConvUp_US
    Decoder = UNET_ConvUp_US( Bottleneck, filters = unet_filters, power_of_two = layers -1 , 
                              block = _blocks[-1] , kernel = unet_kernel, act_func = unet_act_func,
                              p = p)
    i = layers -2
    for _block in _blocks[ -2: : -1]:        
        Decoder = UNET_ConvUp_US( inputs = Decoder, filters = unet_filters, power_of_two = i, 
                                  block = _block , kernel = unet_kernel, act_func = unet_act_func,
                                   p = p) 
        i -= 1

    Final_Stage = keras.layers.Conv1D( filters = 1, 
                                     kernel_size = 1, 
                                     activation = final_func_act)( Up_Match_USConv(Decoder, inputs) )

    return Final_Stage


def G_F_UNET(inputs: tf.Tensor, layers: list, unet_kernel = 3, unet_act_func = 'relu' , p: bool = False, pool: int = 12 ,
        stride: int = 3, final_func_act: str = 'relu' ):
    '''
    Description:
        This function set a General U-NET architecture using the UNET_ConvDown_AP and UNET_ConvUP_UP. The dimension for the UpSampling
        is automatically done by infering the dimensions of each layer to be concatened with. The argument layers referes to the total 
        number of blocks in each section (Encoder and Decoder). The layers argument is modified to put different filters among the model. 

        The information flux is as follows:

            input -> Enconder (layers Blocks) -> Bottleneck (1 Block) -> Decoder (layers Blocks) -> output

    Args:
        input_size (tuple): This parameters indicates the data dimension like (911,1) or (13301, 1) for usual Spectral data. 
        unet_filters (int): Refers to the initial number of filters in the U-NET. This filters will duplicated with respected of the later
                            one.

        layers: Refers to the list where each element corresponds to a convolutional filter in the Encoder and bottleneck 
        
        unet_kernel (int): Refers to the kernel window in the convolutinal, by default is 5
        unet_act_func (str): Refers to the activation function to be used during the Convolutional, by default is leaky relu
        p (bool): Must I show dimensions?
        pool (int): Pool size for the AveragePooling, by default is set to 4. 
        stride (int): Stride size for the AveragePooling, by default is set to 4
        final func act (str): Final function activation to be employed in the UNET output, by default is relu. 
        
        
    '''
    _blocks = [] # Consider the input signal 

    # Enconder Section ----------------- ConvDown_AP
    Encoder, Concat = UNET_ConvDown_AP(inputs, filters = layers[0], power_of_two = 0, 
                                    kernel = unet_kernel, act_func = unet_act_func, p = p, 
                                    pool = pool, stride = stride) 
    _blocks.append(Concat) # Save the first block

    for i in range(1, len(layers) -1 ):
        Encoder, Concat = UNET_ConvDown_AP(Encoder, filters = layers[i], power_of_two = 0, 
                                    kernel = unet_kernel, act_func = unet_act_func, p = p, 
                                    pool = pool, stride = stride) 
        _blocks.append(Concat) # Save each block
    
    # Bottleneck -------------------------------------
    Bottleneck = UNET_ConvDown_AP(Encoder, filters = layers[-1], power_of_two = 0, 
                                    kernel = unet_kernel, act_func = unet_act_func, p = p, 
                                    pool = pool, stride = stride, average = False )  
    
    
    # Decoder Section ---------------- ConvUp_US
    Decoder = UNET_ConvUp_US( Bottleneck, filters = layers[-2], power_of_two = 0 , 
                              block = _blocks[-1] , kernel = unet_kernel, act_func = unet_act_func,
                              p = p)
    i = len(layers) -2
    for _block in _blocks[ -2: : -1]:        
        Decoder = UNET_ConvUp_US( inputs = Decoder, filters = layers[i], power_of_two = 0, 
                                  block = _block , kernel = unet_kernel, act_func = unet_act_func,
                                   p = p) 
        i -= 1

    Final_Stage = keras.layers.Conv1D( filters = 1, 
                                     kernel_size = 1, 
                                     activation = final_func_act)( Up_Match_USConv(Decoder, inputs) )

    return Final_Stage


def G_K_UNET(inputs: tf.Tensor, layers:int, unet_kernel: list, unet_act_func = 'relu' , p: bool = False, pool: int = 12 ,
        stride: int = 3, final_func_act: str = 'relu' ):
    '''
    Description:
        This function set a General U-NET architecture using the UNET_ConvDown_AP and UNET_ConvUP_UP. The dimension for the UpSampling
        is automatically done by infering the dimensions of each layer to be concatened with. The argument layers referes to the total 
        number of blocks in each section (Encoder and Decoder). The layers argument is modified to put different filters among the model. 

        The information flux is as follows:

            input -> Enconder (layers Blocks) -> Bottleneck (1 Block) -> Decoder (layers Blocks) -> output

    Args:
        input_size (tuple): This parameters indicates the data dimension like (911,1) or (13301, 1) for usual Spectral data. 
        unet_filters (int): Refers to the initial number of filters in the U-NET. This filters will duplicated with respected of the later
                            one.

        layers: Refers to the list where each element corresponds to a convolutional filter in the Encoder and bottleneck 
        
        unet_kernel (int): Refers to the kernel window in the convolutinal, by default is 5
        unet_act_func (str): Refers to the activation function to be used during the Convolutional, by default is leaky relu
        p (bool): Must I show dimensions?
        pool (int): Pool size for the AveragePooling, by default is set to 4. 
        stride (int): Stride size for the AveragePooling, by default is set to 4
        final func act (str): Final function activation to be employed in the UNET output, by default is relu. 
        
        
    '''
    _blocks = [] # Consider the input signal 

    # Enconder Section ----------------- ConvDown_AP
    Encoder, Concat = UNET_ConvDown_AP(inputs, filters = layers, power_of_two = 0, 
                                    kernel = unet_kernel[0], act_func = unet_act_func, p = p, 
                                    pool = pool, stride = stride) 
    #print(unet_kernel[0])
    _blocks.append(Concat) # Save the first block

    for i in range(1, len(unet_kernel) - 1 ):
        Encoder, Concat = UNET_ConvDown_AP(Encoder, filters = layers, power_of_two = i, 
                                    kernel = unet_kernel[i], act_func = unet_act_func, p = p, 
                                    pool = pool, stride = stride)
        #print(unet_kernel[i])
        _blocks.append(Concat) # Save each block
    
    # Bottleneck -------------------------------------
    Bottleneck = UNET_ConvDown_AP(Encoder, filters = layers, power_of_two = len(unet_kernel)-1, 
                                    kernel = unet_kernel[-1], act_func = unet_act_func, p = p, 
                                    pool = pool, stride = stride, average = False )
    #print(unet_kernel[-1])
    
    
    
    # Decoder Section ---------------- ConvUp_US
    Decoder = UNET_ConvUp_US( Bottleneck, filters = layers, power_of_two = len(unet_kernel)-2 , 
                              block = _blocks[-1] , kernel = unet_kernel[-2], act_func = unet_act_func,
                              p = p)
    #print(unet_kernel[-2])
    
    
    i = len(unet_kernel) -3
    for _block in _blocks[ -2: : -1]:        
        Decoder = UNET_ConvUp_US( inputs = Decoder, filters = layers, power_of_two = i, 
                                  block = _block , kernel = unet_kernel[i], act_func = unet_act_func,
                                   p = p)
        #print(unet_kernel[i])
        
        i -= 1

    Final_Stage = keras.layers.Conv1D( filters = 1, 
                                     kernel_size = 1, 
                                     activation = final_func_act)( Up_Match_USConv(Decoder, inputs) )

    return Final_Stage


# ---------- Convolutional Neural Network CNN
def CNN_ConvDown_AP(inputs: tf.Tensor, filters: int, kernel: int = 5, act_func: str = 'leaky_relu', 
                pool: int = 4, p: bool = False, stride: int = 4):
    '''
    Description:
        This function is the primary block in the CNN, this block does the Convolution, BN, Activation and
        AveragePooling according to the next information flux:

            input -> Conv1D -> BN -> Act -> AP -> Output

    Args: 
        input (tf.Tensor): Information to be convoluted
        filters (int): Number of filters to be used in the Convolutional layer
        kernel (int): Kernel size of the Conv, by default is 5
        act_func (str): Activation function for the last stage of the flux, by default is leaky relu
        pool (int): Pool size of the AveragePooling, by default is 4
        p (bool): Must I show dimensions? 

    Return:
        tf.Tensor : Corresponding to the processed information.
        
    '''

    Conv = keras.layers.Conv1D(filters = filters, 
                               kernel_size = kernel,
                               padding = 'same')(inputs)

    BN = keras.layers.BatchNormalization()(Conv)
    Act = keras.layers.Activation(activation = act_func)(BN)
    AP = keras.layers.AveragePooling1D( pool_size = pool,
                                        strides = stride, 
                                        padding = 'valid')(Act)
    if p: print(AP.shape)

    return AP



def CNN(inputs: tf.Tensor, cnn_filters: tuple,  nn_nodes: tuple, DP: int, n_final: int, 
        cnn_kernel: int = 5, cnn_act_func = 'leaky_relu', p:bool = False, pool: int = 4, 
        nn_act_func = 'leaky_relu', nn_final_act_func: str = 'softmax', stride: int = 4 ):
        '''
        Description:
            This function corresponds to a Convolutional Neuronal Network, it implements a scable way
            to create several layers of Convolutional blocks and Dense blocks by specifying the number
            of filters and nodes in each layers. This function is highly modificable due to the access
            of each filter, pool, act_func, etc. 

            The information flux is as follows:

             inputs -> Convolutional Blocks (cnn_filters) -> Flat -> Dense Blocks (nn_nodes) -> outputs

        Args: 
            input_size (tuple) : Dimension of the input information
            cnn_filters (tuple) : Filters to be put in each convolutional layer
            nn_nodes (tuple): Nodes to be put in each dense layer
            DP (int): Dropout value to be put in all layers, DP/100. 
            n_final (int): Nodes to be put in the final layer, correspoding to the target variable

            cnn_kernel (int): Kernel size of the convolutional, by default is 5
            cnn_act_func (str): Activation function to be used in the convolutionals
            p (bool): Must I show dimensions?
            pool (int): Pool size of the AveragePooling, by default is 4
            nn_act_func (str): Activation function of the dense layer, by default is leaky relu
            nn_final_act_func (str): Function to be used in the final dense layer, by default is softmax

        Return
            keras.layers.Model  : Correspoding to the desire CNN.
        '''

        _CNN = CNN_ConvDown_AP(inputs = inputs, filters = cnn_filters[0], kernel = cnn_kernel, 
                               act_func = cnn_act_func, pool = pool, p = p, stride = stride)

        for _f in cnn_filters[1:]: 
            _CNN = CNN_ConvDown_AP(_CNN, filters = _f, kernel = cnn_kernel, act_func = cnn_act_func,
                                   pool = pool, p = p, stride = stride)


        _NN = Flat_Dense_layers(inputs = _CNN, nodes = nn_nodes, DP = DP, n_final = n_final, 
                                act_func = nn_act_func, final_act_func = nn_final_act_func)
    
        return _NN


def CNN_K(inputs: tf.Tensor, cnn_filters: tuple,  nn_nodes: tuple, DP: int, n_final: int, 
        cnn_kernel: tuple, cnn_act_func = 'leaky_relu', p:bool = False, pool: int = 4, 
        nn_act_func = 'leaky_relu', nn_final_act_func: str = 'softmax', stride: int = 4 ):
        '''
        Description:
            This function corresponds to a Convolutional Neuronal Network, it implements a scable way
            to create several layers of Convolutional blocks and Dense blocks by specifying the number
            of filters and nodes in each layers. This function is highly modificable due to the access
            of each filter, pool, act_func, etc. 

            The information flux is as follows:

             inputs -> Convolutional Blocks (cnn_filters) -> Flat -> Dense Blocks (nn_nodes) -> outputs

        Args: 
            input_size (tuple) : Dimension of the input information
            cnn_filters (tuple) : Filters to be put in each convolutional layer
            nn_nodes (tuple): Nodes to be put in each dense layer
            DP (int): Dropout value to be put in all layers, DP/100. 
            n_final (int): Nodes to be put in the final layer, correspoding to the target variable

            cnn_kernel (int): Kernel size of the convolutional, by default is 5
            cnn_act_func (str): Activation function to be used in the convolutionals
            p (bool): Must I show dimensions?
            pool (int): Pool size of the AveragePooling, by default is 4
            nn_act_func (str): Activation function of the dense layer, by default is leaky relu
            nn_final_act_func (str): Function to be used in the final dense layer, by default is softmax

        Return
            keras.layers.Model  : Correspoding to the desire CNN.
        '''

        _CNN = CNN_ConvDown_AP(inputs = inputs, filters = cnn_filters[0], kernel = cnn_kernel[0], 
                               act_func = cnn_act_func, pool = pool, p = p, stride = stride)

        for _f, _k in list(zip(cnn_filters[1:], cnn_kernel[1:])):
            if p: print(_f, _k)
            _CNN = CNN_ConvDown_AP(_CNN, filters = _f, kernel = _k, act_func = cnn_act_func,
                                   pool = pool, p = p, stride = stride)


        _NN = Flat_Dense_layers(inputs = _CNN, nodes = nn_nodes, DP = DP, n_final = n_final, 
                                act_func = nn_act_func, final_act_func = nn_final_act_func)
    
        return _NN

# --------- FeedForward - Dense Section
def G_Dense(inputs: tf.Tensor, nodes: list, DP: int, n_final: int,
            act_func: str = 'leaky_relu', final_act_func: str = 'softmax',
            WI: str = 'he_normal', L1: float = 0.0, L2: float = 0.0, 
            use_bias: bool = False) -> tf.Tensor:
    '''
    Description:
        This functions does forward propagation throught a series of dense layers, returning the desired number
        of variables to characterize.


    Args:
        inputs (tf.Tensor): Layer to be processed
        nodes (list): Number of nodes in each dense layer
        DP (int): Dropout probability [0-100]
        n_final (int): Desired number of variables to be target
        act_func (str): Activation function to be used in dense layers, by default is leaky_relu
        final_act_func (str): Activation function to be used in the final layer, by default is softmax

    Output:
        tf.Tensor corresponding to the output of the last dense layer
    '''
    WR = keras.regularizers.L1L2(l1 = L1, l2=L2)
    D = keras.layers.Dense(
        nodes[0],
        activation = act_func,
        kernel_initializer = WI,
        kernel_regularizer =  WR,
        use_bias = use_bias
        )(inputs) # Pass the information through the first dense layer
    Drop = keras.layers.Dropout(DP/100)(D)
    BN = keras.layers.BatchNormalization()(Drop)

    for node in nodes[1:]:
        D = keras.layers.Dense(node,
                               activation = act_func,
                               kernel_initializer = WI,
                               kernel_regularizer =  WR,
                               use_bias = use_bias
                               )(BN) # Pass the information through several dense layers
        Drop = keras.layers.Dropout(DP/100)(D)
        BN = keras.layers.BatchNormalization()(Drop)

    return keras.layers.Dense(n_final, 
                              activation = final_act_func,
                              kernel_initializer = WI, 
                              use_bias = True
                              )(BN) # Return the information with the shape of the target variable
# --------- Convolutional Section
def G_ConvBlock(inputs: tf.Tensor, filters: int, kernel:int, act_func: str, pad_type:str, 
                pool:int, stride:int, WIC:str, WRC:str, stride_conv: int = 1, pool_op = 'AP'):
    '''
    Args: 
        inputs (Tensor): The input information for the Conv1D
        filters (int): The number of filters in the Convolutional
        kernel (int): kernel size in the Conv
        stride_conv (int): strides in the Conv
        WIC (str): Kernel Initializer in the Conv
        WRC (str): Kernel Regulatier in the Conv 
        act_func (str): Activation Function 
        pool (int): pool size in the pooling layer
        stride (int): stride size in the pooling layer

    Flux: 
        Conv->BN->Activation->AP

    Observations: 
        1. The Convolutionals preserves dimension. 
        2. Before is the BatchNormalization and after the Activation. 
        3. The dimension after the Conv is
            (input_shape - pool_size + 1) / strides)

    Test:
        G_ConvBlock_AP(keras.layers.Input((200, 1)), 10, 10, 'relu', 4, 4, 'he_normal', keras.regularizers.L2(0.1))
    '''

    Conv = keras.layers.Conv1D(
        filters = filters,
        kernel_size = kernel,
        strides = stride_conv, 
        padding = 'same',
        kernel_initializer = WIC, 
        kernel_regularizer = WRC
    )(inputs)

    BN = keras.layers.BatchNormalization()(Conv)
    Act = keras.layers.Activation(activation = act_func)(BN)

    if pool_op == 'AP':
        AP = keras.layers.AveragePooling1D(
            pool_size = pool,
            strides = stride, 
            padding = pad_type
            )(Act)
    elif pool_op == 'MP':
        AP = keras.layers.MaxPooling1D(
        pool_size = pool,
        strides = stride, 
        padding = pad_type
        )(Act)

    elif pool_op == None:
        AP = Act

    return AP

def G_DeConvBlock(inputs: tf.Tensor, filters: int, kernel:int, act_func: str, pad_type: str, 
                  stride_deconv:int, WIC:str, WRC:str, out_pad = None):
    '''
    Args: 
        inputs (Tensor): The input information for the Conv1D
        filters (int): The number of filters in the Convolutional
        kernel (int): kernel size in the Conv
        stride_conv (int): strides in the Conv
        WIC (str): Kernel Initializer in the Conv
        WRC (str): Kernel Regulatier in the Conv 
        act_func (str): Activation Function 
        stride (int): stride size in the pooling layer

    Flux: 
        Conv->BN->Activation

    Observations: 
        1. The Convolutionals preserves dimension. 
        2. Before is the BatchNormalization and after the Activation. 
        3. The dimension after the DeConv is
            (input_shape - 1) * S + K

    Test:
        G_DeConvBlock(keras.layers.Input((50,1)), 10, 4, 'relu', 4, 'he_normal', None)
    '''

    Conv = keras.layers.Conv1DTranspose(
            filters = filters,
            kernel_size = kernel,
            strides = stride_deconv, 
            output_padding = out_pad,
            padding = pad_type,
            kernel_initializer = WIC, 
            kernel_regularizer = WRC
        )(inputs)

    BN = keras.layers.BatchNormalization()(Conv)
    Act = keras.layers.Activation(activation = act_func)(BN)
    
    return Act


def G_AE_Conv1D(inputs: tf.Tensor, filters: int, kernel:list, act_func: str, pad_type:str, 
               pool:int, stride:int, WIC:str, WRC:str, stride_conv:int = 1):

    # ------------------- Encoder Input 
    _stage = G_ConvBlock(
        inputs,
        filters,
        kernel[0], 
        act_func, 
        pad_type,
        pool,
        stride,
        WIC,
        WRC,
        stride_conv
    )
    # ------------------ Encoder iterations: 
    _i = 0
    for subkernel in kernel[1:-1]:
        _i+=1
        _stage = G_ConvBlock(
            _stage,
            (filters * 2**_i),
            subkernel, 
            act_func, 
            pad_type,
            pool,
            stride,
            WIC,
            WRC,
            stride_conv
        )
    # --------------- Latent Space: 
    ls = G_ConvBlock(
            _stage,
            (filters * 2** (len(kernel)-1) ),
            kernel[-1], 
            act_func, 
            pad_type,
            pool,
            stride,
            WIC,
            WRC,
            stride_conv,
            None
        )
    # -------------- Decoder Input: [30, 20, 10, 5, 5] -> [5, 5, 10, 20, 30] -> [5, 10, 20, 30]
    _i -+1
    _stage = G_DeConvBlock(
                    ls,
                    (filters * 2**_i),
                    kernel[::-1][1], 
                    act_func, 
                    pad_type,
                    stride,
                    WIC,
                    WRC,
                    None
                )
    
    # ------------ Decoder Flux
    _kernel = kernel[::-1][2:]
    _i -=1
    for subkernel in _kernel:
        _stage = G_DeConvBlock(
                    _stage,
                    (filters * 2**_i),
                    subkernel, 
                    act_func, 
                    pad_type,
                    stride,
                    WIC,
                    WRC,
                    None
                )
        _i -=1

    return keras.layers.Convolution1D(1, 1, padding = 'same')(_stage)


# --------- AutoEncoders

def G_AE_Dense(inputs: tf.Tensor, nodes: list, DP: int, act_func = 'sigmoid', WI: str = 'glorot_normal', WR: float = 1e-4, 
                     final_act_func: str = 'sigmoig'): 
    '''
    Descriptions: 
        This functions generate a Deep Dense AutoEncoder architecture (Symmetric). Each block has the following flux: 

        Dense -> DP -> BN 

    Args:
        inputs (tf.Tensor): Refers to the input shape tensor. 
        nodes (list): Refers to the number of nodes for each dense layer
        DP (int): Dropout probability
        WI (list): Weight Initializer, the first element is the metric used and the second element is the value 
        WR (list): Weight Regulazier, by default we used L1L2, the values of the list are the values for each lambda. 
        act_func (str): Activation function used 
    '''
    
    reg = keras.regularizers.L2(l2=WR)

    # --- Encoder ------------- # [100, 50, 25, 10, 3]
    D = keras.layers.Dense(
        nodes[0], 
        activation = act_func,
        kernel_initializer = WI,
    )(inputs)
    Drop = keras.layers.Dropout(DP/100)(D)
    BN = keras.layers.BatchNormalization()(Drop) 

    # --- Main Cycle of Compresion ---------------
    for node in nodes[1:-1]:
        D = keras.layers.Dense(
            node, 
            activation = act_func,
            kernel_initializer = WI,
        )(BN)
        Drop = keras.layers.Dropout(DP/100)(D)
        BN = keras.layers.BatchNormalization()(Drop)

    # --- Bottleneck ------------------------------
    LS = keras.layers.Dense(
        nodes[-1], 
        activation = 'linear',
        kernel_initializer = WI,
        kernel_regularizer = reg, 
        name = 'latent_space'
    )(BN)
    Drop = keras.layers.Dropout(DP/100)(LS)
    BN = keras.layers.BatchNormalization()(Drop)
    
    # --- Main Cycle of Decompresion ------------- 
    _nodes = nodes[::-1][1:] # [10, 25, 50, 100]
    for node in _nodes: 
        D = keras.layers.Dense(
            node, 
            activation = act_func,
            kernel_initializer = WI,
        )(BN)
        Drop = keras.layers.Dropout(DP/100)(D)
        BN = keras.layers.BatchNormalization()(Drop)

    # --- Decoder -----------------------------
    DF = keras.layers.Dense(
        inputs.shape[1], 
        activation = final_act_func,
        kernel_initializer = WI,
        name = 'reconstruction'
    )(BN)
    
    return DF

def G_AE_Conv1D(inputs: tf.Tensor, filters: int, kernel:list, act_func: str, pad_type:str, 
               pool:int, stride:int, WIC:str, WRC:str, stride_conv:int = 1):

    # ------------------- Encoder Input 
    _stage = G_ConvBlock(
        inputs,
        filters,
        kernel[0], 
        act_func, 
        pad_type,
        pool,
        stride,
        WIC,
        WRC,
        stride_conv
    )
    # ------------------ Encoder iterations: 
    _i = 0
    for subkernel in kernel[1:-1]:
        _i+=1
        _stage = G_ConvBlock(
            _stage,
            (filters * 2**_i),
            subkernel, 
            act_func, 
            pad_type,
            pool,
            stride,
            WIC,
            WRC,
            stride_conv
        )
    # --------------- Latent Space: 
    ls = G_ConvBlock(
            _stage,
            (filters * 2** (len(kernel)-1) ),
            kernel[-1], 
            'relu', 
            pad_type,
            pool,
            stride,
            WIC,
            WRC,
            stride_conv,
            None
        )
    # -------------- Decoder Input: [30, 20, 10, 5, 5] -> [5, 5, 10, 20, 30] -> [5, 10, 20, 30]
    _i -+1
    _stage = G_DeConvBlock(
                    ls,
                    (filters * 2**_i),
                    kernel[::-1][1], 
                    act_func, 
                    pad_type,
                    stride,
                    WIC,
                    WRC,
                    None
                )
    
    # ------------ Decoder Flux
    _kernel = kernel[::-1][2:]
    _i -=1
    for subkernel in _kernel:
        _stage = G_DeConvBlock(
                    _stage,
                    (filters * 2**_i),
                    subkernel, 
                    act_func, 
                    pad_type,
                    stride,
                    WIC,
                    WRC,
                    None
                )
        _i -=1

    return keras.layers.Convolution1D(1, 1, padding = 'same')(_stage)