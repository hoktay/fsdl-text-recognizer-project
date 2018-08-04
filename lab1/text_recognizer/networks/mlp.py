from typing import Tuple

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Lambda,MaxPooling2D, Conv2D


def mlp(input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        layer_size: int=100,
        dropout_amount: float=0.2,
        num_layers: int=3) -> Model:
    """
    Simple multi-layer perceptron: just fully-connected layers with dropout between them, with softmax predictions.
    Creates num_layers layers.
    """
    num_classes = output_shape[0]

    model = Sequential()
    if len(input_shape) < 3:
        model.add(Lambda(lambda x: tf.expand_dims(x, -1), input_shape=input_shape))
        input_shape = (input_shape[0], input_shape[1], 1)

    model.add(Conv2D(32, 
                     kernel_size = (3,3), 
                     strides = (1,1),
                    activation = 'relu',
                    input_shape = input_shape))
    model.add(MaxPooling2D(pool_size = (2,2),
                          strides = (2,2)))
    
    model.add(Conv2D(64, (5, 5), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())


    # Don't forget to pass input_shape to the first layer of the model
    ##### Your code below (Lab 1)

#   model.add(Flatten(input_shape=input_shape))
    #First layer
    model.add(Dense(layer_size, activation='relu'))
    model.add(Dropout(dropout_amount))

    #Second layer
    model.add(Dense(layer_size, activation='relu'))
    model.add(Dropout(dropout_amount))

    #Third layer
    model.add(Dense(layer_size, activation='relu'))
    model.add(Dropout(dropout_amount))

    #Output Layer
    model.add(Dense(num_classes, activation='softmax'))

    ##### Your code above (Lab 1)

    return model

