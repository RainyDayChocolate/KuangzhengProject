
"""This parts provides several utils for further keras neural networks
The neuralnet is a decorator that helps use save and load models

This script is borrowed from
https://github.com/Rochester-NRT/RocAlphaGo/blob/develop/AlphaGo/models/nn_util.py
"""

import json

import keras.backend as K
from keras.layers import Activation, BatchNormalization, Conv1D
from keras.layers.merge import Add
from keras.models import model_from_json
from keras.regularizers import l2


class NeuralNetBase(object):
    """Base class for neural network classes handling feature processing, construction
    of a 'forward' function, etc.
    keep track of subclasses to make generic saving/loading cleaner.
    subclasses can be 'registered' with the @neuralnet decorator
    """

    subclasses = {}

    def __init__(self):
        """create a neural net object that preprocesses according to feature_list and uses
        a neural network specified by keyword arguments (using subclass' create_network())

        optional argument: init_network (boolean). If set to False, skips initializing
        self.model and self.forward and the calling function should set them.
        """
        self.res_layer_num = 6
        self.model = None

    @staticmethod
    def load_model(json_file, weights_file=None):
        """create a new neural net object from the architecture specified in json_file
        """
        with open(json_file, 'r') as f:
            object_specs = json.load(f)

        # Create object; may be a subclass of networks saved in specs['class']
        class_name = object_specs.get('class', 'Default_NN_name')

        if class_name not in NeuralNetBase.subclasses:
            raise ValueError("Unknown neural network type in json file: {}\n"
                             "(was it registered with the @neuralnet decorator?)"
                             .format(class_name))

        network_class = NeuralNetBase.subclasses[class_name]
        # create new object
        if 'res_layer_num' in object_specs:
            new_net = network_class(res_layer_num=object_specs['res_layer_num'])
        else:
            new_net = network_class()

        new_net.model = model_from_json(object_specs['keras_model'])
        if weights_file is not None:
            new_net.model.load_weights(weights_file)
        elif 'weights_file' in object_specs:
            new_net.model.load_weights(object_specs['weights_file'])
        return new_net

    def save_model(self, json_file, weights_file=None):
        """write the network model and preprocessing features to the specified file

        If a weights_file (.hdf5 extension) is also specified, model weights are also
        saved to that file and will be reloaded automatically in a call to load_model
        this looks odd because we are serializing a model with json as a string
        then making that the value of an object which is then serialized as
        json again.
        It's not as crazy as it looks. A Network has 2 moving parts - the
        feature preprocessing and the neural net, each of which gets a top-level
        entry in the saved file. Keras just happens to serialize models with JSON
        as well. Note how this format makes load_model fairly clean as well.
        """

        object_specs = {
            'class': self.__class__.__name__,
            'keras_model': self.model.to_json(),
        }
        if 'res_layer_num' in dir(self):
            object_specs['res_layer_num'] = self.res_layer_num

        if weights_file is None:
            weights_file = './keras_model_weights'
        self.model.save_weights(weights_file)
        object_specs['weights_file'] = weights_file

        # use the json module to write object_specs to file
        with open(json_file, 'w') as f:
            json.dump(object_specs, f)
        print('Model successfully saved to {}'.format(weights_file))

    def set_learning_rate(self, learning_rate):
        """Set learning_rate"""
        K.set_value(self.model.optimizer.lr, learning_rate)


def neuralnet(cls):
    """Class decorator for registering subclasses of NeuralNetBase
    """
    NeuralNetBase.subclasses[cls.__name__] = cls
    return cls


def add_conv(neural_network, filter_nb=32, filter_width=3, l2_reg=l2(0.0001)):
    """Add 1D-convolution layers to Neural networks structure

    Params:
    ------
    neural_network(Keras.model):
        To be appended Keras model
    filter_nb(Int):
        Decide the number of filters
    filter_width(Int):
        Decide the width of each filter
    l2_reg(l2 regulazation):
        Decide the L2 regularization coefficien

    Returns:
    ------
    neural_network(Keras.model):
        Added Neural network
    """

    neural_network = Conv1D(
        filters=filter_nb,
        kernel_size=filter_width,
        kernel_initializer='uniform',
        padding='same',
        trainable=True,
        strides=1,
        use_bias=False,
        kernel_regularizer=l2_reg)(neural_network)

    return neural_network


def build_residual_block(network, index=0):

    """Add one residual block to Neural networks structure

    Params:
    ------
    network(Keras.model):
        To be appended Keras model
    index(Int):
        Index of residual block

    Returns:
    ------
    network(Keras.model):
        Added Neural network
    """

    in_network = network
    res_name = "res"+str(index)

    network = add_conv(network)
    network = Activation("relu", name=res_name+"_relu1")(network)
    network = BatchNormalization(name=res_name+"_batchnorm1")(network)
    network = add_conv(network)
    network = BatchNormalization(name="res"+str(index)+"_batchnorm2")(network)
    network = Add(name=res_name+"_add")([in_network, network])
    network = Activation("relu", name=res_name+"_relu2")(network)

    return network
