"""This question provides a policy for the Miniddz and corresponding data processing
method.

The policy mainly relies on a Residual neural network that contains several
1D convolution layers, Batchnormalization and Rectifier Activate function.
"""
from random import sample

import keras.backend as K
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Activation, BatchNormalization, Dense, Flatten, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from pyminiddz.miniddz import (PASS, GameState, GameStatePov, Move,
                               move_decoder, move_encoder)
from pyminiddz.utils import MOVE_ENCODING
from pymodels.nn_utils import (NeuralNetBase, add_conv, build_residual_block,
                               neuralnet)
from pymodels.policy.residual_policy import Preprocess


@neuralnet
class PolicyNetwork(NeuralNetBase):
    def __init__(self, preprocessor=Preprocess, res_layer_num=6, model_path=None):
        """Params
        ------
        res_layer_num(Int):
            Decides the depth of residual blocks
        model_path(None or String):
            if model_path is not None:
            the model_json = model_path.model.
            the weight_json = model_path.weights.
            Or the poilcy create a new keras model
        """
        self.preprocessor = preprocessor
        self.res_layer_num = res_layer_num
        if model_path:
            model_json = '{}.model'.format(model_path)
            weights_json = '{}.weights'.format(model_path)
            self.model = self.load_model(model_json, weights_json).model
        else:
            self.model = self.create_model(res_layer_num=res_layer_num)

    @staticmethod
    def create_model(res_layer_num):
        """Create a keras model for this policy

        Params:
        ------
        res_layer_num(Int):
            Decides the number of residual blocks

        Returns:
        ------
        model(Keras.model)
            A double-output Neural Network
        """
        n_labels = 1+ 7 + 7 + 10 + 1
        cards_input = network = Input((8, 8), name='cards')
        l2_reg = l2(0.0001)
        network = add_conv(network, l2_reg=l2_reg)
        network = BatchNormalization(name="input_batchnorm")(network)
        network = Activation("relu", name="input_relu")(network)

        for i in range(res_layer_num):
            network = build_residual_block(network, i + 1)

        #for policy output
        policy_out = add_conv(network)
        policy_out = BatchNormalization()(policy_out)
        policy_out = Activation("relu")(policy_out)
        policy_out = Flatten(name='policy_flattern')(policy_out)
        policy_out = Dense(n_labels, kernel_regularizer=l2_reg,
                           activation="softmax", name="policy_out")(policy_out)

        #Compiling model
        model = Model(cards_input, policy_out, name="Minidou_model")
        return model

    def _batch_get_policy(self, states):
        """Batch predict for states

        Params:
        ------
        states(List):
            A list of GameState

        Returns:
        ------
        policies(List):
            A list of policy for given states
        """
        is_predicts, to_predicts, to_predicts_moves = [], [], []
        unpredict_moves = []
        for state in states:
            moves = state.get_legal_moves()
            move_num = len(moves)
            is_predicts.append(move_num > 1)
            if move_num > 1:
                to_predicts.append(state)
                to_predicts_moves.append(moves)
            else:
                unpredict_moves.append(moves[0])
        if to_predicts:
            states_tensor = np.concatenate([self.preprocessor.state_tensor(to_pred_state)
                                            for to_pred_state in to_predicts])
            policy_nn_outputs = self.model.predict(states_tensor)
            nn_policy = [self._nn_output_to_policy(out, moves=moves)
                         for out, moves in zip(policy_nn_outputs, to_predicts_moves)]
        else:
            nn_policy = []

        policies = [nn_policy.pop(0)
                    if is_predict
                    else {unpredict_moves.pop(0): 1.0}
                    for is_predict in is_predicts]
        return policies

    def _nn_output_to_policy(self, nn_output, state=None, moves=None):
        policy = np.clip(nn_output, 1e-8, 1)
        if not moves and state:
            moves = state.get_legal_moves()
        moves_mask = np.zeros(len(MOVE_ENCODING), float)
        for move in moves:
            moves_mask[move_encoder(move)] += 1
        policy *= moves_mask
        policy /= policy.sum()
        move_prob = {move_decoder(move_idx): prob
                     for move_idx, prob in enumerate(policy)
                     if moves_mask[move_idx]}
        return move_prob

    def get_policy(self, state):
        """Only get policy"""
        if isinstance(state, list):
            return self._batch_get_policy(state)

        state_tensor = self.preprocessor.state_tensor(state)
        nn_output = self.model.predict(state_tensor)[0]
        return self._nn_output_to_policy(nn_output, state)

    def fit(self, states, moves_probs, **kwargs):
        """The policy model training

        Params:
        ------
        states(List):
            A list of states(GameState) as inputs
        moves_probs(List):
            A list of Dicts{Move: prob} as output label

        Returns:
        ------
        None
        """
        states_feature = np.concatenate([self.preprocessor.state_tensor(state)
                                         for state in states])
        moves_label = np.concatenate([self.preprocessor.move_prob_tensor(move_prob)
                                      for move_prob in moves_probs])

        early_stopping = EarlyStopping(monitor='val_loss',
                                       restore_best_weights=True,
                                       patience=20)
        self.model.fit(states_feature,
                       moves_label,
                       callbacks=[early_stopping],
                       **kwargs)


@neuralnet
class ValueNetwork(NeuralNetBase):
    """Building for Mini-ddz, using Residual Neural Network

    """

    def __init__(self, preprocessor=Preprocess, res_layer_num=6, model_path=None):
        """Params
        ------
        res_layer_num(Int):
            Decides the depth of residual blocks
        model_path(None or String):
            if model_path is not None:
            the model_json = model_path.model.
            the weight_json = model_path.weights.
            Or the poilcy create a new keras model
        """
        self.preprocessor = preprocessor
        self.res_layer_num = res_layer_num
        if model_path:
            model_json = '{}.model_value'.format(model_path)
            weights_json = '{}.weights_value'.format(model_path)
            self.model = self.load_model(model_json, weights_json).model
        else:
            self.model = self.create_model(res_layer_num=res_layer_num)

    @staticmethod
    def create_model(res_layer_num):

        n_labels = 1+ 7 + 7 + 10 + 1
        cards_input = network = Input((8, 8), name='cards')
        l2_reg = l2(0.0001)
        network = add_conv(network, l2_reg=l2_reg)
        network = BatchNormalization(name="input_batchnorm")(network)
        network = Activation("relu", name="input_relu")(network)

        for i in range(res_layer_num):
            network = build_residual_block(network, i + 1)

        # for value output
        value_out = add_conv(network)
        value_out = BatchNormalization()(value_out)
        value_out = Activation("relu")(value_out)
        value_out = Flatten(name='value_flattern')(value_out)
        value_out = Dense(64, kernel_regularizer=l2_reg, activation="relu")(value_out)
        value_out = Dense(1, kernel_regularizer=l2_reg, activation="tanh",
                          name="value_out")(value_out)

        model = Model(cards_input, value_out)

        return model

    def fit(self, states, values, **kwargs):
        """
        Params:
        ------
        states(List):
            A list of states(GameState) as inputs
        values(List):
            A list of final result(float) as output label

        Returns:
        ------
            None
        """
        states_feature = np.concatenate([self.preprocessor.state_tensor(state)
                                         for state in states])
        values_label = np.concatenate([self.preprocessor.value_tensor(value)
                                       for value in values])
        early_stopping = EarlyStopping(monitor='val_loss',
                                       restore_best_weights=True,
                                       patience=20)
        self.model.fit(states_feature,
                       values_label,
                       callbacks=[early_stopping],
                       **kwargs)

    def get_value(self, state):
        """State to policy and value

        Params:
        ------
        state(GameState, GameStatePov):
            Represent Current state, if Gamestate is an instance of GameStatePov
            state.current_player should be equal to state.pov

        Returns:
        ------
        value(Float):
            State value
        """
        feature_input = self.preprocessor.state_tensor(state)
        value = self.model.predict(feature_input)
        value = value[0][0] * 2
        return value


@neuralnet
class SepPolicyValue(NeuralNetBase):
    """Building for Mini-ddz, using Residual Neural Network

    """

    def __init__(self, preprocessor=Preprocess, res_layer_num=6, model_path=None):
        """Params
        ------
        res_layer_num(Int):
            Decides the depth of residual blocks
        model_path(None or String):
            if model_path is not None:
            the model_json = model_path.model.
            the weight_json = model_path.weights.
            Or the poilcy create a new keras model
        """
        self.preprocessor = preprocessor
        self.res_layer_num = res_layer_num
        if model_path:
            model_json = '{}.model'.format(model_path)
            weights_json = '{}.weights'.format(model_path)
            self._policy = self.load_model(model_json, weights_json)

            value_model_json = '{}.model_value'.format(model_path)
            value_weights_json = '{}.weights_value'.format(model_path)
            self._value = self.load_model(value_model_json, value_weights_json)

        else:
            self._policy = PolicyNetwork()
            self._value = ValueNetwork()

        self.compile()

    def compile(self):
        self._policy.model.compile(optimizer=Adam(lr=0.05),
                                   loss=['categorical_crossentropy'],
                                   metrics=['acc'])

        self._value.model.compile(optimizer=Adam(lr=0.05),
                                  loss=['mse'],
                                  metrics=['mse'])

    def get_policy_value(self, state):
        """State to policy and value

        Params:
        ------
        state(GameState, GameStatePov):
            Represent Current state, if Gamestate is an instance of GameStatePov
            state.current_player should be equal to state.pov

        Returns:
        ------
        move_prob(Dict):
            A normalized move prob like {action_A: prob_A, ......}
            sum(Probs) = 1
        """
        return self._policy.get_policy(state), self._value.get_value(state)

    def get_policy(self, state):
        """Only get policy"""
        return self._policy.get_policy(state)

    def fit(self, states, moves_probs, values,
            **kwargs):
        """The policy model training

        Params:
        ------
        states(List):
            A list of states(GameState) as inputs
        moves_probs(List):
            A list of Dicts{Move: prob} as output label
        values(List):
            A list of final result(float) as output label

        Returns:
        ------
        None
        """
        non_pass_states, non_pass_move_probs = [], []
        for state, move_prob in zip(states, moves_probs):
            legal_moves = state.get_legal_moves()
            if len(legal_moves) == 1 and legal_moves[0].is_pass():
                continue
            non_pass_states.append(state)
            non_pass_move_probs.append(move_prob)

        self._policy.fit(non_pass_states, non_pass_move_probs, **kwargs)
        self._value.fit(states, values, **kwargs)

    def save_model(self, model_path, weights_file):
        self._policy.save_model(model_path, weights_file)
        self._value.save_model(model_path + '_value', weights_file + '_value')

    def set_learning_rate(self, learning_rate):
        self._policy.set_learning_rate(learning_rate)
        self._value.set_learning_rate(learning_rate)
