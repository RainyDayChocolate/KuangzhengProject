"""This question provides a policy for the Miniddz and corresponding data processing
method.

The policy mainly relies on a Residual neural network that contains several
1D convolution layers, Batchnormalization and Rectifier Activate function.
"""
from random import sample

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Activation, BatchNormalization, Dense, Flatten, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
import keras.backend as K

from pyminiddz.miniddz import (PASS, GameState, GameStatePov, Move,
                               move_decoder, move_encoder)
from pyminiddz.utils import MOVE_ENCODING
from pymodels.nn_utils import (NeuralNetBase, add_conv, build_residual_block,
                               neuralnet)


class Preprocess:
    """a class to convert from AlphaGo GameState objects to tensors of one-hot
    features for NN inputs
    """

    @staticmethod
    def state_tensor(state, history_len=3):
        """Convert a game state to a tensorflow-compatible tensor

        Params:
        ------
        state(GameState or GameStatePov):
            Current state that the player should make a choice
        history_len(Int):
            Decide the number of history steps that should involved in the feature
            input

        Returns:
        -----
        state_tensor(np.ndarray(3D)):
            A list of state tensor For example
            [state_tensor = [my_cards, other_cards,....]]
            if history_len = 3. The state is a 1 X 8 X 7 tensor
        """
        history = [PASS for _ in range(history_len)] + state.get_history()
        my_played_sum = state.get_cards_played(state.get_current_player())
        lower_played_sum = state.get_cards_played(state.next_player())
        upper_played_sum = state.get_cards_played(state.upper_player())
        # 其余两家出牌的总和
        if isinstance(state, GameStatePov):
            if state.get_pov() != state.get_current_player():
                raise ValueError('The position should be equal to the current player')
            other_cards = state.get_other_cards()
        elif isinstance(state, GameState):
            other_cards = state.get_player_cards([state.next_player(),
                                                  state.upper_player()])

        input_feature = np.vstack([state.get_current_player_card(), other_cards,
                                   my_played_sum, lower_played_sum, upper_played_sum] +
                                  [play.get_vec()
                                   for play in history[-history_len:]])

        state_tensor = input_feature.transpose()
        state_tensor = np.expand_dims(state_tensor, axis=0)
        return state_tensor

    @staticmethod
    def move_prob_tensor(move_prob):
        """Convert a Move object to a tensorflow-compatible tensor

        Params:
        ------
        move_prob(Dict):
            A move which could be considered as move label for
            training data

        Returns:
        -----
        move_tensor(np.ndarray(2D)):
            A list contains encode_moves(24 dim vector)
            like [encode_move_a, encode_move_b, encode_move_c, ...]
        """
        move_tensor = np.zeros(len(MOVE_ENCODING), float)

        for move, prob in move_prob.items():
            _encode = move_encoder(move)
            move_tensor[_encode] += prob

        move_tensor /= move_tensor.sum()
        move_tensor = np.expand_dims(move_tensor, axis=0)

        return move_tensor

    @staticmethod
    def value_tensor(value):
        """Convert a value to a tensorflow-compatible tensor

        Params:
        ------
        value(Float):
            A value which could be considered as the value label
            for training data

        Returns:
        -----
        value_tensor(np.ndarray(2D)):
            A list contains encode_moves(24 dim vector)
            like [[value_1], [value_2], ...]
        """

        return np.array([[float(value / 2.0)]])


@neuralnet
class ResidualPolicyValue(NeuralNetBase):
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
            self.model = self.load_model(model_json, weights_json).model
        else:
            self.model = self.create_model(res_layer_num=res_layer_num)#(res_layer_num)
        self.compile(self.model)

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

        # for value output
        value_out = add_conv(network)
        value_out = BatchNormalization()(value_out)
        value_out = Activation("relu")(value_out)
        value_out = Flatten(name='value_flattern')(value_out)
        value_out = Dense(64, kernel_regularizer=l2_reg, activation="relu")(value_out)
        value_out = Dense(1, kernel_regularizer=l2_reg, activation="tanh",
                          name="value_out")(value_out)

        #Compiling model
        model = Model(cards_input, [policy_out, value_out], name="Minidou_model")
        return model

    @staticmethod
    def compile(model):
        """Compling models with Adam
        optimizer: Adam
        losses: For policy: cross Entropy, Value: Mean_Square_Error
        loss_weights: policy: Value = 1:1
        metrics: both acc

        Params:
        ------
        model(Keras.Model):
            A keras fuctional Model
        """
        model.compile(optimizer=Adam(lr=0.05),
                      loss=['categorical_crossentropy', 'mean_squared_error'],
                      loss_weights=[1, 1.5],
                      metrics=['acc'])

        return model

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
        feature_input = self.preprocessor.state_tensor(state)
        policy, value = self.model.predict(feature_input)
        policy, value = policy[0], value[0][0] * 2
        move_prob = self._nn_output_to_policy(policy, state)
        return move_prob, value

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
            nn_outputs = self.model.predict(states_tensor)
            policy_nn_outputs = nn_outputs[0]
            nn_policy = [self._nn_output_to_policy(out, moves=moves)
                        for out, moves in zip(policy_nn_outputs, to_predicts_moves)]
        else:
            nn_policy = []

        policies = [nn_policy.pop(0)
                    if is_predict
                    else {unpredict_moves.pop(0): 1.0}
                    for is_predict in is_predicts]
        return policies

    def get_policy(self, state):
        """Only get policy"""
        if isinstance(state, list):
            return self._batch_get_policy(state)
        return self.get_policy_value(state)[0]

    def set_learning_rate(self, learning_rate):
        """Set learning_rate"""
        K.set_value(self.model.optimizer.lr, learning_rate)

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
        self.model = self.compile(self.model)
        states_feature = np.concatenate([self.preprocessor.state_tensor(state)
                                         for state in states])
        moves_label = np.concatenate([self.preprocessor.move_prob_tensor(move_prob)
                                      for move_prob in moves_probs])
        values_label = np.concatenate([self.preprocessor.value_tensor(value)
                                       for value in values])

        early_stopping = EarlyStopping(monitor='val_loss',
                                       restore_best_weights=True,
                                       patience=20)
        self.model.fit(states_feature,
                       [moves_label, values_label],
                       callbacks=[early_stopping],
                       **kwargs)
