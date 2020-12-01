"""This script is used for sampling filter.

Limited resources lead us to filter out impossible samples.
Then set it to the appendix bayesian parts.
"""
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Activation, BatchNormalization, Dense, Flatten, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from pyminiddz.miniddz import PASS, GameState, GameStatePov
from pymodels.nn_utils import NeuralNetBase, add_conv, neuralnet


class Preprocess:
    """a class to convert from AlphaGo GameState objects to tensors of one-hot
    features for NN inputs
    """

    @staticmethod
    def state_to_tensor(state, history_len=3):
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
        cards_played = np.array([my_played_sum, lower_played_sum, upper_played_sum])
        # 其余两家出牌的总和
        if isinstance(state, GameStatePov):
            raise ValueError('The position should be equal to the current player')
        player_cards = state.player_cards
        recent_history = np.array([play.get_vec()
                                  for play in history[-history_len:]])
        input_feature = np.concatenate([cards_played,
                                        player_cards,
                                        recent_history])
        state_tensor = input_feature.transpose()
        state_tensor = np.expand_dims(state_tensor, axis=0)
        return state_tensor

    @staticmethod
    def rationality_to_tensor(rationality):
        """rationality to tensor

        Params:
        ------
        rationality(Boolern):
            The rationality of a sample
        """
        return np.array([[rationality]])


@neuralnet
class SampleFilter(NeuralNetBase):

    def __init__(self):
        self.preprocessor = Preprocess()
        model = self.create_model()
        self.model = self.compile(model)

    def create_model(self):
        x = cards_input = Input((8, 9))
        l2_reg = l2(0.0001)
        conv_num = 6

        for num in range(conv_num):
            x = add_conv(x, filter_nb=64, filter_width=2)
            x = Activation('relu', name='relu_{}'.format(num))(x)
            x = BatchNormalization(name='batchnorm_{}'.format(num))(x)

        x = Flatten(name='flatter')(x)
        x = Dense(64, kernel_regularizer=l2_reg, activation="relu")(x)
        sample_filter = Dense(1, kernel_regularizer=l2_reg, activation='sigmoid',
                              name='sample_val')(x)

        model = Model(cards_input, sample_filter, name='sample_val')
        return model

    @staticmethod
    def compile(model, optimizer=Adam(lr=0.001),
                    loss='binary_crossentropy',
                    metrics=['acc']):

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model

    def fit(self, states, rationalities, **kwargs):
        self.model = self.compile(self.model)
        states_feature = np.concatenate([self.preprocessor.state_to_tensor(state)
                                         for state in states])
        rational_labels = np.concatenate([self.preprocessor.rationality_to_tensor(rationality)
                                      for rationality in rationalities])

        early_stopping = EarlyStopping(monitor='val_loss',
                                       restore_best_weights=True,
                                       patience=20)
        self.model.fit(states_feature,
                       rational_labels,
                       callbacks=[early_stopping],
                       **kwargs)

    def predict(self, state_input):
        if isinstance(state_input, list):
            return self.batch_predict(state_input)

        state_tensor = self.lower_upper_processor(state_input)
        rationality = self.model.predict(state_tensor)[0][0]
        return rationality

    def batch_predict(self, states):

        batch_tensors = np.concatenate([self.preprocessor.state_to_tensor(state)
                                        for state in states])
        rationalities = self.model.predict(batch_tensors)
        rationalities = np.reshape(rationalities,
                                   len(rationalities))
        return rationalities

    def filter_out(self, samples, num):
        """Filter out top K samples

        Params:
        ------
        samples(List):
            A list of GameState
        num(Int):
            samples with highest rationalities

        Returns:
        ------
        filter_out_samples(List):
            samples less than num
        """
        rationality = self.batch_predict(samples)
        sample_rationality = list(zip(samples, rationality))
        sample_rationality = sorted(sample_rationality, key=lambda s: s[1])[-num:]
        filter_out_samples = [sample for sample, _ in sample_rationality]
        return filter_out_samples
