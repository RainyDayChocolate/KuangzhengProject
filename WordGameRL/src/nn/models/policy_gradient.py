import random

import numpy.linalg as LA
import tensorflow as tf

from src.nn import model
from src.nn.bert_config import flags
from src.nn.models.model_utils import BasicNNModel

FLAGS = flags.FLAGS

bert_to_strategy = model.get_masked_lm_output # softmax output.

class PolicyGradientModel(BasicNNModel):

    def __init__(self, **params):
        params['bert_task'] = bert_to_strategy
        super(PolicyGradientModel, self).__init__(**params)
        (self.log_probs, self.per_exmample_loss, self.total_loss, self.train_op, self.input_gradient), self.placeholders =\
                                         self.create_model(scope=params['player_tag'], 
                                                            to_restore=False, 
                                                            to_train=True)
        self._cache = {}
        self.copy_scope_to(params['player_tag'])    
        print("Policy Gradient Model has been created successfully")
    