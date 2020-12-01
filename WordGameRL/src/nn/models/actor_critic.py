

import numpy as np
import tensorflow as tf
import os

from src.nn import bert, model, optimization, preprocess
from src.nn.models.model_utils import OPTIMIZER, BasicNNModel

#from src.nn.bert_config import flags

GRAPH = tf.get_default_graph()
SESSION = tf.Session(graph=GRAPH)

class ActorCriticModel(BasicNNModel):

    def __init__(self, player_tag, 
                 is_giver=True, 
                 restrict_value=True,
                 **params):
        params['player_tag'] = player_tag
        self.is_giver = is_giver
        self.restrict_value = restrict_value
        super(ActorCriticModel, self).__init__(**params)
        self.build_model()

    def build_model(self):
        self.graph = GRAPH
        self.session = SESSION
    
        with self.graph.as_default():
            with tf.variable_scope(self.player_tag):
                self.placeholders = self.create_placeholders()
                self.bert_model = self.init_bert()
                self.log_probs = self.bert_to_policy()
                self.state_value = self.bert_to_state_value()
                self.policy_optimizer = self.get_policy_optimizer()
                self.value_optimizer = self.get_value_optimizer()
                self.input_gradient = self._input_gradient()
                self.action_log_probs = self.get_action_log_prob()
                self.initialize_all_variables()
        print("Built Model Successfully")

    def create_placeholders(self):
        placeholders = {}     
        input_ids = tf.placeholder(name='input_ids', dtype=tf.int32, shape=[None, self.max_len])
        input_mask = tf.placeholder(name='input_mask', dtype=tf.int32, shape=[None, self.max_len])
        segment_ids = tf.placeholder(name='segment_ids', dtype=tf.int32, shape=[None, self.max_len])
        masked_lm_positions = tf.placeholder(name='masked_lm_positions', dtype=tf.int32, shape=[None])
        masked_lm_ids = tf.placeholder(name='masked_lm_ids', dtype=tf.int32, shape=[None])
        masked_lm_weights = tf.placeholder(name='masked_lm_weights', dtype=tf.float32, shape=[None ])
        masked_lm_target_state_value = tf.placeholder(name='target_state_value', dtype=tf.float32, shape=[None ])
        masked_lm_action_indexes = tf.placeholder(name='action_index', dtype=tf.int32, shape=[None])

        placeholders['input_ids'] = input_ids
        placeholders['input_mask'] = input_mask
        placeholders['segment_ids'] = segment_ids
        placeholders['masked_lm_positions'] = masked_lm_positions
        placeholders['masked_lm_ids'] = masked_lm_ids # action
        placeholders['masked_lm_weights'] = masked_lm_weights # lm_weights
        placeholders['target_state_value'] = masked_lm_target_state_value
        placeholders['action_indexes'] = masked_lm_action_indexes
        return placeholders

    def init_bert(self):
        input_ids = self.placeholders['input_ids']
        input_mask = self.placeholders['input_mask']
        segment_ids = self.placeholders['segment_ids']

        bert_model = bert.BertModel(
            config=self.bert_config,
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False)

        return bert_model

    def bert_to_state_value(self):
        """Get action value approximation for the DQN part
        label_weights here should be viewed as r + γmaxQ(S', a')
        """
        positions = self.placeholders['masked_lm_positions']
        input_tensor = self.bert_model.get_sequence_output()
        output_weights = self.bert_model.get_embedding_table()

        input_tensor = model.gather_indexes(input_tensor, positions)
        with tf.variable_scope("value"):
            # FC to state value
            input_tensor = tf.layers.dense(
                input_tensor,
                units=self.bert_config.hidden_size,
                activation=bert.get_activation(self.bert_config.hidden_act),
                kernel_initializer=bert.create_initializer(
                    self.bert_config.initializer_range))

            input_tensor = bert.layer_norm(input_tensor)
        
            output = tf.matmul(input_tensor, output_weights, transpose_b=True)
            output_bias = tf.get_variable(
                "output_bias",
                shape=[self.bert_config.vocab_size],
                initializer=tf.zeros_initializer())
    
            output = tf.nn.bias_add(output, output_bias)
            if self.restrict_value:
                state_value = tf.layers.dense(
                    output,
                    units=1,
                    activation=tf.math.sigmoid)
            else:
                state_value = tf.layers.dense(
                    output,
                    units=1)

        return state_value
    
    def bert_to_policy(self):
        positions = self.placeholders['masked_lm_positions']
        input_tensor = self.bert_model.get_sequence_output()
        output_weights = self.bert_model.get_embedding_table()
        input_tensor = model.gather_indexes(input_tensor, positions)
        bert_config = self.bert_config
    
        with tf.variable_scope("cls/predictions"):
            with tf.variable_scope("transform"):
            # FC to policy softmax
                input_tensor = tf.layers.dense(
                    input_tensor,
                    units=bert_config.hidden_size,
                    activation=bert.get_activation(bert_config.hidden_act),
                    kernel_initializer=bert.create_initializer(
                        bert_config.initializer_range))
                input_tensor = bert.layer_norm(input_tensor)
    
            output_bias = tf.get_variable(
                "output_bias",
                shape=[bert_config.vocab_size],
                initializer=tf.zeros_initializer())
            logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)


        names_map = model.get_variable_restore_map(self.player_tag, 
                                        self.FLAGS.init_checkpoint)
        if self.FLAGS.init_checkpoint:
            saver = tf.compat.v1.train.Saver(var_list=names_map)
            saver.restore(self.session, self.FLAGS.init_checkpoint)
        return log_probs

    def get_policy_optimizer(self):
        label_ids = self.placeholders['masked_lm_ids']
    
        td_state_value = self.placeholders['masked_lm_weights']
        # R + rV(s')[Q(s, a)] - V(s) could replace the Q(s, a) - b(s)
        bert_config = self.bert_config

        label_ids = tf.reshape(label_ids, [-1])
        td_state_value = tf.reshape(td_state_value, [-1])
    
        one_hot_labels = tf.one_hot(label_ids, depth=bert_config.vocab_size, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(self.log_probs * one_hot_labels, axis=[-1])
        policy_improvement = tf.reduce_mean(td_state_value * per_example_loss)
        with tf.variable_scope("policy"):
            policy_op = optimization.create_optimizer(
                loss=policy_improvement,
                init_lr=self.FLAGS.learning_rate,
                num_train_steps=50000,
                num_warmup_steps=100,
                use_tpu=self.FLAGS.use_tpu)
        return policy_op
    
    def get_policy_gradient(self):
        label_ids = self.placeholders['mask']

    def get_value_optimizer(self):
    
        td_state_value = self.placeholders['target_state_value']
        # R + rV(s')[Q(s, a)] - V(s) could replace the Q(s, a) - b(s)

        #td_state_value = tf.reshape(state_value, [-1])
    
        per_example_loss = tf.subtract(self.state_value, td_state_value)
        per_example_loss = tf.square(per_example_loss)
        value_improvement = tf.reduce_mean(td_state_value * per_example_loss)
        with tf.variable_scope("value"):
            value_op = optimization.create_optimizer(
                loss=value_improvement,
                init_lr=self.FLAGS.learning_rate,
                num_train_steps=50000,
                num_warmup_steps=100,
                use_tpu=self.FLAGS.use_tpu)
        return value_op

    def _input_gradient(self):
        action_indexes = self.placeholders['action_indexes']
        embedding_output = self.bert_model.get_embedding_output()
        self.log_probs = tf.expand_dims(self.log_probs, axis=-1)
        action_log_probs = model.gather_indexes(self.log_probs,
                                                 action_indexes)
        gradient = tf.gradients(action_log_probs, 
                                                embedding_output)
        return gradient

    def get_action_log_prob(self):
        action_indexes = self.placeholders['action_indexes']
        action_proba = tf.gather(self.log_probs, action_indexes)
        return action_proba

    def initialize_all_variables(self):
        with self.graph.as_default():
            uninitialized_vars = []
            for var in tf.all_variables():
                try:
                    self.session.run(var)
                except tf.errors.FailedPreconditionError:
                    uninitialized_vars.append(var)
            to_init_new_vars_op = tf.initialize_variables(uninitialized_vars)
            self.session.run(to_init_new_vars_op)

    def get_policy(self, state):
        instance = preprocess.game_state_to_predict_instance(state, 
                                                             is_giver=self.is_giver)
        features = preprocess.convert_instances_to_features([instance], 
                                                           max_seq_length=self.max_len,
                                                           tokenizer=self.tokenizer)
        features = model.get_train_batches(features)
        fd  = {
            self.placeholders['input_ids']: features[0],
            self.placeholders['input_mask']: features[1],
            self.placeholders['segment_ids']: features[2],
            self.placeholders['masked_lm_positions']: features[3],
        }

        policy = self.session.run(self.log_probs, feed_dict=fd)[0]
        policy = np.exp(policy)
        policy = policy.reshape(1, -1)[0]
        return policy
    
    def get_state_value(self, state):
        if not state:
            return 0

        instance = preprocess.game_state_to_predict_instance(state, 
                                                             is_giver=self.is_giver)
        features = preprocess.convert_instances_to_features([instance], 
                                                           max_seq_length=self.max_len,
                                                           tokenizer=self.tokenizer)
        features = model.get_train_batches(features)
        fd = {
            self.placeholders['input_ids']: features[0],
            self.placeholders['input_mask']: features[1],
            self.placeholders['segment_ids']: features[2],
            self.placeholders['masked_lm_positions']: features[3],
        }
        state_value = self.session.run(self.state_value, feed_dict=fd)
        
        return state_value[0][0]

    def get_input_gradient(self, state, action):
        action_index = self.tokenizer.vocab[action]
        instance = preprocess.game_state_to_predict_instance(state, 
                                                             is_giver=self.is_giver)
        features = preprocess.convert_instances_to_features([instance], 
                                                           max_seq_length=self.max_len,
                                                           tokenizer=self.tokenizer)
        features = model.get_train_batches(features)
        fd = {
            self.placeholders['input_ids']: features[0],
            self.placeholders['input_mask']: features[1],
            self.placeholders['segment_ids']: features[2],
            self.placeholders['masked_lm_positions']: features[3],
            self.placeholders['action_indexes']: np.asarray([action_index])
        }
        input_gradient = self.session.run(self.input_gradient, feed_dict=fd)[0][0]#[0]
        attention = self.input_gradient_to_attention(instance, gradient=input_gradient)
        
        return attention

    def state_tuple_to_train_instance(self, state_tuple):
        current_state, action, reward, next_state = state_tuple
        current_state_value = self.get_state_value(current_state)
        target_state_value = self.get_state_value(next_state)
        instance = preprocess.game_state_to_instance(current_state,
                                                     winlose=False,
                                                     rewards=[reward, target_state_value],
                                                     gamma=self.gamma,
                                                     to_predict=False,
                                                     is_giver=self.is_giver,
                                                     current_state_value=current_state_value,
                                                     action=action)
        
        return instance
    

    def train(self, trajs):
        all_state_tuples = []
        for traj in trajs:
            state_tuples = preprocess.game_trajectory_to_states_tuples(
                                            traj,
                                            gamma=self.gamma,
                                            is_giver=self.is_giver)
            all_state_tuples.extend(state_tuples)
    
        to_train_instances = [self.state_tuple_to_train_instance(state_tuple)
                              for state_tuple in all_state_tuples]
        
        features = preprocess.convert_instances_to_features(to_train_instances,
                                                            max_seq_length=self.max_len,
                                                            tokenizer=self.tokenizer)
        features = model.get_train_batches(features)
        features[5][np.isnan(features[5])] = 0
        fd = {
            self.placeholders['input_ids']: features[0],
            self.placeholders['input_mask']: features[1],
            self.placeholders['segment_ids']: features[2],
            self.placeholders['masked_lm_positions']: features[3],
            self.placeholders['masked_lm_ids']: features[4],
            self.placeholders['masked_lm_weights']: features[5],
            self.placeholders['target_state_value']: features[6],
            self.placeholders['action_indexes']: features[7]
        }
        # train_policy
        self.session.run(self.policy_optimizer, feed_dict=fd)        
        self.session.run(self.value_optimizer, feed_dict=fd)
    
    def save_model(self, epoch, path):
        path = os.path.join(path, self.player_tag + '_' + str(epoch))

        saver = tf.compat.v1.train.Saver(
            var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                                        self.player_tag))
        print("Model successful saved to {}".format(path))
        saver.save(self.session, path)
    

    def load_model(self, ckpt_path):
        print("Begin loading Model {}".format(ckpt_path))
        saver = tf.compat.v1.train.Saver()
        saver.restore(self.session, ckpt_path)
        print("Load Model successfully")
