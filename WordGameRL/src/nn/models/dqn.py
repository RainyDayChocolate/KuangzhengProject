import tensorflow as tf

from src.nn import bert, model
from src.nn.models.model_utils import BasicNNModel

def bert_to_action_value(bert_config,
                         input_tensor,
                         output_weights,
                         positions,
                         label_ids,
                         label_weights):
    """Get action value approximation for the DQN part
    label_weights here should be viewed as r + γmaxQ(S', a')
    """
    target_q_values = label_weights
    input_tensor = model.gather_indexes(input_tensor, positions)

    with tf.variable_scope("cls/predictions"):
        with tf.variable_scope("transform"):
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
        q_values = tf.matmul(input_tensor, output_weights, transpose_b=True)
        q_values = tf.nn.bias_add(q_values, output_bias)
        #q_values = 5 * tf.math.tanh(q_values)
        q_values = tf.math.sigmoid(q_values)

        label_ids = tf.reshape(label_ids, [-1])
        target_q_values = tf.reshape(target_q_values, [-1]) # r + 

        square_errors = tf.subtract(q_values, target_q_values)
        square_errors = tf.square(square_errors)
        one_hot_labels = tf.one_hot(
            label_ids, depth=bert_config.vocab_size, dtype=tf.float32)
        
        # Each example loss is set as an array.
        per_example_loss = tf.reduce_sum(square_errors * one_hot_labels, axis=[-1])
        loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, q_values)

class DQNModel(BasicNNModel):

    def __init__(self, **params):

        params['bert_task'] = bert_to_action_value
        super(DQNModel, self).__init__(**params)
        self.q_network_tag = params['player_tag'] + '_Q'
                
        (self.q_values, self.per_exmample_loss, self.total_loss, self.train_op, self.input_gradient), self.placeholders =\
                                                self.create_model(self.q_network_tag,                           
                                                                to_restore=True, 
                                                                to_train=True)
        print("The Q model have been created successfully")

        self.target_q_network_tag = params['player_tag'] + '_target_Q'
        (self.target_q_values, _, _, _, _), self.target_placeholders = self.create_model(self.target_q_network_tag, 
                                                        to_restore=True, 
                                                        to_train=False)
        #res_1 = self.get_wtf(prev_cues=('cash',), guesses=tuple())
        print("The target_Q model have been created successfully")

    def _copy_weights(self):
        #self.copy_scope_to(self.q_network_tag)
        self.copy_scope_to(self.target_q_network_tag, self.q_network_tag)

    def update_target_q_network(self):
        # the trained data is different from the one which copied from Q_network
        self.copy_scope_to(self.target_q_network_tag, self.q_network_tag)
