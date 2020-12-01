import json
import math
import random

import numpy as np
import numpy.linalg as LA
import tensorflow as tf
from scipy import exp
from scipy.special import logsumexp
from tqdm import tqdm

from src.agents.agentutil import similarToAnyInList
from src.games.game_util import target_lematizer as LEMMATIZER
from src.nn import bert, model, optimization, preprocess, tokenization
from src.nn.bert_config import flags
from src.nn.preprocess import convert_instances_to_features
from src.resource.evocation import free_association

FLAGS = flags.FLAGS

GRAPH = tf.get_default_graph()
SESSION = tf.Session(graph=GRAPH)

OPTIMIZER = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1)

def dict_proba_normalizer(dict_probas):
    sum_proba = sum(dict_probas.values())
    return {action: proba / sum_proba for action, proba in dict_probas.items()}


# Wei's Probability Gain Guesser
class Guesser_Model:
    def __init__(self):
        self.data = free_association

    def get_loglikelihood(self, cue, prevcues, gamma):

        loglikelihood = math.log(gamma)  # return very small value

        if len(prevcues)==0:
            return loglikelihood

        for c in prevcues:
            fsg = self.data.get_fsg(c, cue) or gamma
            loglikelihood += math.log(fsg)
        return loglikelihood
            
    def cf_act(self, target, cues, guesses, 
                bert_scores, gamma=5e-6, topics=None):
        '''
        act counterfactually for each action
        :param target:
        :param cues:
        :param guesses:
        :return:
        '''
        candidates = self.data.get_bsg_rank_list_with_fsg(target)
        loglikelihoods = {}
        # P(t|c) = P(t | topic, clue)
        scores = {}
        topic_likelihoods = {}
        for cuec, _ in candidates.items():
            # how likely the clue is leading to target
            # naive: c -> topic, topic 50, 50
            # transform, 
            scores[cuec] = math.log(self.data.get_fsg(cuec, target))
            # how surprising the clue is, given previous clues
            loglikelihoods[cuec] =  self.get_loglikelihood(cuec, cues, gamma)
            if topics:
                topic_likelihoods[cuec] = self.get_loglikelihood(cuec, 
                                                                topics,
                                                                gamma)
            
        likelihoodssum = logsumexp([ s for _, s in loglikelihoods.items()] )
        scorelikelihoodssum = logsumexp([ s for _, s in scores.items()] )
        if topics:
            topic_likelihoodssum = logsumexp([s for _, s 
                                                in topic_likelihoods.items()])
            topic_likelihoods = {k: v - topic_likelihoodssum
                                 for k, v in topic_likelihoods.items()}
    
        #log P(c|C)
        loglikelihoods = {c: s - likelihoodssum for c, s in loglikelihoods.items()}
        scoreloglikelihoods = {c: s - scorelikelihoodssum for c, s in scores.items()}
        scores = {c: scoreloglikelihoods[c] - loglikelihoods[c] for c,s in loglikelihoods.items()}

        scorevalues = list(scores.values())
        smean, sstd = np.mean(scorevalues), np.std(scorevalues)
        scores = { c: (s-smean)/sstd for c,s in scores.items()}

        if topics:
            clues_scores = list(topic_likelihoods.values())
            smean, sstd = np.mean(clues_scores), np.std(clues_scores)
            topic_likelihoods = {c: (s - smean) / sstd 
                                 for c, s in topic_likelihoods.items()}
            merged = {}
            for key in scores:
                if key not in topic_likelihoods:
                    merged[key] = 0.8 * scores[key]
                else:
                    merged[key] = 0.8 * scores[key] + 0.2 * topic_likelihoods[key]
            return merged
    
        return scores

    def cluescore(self, cue, target, prevcues, prevguesses, gamma=5e-6):
        '''
        used for provide reward for giver cues.
        :param cue:
        :param target:
        :param prevcues:
        :param prevguesses:
        :return:
        '''
        if len(prevcues)==0:
            return self.data.get_fsg(cue,target) or gamma

        score = math.log( self.data.get_fsg(cue, target) or gamma )
        ll = self.get_loglikelihood(cue, prevcues, gamma)

        return math.exp(score - ll)

guesser_model = Guesser_Model()

def get_probability_gain(target,
                         prev_cues,
                         guesses,
                         tokenizer,
                         topics=None):
    candidates = set(tokenizer.vocab.keys())
    # off policy sampler
    bert_scores = guesser_model.cf_act(target, prev_cues, guesses,
                                       candidates, topics=topics)
    cue_bsg_pair_list = sorted(
        bert_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    return cue_bsg_pair_list
                         
def probability_gain_policy(target,
                     prev_cues,
                     guesses,
                     tokenizer,
                     cue_bsg_pair_list,
                     random_top_K=3,
                     verbose=True):

    prob = 1.0 / random_top_K
    if random_top_K:
        for cue, _ in cue_bsg_pair_list:
            if cue in prev_cues: continue
            if similarToAnyInList(cue, prev_cues): continue
            if cue == target: continue
            #if verbose: print('clue: ', cue)
            # random sample
            if random.random()< prob:
                return cue
        return None



class BasicNNModel:

    def __init__(self, bert_task=None, 
                       player_tag=None, 
                       model_flags=None, 
                       evo_data=None, 
                       gamma=1):

        if player_tag is None:
            raise ValueError("The model should be specified by a tag")
    
        self.player_tag = player_tag
        self.evo_data = evo_data
        #cls + target+ sep + clues *2 + mask + sep
        # max_seq_length=1+2+len(prev_cues)*2*2+2, #cls + target+ sep + clues *2 + mask + sep
        self.max_len = 23
        self.FLAGS = model_flags or FLAGS
        self.bert_config = bert.BertConfig.from_json_file(FLAGS.bert_config_file)# main
        self.gamma = gamma
        self.lemmatizer = LEMMATIZER
        self._cache = {}

        if FLAGS.max_seq_length > self.bert_config.max_position_embeddings: # bert sentence
            raise ValueError(
                "Cannot use sequence length %d because the BERT model "
                "was only trained up to sequence length %d" %
                (FLAGS.max_seq_length, self.bert_config.max_position_embeddings))

        tf.gfile.MakeDirs(FLAGS.output_dir)

        self.init_tokenizer()
        self.lemmatizer = LEMMATIZER
        self.bert_task = bert_task
        self.session = SESSION
        self.graph = GRAPH
        # Not implemented in basic model    
        self.placeholders = {}
        self.train_op = None
        self.input_gradient = None
        #self.embedding_matrix = self.loadEmbedding()
        #(self._wtf, _, _, _, _), self.wtf_placeholders = self.init_bert_part()

    def init_tokenizer(self):
        self.tokenizer = tokenization.FullTokenizer(self.FLAGS.vocab_file) # target
        tokenization.validate_case_matches_checkpoint(
            FLAGS.do_lower_case,
            FLAGS.init_checkpoint)
        self.vocab_words = list(self.tokenizer.vocab.keys()) # tokenizer

    def loadEmbedding(self, embedding_path=None):
        if embedding_path is None:
            embedding_path = './glove.6B.50d.txt'
    
        print('loading word embeddings ...')
        _filter = set(self.vocab_words)
        f = open(embedding_path)
        lines = f.readlines()
        embedding_matrix = {}

        for lineid in tqdm(range(len(lines))):
            line = lines[lineid]
            values = line.strip().split()
            word = values[0]
            if type(word) == bytes:
                word = word.decode('utf-8')

            if _filter and word not in _filter:
                continue
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_matrix[word] = coefs

        print('word embeddings loaded...')

        return embedding_matrix


    def copy_scope_to(self, dest_scope, source_scope='bert'):
    
        """
        1. Initalized Bert part for all new NN 
        2. Update Target Q network with current Q network
        """
        with self.graph.as_default():
            source_params = tf.get_collection(tf.GraphKeys.VARIABLES, 
                                            scope=source_scope)
            dest_params = tf.get_collection(tf.GraphKeys.VARIABLES, 
                                            scope=dest_scope)
            
            def helper(x, tag):
                splied = x.split(tag)
                # Optimizer
                if len(splied) == 3: # optimizwe Q/Q/adam;balabla
                    return ''
                return splied[-1]
        
            # target_Q/bert/....
            # target_Q/classifcation

            # Q/bert/....
            # target_Q/classifcation
            if source_scope == 'bert':
                source_params_dict = {param.name: param for param in source_params}
            else:
                source_params_dict = {helper(param.name, source_scope + '/'): param 
                                                        for param in source_params}
            
            dest_params_dict = {helper(param.name, dest_scope + '/'): param 
                                                        for param in dest_params}
            for source_param in source_params_dict:
                if source_param in dest_params_dict:
                    self.session.run(tf.compat.v1.assign(dest_params_dict[source_param], 
                                                        source_params_dict[source_param]))

    def create_model(self, scope, to_restore, to_train):
        with self.graph.as_default():
            placeholders = {}
            with tf.variable_scope(scope):
                input_ids = tf.placeholder(name='input_ids', dtype=tf.int32, shape=[None, self.max_len])
                input_mask = tf.placeholder(name='input_mask', dtype=tf.int32, shape=[None, self.max_len])
                segment_ids = tf.placeholder(name='segment_ids', dtype=tf.int32, shape=[None, self.max_len])
                masked_lm_positions = tf.placeholder(name='masked_lm_positions', dtype=tf.int32, shape=[None])
                masked_lm_ids = tf.placeholder(name='masked_lm_ids', dtype=tf.int32, shape=[None])
                masked_lm_weights = tf.placeholder(name='masked_lm_weights', dtype=tf.float32, shape=[None ])

                placeholders['input_ids'] = input_ids
                placeholders['input_mask'] = input_mask
                placeholders['segment_ids'] = segment_ids
                placeholders['masked_lm_positions'] = masked_lm_positions
                placeholders['masked_lm_ids'] = masked_lm_ids
                placeholders['masked_lm_weights'] = masked_lm_weights

                _model = model.build_model(self.session,
                                            placeholders,
                                            self.bert_config,
                                            FLAGS,
                                            scope=scope,
                                            to_train=to_train,
                                            to_restore=to_restore,
                                            bert_task=self.bert_task)

        return _model, placeholders

    def train_instances(self, instances):
        features = convert_instances_to_features(instances,
                                                  self.max_len,
                                                  self.tokenizer)

        features = model.get_train_batches(features)
        fd  = {
            self.placeholders['input_ids']: features[0],
            self.placeholders['input_mask']: features[1],
            self.placeholders['segment_ids']: features[2],
            self.placeholders['masked_lm_positions']: features[3],
            self.placeholders['masked_lm_ids']: features[4],
            self.placeholders['masked_lm_weights']: features[5]
        }
        self.session.run(self.train_op, feed_dict=fd) # Gradient descent part

    def get_input_gradient(self, instances):
        features = convert_instances_to_features(instances,
                                                  self.max_len,
                                                  self.tokenizer)

        features = model.get_train_batches(features)
        fd  = {
            self.placeholders['input_ids']: features[0],
            self.placeholders['input_mask']: features[1],
            self.placeholders['segment_ids']: features[2],
            self.placeholders['masked_lm_positions']: features[3],
            self.placeholders['masked_lm_ids']: features[4],
            self.placeholders['masked_lm_weights']: features[5]
        }
        input_gradients = self.session.run(self.input_gradient, feed_dict=fd)
        return [gradient[0][0] for gradient in input_gradients]

    @staticmethod
    def input_gradient_to_attention(instance, gradient, to_filter=True):
        attentions = {}
        tokens = instance.tokens

        gradient_norm = LA.norm(gradient, axis=1)
        gradient_norm_sum = gradient_norm.sum()
        gradient_norm /= gradient_norm_sum

        to_filter_out= {"[MASK]", "[CLS]", "[SEP]"}
        for token, attention in zip(tokens, gradient_norm):
            if to_filter and token in to_filter_out:
                continue

            if token not in attentions:
                attentions[token] = attention

        attention_sum = sum(attentions.values())
        attentions = {token: atten / attention_sum 
                      for token, atten in attentions.items()}
        
        return attentions

    def scorefunc(self, w, res):
        if w not in self.tokenizer.vocab:
            return 0
        _id = self.tokenizer.vocab[w]
        return res[_id]
    
    def set_role(self, is_giver):
        self.is_giver = is_giver

    def giver_get_probability_gain(self, target, 
                                   prev_clues, prev_guesses, topics=None):
        if not self.is_giver:
            raise ValueError("Only giver could visit get probability gain")
        return get_probability_gain(target, prev_clues, 
                                    prev_guesses, self.tokenizer,
                                    topics=topics)

    def giver_probability_gain_policy(self, target, 
                                      prev_clues, prev_guesses, top_K=5):
        if not self.is_giver:
            raise ValueError("Only giver could visit this method")
        cue_bsg_pair_list = self.giver_get_probability_gain(target, 
                                                            prev_clues,
                                                            prev_guesses)
        tokenizer = self.tokenizer
        clue = probability_gain_policy(target, 
                                       prev_clues, 
                                       prev_guesses, 
                                       tokenizer,
                                       cue_bsg_pair_list,
                                       random_top_K=top_K)
        return clue
