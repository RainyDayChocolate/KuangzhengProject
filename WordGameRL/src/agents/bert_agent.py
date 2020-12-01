
import math
import random
from collections import deque
from enum import Enum

import numpy as np

from src.nn.models.actor_critic import ActorCriticModel
from src.nn.preprocess import game_trajectory_to_states_tuples
from src.resource.evocation import free_association


class EvoType(Enum):
    SWOW=1
    FA=2
    EAT=3

class EvocationDataAgent:

    def __init__(self, strategy=None,
                       data_source=None,
                       verbose=False,
                       FSG=False,
                       data_type=EvoType.SWOW,
                       REV_FSGRANK=True,
                       REV_FSG_NORM=False,
                       evofactor=1):

        self.evofactor = evofactor
        self.cand_cache = {}
        self.verbose = verbose
        if data_type == EvoType.SWOW:
            self.data = data_source or free_association
        self.strength_dict = self.init_strength_dict(FSG, 
                                                     REV_FSGRANK, 
                                                     REV_FSG_NORM)
        self.is_strength_dict_normalized = {k: False for k in self.strength_dict}
        self.target = None

    def init_strength_dict(self, FSG, REV_FSGRANK, REV_FSG_NORM):
        # Might be modified here.
        if FSG:
            strength_dict = self.data.normwords_fsg_dict
        else:
            if REV_FSGRANK:
                strength_dict = self.data.revwords_fsg_rank_dict
            else:
                strength_dict = self.data.bsg_cnt_dict
            if REV_FSG_NORM: # over power the REV_FSG_RANK option!
                strength_dict = self.data.normrevwords_fsg_dict
        return strength_dict
    
    def set_target(self, target):
        self.target = target

    def get_all_targets(self, sort=True):
        if sort:
            return sorted(
                self.data.target_freq.items(),
                key=lambda x: x[1],
                reverse=True
            )
        return self.data.target_freq.items()

    def set_possile_target(self, possible_targets):
        self.possible_targets = possible_targets
    
    def normalize_strength_dict(self, _key):
        self.is_strength_dict_normalized[_key] = True
        helper_dict = self.strength_dict[_key]
        _max_key = max(helper_dict, key=helper_dict.get)
        _max = helper_dict[_max_key]
        self.strength_dict[_key] = {k: v / _max 
                                    for k, v in helper_dict.items()}


class BERTPolicyAgent(EvocationDataAgent):

    def __init__(self, 
                model,
                player_tag,
                memory_size=2000,
                gamma=0.7,
                **params):

        super(BERTPolicyAgent, self).__init__(**params)

        self.memory_size = memory_size
        self.memory = deque([], memory_size)
        self.model = model(player_tag=player_tag)
        self.gamma = gamma

    def get_trajectory_immediate_rewards(self, target, clues, guesses):
        raise NotImplementedError

    def update_memory(self, target, clues, guesses, winlose):
        """Update a game memeory from the end of game"""
        immediate_rewards = self.get_trajectory_immediate_rewards(target, 
                                                                  clues,
                                                                  guesses)
        
        self.memory.append((target, 
                            clues, 
                            guesses, 
                            winlose, 
                            immediate_rewards))

    def train_memory_with_evocation(self):
        '''
        Train with memory and pre-train with evo
        A memory is a dict of {target: (clues, guesses, reward)}
        :return:
        '''
        self.model.train_memory_with_evocation(self.memory)
        self.memory = deque([], self.memory_size)


class BERTDQNAgent(EvocationDataAgent):

    def __init__(self,
                 model,
                 player_tag,
                 sample_size=500,
                 memory_size=5000,
                 gamma=1,
                 update_interval=5,
                 **params):

        super(BERTDQNAgent, self).__init__(**params)

        self.gamma = gamma
        self.memory_size = memory_size
        self.memory = deque([], maxlen=memory_size)
        self.sample_size = sample_size
        self.update_interval = update_interval
        self.trained_counter = 0
        self.epsilon = 0.5
        self.model = model(player_tag=player_tag)

    def update_memory(self, target, clues, guesses, winlose):
        raise NotImplementedError
    
    def get_trajectory_immediate_rewards(self, clues, guesses):
        raise NotImplementedError

    def importance_assignment(self, 
                              current_state, 
                              action,
                              reward,
                              next_state):
        if next_state:
            target_Q_values = self.model.get_q_values(*next_state, 
                                                    self.possible_targets,
                                                    is_to_predict=False)

            max_Q_action = max(target_Q_values, key=target_Q_values.get)
            _target = target_Q_values[max_Q_action]
            _target *= self.gamma
        else:
            _target = 0
        _target += reward
        current_prediction = self.model.get_q_values(*current_state, 
                                                      self.possible_targets, 
                                                      is_to_predict=True)
        
        importance = abs(current_prediction[action] - _target)
        
        return importance

    def importance_sampling(self, temperature=2):
        sample_size = min(self.sample_size, self.memory_size)
        memory_ids = list(range(len(self.memory)))
        # memory_weights = np.array([self.importance_assignment(*memory)
        #                            for memory in self.memory])
        # memory_weights = memory_weights ** temperature
        # memory_weights /= memory_weights.sum()
        sampled_ids = np.random.choice(memory_ids, sample_size)#, p=memory_weights)
        print("Importance sampling finished")
        return [self.memory[_id] for _id in sampled_ids]

    def train_memory_with_evocation(self):
        '''
        Train with memory and pre-train with evo
        A memory is a dict of {target: (clues, guesses, reward)}
        :return:
        '''
        
        sampled = self.importance_sampling()
        self.model.train_memory_with_evocation(sampled, 
                                                self.possible_targets)
                                
        self.trained_counter += 1
        if (self.trained_counter % self.update_interval == \
                self.update_interval - 1):
            self.model.update_target_q_network()
        self.epsilon = (100 - self.trained_counter) / 100 * 0.5


class ActorCriticAgent(EvocationDataAgent):

    def __init__(self, player_tag, 
                       is_giver, 
                       gamma=1,
                       restrict_value=False,
                       **params):
        
        self.model = ActorCriticModel(player_tag=player_tag,
                                      is_giver=is_giver,
                                      gamma=gamma,
                                      restrict_value=restrict_value)
        if not is_giver:
            params['FSG'] = True

        super(ActorCriticAgent, self).__init__(**params)

        self.gamma = gamma
        self.is_giver = is_giver
            
        self.memory = [] # The memory would be evacuted after
        self.target = None
        # init max_clues and max_mask
        self.set_max_possible_clues()
        self.set_max_possible_guesses()
        self.clear_possible_targets()
        self.training_counter = 0

    def update_memory(self, trajectory):
        self.memory.append(trajectory)

    def set_target(self, target):
        self.target = target
        self.set_possible_clues()

    def get_output_mask(self, candidates):
        verb_list = self.model.tokenizer.inv_vocab
        mask = np.zeros(len(verb_list))
        for ind in range(len(verb_list)):
            word = verb_list[ind]
            mask[ind] = (word in candidates)
        return mask

    def set_max_possible_clues(self):
        self.max_cand_clues = self.data.cue_freq
        self.max_mask = self.get_output_mask(self.max_cand_clues)

    def set_max_possible_guesses(self):
        self.max_cand_target = {word for word, freq in self.data.target_freq.items() if freq > 20}
        self.target_max_mask = self.get_output_mask(self.max_cand_target)

    def clear_possible_targets(self):
        self.possible_targets = set()

    def add_possible_targets(self, clue):
        cand_targets = self.strength_dict[clue]
        new_possible_targets = dict(sorted(cand_targets.items(), 
                                    key=lambda  x: x[1])[-30:])
        new_possible_targets = set(new_possible_targets.keys())
        self.possible_targets = self.possible_targets.union(new_possible_targets)
    
    def set_possible_clues(self):
        cand_clues = self.strength_dict[self.target]
        self.possible_clues = dict(sorted(cand_clues.items(), 
                                key=lambda x: x[1])[-50:])
        
    def clear_memory(self):
        self.memory = []

    def add_train_counter(self):
        self.training_counter += 1
        
    def train(self):
        self.model.train(self.memory)
        self.clear_memory()

    def guess(self, prev_clues, prev_guesses, to_train=True):
        if len(prev_clues) == 1:
            self.clear_possible_targets()
        if self.is_giver:
            raise ValueError("Should be guesser")
        current_state = (prev_clues, prev_guesses)
        policy = self.model.get_policy(current_state)

        verb_list = self.model.tokenizer.inv_vocab
        lemmatizer = self.model.lemmatizer
        self.add_possible_targets(prev_clues[-1])
        current_mask = self.get_output_mask(self.possible_targets)
        unfiltered = np.array([1 for _ in current_mask])
        decay = self.training_counter / 2000
        current_mask = (1 - decay) * current_mask + decay * unfiltered
        
        for ind, _ in enumerate(policy):
            word = verb_list[ind]
            if to_train:
                if current_mask[ind] == 0:
                    policy[ind] = 0
                    continue
                if ((word not in self.possible_targets) or
                    lemmatizer.is_contained_in(word, prev_guesses) or 
                    lemmatizer.is_contained_in(word, prev_clues)):
                    policy[ind] = 0
                    continue

            else:
                if self.target_max_mask[ind] == 0:
                    policy[ind] = 0
                    continue
                if ((word not in self.max_cand_target) or
                    lemmatizer.is_contained_in(word, prev_guesses) or 
                    lemmatizer.is_contained_in(word, prev_clues)):
                    policy[ind] = 0
                    continue
        
        if to_train:
            policy *= current_mask
        policy *= self.target_max_mask
        policy /= policy.sum()

        words = ['' for _ in verb_list]
        for ind in range(len(words)):
            if policy[ind]:
                words[ind] = verb_list[ind]
        while True:
            action = np.random.choice(words, p=policy)
            if action != '':
                break
        
        if not to_train:
            attention = self.model.get_input_gradient(current_state, action)
            return action, attention
        else:
            #print(self.model.get_input_gradient(current_state, action))
            return action            

    def give(self, prev_clues, prev_guesses, to_train=True, only_policy=False):
        current_state = (self.target, prev_clues, prev_guesses)
        policy = self.model.get_policy(current_state)
        
        verb_list = self.model.tokenizer.inv_vocab
        lemmatizer = self.model.lemmatizer

        current_mask = self.get_output_mask(self.possible_clues)
        current_mask *= (1 - self.training_counter / 2000)
        decay = (self.training_counter / 2000)
        unfiltered = np.array([decay for _ in current_mask])
        
        current_mask += unfiltered
        current_mask *= self.max_mask

        for ind in range(len(policy)):
            word = verb_list[ind]
            if to_train:
                if current_mask[ind] == 0:
                    policy[ind] = 0
                    continue
                if (lemmatizer.is_contained_in(word, prev_clues) or
                    lemmatizer.is_contained_in(word, prev_guesses) or
                    lemmatizer.is_same(word, self.target)):
                    policy[ind] = 0
                
            else:
                if self.max_mask[ind] == 0:
                    policy[ind] = 0
                    continue

                if ((word not in self.max_cand_clues) or
                    lemmatizer.is_contained_in(word, prev_clues) or
                    lemmatizer.is_contained_in(word, prev_guesses) or
                    lemmatizer.is_same(word, self.target)):
                    policy[ind] = 0
                
        if to_train:
            policy *= current_mask
        policy *= self.max_mask
        policy_norm = policy / policy.sum()
    
        words = ['' for _ in verb_list]
        for ind in range(len(words)):
            if policy[ind]:
                words[ind] = verb_list[ind]
        if only_policy:
            return words, policy
        while True:
            try:
                action = np.random.choice(words, p=policy_norm)
            except:
                import pdb; pdb.set_trace()
            if action != '':
                break

        if not to_train:
            attention = self.model.get_input_gradient(current_state, action)
            return action, attention
        else:
            #print(self.model.get_input_gradient(current_state, action))
            return action
    

class PGACAgent(ActorCriticAgent):

    def __init__(self, **params):
        super(PGACAgent, self).__init__(**params)

    def off_policy_give(self, prev_clues, prev_guesses):
        action = self.model.giver_probability_gain_policy(
            self.target,
            prev_clues,
            prev_guesses
        )
        cand = self.strength_dict[self.target]
        cand = sorted(cand.items(), key=lambda x: x[1])[-10:]
        return action
    
    def get_probability_gain(self, prev_clues, prev_guesses,
                             action, guesser_topics=None):
        if not self.is_giver:
            raise ValueError("Only Giver could access to it")
        pg = self.model.giver_get_probability_gain(self.target, 
                                                   prev_clues, 
                                                   prev_guesses,
                                                   topics=guesser_topics)
        
        pg = dict(pg)
        pg = np.clip(pg.get(action, 0), 0, 5)
        return pg


    