"""Probability Gain
"""


import os
import random
from operator import itemgetter

import numpy as np

from experiments.utils import (CURRENT_TIME, deterministic_guessers,
                               play_a_word, test_on_words)
from src.agents.bert_agent import ActorCriticAgent, PGACAgent


def get_immediate_rewards(player, clues, guesses, target=None):
    rewards = []
    for i in range(len(clues)):
        prev_clues = clues[:i]
        prev_guesses = clues[:i]
        action = clues[i]
        reward = player.get_probability_gain(prev_clues, 
                                             prev_guesses, 
                                             action)
        if np.isnan(reward):
            reward = 0
        reward = np.clip(reward, 0, 5)
        rewards.append(reward)
    
    return np.array(rewards)

def add_experience(giver, guesser, words):
    word = random.choice(words)
    try:
        _, winlose, clues, guesses = play_a_word(word, giver, guesser)
    except ValueError:
        return
    except KeyError:
        return
    if isinstance(guesser, PGACAgent):
        rewards = get_immediate_rewards(guesser, clues, guesses)
        guesser.update_memory((clues, guesses, winlose, rewards))

    if isinstance(giver, PGACAgent):
        rewards = get_immediate_rewards(giver, clues, guesses)
        
        giver.update_memory((word, clues, guesses, winlose, rewards))


def main():
    training_size = 1000
    index= range(training_size)
    epochs = 1000
    one_epoch_word_num = training_size * 3
    giver = PGACAgent(player_tag='giver', is_giver=True, 
                     restrict_value=False)
    all_words = giver.get_all_targets()
    words = itemgetter(*index)(all_words)
    words = [w[0] for w in words]
    res_dir = "./single_pg_res"
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    path = 'single_pg_{0}.json'.format(CURRENT_TIME)
    path = os.path.join(res_dir, path)
    to_save_path = os.path.join(res_dir, 'model_{}'.format(CURRENT_TIME))

    for epoch in range(epochs):
        print("Current Epoch {}".format(epoch))
        for guesser in deterministic_guessers:
            print("In the testing set")
            if epoch:
                test_on_words(words, giver, guesser, epoch=epoch, path=path)    
        for num in range(one_epoch_word_num):
            guesser = random.choice(deterministic_guessers)
            add_experience(giver, guesser, words)
            try:
                if num % 100 == 99:
                    print("Finished a training process")
                    giver.train()
            except:
                import pdb; pdb.set_trace()
        giver.model.save_model(epoch=epoch, 
                               path=to_save_path)
        print('=======================')
        # testing
        

if __name__ == "__main__":

    main()
