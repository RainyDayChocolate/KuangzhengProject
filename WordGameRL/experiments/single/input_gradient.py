"""Probability Gain
"""


import random
from operator import itemgetter

import numpy as np

from experiments.utils import (deterministic_guessers, play_a_word,
                               test_on_words)
from src.agents.bert_agent import PGACAgent


import random
from operator import itemgetter

import numpy as np

from experiments.utils import (deterministic_guessers, play_a_word,
                               test_on_words)
from src.agents.bert_agent import PGACAgent


def get_immediate_rewards(player, clues, guesses, target=None):
    proba_gains, rewards = [], []
    target = player.target
    for i in range(len(clues)):
        prev_clues = clues[:i]
        prev_guesses = clues[:i]
        action = clues[i]
        current_state = (target, 
                         tuple(prev_clues), 
                         tuple(guesses))
        
        attention = player.model.get_input_gradient(current_state, action)
        proba_gain = player.get_probability_gain(prev_clues, 
                                             prev_guesses, 
                                             action)
        prev_proba_gain = 0
        for ind, clue in enumerate(prev_clues):
            prev_proba_gain += attention[clue] * proba_gains[ind]
        rewards.append(proba_gain - prev_proba_gain)
        proba_gains.append(proba_gain)
    
    return np.array(rewards) #+ np.array(proba_gains)

def add_experience(giver, guesser, words):
    word = random.choice(words)
    try:
        _, winlose, clues, guesses = play_a_word(word, giver, guesser)
    except ValueError:
        return
    if isinstance(guesser, PGACAgent):
        rewards = get_immediate_rewards(guesser, clues, guesses)
        guesser.update_memory((clues, guesses, winlose, rewards))

    if isinstance(giver, PGACAgent):
        rewards = get_immediate_rewards(giver, clues, guesses)
        giver.update_memory((word, clues, guesses, winlose, rewards))


def main():
    training_size = 1000
    index, test_index = range(training_size), range(training_size, training_size + 59)
    epochs = 1000
    one_epoch_word_num = training_size * 3
    giver = PGACAgent(player_tag='giver', is_giver=True, 
                             restrict_value=False)
    all_words = giver.get_all_targets()
    words = itemgetter(*index)(all_words)
    words = [w[0] for w in words]

    test_words = itemgetter(*test_index)(all_words)
    test_words = [w[0] for w in test_words]

    for epoch in range(epochs):
        print("Current Epoch {}".format(epoch))
        for guesser in deterministic_guessers:
            print("In the testing set")
            if epoch:
                test_on_words(test_words, giver, guesser)
    
        for num in range(one_epoch_word_num):
            guesser = random.choice(deterministic_guessers)
            add_experience(giver, guesser, words)
            try:
                if num % 100 == 99:
                    giver.train()
            except:
                continue
        print('=======================')
        # testing
        

if __name__ == "__main__":

    main()
