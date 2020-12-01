import os
import random
from operator import itemgetter

import numpy as np
from sklearn.model_selection import train_test_split

from experiments.utils import (CURRENT_TIME, deterministic_guessers, giver12,
                               giver21, giver22, giver31, play_a_word,
                               test_on_words)
from src.agents.bert_agent import ActorCriticAgent

deter_givers = [giver12, giver21, giver22, giver31]

def get_immediate_rewards(player, clues, guesses, target=None):
    return np.array([0 for _ in guesses])

def add_experience(giver, guesser, words):
    word = random.choice(words)
    try:
        _, winlose, clues, guesses = play_a_word(word, giver, guesser,  to_train=True)
    except ValueError:
        return 
    except KeyError:
        return
    if isinstance(guesser, ActorCriticAgent):
        rewards = get_immediate_rewards(guesser, clues, guesses)
        guesser.update_memory((clues, guesses, winlose, rewards))

    if isinstance(giver, ActorCriticAgent):
        rewards = get_immediate_rewards(giver, clues, guesses)
        giver.update_memory((word, clues, guesses, winlose, rewards))

## For testing need to delete

def main():
    guesser = ActorCriticAgent(player_tag='guesser', 
                             is_giver=False,
                             restrict_value=True)
    training_size = 1000
    epochs = 1000
    one_epoch_word_num = training_size * 3
    all_words = guesser.get_all_targets()
    words = all_words[:training_size]
    words = [w[0] for w in words]
    res_dir = "./single_naive_gueeser_res"
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    path = 'single_naive_gueeser_{0}.json'.format(CURRENT_TIME)
    path = os.path.join(res_dir, path)
    to_save_path = os.path.join(res_dir, 'model_{}'.format(CURRENT_TIME))

    for epoch in range(epochs):
        print("Current Epoch {}".format(epoch))
    
        for giver in deter_givers:
            # print("In the training set")
            # test_on_words(words, giver, guesser)
            print("In the testing set")
            if epoch:
                test_on_words(words, giver, guesser, epoch=epoch, path=path)
    
        for num in range(one_epoch_word_num):
            giver = random.choice(deter_givers)
            #guesser = deterministic_guessers[0]
            add_experience(giver, guesser, words)
            if num % 100 == 99:
                print("Trained on a batch")
                guesser.train()
        
        guesser.model.save_model(epoch=epoch, path=to_save_path)
        print('=======================')
        # testing


if __name__ == "__main__":

    main()
