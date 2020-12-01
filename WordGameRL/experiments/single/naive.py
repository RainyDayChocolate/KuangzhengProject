import random
from operator import itemgetter
import os

import numpy as np
from sklearn.model_selection import train_test_split

from experiments.utils import (deterministic_guessers, play_a_word, test_on_words, CURRENT_TIME)
from src.agents.bert_agent import ActorCriticAgent


def get_immediate_rewards(player, clues, guesses, target=None):
    return np.array([0 for _ in clues])

def add_experience(giver, guesser, words):
    word = random.choice(words)
    try:
        _, winlose, clues, guesses = play_a_word(word, giver, guesser,  to_train=True)
        print(winlose, clues, guesses)
    except ValueError:
        return 
    if isinstance(guesser, ActorCriticAgent):
        rewards = get_immediate_rewards(guesser, clues, guesses)
        guesser.update_memory((clues, guesses, winlose, rewards))

    if isinstance(giver, ActorCriticAgent):
        rewards = get_immediate_rewards(giver, clues, guesses)
        giver.update_memory((word, clues, guesses, winlose, rewards))

## For testing need to delete

def main():
    giver = ActorCriticAgent(player_tag='giver', 
                             is_giver=True,
                             restrict_value=True)
    training_size = 1000
    epochs = 1000
    one_epoch_word_num = training_size * 3
    all_words = giver.get_all_targets()
    words = all_words[:training_size]
    words = [w[0] for w in words]
    res_dir = "./single_naive_res"
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    path = 'single_naive_{0}.json'.format(CURRENT_TIME)
    path = os.path.join(res_dir, path)
    to_save_path = os.path.join(res_dir, 'model_{}'.format(CURRENT_TIME))
    for epoch in range(epochs):
        print("Current Epoch {}".format(epoch))
        
        guesser = deterministic_guessers[0]
        for guesser in deterministic_guessers:
            # print("In the training set")
            # test_on_words(words, giver, guesser)
            print("In the testing set")
            if epoch:
                test_on_words(words, giver, guesser, epoch=epoch, path=path)
    
        for num in range(one_epoch_word_num):
            guesser = random.choice(deterministic_guessers)
            #guesser = deterministic_guessers[0]
            add_experience(giver, guesser, words)
            if num % 100 == 99:
                print("Trained on a batch")
                giver.train()
            if num == (one_epoch_word_num - 1):
                giver.add_train_counter()
        
        giver.model.save_model(epoch=epoch, path=to_save_path)
        print('=======================')
        # testing


if __name__ == "__main__":

    main()
