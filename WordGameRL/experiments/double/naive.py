import random
from operator import itemgetter

import numpy as np
import os
from sklearn.model_selection import train_test_split

from experiments.utils import (deterministic_guessers, play_a_word,
                               test_on_words, CURRENT_TIME)
from src.agents.bert_agent import PGACAgent



def get_immediate_rewards(player, clues, guesses, target=None):
    # 
    return np.array([0 for _ in clues])

def add_experience(giver, guesser, words):
    word = random.choice(words)
    try:
        _, winlose, clues, guesses = play_a_word(word, giver, guesser)
    except ValueError:
        return 
    rewards = get_immediate_rewards(guesser, clues, guesses)
    guesser.update_memory((clues, guesses, winlose, rewards))

    rewards = get_immediate_rewards(giver, clues, guesses)
    giver.update_memory((word, clues, guesses, winlose, rewards))


def main():
    training_size = 1000
    giver = PGACAgent(player_tag='giver', 
                        is_giver=True,
                        restrict_value=True)
    
    guesser = PGACAgent(player_tag='guesser', 
                        is_giver=False, FSG=True, 
                        restrict_value=True,
                        REV_FSG_NORM=True)
    epochs = 1000
    index = list(range(training_size))
    one_epoch_word_num = training_size * 3
    all_words = giver.get_all_targets()
    words = itemgetter(*index)(all_words)
    words = [w[0] for w in words]
    deterministic_guessers.append(guesser)
    res_dir = "./double_naive_res"
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    path = 'double_naive_{0}.json'.format(CURRENT_TIME)
    path = os.path.join(res_dir, path)
    to_save_path = os.path.join(res_dir, 'model_{}'.format(CURRENT_TIME))

    for epoch in range(epochs):
        print("Current Epoch {}".format(epoch))
        if epoch:
            for to_test_guesser in deterministic_guessers:
                test_on_words(words, giver, to_test_guesser, epoch=epoch, path=path)
        for num in range(one_epoch_word_num):
            add_experience(giver, guesser, words)
            if num % 100 == 99:
                print("Trained on a batch")
                giver.train()
                guesser.train()

        giver.model.save_model(epoch=epoch, 
                               path=to_save_path)
        guesser.model.save_model(epoch=epoch, 
                                 path=to_save_path)
        print('=======================')
        # testing
        

if __name__ == "__main__":

    main()
