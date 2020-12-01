import os
import random
from operator import itemgetter

import numpy as np
from sklearn.model_selection import train_test_split

from experiments.utils import (CURRENT_TIME, deterministic_guessers,
                               play_a_word, test_on_words)
from src.agents.bert_agent import PGACAgent


def normalize_dict(value_dict):
    _max = max(value_dict.values())
    value_dict = {k: v / _max for k, v in value_dict.items()}
    return value_dict

def target_in_guesser_view(giver, guesser, clues, guesses):
    1

def get_reward(player, clues, guess):
    # Get one step reward
    reward, discount = 0, 0.9
    discounted, avg_factor = 1, 0

    for clue in clues[::-1]:
        if clue not in player.strength_dict:
            r = 0
        else:
            if not player.is_strength_dict_normalized[clue]:
                player.strength_dict[clue] = normalize_dict(player.strength_dict[clue])
            r = player.strength_dict[clue].get(guess, 0)

        reward += discounted * r
        avg_factor += discounted
        discounted *= discount
    return reward


def get_immediate_rewards(player, clues, guesses, target=None):
    # 
    rewards = []
    if player.is_giver:
        for i in range(len(clues)):
            prev_clues = clues[:i]
            prev_guesses = clues[:i]
            action = clues[i]
            reward = player.get_probability_gain(prev_clues, 
                                                prev_guesses, 
                                                action)
            rewards.append(reward)
        
        return np.array(rewards)
    else:
        for i in range(len(guesses)):
            current_guess = guesses[i]
            prev_clues = clues[:i + 1]
            r = get_reward(player, prev_clues, current_guess)
            rewards.append(r)
        return np.array(rewards)

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
                        restrict_value=False)
    
    guesser = PGACAgent(player_tag='guesser', 
                        is_giver=False, FSG=True, 
                        restrict_value=False,
                        REV_FSG_NORM=True)
    epochs = 1000
    index = list(range(training_size))
    one_epoch_word_num = training_size * 3
    all_words = giver.get_all_targets()
    words = itemgetter(*index)(all_words)
    words = [w[0] for w in words]
    deterministic_guessers.append(guesser)
    res_dir = "./double_pg_res"
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    path = 'double_pg_{0}.json'.format(CURRENT_TIME)
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
        giver.model.save_model(epoch=epoch, path=to_save_path)
        guesser.model.save_model(epoch, path=to_save_path)
        print('=======================')
        # testing
        

if __name__ == "__main__":

    main()
