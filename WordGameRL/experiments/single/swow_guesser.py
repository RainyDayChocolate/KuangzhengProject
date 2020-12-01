import os
import random
from operator import itemgetter

import numpy as np

from experiments.utils import (CURRENT_TIME, deterministic_guessers, giver12,
                               giver21, giver22, giver31, play_a_word,
                               test_on_words)
from src.agents.bert_agent import ActorCriticAgent, PGACAgent

deter_givers = [giver12, giver21, giver22, giver31]

def normalize_dict(value_dict):
    _max = max(value_dict.values())
    value_dict = {k: v / _max for k, v in value_dict.items()}
    return value_dict


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
    except KeyError:
        return 

    if isinstance(guesser, PGACAgent):
        rewards = get_immediate_rewards(guesser, clues, guesses)
        guesser.update_memory((clues, guesses, winlose, rewards))

def main():
    training_size = 1000
    index= range(training_size)
    epochs = 1000
    one_epoch_word_num = training_size * 3

    guesser = PGACAgent(player_tag='guesser', is_giver=False, 
                        restrict_value=False)
    all_words = guesser.get_all_targets()

    words = itemgetter(*index)(all_words)
    words = [w[0] for w in words]
    res_dir = "./single_naive_guesser_res"
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    path = 'single_guesser_{0}.json'.format(CURRENT_TIME)
    path = os.path.join(res_dir, path)
    to_save_path = os.path.join(res_dir, 'model_{}'.format(CURRENT_TIME))

    for epoch in range(epochs):
        print("Current Epoch {}".format(epoch))
        for giver in deter_givers:
            print("In the testing set")
            if epoch:
                test_on_words(words, giver, guesser, epoch=epoch, path=path)    
    
        for num in range(one_epoch_word_num):
            giver = random.choice(deter_givers)
            add_experience(giver, guesser, words)
            try:
                if num % 100 == 99:
                    print("Finished a training process")
                    guesser.train()
            except:
                guesser.memory.clear()
        guesser.model.save_model(epoch=epoch, 
                               path=to_save_path)
        print('=======================')
        # testing
        

if __name__ == "__main__":

    main()
