"""Probability Gain
"""


import os
import random
from operator import itemgetter

import numpy as np

from experiments.single.pg_topic import get_immediate_rewards
from experiments.utils import (CURRENT_TIME, deterministic_guessers,
                               play_a_word, test_on_words, topic_guesseres, make_new_topic)
from src.agents.bert_agent import ActorCriticAgent, PGACAgent
from src.agents.zoo import (giver12, giver21, giver22, giver31, giver_bert,
                            guesser11, guesser12, guesser12_clue, guesser21,
                            guesser22, guesser22_clue, guesser31, guesser32,
                            guesser32_clue, guesser41, guesser42,
                            guesser42_clue, guesser51, guesser52,
                            guesser52_clue)

clue_topic_guessers = [guesser12_clue, guesser22_clue, guesser32_clue,
                 guesser42_clue, guesser52_clue]

def set_topics(topic):
    for guesser in clue_topic_guessers:
        guesser.set_topic(topic)


def add_experience(giver, guesser, words):
    word = random.choice(words)
    topic = make_new_topic(word, 
                           giver.strength_dict)
    guesser.set_topic(topic)
    try:
        _, winlose, clues, guesses = play_a_word(word, giver, guesser)
    except ValueError:
        return
    if isinstance(giver, PGACAgent):
        rewards = get_immediate_rewards(giver, clues, guesses)
        giver.update_memory((word, clues, guesses, 3 * winlose, rewards))


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
    res_dir = "./single_pg_clue_topic_res"
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    path = 'single_pg_clue_topic_{0}.json'.format(CURRENT_TIME)
    path = os.path.join(res_dir, path)
    to_save_path = os.path.join(res_dir, 'model_{}'.format(CURRENT_TIME))

    for epoch in range(epochs):
        print("Current Epoch {}".format(epoch))
        for guesser in deterministic_guessers:
            print("In the testing set")
            if epoch:
                test_words = random.sample(words, int(training_size / 5))
                test_on_words(test_words, giver, guesser, epoch=epoch, path=path)    
        for num in range(one_epoch_word_num):
            guesser = random.choice(clue_topic_guessers)
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
