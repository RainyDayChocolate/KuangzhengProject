

from experiments.utils import (CURRENT_TIME, giver12, guesser12,
                               guesser12_family, guesser22, guesser22_family,
                               guesser32, guesser32_family, guesser42,
                               guesser42_family, guesser52, guesser52_family,
                               test_on_words)
from src.agents.bert_agent import PGACAgent


def topic_guesses_checker():
    giver = giver12
    training_size = 200
    all_words = giver.get_all_targets()
    words = all_words[:training_size]
    words = [w[0] for w in words]
    print("Test for 1")
    giver = PGACAgent(player_tag='giver', is_giver=True, 
                    restrict_value=False)
    giver.model.load_model('./remote_models/single_pg/giver')

        
    test_on_words(words, giver, guesser12, epoch=0, to_print_traj=False)
    print("Test for Topic 1")
    for guesser in guesser12_family:
        test_on_words(words, giver, guesser, epoch=0, to_print_traj=False)
    print("Test for 2")
    test_on_words(words, giver, guesser22, epoch=0, to_print_traj=False)
    print("Test for Topic 2")
    for guesser in guesser22_family:
        test_on_words(words, giver, guesser, epoch=0, to_print_traj=False)
    print("Test for 3")
    test_on_words(words, giver, guesser32, epoch=0, to_print_traj=False)
    print("Test for Topic 3")
    for guesser in guesser32_family:
        test_on_words(words, giver, guesser, epoch=0, to_print_traj=False)
    # testing
    print("Test for 4")
    test_on_words(words, giver, guesser42, epoch=0, to_print_traj=False)
    print("Test for Topic 4")
    for guesser in guesser42_family:
        test_on_words(words, giver, guesser, epoch=0, to_print_traj=False)
    print("Test for 5")
    test_on_words(words, giver, guesser52, epoch=0, to_print_traj=False)
    print("Test for Topic 5")
    for guesser in guesser52_family:
        test_on_words(words, giver, guesser, epoch=0, to_print_traj=False)

def the_rank_in_fsg_bsg():

    giver = giver12
    training_size = 200
    all_words = giver.get_all_targets()
    words = all_words[:training_size]
    words = [w[0] for w in words]
    giver.data.ge

if __name__ == "__main__":

    main()
