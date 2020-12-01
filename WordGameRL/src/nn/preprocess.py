from src.nn import model


def giver_game_trajectory_to_states(traj_tuple, gamma):

    target, clues, guesses, winlose, rewards = traj_tuple
    states_tuples = []
    for ind in range(len(clues)):
        current_game_state = (target, tuple(clues[:ind]), 
                              tuple(guesses[:ind]))
        action = clues[ind]
        if ind == (len(guesses) - 1):
            next_game_state = tuple()
            # Trick to tune final reward

            immediate_reward = rewards[ind] + winlose
        else:
            immediate_reward = rewards[ind]
            next_game_state = (target, tuple(clues[:ind + 1]),
                               tuple(guesses[:ind + 1]))
        state_tuple = (current_game_state, action,
                       immediate_reward, next_game_state)

        states_tuples.append(state_tuple)
    return states_tuples


def guesser_game_trajectory_to_states(traj_tuple, gamma):
    clues, guesses, winlose, rewards = traj_tuple
    states_tuples = []
    if winlose:
        rewards[-1] = 0
    for ind in range(len(clues)):
        current_game_state = (tuple(clues[:ind + 1]), 
                              tuple(guesses[:ind]))
        action = guesses[ind]
        if ind == (len(clues) - 1):
            next_game_state = tuple()
            # Trick to tune final reward

            immediate_reward = rewards[ind] + winlose
        else:
            try:
                immediate_reward = rewards[ind]
            except:
                import pdb; pdb.set_trace()
            next_game_state = (tuple(clues[:ind + 2]),
                            tuple(guesses[:ind + 1]))
        state_tuple = (current_game_state,
                        action,
                        immediate_reward,
                        next_game_state)
        states_tuples.append(state_tuple)
    return states_tuples



def giver_game_state_to_token_segments(target, clues, guesses):
    
    SEP_token = "[SEP]"
    tokens = ["[CLS]", SEP_token, target, SEP_token]
    GIVER_repr = 1
    GUESSER_repr = 2
    TARGET_repr = 3
    segment_ids = [0, 0, TARGET_repr, 0]
    # 0: default tokens, 1 giver, 2 guess
    if isinstance(clues, list):
        clues = clues.copy()
    
    for clue, guess in zip(clues, guesses):
        tokens.extend([clue, SEP_token, guess, SEP_token])
        segment_ids.extend([GIVER_repr, 0, GUESSER_repr, 0])
    if len(clues) == (len(guesses) + 1):
        tokens.extend([clues[-1], SEP_token])
    else:
        # For prediction
        tokens.extend(["[MASK]", SEP_token])

    segment_ids.extend([GIVER_repr, 0])
    return tokens, segment_ids


def guesser_game_state_to_token_segments(clues, guesses):
    # for guesser
    tokens = ["[CLS]"]
    SEP_token = "[SEP]"
    
    # 0: default tokens, 1 giver, 2 guess
    GIVER_repr = 1
    GUESSER_repr = 2
    segment_ids = [0]

    # to predict
    if (len(clues) - 1) == len(guesses):
        if isinstance(guesses, list):
            guesses = guesses.copy()
            guesses.append("[MASK]")
        else:
            guesses = guesses + ("[MASK]", )

    for clue, guess in zip(clues, guesses):
        tokens.extend([clue, SEP_token, guess, SEP_token])
        segment_ids.extend([GIVER_repr, 0, GUESSER_repr, 0])
    return tokens, segment_ids


def game_state_to_token_segments(target, clues, guesses, is_giver):
    if is_giver:
        return giver_game_state_to_token_segments(target, clues, guesses)
    else:
        return guesser_game_state_to_token_segments(clues, guesses)


def game_state_to_giver_instance(target, clues, guesses, winlose, 
                                 rewards, gamma, to_predict=False, 
                                 current_state_value = 0, action=None):
    """Game State was represented as clues and guesses
    The input token would be displayed as 
    [[CLS], clue, [SEP], guess, [SEP], clue, [SEP], [MASK], [SEP]]
    The output would be the real guess which was replaced by [MASK]
    Classification problem.
    """
    if not isinstance(clues, tuple):
        clues = tuple(clues)
    if not isinstance(guesses, tuple):
        guesses = tuple(guesses)

    if not to_predict:
        clues = clues + tuple([action])
    tokens, segment_ids = giver_game_state_to_token_segments(target, clues, guesses)    
    tokens, masked_lm_positions, masked_lm_labels  =\
                             model.create_masked_lm_predictions(tokens)
    if not to_predict:
        estimate_values = get_estimate_value(rewards, winlose, gamma)
    else:
        estimate_values = 0
    training_instance = model.TrainingInstance(
                        tokens=tokens,
                        segment_ids=segment_ids,
                        masked_lm_labels=masked_lm_labels,
                        masked_lm_positions=masked_lm_positions,
                        reward=[[estimate_values - current_state_value]],
                        target_state_value=estimate_values)
    return training_instance

def game_state_to_guesser_instance(clues, guesses, winlose, 
                                   rewards, gamma, to_predict=False, 
                                   current_state_value=0, action=None):
    """Game State was represented as clues and guesses
    The input token would be displayed as 
    [[CLS], clue, [SEP], guess, [SEP], clue, [SEP], [MASK], [SEP]]
    The output would be the real guess which was replaced by [MASK]
    Classification problem.
    """
    if not to_predict:
        guesses += tuple([action])
    tokens, segment_ids = guesser_game_state_to_token_segments(clues, guesses)

    tokens, masked_lm_positions, masked_lm_labels  =\
                             model.create_masked_lm_predictions(tokens)
    if not to_predict:
        estimate_values = get_estimate_value(rewards, winlose, gamma)
    else:
        estimate_values = 0

    training_instance = model.TrainingInstance(
                        tokens=tokens,
                        segment_ids=segment_ids,
                        masked_lm_labels=masked_lm_labels,
                        masked_lm_positions=masked_lm_positions,
                        reward=[[estimate_values - current_state_value]],
                        target_state_value=estimate_values)
    return training_instance


def get_to_train_giver_instances(traj_tuple, gamma):
    """Creates training instances for a guesser of game play
    """
    target, clues, guesses, winlose, rewards = traj_tuple
    training_instances = []
    for ind in range(len(clues)):
        training_instance = game_state_to_giver_instance(
                            target=target,
                            clues=clues[:ind + 1], 
                            guesses=guesses[:ind], 
                            winlose=winlose, 
                            rewards=rewards[ind:],
                            gamma=gamma)
        training_instances.append(training_instance)
    return training_instances


def get_to_train_guesser_instances(traj_tuple, gamma):
    """Creates training instances for a guesser of game play
    """
    clues, guesses, winlose, rewards = traj_tuple
    training_instances = []
    for ind in range(len(clues)):
        training_instance = game_state_to_guesser_instance(
                            clues=clues[:ind + 1], 
                            guesses=guesses[:ind + 1], 
                            winlose=winlose, 
                            rewards=rewards[ind:],
                            gamma=gamma)
        training_instances.append(training_instance)
    return training_instances


def get_to_predict_giver_instance(traj_tuple):
    target, clues, guesses, winlose, rewards = traj_tuple
    return game_state_to_giver_instance(
                            target=target,
                            clues=clues, 
                            guesses=guesses, 
                            winlose=winlose, 
                            rewards=rewards,
                            gamma=0,
                            to_predict=True)


def get_to_predict_guesser_instance(traj_tuple):
    clues, guesses, winlose, rewards = traj_tuple
    return game_state_to_guesser_instance(
                            clues=clues, 
                            guesses=guesses, 
                            winlose=winlose, 
                            rewards=rewards,
                            gamma=0,
                            to_predict=True)


def get_estimate_value(immediate_rewards, winlose, gamma):
    discounted = 1
    estimates_value = 0
    for ind, reward in enumerate(immediate_rewards):
        if ind != 0:
            discounted *= gamma
        estimates_value += discounted * reward
    estimates_value += winlose * discounted
    return estimates_value


def convert_instances_to_features(instances, max_seq_length, tokenizer):
    """Convert tokens to numercial features.
    Instances is a list of training instance defined in model
    """
    features = []
    for instance in instances:
        input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
        input_mask = [1 for _ in input_ids]
        segment_ids = instance.segment_ids.copy()
        to_filled_num = max_seq_length - len(segment_ids)
        for _ in range(to_filled_num):
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        masked_lm_positions = instance.masked_lm_positions
        masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = instance.reward
        action_indexes = instance.action_indexes
        feature = model.InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            masked_lm_positions=masked_lm_positions,
            masked_lm_ids=masked_lm_ids,
            masked_lm_weights=masked_lm_weights,
            action_indexes=action_indexes)
        features.append(feature)
    return features


def get_DQN_target(DQN_model, next_state, 
                   immediate_reward, possible_targets):
    if next_state:
        target_Q_values = DQN_model.get_q_values(*next_state, 
                                            possible_targets,
                                            is_to_predict=False)
        # to approxiamte r + gamma * target(Q)
        to_approx = max(target_Q_values, key=target_Q_values.get)
        to_approx = target_Q_values[to_approx]
        to_approx *= DQN_model.gamma
    else:
        to_approx = 0
    to_approx += immediate_reward
    return to_approx

def get_DQN_giver_training_instance(
                              current_state, 
                              action, 
                              to_approx):
    """A simplified GAME STATE for giver 
        should be viewed as a tuple(target, prev_cues, prev_guesses)
        In a word the observation of giver
        immediate_reward might be set as 0
        target_Q_network is an instance of giverDQNModel
        action is a word
        A big tuple memorized is like
        (current_state(defined as above), action, immediate_reward, next_state)

    """

    target, clues, guesses = current_state
    clues = clues + (action,)

    # After Q(s, a)
    tokens, segment_ids = giver_game_state_to_token_segments(target, clues, guesses)

    tokens, masked_lm_positions, masked_lm_labels  =\
                            model.create_masked_lm_predictions(tokens)
    training_instance = model.TrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    masked_lm_labels=masked_lm_labels,
                    masked_lm_positions=masked_lm_positions,
                    reward=[[to_approx]])
    return training_instance


def get_DQN_guesser_training_instance(
                              current_state, 
                              action, 
                              to_approx):
    """A simplified GAME STATE for guesser 
        should be viewed as a tuple(prev_cues, prev_guesses)
        immediate_reward might be set as 0
        target_Q_network is an instance of GuesserDQNModel
        action is a word
        A big tuple memorized is like
        (current_state(defined as above), action, immediate_reward, next_state)

    """

    clues, guesses = current_state
    if isinstance(guesses, list):
        guesses = guesses.copy()
        guesses.append(action)
    elif isinstance(guesses, tuple):
        guesses = guesses + (action,)

    tokens, segment_ids = guesser_game_state_to_token_segments(clues, 
                                                               guesses)

    tokens, masked_lm_positions, masked_lm_labels  =\
                            model.create_masked_lm_predictions(tokens)
    training_instance = model.TrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    masked_lm_labels=masked_lm_labels,
                    masked_lm_positions=masked_lm_positions,
                    reward=[[to_approx]])
    return training_instance



def game_trajectory_to_states_tuples(traj_tuple, gamma, is_giver):
    """Tuples are (state, action, reward, next_state)
    """
    if is_giver:
        return giver_game_trajectory_to_states(traj_tuple, gamma)
    else:
        return guesser_game_trajectory_to_states(traj_tuple, game_state_to_guesser_instance) 

# Actor Critical use
def game_state_to_instance(state, winlose, 
                           rewards, gamma, to_predict, is_giver, 
                           current_state_value, action=None):
    if is_giver:
        target, clues, guesses = state
        instance = game_state_to_giver_instance(target, clues, guesses, winlose, 
                           rewards, gamma=gamma, to_predict=to_predict, 
                           current_state_value=current_state_value,
                           action=action)
        return instance
    else:
        clues, guesses = state
        return game_state_to_guesser_instance(clues, guesses, winlose, 
                           rewards, gamma=gamma, to_predict=to_predict,
                           current_state_value=current_state_value,
                           action=action)

def game_state_to_predict_instance(state, is_giver):
    
    helper = 0
    if is_giver:
        target, clues, guesses = state
        return game_state_to_giver_instance(target, clues, guesses, helper, 
                           helper, gamma=helper, to_predict=True)
    else:
        clues, guesses = state
        return game_state_to_guesser_instance(clues, guesses, helper, 
                           helper, gamma=helper, to_predict=True)

    