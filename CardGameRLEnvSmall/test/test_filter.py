from pyminiddz.miniddz import GameState
from pymodels.history.filters.naive_filter import SampleFilter
import json

def parse_filter_data(data):
    records = data['data']
    #for C, D, E
    empty_recorders = lambda: [[], [], []]
    states, rationalities = [empty_recorders()
                                for _ in range(2)]

    game_state_helper = GameState()
    for rec in records:
        player_idx = rec['player']
        game_state_helper.from_dict(rec['real_state'])
        states[player_idx].append(game_state_helper.copy())
        rationalities[player_idx].append(1)
        states[player_idx].append(game_state_helper.sample_upper_lower())
        rationalities[player_idx].append(0)

    return states, rationalities

def making_filter_training_data(game_datas):
    empty_recorders = lambda: [[], [], []]
    all_states, all_rationalities = [empty_recorders()
                                        for _ in range(2)]

    def _update(main_records, to_extend):
        for player, infos in enumerate(to_extend):
            main_records[player].extend(infos)

    for data in game_datas:
        states, rations = parse_filter_data(data)
        _update(all_states, states)
        _update(all_rationalities, rations)

    return all_states, all_rationalities

sample_filter = SampleFilter()

all_game_datas = []
game_file = open('./data/test_filter.json')
for i, line in enumerate(game_file.readlines()):
    if i == 30000:
        break
    #try:
    if i % 1000 == 0 and i:
        print('{} games has processed'.format(i))
    all_game_datas.append(json.loads(line))
    #except:
    #    continue
game_file.close()

states, rations = making_filter_training_data(all_game_datas)
sample_filter.fit(states[0], rations[0])



