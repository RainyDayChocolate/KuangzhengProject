#from train.fpmcts_selfplay_training import Trainer
#import pprint
#import json
#from agents.fpmcts_agent import FPMCTSPlayer

##players = [FPMCTSPlayer(0), FPMCTSPlayer(1), FPMCTSPlayer(2)]
#trainer = Trainer(model_id='20190418-132807')
#trainer.learning_from_file('/Users/kuangzheng/Programming/miniddz-env/fpmcts_selfplay/game_data/20190409-183049')

""" for _ in range(100):
    trainer.run_a_game()
with open('./20190413-175050') as f:
    lines = f.readlines()

for line in lines:
    game_data = json.loads(line)
    data = trainer.parse_data(game_data)
    print('-------------- ')
    for x, y, z in data:
        print(x)
        print('Policy')
        print(y)
        print('Result')
        print(z)

        import pdb; pdb.set_trace()
"""
import os
import time

for _ in range(10):
    print(os.listdir('.'))
    time.sleep(10)

