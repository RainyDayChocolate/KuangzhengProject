from pyminiddz.miniddz import GameState

game_state = GameState()
moves = game_state.get_legal_moves()

from pymodels.policy.residual_policy import ResidualPolicyValue

policy_value = ResidualPolicyValue()

policy_value.fit([game_state] * 1000, [{moves[0]: 1.0, moves[1]: 1}] * 1000, [1] * 1000, epochs=20)