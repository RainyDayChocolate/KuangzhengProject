from pymodels.search.psmcts import PIMC


class PIMCPlayer:

    def __init__(self, **kwargs):

        self._pimc = PIMC(**kwargs)

    def get_move(self, state):

        return self._pimc.get_move(state)

    def get_pimc(self):

        return self._pimc
