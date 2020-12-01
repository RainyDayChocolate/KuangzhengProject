from nltk.stem import WordNetLemmatizer

from src.strutil import common_prefix


def hit( guess, target):
    _guess = guess.strip().lower()
    _target = target.strip().lower()
    if _guess == _target:
        return True, 1
    from difflib import SequenceMatcher
    ratio = SequenceMatcher(None, _guess, _target).ratio()
    if ratio > 0.9 or (_target in _guess and len(_guess) > 0):
        return True, ratio
    return False, 0

# So what is the giver's policyt
class TargetLemmatizer:

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self._cache = {}

    def is_same(self, word_one, word_two):

        to_judge = tuple((word_one, word_two))
        if to_judge not in self._cache:    
            word_one = self.lemmatizer.lemmatize(word_one)
            word_two = self.lemmatizer.lemmatize(word_two)
            self._cache[to_judge] = (word_one == word_two)
        return self._cache[to_judge]

    def lemmatize(self, word):
        return self.lemmatizer.lemmatize(word)
    
    def is_contained_in(self, word, word_list):
        return any([self.is_same(word, w) for w in word_list])



target_lematizer = TargetLemmatizer()
hit = target_lematizer.is_same

if __name__ == '__main__':
    print(hit('sea', 'sex'))
