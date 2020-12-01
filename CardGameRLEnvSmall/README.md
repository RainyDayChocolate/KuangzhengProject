## miniddz-env
This miniddz-environment provides a mini-game of Doudizhu

* ### A deck of Cards
    B222AAAKKKQQQJJJTTT999

* ### Pattern
    * **Pass**
    * **Single**: 2 > A > K > Q > J > T > 9
    * **Pair**:   22 > AA > KK > QQ > JJ > TT > 99
    * **Straight**:
        * **3-straight**: AKQ > KQJ > QJT > JT9
        * **4-straight**: AKQJ > KQJT > QJT9
        * **5-straight**: AKQJT > KQJT9
        * **6-straight**: AKQJT9

    * **Super**: B > EVERYTHING

* ### Players
    * **Members**:
        * **Landlord**: C
        * **Peasants**: D, E
    * **Order from the start of each game**: C -> D -> E -> C -> ……

    * **Cards**: The initial cards of each player were distributed randomly.
    Numbers of cards owned by each player are: C: 8, D: 7, E: 7

* ### Result
    * **Winner decision**: If the Landlord played out all his cards first, the Landlord wins the game; the peasants win otherwise.
    * **Score computation**:The BASE SCORE for each player is 1(win) or -1(loss).
        * **Super was played out**: FINAL SCORE = BASE SCORE X 2
        * **Super was not played out**: FINAL SCORE = BASE SCORE

* ### Other Hints
    * **Lead or Follow Round**: If the current player is the last one who not play pass, he could choose any legal play or he should pass or play a legal play that bigger than the last non-pass.
    * **Bidding**: To reduce the complexity of game, the bidding phase is not involved.

## Quick start

    git clone https://github.com/dummycat/miniddz-env.git

    cd miniddz-env

    python3 example_game.py
