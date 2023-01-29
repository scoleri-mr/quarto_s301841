Final project for the Masters' Degree course Computational Intelligence at Politecnico di Torino. 
The goal of this project is to develop an agent able to play the game Quarto. Two approaches were considered: Reinforcement Learning and MinMax.

# Min Max Player -> Champion
The best strategy to win at Quarto proves to be the Min Max. Due to the high complexity of the game, building a complete tree proved to be impossible. We implemented a min max strategy with different tecqniques that in the end allow us to win around 99.8% of the matches against a random player.  
Our player can use three different strategies, uses a heuristic called Quarticity found in [1](#1), and includes a tweaked random opening move. 


## Minmax strategies
We developed three different strategies that can be selected using the strategy parameter of the minmax player. The bound_value parameter has different meaning based on the selected strategy but in general it represents how deep will be the search on the tree performed by the minmax.
- **Strategy 1**: set with strategy=1 and a bound_value of choice. In this mode the bound_value is the actual depth of the tree that the minmax will explore for every move. When using strategy=1 we usually set bound_value=1, meaning that the minmax will visit the tree using a maximum depth of 1.
- **Strategy 2**: set with strategy=2 and a bound_value of choice. When this mode is selected the minmax will run without a bounded depth if the complete tree from the given state visits a number of states (or combinations) that is less than the value found in bound_value. Basically, with this mode, when the functions `choose_piece` and `place_piece` are used, the `tweak_depth` function is called. The `tweak_depth` function calls `estimate_tree_complexity` which, as the names suggests, gives as estimate of the tree complexity starting from the given state. If this complexity is less than the complexity chosen and passed as parameter, the minmax will visit the complete tree, otherwise it will set a maximum depth=1.  
The `estimate_tree_complexity` works as follows: it counts how many free places are left on the board and based on that counts how many possible states the game (and the minmax tree as well) will have. It's an estimate because our minmax uses the alfa-beta pruning technique which means that the actual number of visited states could be inferior to the one given by the `estimate_tree_complexity` function.
- **Strategy 3**: set with strategy=3 and a bound_value of choice. When this mode is selected the minmax will always go at a depth that will explore a number of states as close as possible (but inferior) to the value passed as parameter in bound_value. Similarly to the previous mode, the `tweak_depth` function is called. This time it calls the `find_depth` function that based on the remaining turns and the given state calculates a depth for the minmax. This depth is the maximum possible depth that visits a number of states less than what is passed in bound_value.  
  
FOR THE TOURNAMENT: the best player is the minmax player that uses the third strategy with bound_value=math.factorial(5)**2. These are already the default parameters.

## The choice of bound_value
Suppose we have a board with 5 free spots to place a piece and that the game is going to end in a draw (worst case scenario in terms of number of moves). It means we are given 1 of 5 pieces to place on 5 possible places (5·5), then we can choose between 4 pieces for our opponent which will have 4 places left to choose from to place the piece (4·4) and so on. This means that if there are 5 moves left to be made we have 5·5·4·4·3·3·2·2 possible combinations which is (5!)^2.  
Choosing a bound value of (5!)^2 means that the minmax will be able to visit the complete tree if there are only 5 missing moves. This allows the minmax to win most of the matches since it can visit a winning state. Increasing the bound value would of course increase the performance as well but it can become very time consuming. We tried several values following this strategy and the best tradeoff between performance and time to make a move was found to be with (5!)^2. With a minmax of strategy 3 and this bound_value, an average game is played in under 4 seconds. 

## Heuristic
As mentioned before, the heuristic we implemented was found in [1](#1) and it's called Quarticity. To win at quarto we need to have four pieces in a row with at least one common feature, we will call this winning sequence of four pieces a 'quarto'. The key idea of this heuristic is that to have a quarto we need to have a 'terzo' first (a sequence of three pieces with a common feature). Basically we decide to assign a positive (or negative if the minmax is minimizing) score to each state that contains at least a 'terzo'. In particular for each 'terzo' we can find in the board's four rows, four columns and two diagonals we increase the score of 1. This means that when the heuristic is called each state is evaluated with a score that can go from 0 to 10 (in absolute value): it counts how many 'terzo' are present in the board.  
Strictly tied to this heuristic is the choice to assign a score of 11 to a state that corresponds to a draw and 12 to a state that corresponds to a win. Choosing to evaluate 11 a state that corresponds to a draw means that the MinMax algorithm will prefer to have a draw with respect to choosing a move that can lead to both a win or a loss. If we decided to evaluate a draw as 0 for example, the MinMax would choose a move that leads to a 'terzo' rather than a move that leads to a draw. This could be dangerous especially if the opponent is another intelligent strategy because it could lead to a loss. It's a "conservative" choice but it doesn't impact too much the performances against the random, especially with strategy 2 and 3 of the minmax. If we are playing against the random player and we are in a state that can lead to a draw it means that we are probably close to a state that can lead to a win or a loss a well. This could be the reason why playing against the random player assigning 0 or 11 to a draw does not impact the performance. It would be a different matter if the opponent was an intelligent strategy.

## Random first move and first placement on the diagonal
The minmax strategy is deterministic, meaning that if we could build a complete tree and make two optimal players play against each other we would always have the same game. This is also linked to the fact that if we have equivalent states the minmax will choose the move that leads to the first state in the tree. 
In the beginning of the game (due to the bounded search we are forced to have) we will always have states with score=0 because we can not visit a state that can lead to a win or a draw and we can't even visit a state that can lead to a 'terzo'. This means that at the beginning the MinMax will always make the same choices: for example if the minmax plays against the random and it's the first player, it will always choose the piece 0. If it's the second player the first placement will always be on the top left corner of the board. For this reason introducing a random move in the beginning of the game can lead the minmax to visit an area of the tree that would otherwise always be neglected.  
In addition, if we are the second player (meaning that we are the first player to place a piece on the board), we decide to place the piece randomly but with the constraint to always place it on one of the two diagonals. This choice allows us to have two more chances to make a 'terzo' and then a 'quarto' faster. Introducing the random move and the placement of the diagonals makes our player stronger and more efficient (as in it wins with less turns).

## Results
In the following section we show a summary of the results obtained making the minmax play against the random and against itself in different configurations with a maximum number of combinations equal to math.factorial(5)**2.  
NB: 'wins' refers to the number of games won by the first strategy displayed: S1 vs S2: wins=90% means that S1 has won 90% of the matches).  
Tests against the random player:
- STRATEGY 1(with randomness) vs RANDOM: wins=90.9%, draws=2.8%, average turns=9.8 on 1000 matches
- STRATEGY 2(with randomness) vs RANDOM: wins=95.7%, draws=1.4%, average turns=9.9 on 1000 matches
- STRATEGY 3(with randomness) vs RANDOM: wins=99.8%, draws=0.0%, average turns=8.7 on 1000 matches

MinMax VS MinMax strategies:
- STRATEGY 1(with randomness) vs STRATEGY 2(with randomness): wins=54.0%, draws=1.0%, average turns=10.6 on 100 matches
- STRATEGY 3(with randomness) vs STRATEGY 1(with randomness): wins=95.0%, draws=0.0%, average turns=10.5 on 100 matches
- STRATEGY 3(with randomness) vs STRATEGY 2(with randomness): wins=95.0%, draws=1.0%, average turns=10.12 on 100 matches
- STRATEGY 3(with randomness) vs STRATEGY 2(with randomness and math.factorial(6)**2) : wins=85.0%, draws=8.0%, average turns=10.11 on 100 matches

After these trials we consider the STRATEGY 3 minmax to be our champion. The final test consists in challenging our champion against itself without the randomness discussed above:
- STRATEGY 3(without randomness) vs STRATEGY 3(with randomness): wins=18%, draws=34%, average turns=14.29 on 100 matches.  

It's interesting to see that the minmax without randomness wins 18% of the games and loses 48%, randomness makes the player win twice the matches. In addition we can also see that the average turns is greater than 14 since we are using two players that are very close to optimal.

# Reinforcement Learning Player
In this section we are going to explain the features and implementation choices that concern the Reinforcement Learning Player along with details that help understanding the code.  

## Reinforcement Learning parameters
This RL uses several parameters that can be used fixed or tuned.
- Learning rate (alfa): different from the learning rate of a neural network, it's a coefficient used to update the rewards. This parameter has been tuned and the best value was found to be 1.
- Discount factor (gamma): similarly to alfa is a coefficient used to update the rewards. It's used to weight the MaxQ value (explained later). This parameter has been tuned and the best value was found to be 1.
- Random factor: it's the probability with which the algorithm will choose a random move over the move with the highest reward. This parameter is initialized to 1, meaning that at the beginning of the training the moves will be chosen randomly (exploration). While training, this parameter is slowly decreased multiplying it by the rf_decay which is set to 0.95. The random factor will keep decreasing during training until it reaches a minimum of 0.1 and at that point it stops decreasing (exploitation).  
- Reward Coefficient: it's a parameter that defines how big is the reward for a winning state-action and symmetrically how small it is the reward for a state-action that leads to a loss. This parameter has been tuned and the best value was found to be 100.

## Q-table
This player uses a state-action q-table in order to learn how to play the quarto game.
To understand well the structure of the q-table let's define the state and the action:
- state: it's a touple in which the first element is the current game board and the second element is the piece chosen by the opponent for us to place; so `state = (board, piece to place)`
- action: it's a tuple in which the first element is the piece to choose for the opponent to place and the second element is a tuple that contains the "coordinates" in which we place the piece assigned to us; so `action = (piece to choose, (x,y))`

The q-table is a nested dictionary in which the key (of the external dictionary) is the state. For each state a dictionary is defined in which we have the possible actions (or moves) that can be made from that state associated with a reward.  
So the final structure can be summarized as follows:  
`q_table = {state : {action : reward}}`  
or in a more exhaustive way:  
`q_table = {(board, piece to place) : {(piece to choose, (x,y)) : reward}}`  

## states-history
The variable states-history shown in the code saves the sequence of states that represent the game along with all the moves made by the RLPlayer. The state and action used in this variable are the same explained in the section above about the q-table.  
The structure is the follwing:  
`states-history = [ ({state : action}, reward) ]`  

States-history is used to update the rewards of the states-actions that unroll during one game in order to update the rewards in the q-table and make the agent learn. At every game the states-history is updated in two cases:
- every time it's the RL's turn: the current state (before making the move) and the action chosen by the RL are saved in the states-history. If the move leads to winning the game, a positive reward is assigned to the state-action that led to this situation, otherwise the reward is set to 0.
- when it's the opponent's turn and it wins: the RL's previous choice and state that led to the opponent winning are saved in the states-history with a negative reward.

## Q-table update
Once the game is finished the function `update_q_table` is called. This function goes through the states-history in reversed order (so starting from the last state-action going backwards to the first) and updates the rewards of the states-actions of the game.  
The function uses the Bellman equation:
$$
NewQ(s,a)=Q(s,a)+\alpha[R(s,a)+\gamma maxQ'(s',a')-Q(s,a)]
$$
where:
- $NewQ(s,a)$ is the updated value of the reward
- $Q(s,a)$ is the current reward for this state-action found in the q-table (before the update)
- $\alpha$ is the learning rate
- $R(s,a)$ is the reward assigned to this state-action in the states-history (associated to the current game)
- $\gamma$ is the discount factor
- $maxQ'(s',a')$ is the maximum reward associated with this state that can be found in the q-table

## Results
This player unfortunately does not perform as well as anticipated and only manages to be slightly better than a random player. With the best configuration found using the tuning the RL player can win around 60% of the matches against a random strategy.  
Another drawback of this technique  is that the training requires around ten minutes and even loading an already full q-table can take up to a couple of minutes. Once the table is loaded the RL Player is as fast as the Random Player.


## References
[1] A FLEXIBLE EXTENDED QUARTO! IMPLEMENTATION BASED ON COMBINATORIAL ANALYSIS, Daniel Castro Silva and Vasco Vinhas, IADIS International Conference Gaming 2008
