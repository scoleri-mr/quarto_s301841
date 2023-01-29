from copy import deepcopy
import math
from .objects import Quarto, Player
import random
import pickle
from matplotlib import pyplot as plt
import logging
from functools import reduce
from operator import and_
from tqdm import tqdm

class RLPlayer(Player):
    '''Reinforcement Learning Player'''
    def __init__(self, quarto: Quarto, alfa = 1, gamma = 1, reward_coefficient = 100) -> None:
        super().__init__(quarto)

        #if the payer is in train mode it will update its Q-table
        #The Q-table is a state-action table: { (board, piece to place) : {(piece to choose, (x,y)) : reward} }
        #which means the types are q_table = dict{} tuple(4x4 np.array,int) : dict{ tuple(int, tuple(x,y)) : float} }
        self.q_table = {}

        #set the reinforcement learning parameters
        self.rf = 1             #random factor
        self.min_rf = 0.1
        self.rf_decay = 0.95
        self.alfa = alfa        #learning rate
        self.gamma = gamma      #discount factor
        self.reward_coefficient = reward_coefficient

        #pieces available to be selected for the opponent and available places to put the piece given by the opponent
        #valid choice + valid place = action
        self.valid_choices = list(i for i in range(self.get_game().BOARD_SIDE ** 2) if i not in self.get_game().get_board_status() and i != self.get_game().get_selected_piece())
        self.valid_places = list((j % self.get_game().BOARD_SIDE, j // self.get_game().BOARD_SIDE) for j in range(self.get_game().BOARD_SIDE ** 2) if self.get_game().get_board_status()[j // self.get_game().BOARD_SIDE][j % self.get_game().BOARD_SIDE] < 0 )

        #helper variables
        self.quarto_temp = Quarto()
        self.prev_state = None
        self.states_history = list() #list of tuples with ({state : action}, reward)
        self.train_mode = False

    def update_valid_choices(self, game: Quarto):
        self.valid_choices = list(i for i in range(game.BOARD_SIDE ** 2) if i not in game.get_board_status() and i != game.get_selected_piece())

    def update_valid_places(self, game:Quarto):
        self.valid_places = list((j % game.BOARD_SIDE, j // game.BOARD_SIDE) for j in range(game.BOARD_SIDE ** 2) if game.get_board_status()[j // game.BOARD_SIDE][j % game.BOARD_SIDE] < 0 )

    def set_q_table(self, q_table):
        '''set the q-table. Use it to test a trained player'''
        self.set_play_mode()
        self.q_table = q_table
    
    def reset_states_history(self):
        self.states_history = list()

    def set_train_mode(self):
        '''put the player in training mode'''
        self.train_mode=True

    def set_play_mode(self):
        '''put the player in playing mode, q-table is not updated'''
        self.train_mode=False

    def save_table(self, filename = 'q_table'):
        data = open(filename, 'wb')
        pickle.dump(self.q_table, data)
        data.close

    def load_q_table(self, filename: str = 'q_table'):
        data = open(filename, 'rb')
        self.q_table = pickle.load(data)
        data.close()
    
    def get_best_move(self) -> tuple:
        '''finds the best move from the q_table'''
        board = tuple(map(tuple, self.get_game().get_board_status()))
        piece_to_place = self.get_game().get_selected_piece()
        state = (board, piece_to_place)
        #create a mini Q-table that only contais state-action for the current state
        mini_qt = self.q_table[state]
        best_move = max(mini_qt, key=mini_qt.get)
        return best_move

    def choose_piece(self) -> int:
        if not self.train_mode:
            if len(self.valid_choices) == 16: #I'm on the first turn, return a random choice
                return random.choice(self.valid_choices)
            move = self.get_best_move()
            return move[0]
        else: return self.choose_action[0]

    def place_piece(self) -> tuple[int,int]:
        if not self.train_mode:
            if len(self.valid_places) == 16: #I'm on the first turn, return a random choice
                return random.choice(self.valid_places)
            move = self.get_best_move()
            return move[1]
        else: return self.choose_action[1]
        
    def choose_action(self):
        self.prev_state = (tuple(map(tuple,self.get_game().get_board_status())), self.get_game().get_selected_piece() )
        self.quarto_temp = deepcopy(self.get_game())
        maxG = -10e15
        if random.random() < self.rf:
            #choose random action; the choices for piece and place are independent
            chosen_piece = random.choice(self.valid_choices)
            place_piece = random.choice(self.valid_places)
            action = (chosen_piece, place_piece)
            self.quarto_temp.place(place_piece[0], place_piece[1])
            self.quarto_temp.select(chosen_piece)

            #the board is a numpy array which is not hashable. Change it to tuple using map
            state_temp = (tuple(map(tuple, self.quarto_temp.get_board_status())) , self.quarto_temp.get_selected_piece() )

            if state_temp not in self.q_table or action not in self.q_table[state_temp]:
                self.q_table.update({state_temp:{action:0.01}})
            else: None
            return action
        else: 
            #if exploiting, chose the action with the highest reward 
            for valid_choice in self.valid_choices:
                for valid_place in self.valid_places:
                    self.quarto_temp = deepcopy(self.get_game())
                    action = (valid_choice, valid_place)
                    self.quarto_temp.place(valid_place[0], valid_place[1])
                    self.quarto_temp.select(valid_choice)
                    state_temp = ( tuple(map(tuple,self.quarto_temp.get_board_status())), self.quarto_temp.get_selected_piece() )
                    if state_temp not in self.q_table or action not in self.q_table[state_temp]:
                        self.q_table.update({state_temp:{action:0.01}})
                    if self.q_table[state_temp][action] >= maxG:
                        best_action = action
                        maxG = self.q_table[state_temp][action]
            return best_action

    def reinforcement_learning(self, opponent: Player, epochs: int = 1000, plot_wins: bool = True):
        self.set_train_mode()
        epochs_num = list()
        wins_history = list()
        wins = 0

        for i in tqdm(range(epochs), ncols=70):
            self.reset_states_history()
            self.get_game().reset()
            self.update_valid_choices(self.get_game())
            self.update_valid_places(self.get_game())

            if i%2 == 0:
                #I start
                current_player = 0
            else: 
                #the opponent starts  
                current_player = 1       

            #save my previous move; if it leads to a loss assign negative reward
            prev_choice = None
            prev_place = None

            winner = -1
            turn = 0

            while winner == -1 and not self.get_game().check_finished():
                if current_player == 1: #it's my opponent's turn
                    if turn == 0: #if the game is starting now (turn = 0) the opponent only has to choose a move for me (I'm rl)
                        chosen_piece = opponent.choose_piece()
                        self.get_game().select(chosen_piece)
                        self.update_valid_choices(self.get_game())
                        current_player = 1 - current_player
                        turn+=1
                        continue
                    turn+=1
                    chosen_place = opponent.place_piece()

                    #apply my opponent's moves on a temporary game
                    tmp = deepcopy(self.get_game())
                    tmp.place(chosen_place[0], chosen_place[1])

                    winner = tmp.check_winner()
                    if winner != -1:  #if we have a winner here it means the opponent has won
                        #I need to assign a negative reward to the state-action that led here
                        if prev_choice is not None and prev_place is not None:
                            state = (tuple(map(tuple,self.get_game().get_board_status())), self.get_game().get_selected_piece())
                            action = (prev_choice, prev_place)
                            reward = -self.reward_coefficient #negative reward
                            self.states_history.append(({state:action},reward))
                            logging.debug(f'game number {i}: states_history, opponent has won after {turn} turns: ')
                            logging.debug(f'final board: \n{self.get_game().get_board_status()}')

                    #if the opponent has not won it has to:
                    # 1. place the piece I assigned and update valid_places
                    self.get_game().place(chosen_place[0], chosen_place[1])
                    self.update_valid_places(self.get_game())

                    # 2. choose a piece for me to place and update valid_choices
                    try: #can go wrong if we don't have any more pieces to choose from (we have a draw)
                        chosen_piece = opponent.choose_piece()                    
                    except:
                        logging.debug('Error: no pieces left to choose from => draw. Ignore')
                        break

                    self.get_game().select(chosen_piece)
                    self.update_valid_choices(self.get_game())
                    
                else:   #My turn
                    #if it's the first turn and I have to play I can choose a random piece
                    #I choose not to include this first choice in the states_history: I don't want to give an evaluation to the first choice
                    if turn==0:
                        turn+=1
                        chosen_piece = random.choice(self.valid_choices)
                        self.get_game().select(chosen_piece)
                        self.update_valid_choices(self.get_game())
                        current_player = 1 - current_player   
                        continue
                    turn+=1
                    #I have to:
                    # 1. Choose the action to perform
                    try: #can go wrong if we don't have any more pieces to choose from (we have a draw)
                        action = self.choose_action()
                    except:
                        logging.debug('Error: no pieces left to choose from => draw. Ignore')
                        break

                    # 2. Place the piece the opponent has assigned to me and update valid_places
                    self.get_game().place(action[1][0], action[1][1])
                    self.update_valid_places(self.get_game())
                    
                    # 3. give a reward = 0 to this state-action if we don't win, 1 otherwise
                    winner = self.get_game().check_winner()
                    if winner != -1: #If we are here I've won
                        wins+=1
                        reward = self.reward_coefficient #positive reward
                        logging.debug(f'game number {i}: states_history, I have won after {turn} turns: ')
                        logging.debug(f'final board: \n {self.get_game().get_board_status()}')
                    else:
                        reward = 0
                    
                    prev_choice = action[0]
                    prev_place = action[1]  

                    # 4. Select the piece for my opponent
                    if winner == -1: 
                        self.get_game().select(action[0])
                        self.update_valid_choices(self.get_game())

                    state = (tuple(map(tuple,self.get_game().get_board_status())), self.get_game().get_selected_piece())
                    self.states_history.append(({state:action},reward))

                current_player = 1 - current_player
            
            self.update_q_table()
            self.reset_states_history()
            self.get_game().reset()
            self.update_valid_choices(self.get_game())
            self.update_valid_places(self.get_game())

            if (i+1) % 20 == 0:
                wins_history.append(100*wins/(i+1))
                epochs_num.append(i)

        print('TRAINING COMPLETE')
        table_name = 'q_table_'+'epochs'+str(epochs)+'_alfa'+str(self.alfa)+'_gamma'+str(self.gamma)+'_rc'+str(self.reward_coefficient)
        self.save_table(table_name)

        if plot_wins: 
            plt.semilogy(epochs_num, wins_history, "b")
            title = 'epochs='+str(epochs)+', alfa='+str(self.alfa)+', gamma='+str(self.gamma)+', rc='+str(self.reward_coefficient)
            plt.title(title)
            plt.xlabel('epochs')
            plt.ylabel('won games %')
            filename = 'plot_'+'epochs'+str(epochs)+'_alfa'+str(self.alfa)+'_gamma'+str(self.gamma)+'_rc'+str(self.reward_coefficient)+'.jpg'
            plt.savefig(filename)
            plt.show()
        return self.q_table

    def update_q_table(self):
        #target = 0
        for el in reversed(self.states_history):
            state = list(el[0].keys())[0] #it's just one entry in the dictionary because one state does not repeat itself in the same game
            action = list(el[0].values())[0] 
            reward = el[1]
            #if the state-action is already present in the q_table I can just update the reward. 
            #If it's not there I need to add it. I can use the reward in states_history           
            if self.q_table.get(state):
                if self.q_table[state].get(action):
                    maxQ = max(self.q_table[state].values())
                    self.q_table[state][action] += self.alfa*(reward + self.gamma*maxQ - self.q_table[state][action])
            else:
                self.q_table.update({state:{action:reward}})
            #target += el[1]
        self.rf = max(self.min_rf, self.rf * self.rf_decay)

class RandomPlayer(Player):
    """Random player"""

    def __init__(self, quarto: Quarto) -> None:
        super().__init__(quarto)

    def choose_piece(self) -> int:
        return random.choice(list(i for i in range(self.get_game().BOARD_SIDE ** 2) if i not in self.get_game().get_board_status() and i != self.get_game().get_selected_piece()))
        # return random.randint(0, 15)

    def place_piece(self) -> tuple[int, int]:
        return random.choice(list((j % self.get_game().BOARD_SIDE, j // self.get_game().BOARD_SIDE) for j in range(self.get_game().BOARD_SIDE ** 2) if self.get_game().get_board_status()[j // self.get_game().BOARD_SIDE][j % self.get_game().BOARD_SIDE] < 0))
        # return random.randint(0, 3), random.randint(0, 3)
        # (j % self.get_game().BOARD_SIDE, j // self.get_game().BOARD_SIDE)   --->  it's a way to run through the matrix by columns

class MinMaxPlayer(Player):
    """MinMax player"""

    #possible strategies
    FIXED_DEPTH = 1
    FIXED_COMPLEXITY = 2
    VARIABLE_COMPLEXITY = 3

    WIN_SCORE = 12
    DRAW_SCORE = 11

    def __init__(self, quarto: Quarto, strategy: int = 3, bound_value: int = math.factorial(5) ** 2, randomness: bool = True) -> None:
        super().__init__(quarto)
        self.best_move = None
        self.max_depth = None
        self.max_combs = None
        self.randomness = randomness
        self.__set_bound(strategy, bound_value)

    def choose_piece(self) -> int:
        if self.randomness:
            av_pieces = list(i for i in range(self.get_game().BOARD_SIDE ** 2) if i not in self.get_game().get_board_status() and i != self.get_game().get_selected_piece())
            if len(av_pieces) == 16:
                logging.debug('MinMax plays first turn: choose a random piece')
                return random.choice(av_pieces)
        if not self.best_move:
            self.__tweak_depth(self.get_game())
            self.best_move = self.__minmax(self.get_game(), (-math.inf, ((-math.inf, -math.inf), -math.inf)), (math.inf, ((math.inf, math.inf), math.inf)), 0)

        m = self.best_move[1][1]
        self.best_move = None
        return m

    def place_piece(self) -> tuple[int, int]:
        if self.randomness:
            av_places = list((j % self.get_game().BOARD_SIDE, j // self.get_game().BOARD_SIDE) for j in range(self.get_game().BOARD_SIDE ** 2) if self.get_game().get_board_status()[j // self.get_game().BOARD_SIDE][j % self.get_game().BOARD_SIDE] < 0)
            diag = [el for el in av_places if el[0]==el[1]] + [el for el in av_places if el[0]+el[1]==3]
            #               diagonal                        +                      antidiagonal
            if len(av_places) == 16:
                logging.debug('MinMax plays first turn: choose a random place')
                return random.choice(diag)
            #the first placement can be random but if I put the piece in the diagonal or antidiagonal the game could go a bit faster
            #I have 10 chances to make a quarto (4 columns, 4 rows, 2 diagonals), if I start with the diagonal I start having 2 more 
            #choices than if I don't

        if not self.best_move:
            self.__tweak_depth(self.get_game())     
            self.best_move = self.__minmax(self.get_game(), (-math.inf, ((-math.inf, -math.inf), -math.inf)), (math.inf, ((math.inf, math.inf), math.inf)), 0)

        return self.best_move[1][0]

    def __set_bound(self, strategy: int, bound_value: int):
        if strategy == self.FIXED_DEPTH:
            self.max_depth = bound_value
        elif strategy == self.FIXED_COMPLEXITY or strategy == self.VARIABLE_COMPLEXITY: 
            self.max_combs = bound_value
        else:
            raise ValueError(f'Invalid strategy: {strategy}. Choose between FIXED DEPTH = 1, FIXED COMPLEXITY = 2 and VARIABLE_ COMPLEXITY = 3')
        self.strategy = strategy

    def __tweak_depth(self, game: Quarto) -> None:
        if self.strategy == self.FIXED_COMPLEXITY: 
            self.max_depth = math.inf if self.estimate_tree_complexity(game) <= self.max_combs else 1
            #if the number of total combinations from the current state of the game is less than the max number of combinations given
            #I visit the complete tree (math.inf) else I use the minmax with a bound of 1
        elif self.strategy == self.VARIABLE_COMPLEXITY: 
            self.max_depth = self.find_depth((game.get_board_status() == -1).sum(), self.max_combs)        
            #game.get_board_status()==-1 gives a np matrix (like board) containing true if in the board we have -1, false otherwise
            #the first sum gives a np vector long as the row of the board with the sum of true for each column
            #the second sum gives us how many pieces are left to be placed/how many places left to place a piece

    def __heuristic(self, state: Quarto):
        score = 0
        board = state.get_board_status()

        for row in board:
            useful_pieces = row != -1   #mask with true if there is a piece, false if not
            if sum(useful_pieces) == 3:
                if reduce(and_, row[useful_pieces]) != 0 or reduce(and_, row[useful_pieces]^15) != 0:
                    #reduce allows us to perform the consecutive bitwise and between each piece in row[useful_pieces]
                    #first I check if the pieces have a 1 (a feature) in common, if they don't I check if they have a 0 in common (still a feature)
                    #to check the 0s I have to perform the bitwise not. I use the xor operator (^) with 15 (in binary it's 1111) to obtain the bitwise not
                    score += 1

        for col in board.T:
            useful_pieces = col != -1
            if sum(useful_pieces) == 3:
                if reduce(and_, col[useful_pieces]) != 0 or reduce(and_, col[useful_pieces]^15) != 0:
                    score += 1        

        for diag in [board.diagonal(), board[::-1].diagonal()]:
            useful_pieces = diag != -1
            if sum(useful_pieces) == 3:
                if reduce(and_, diag[useful_pieces]) != 0 or reduce(and_, diag[useful_pieces]^15) != 0:
                    score += 1       
        
        return -score if state.get_current_player() == self.get_game().get_current_player() else score

    def __minmax(self, state: Quarto, alpha, beta, depth=0):
        if depth >= self.max_depth:
            return self.__heuristic(state), None    
        
        available_pieces = list(i for i in range(state.BOARD_SIDE ** 2) if i not in state.get_board_status() and i != state.get_selected_piece())
        available_positions = list((j % state.BOARD_SIDE, j // state.BOARD_SIDE) for j in range(state.BOARD_SIDE ** 2) if state.get_board_status()[j // state.BOARD_SIDE][j % state.BOARD_SIDE] < 0)

        minimizing = state.get_current_player() == self.get_game().get_current_player() #minimize at my turn, maximize at the opponent's turn

        value = (math.inf, ((math.inf, math.inf), math.inf)) if minimizing else (-math.inf, ((-math.inf, -math.inf), -math.inf))
        for x, y in available_positions:
            tmp = deepcopy(state)

            if tmp.get_selected_piece() not in tmp.get_board_status():
                assert tmp.place(x, y), 'Error placing'
            
                winner = tmp.check_winner()  
                if tmp.check_finished() or winner != -1: #if the game is finished
                    if winner == -1:    #if we have a draw
                        score = -MinMaxPlayer.DRAW_SCORE if minimizing else MinMaxPlayer.DRAW_SCORE
                    else:               #if we have a winner
                        winner = 1 - winner if self.get_game().get_current_player() == 1 else winner
                        score = -MinMaxPlayer.WIN_SCORE if winner == 0 else MinMaxPlayer.WIN_SCORE
                    value = min(value, (score, ((x, y), -1))) if minimizing else max(value, (score, ((x, y), -1)))

                    if minimizing:
                        beta = min(beta, value)
                    else:
                        alpha = max(alpha, value)

                    if beta <= alpha:
                        break
                    continue   

            for p in available_pieces:
                new_state = deepcopy(tmp)
                assert new_state.select(p), 'Error selecting'
                new_state._current_player = (state._current_player + 1) % state.MAX_PLAYERS #change the player
                val, _ = self.__minmax(new_state, alpha, beta, depth + 1)
                #logging.debug(f'move and score: {(val, ((y, x), p))}')
                value = min(value, (val, ((x, y), p))) if minimizing else max(value, (val, ((x, y), p)))

                if minimizing:
                    beta = min(beta, value)
                else:
                    alpha = max(alpha, value)

                if beta <= alpha:
                    break   
        
        return value
    
    @staticmethod
    def estimate_tree_complexity(game: Quarto):
        logging.debug(f'Tree complexity: {math.factorial((game.get_board_status() == -1).sum()) ** 2}')
        return math.factorial((game.get_board_status() == -1).sum()) ** 2

    @staticmethod
    def find_depth(remaining_turns, max_complexity):
        product = 1
        for i, v in enumerate(range(remaining_turns, 0, -1)):
            product *= v**2
            if product > max_complexity:
                logging.debug(f'Depth found: {max(i,1)}')
                return max(i,1)
        return remaining_turns