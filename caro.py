import numpy as np
import json

from tqdm import tqdm

from time import sleep

class Board:
    def __init__(self):
        self.board = '.........'
        self.turn = 'X'
        self.winner = None
        self.done = False
        
    def get_legal_moves(self):
        return [i for i in range(9) if self.board[i] == '.']
    
    def get_state(self):
        return self.board
    
    def update_win(self):
        lines = [[0,1,2], [3,4,5], [6,7,8], 
                 [0,3,6], [1,4,7], [2,5,8], 
                 [0,4,8], [2,4,6]]
        
        for line in lines:
            line_state = self.board[line[0]] + self.board[line[1]] + self.board[line[2]]
            if line_state == 'XXX':
                self.winner = 'X'
                self.done = True
                return True
            elif line_state == 'OOO':
                self.winner = 'O'
                self.done = True
                return True
        return False
    
    def update_draw(self):
        if '.' not in self.board:
            self.done = True
            self.winner = 'draw'
            return True
        return False
            
    # def get_reward(self):
    #     if self.winner == 'X':
    #         return 1
    #     elif self.winner == 'O':
    #         return -1
    #     else:
    #         return 0
    
    def move(self, next_state):
        self.board = next_state
        self.update_win() or self.update_draw()
        self.turn = 'X' if self.turn == 'O' else 'O'
        return self.board, self.done
    
    
    
class Player:
    def __init__(self, turn):
        self.turn = turn
        self.iq = {}
        
    
    def value_of_state(self, state, action):
        if (state,action) not in self.iq:
            self.iq[(state,action)] = 0
            
        return self.iq[(state,action)]
    
    def learn_n_times(self, n):
        espilon = 0.9
        time_to_reduce = n // 2
        redu = espilon / time_to_reduce
        for i in tqdm(range(n)):
            self.learn(espilon=espilon)
            if i > time_to_reduce:
                espilon -= redu
    
    def learn(self,espilon = 0.1, alpha=0.1, gamma=0.9):
        board = Board()
        done = False
        current_state = board.get_state()
        while not board.done:
            _, selected_action = self.select_move(board, espilon=espilon)
            next_state = current_state[:selected_action] + board.turn + current_state[selected_action+1:]
            next_state, done = board.move(next_state)
            reward = self.reward(board)
            next_state_value = 0
            if not done:
                best_action, _ = self.select_move(board, espilon=0)
                next_state_value = self.value_of_state(next_state, best_action)
            self.iq[(current_state, selected_action)] += alpha * (reward + gamma * next_state_value - self.value_of_state(current_state, selected_action))
            current_state = next_state
            
        # print(max(self.iq.values()))
        
        
        
    def select_move(self, board, espilon=0.1):
        legal_moves = board.get_legal_moves()
        value = [self.value_of_state(board.get_state(), action) for action in legal_moves]
        if board.turn == self.turn:
            best_value = np.argmax(value)
        else:
            best_value = np.argmin(value)
        best_action = legal_moves[best_value]
        selected_action = best_action
        # print(value)
        if np.random.random() < espilon:
            selected_action = np.random.choice(legal_moves)
        return best_action, selected_action
    
    def reward(self, board):
        if not board.done:
            return 0
        if board.winner == self.turn:
            return 1
        elif board.winner == 'draw':
            return 0
        else:
            return -1
        
    def save(self):
        with open('iq.json', 'w') as f:
            json.dump(self.iq, f)
            
    def load(self):
        with open('iq.json', 'r') as f:
            self.iq = json.load(f)

alientX = Player('X')
alientX.learn_n_times(50000)
alientO = Player('O')
alientO.learn_n_times(50000)

def play_game(playerX, playerO=None, verbose=False):
    board = Board()
    while not board.done:
        if verbose:
            print(" \nTurn {}\n".format(board.turn))
            print(board.board)
            if board.turn == playerX.turn:
                best_action, selected_action = playerX.select_move(board, espilon=0)
            else:
                best_action = np.random.choice(board.get_legal_moves())     
        else:
            best_action, selected_action = playerX.select_move(board, espilon=0)
        next_state = board.board[:best_action] + board.turn + board.board[best_action+1:]
        board.move(next_state)
    if verbose:
        print(" \nTurn {}\n".format(board.turn))
        print(board.board)
    if board.winner == 'X':
        print('X wins')
    elif board.winner == 'O':
        print('O wins')
    else:
        print('draw')

        
play_game(alientX, alientO, verbose=True)

            