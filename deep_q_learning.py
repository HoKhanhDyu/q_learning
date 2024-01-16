import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np

from tqdm import tqdm

import tensorflow as tf
# print(tf.config.list_physical_devices('GPU'))
from collections import deque


class board:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.board = [[0 for i in range(w)] for j in range(h)]
        self.turn = 1
        self.winner = None
        self.done = False

    def get_legal_moves(self):
        return [self.w*i+j for i in range(self.h) for j in range(self.w) if self.board[i][j] == 0]
    
    def get_state(self):
        return self.board
    
    def move(self, action):
        i = action//self.w
        j = action%self.w
        self.board[i][j] = self.turn
        self.turn = 1 if self.turn == 2 else 2
    
    def is_win(self, action):
        i = action//self.w
        j = action%self.w
        lines = [[(-1,1),(1,-1)],[(0,1),(0,-1)],[(1,1),(-1,-1)],[(1,0),(-1,0)]]
        for line in lines:
            count = 1
            for step in line:
                x = i
                y = j
                while 0<=x+step[0]<self.h and 0<=y+step[1]<self.w and self.board[x+step[0]][y+step[1]] == self.board[i][j]:
                    count += 1
                    x += step[0]
                    y += step[1]
                    
            
            if count >= 5:
                # print(line)
                return True, self.board[i][j]
        
        if len(self.get_legal_moves()) <= 0:
            return True, 0
        
        return False, 0
    
    def make_action(self, action):
        self.move(action)
        self.done, self.winner = self.is_win(action)
        
        return self.get_state(), self.done, self.winner
    
    def reset(self):
        self.board = [[0 for i in range(self.w)] for j in range(self.h)]
        self.turn = 1
        self.winner = None
        self.done = False
        
    def render(self):
        for i in range(self.h):
            for j in range(self.w):
                if self.board[i][j] == 0:
                    print('-', end=' ')
                elif self.board[i][j] == 1:
                    print('X', end=' ')
                else:
                    print('O', end=' ')
            print()
    

class AlienX:
    def __init__(self, w, h, turn=1):
        self.actions = w*h
        self.state_size = w*h
        self.turn = turn
        
        self.batch_size = 32
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.001
        
        self.memory = deque(maxlen=500000)
        
        self.train_model = self.model()
        self.target_model = self.model()
        
        self.target_model.set_weights(self.train_model.get_weights())
        
    def model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam())
        return model
        
    def decision(self, board, epsilon):
        if np.random.rand() <= epsilon:
            return np.random.choice(board.get_legal_moves())
        else:
            legal_moves = board.get_legal_moves()
            if self.turn == board.turn:
                q_table = self.train_model.predict(np.array([board.get_state()]).reshape(1, self.state_size), verbose=0)[0]
                q_table = [q_table[i] for i in legal_moves]
                return legal_moves[np.argmax(q_table)]
            else:
                q_table = self.target_model.predict(np.array([board.get_state()]).reshape(1, self.state_size), verbose=0)[0]
                q_table = [q_table[i] for i in legal_moves]
                return legal_moves[np.argmin(q_table)]
            
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def get_batch(self):
        batch = random.sample(self.memory, self.batch_size)
        state = np.array([i[0] for i in batch]).reshape(self.batch_size, self.state_size)
        action = np.array([i[1] for i in batch])
        reward = np.array([i[2] for i in batch])
        next_state = np.array([i[3] for i in batch]).reshape(self.batch_size, self.state_size)
        done = np.array([i[4] for i in batch])
        
        return state, action, reward, next_state, done
    
    def train_alien(self):
        if len(self.memory) < self.batch_size:
            return
        
        state, action, reward, next_state, done = self.get_batch()
        
        target = self.train_model.predict(state, verbose=0)
        target_val = self.target_model.predict(next_state, verbose=0)
        
        for i in range(self.batch_size):
            target[i][action[i]] = reward[i] if done[i] else reward[i] + self.gamma * (np.amax(target_val[i]))
            
        self.train_model.fit(state, target, batch_size=self.batch_size, verbose=0, epochs=10)
        loss = self.train_model.evaluate(state, target, verbose=0)
        return loss
    
    def update_target_model(self):
        self.target_model.set_weights(self.train_model.get_weights())
        
    def save_model(self, fn):
        self.train_model.save(fn)
        
    def load_model(self, fn):
        self.train_model.load_weights(fn)
        self.target_model.load_weights(fn)
        
    def test_alient(self, n=100):
        env = board(10, 10)
        self.load_model('alienx.h5')
        win = 0
        for i in tqdm(range(n)):
            env.reset()
            done = False
            while not done:
                if env.turn == self.turn:
                    action = self.decision(env, 0)
                else:
                    action = np.random.choice(env.get_legal_moves())
                # print(action//env.w, action%env.w)
                next_state, done, winner = env.make_action(action)
            # env.render()
            # print('--------------------------------')
            if winner == self.turn:
                win += 1
        print('win: {}/{}'.format(win, n))
        
def train_model_alienX():
    env = board(10, 10)
    agent = AlienX(10, 10)
    episodes = 5000
    loss_list = []
    history = deque(maxlen=100)
    for e in range(episodes):
        env.reset()
        state = env.get_state()
        done = False
        while not done:
            action = agent.decision(env, agent.epsilon)
            next_state, done, winner = env.make_action(action)
            # if done and winner == agent.turn:
            #     env.render()
            #     print(action)
            #     print(f'win at action {action//env.w} {action%env.w}')
            reward = 1 if winner == agent.turn else -1 if winner != 0 else 0
            agent.remember(state, action, reward, next_state, done)
            state = next_state
        loss = agent.train_alien()
        loss_list.append(loss)
        history.append(winner)
            
        if e % 25 == 0:
            agent.update_target_model()
            
        if e % 100 == 0 and e > 0:
            print(f'win rate: {history.count(agent.turn)}/{len(history)}, draw rate: {history.count(0)}/{len(history)}')
            
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            
        if e % 1000 == 0:
            print('save model')
            agent.save_model('alienx.keras')
            
        print('episode: {}/{}, loss: {}, e: {:.2}'.format(e, episodes, loss, agent.epsilon))

    #plot loss
    import matplotlib.pyplot as plt
    plt.plot(loss_list)
    plt.show()


# print(f'Num GPUs Available: {len(tf.config.experimental.list_physical_devices("GPU"))}')

train_model_alienX()