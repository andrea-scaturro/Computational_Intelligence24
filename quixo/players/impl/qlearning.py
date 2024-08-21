import numpy as np
from tqdm import tqdm
from game import Game, Move
import random
import sys
import struct

class QLearning:
    def __init__(self, alpha, gamma, epsilon, player):
        self.learning_rate = alpha
        self.discount_factor = gamma
        self.exploration_rate = epsilon
        self.q_table = {}
        self.player = player

    def set_exploration_rate(self, epsilon):
        self.exploration_rate = epsilon

    def get_exploration_rate(self):
        return self.exploration_rate

    def get_q_table(self):
        return self.q_table

    def get_q_value(self, state, action):
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0
        return self.q_table[(state, action)]

    def choose_action(self, state, actions):
        if np.random.uniform() < self.exploration_rate:
            return actions[np.random.choice(range(len(actions)))]
        else:
            state = self.compact_string(state)
            actions = [self.compact_move(action) for action in actions]
            q_values = np.array([self.get_q_value(state, action) for action in actions])
            maximum = np.max(q_values)
            return self.decode_move(actions[np.random.choice(np.where(q_values == maximum)[0])])

    def calculate_reward(self, win):
        if win == self.player:
            return 1
        elif win == 1 - self.player:
            return -1
        else:
            return 0
        
    def compact_string(self, matrix):
        matrix = matrix.flatten()
        compressed_data = struct.pack(f">{len(matrix)}b", *matrix)
        return compressed_data

    def compact_move(self, t):
        string = f"{t[0][0]}{t[0][1]}{t[1].value}"
        string = string.encode('utf-8')
        return string

    def decode_move(self, t):
        t = t.decode('utf-8')
        return (int(t[0]), int(t[1])), Move(int(t[2]))

    def update_q_values(self, trajectory, reward):
        for state, action in trajectory:
            state = self.compact_string(state)
            action = self.compact_move(action)
            self.q_table[(state, action)] = self.get_q_value(state, action) + \
                self.learning_rate * (reward - self.get_q_value(state, action))
            reward = reward * self.discount_factor

if __name__ == '__main__':
    # Training Player 0
    q_agent = QLearning(0.5, 0.95, 1, 0)  # Inizializza l'agente come Player 0
    num_games = 5000
    gamma_epsilon = 0.9999
    min_eps = 0.15

    print("Training Player 0")
    for i in tqdm(range(num_games)):
        if q_agent.get_exploration_rate() > min_eps:  # Minimum exploration
            q_agent.set_exploration_rate(q_agent.get_exploration_rate() * gamma_epsilon)
        current_game = Game()
        current_player = 0
        current_trajectory = []

        while current_game.check_winner() == -1:
            current_state = current_game.get_board()
            available_actions = current_game.available_moves(current_player)
            chosen_action = q_agent.choose_action(current_state, available_actions)
            current_game.qlearning_move(chosen_action[0], chosen_action[1], current_player)
            current_trajectory.append((current_state, chosen_action))

            if current_game.check_winner() != -1:
                break

            current_player = 1 - current_player  # Passa al giocatore casuale

            current_state = current_game.get_board()
            available_actions = current_game.available_moves(current_player)
            chosen_action = random.choice(available_actions)
            current_game.qlearning_move(chosen_action[0], chosen_action[1], current_player)

            if current_game.check_winner() != -1:
                break

            current_player = 1 - current_player

        q_agent.update_q_values(current_trajectory, q_agent.calculate_reward(current_game.check_winner()))
        current_trajectory = []

    # Salva la Q-table di Player 0
    with open('players/impl/Q_table0.txt', 'w') as file:
        original_stdout = sys.stdout  # Salva il riferimento a stdout
        sys.stdout = file
        for key, value in q_agent.get_q_table().items():
            print(key[0], key[1], value)
        sys.stdout = original_stdout  # Ripristina stdout

    # Training Player 1
    q_agent1 = QLearning(0.5, 0.95, 1, 1)
    
    print("Training Player 1")
    for i in tqdm(range(num_games)):
        if q_agent1.get_exploration_rate() > min_eps:
            q_agent1.set_exploration_rate(q_agent1.get_exploration_rate() * gamma_epsilon)
        current_game = Game()
        current_player = 0
        current_trajectory = []

        while current_game.check_winner() == -1:
            current_state = current_game.get_board()
            available_actions = current_game.available_moves(current_player)
            chosen_action = random.choice(available_actions)
            current_game.qlearning_move(chosen_action[0], chosen_action[1], current_player)

            if current_game.check_winner() != -1:
                break

            current_player = 1 - current_player
            current_state = current_game.get_board()
            available_actions = current_game.available_moves(current_player)
            chosen_action = q_agent1.choose_action(current_state, available_actions)
            current_game.qlearning_move(chosen_action[0], chosen_action[1], current_player)
            current_trajectory.append((current_state, chosen_action))
            current_player = 1 - current_player

        q_agent1.update_q_values(current_trajectory, q_agent1.calculate_reward(current_game.check_winner()))
        current_trajectory = []

    # Salva la Q-table di Player 1
    with open('players/impl/Q_table1.txt', 'w') as file:
        original_stdout = sys.stdout
        sys.stdout = file
        for key, value in q_agent1.get_q_table().items():
            print(key[0], key[1], value)
        sys.stdout = original_stdout
