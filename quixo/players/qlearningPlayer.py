import random
from game import Game, Move, Player
import numpy as np
import struct

class QLearningPlayer(Player):
    def __init__(self, player: int) -> None:
        super().__init__()
        self.player = player
        self.q_table = {}
        self.load_q_table()

    def load_q_table(self):
        filename = f"players/impl/Q_table{self.player}.txt" 
        
        
        with open(filename, 'r') as f:
                for line in f:
                    state_action, value = line.rsplit(" ", 1)
                    state, action = state_action.split(" ")
                    self.q_table[(state, action)] = float(value)
       

    def compact_state(self, matrix):
        return struct.pack(f">{len(matrix.flatten())}b", *matrix.flatten())

    def compact_move(self, move):
        return f"{move[0][0]}{move[0][1]}{move[1].value}".encode('utf-8')

    def decode_move(self, encoded_move):
        decoded = encoded_move.decode('utf-8')
        return (int(decoded[0]), int(decoded[1])), Move(int(decoded[2]))

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0)

    def make_move(self, game: Game) -> tuple[tuple[int, int], Move]:
        player = game.get_current_player()
        actions = game.possible_moves(player)
        state = game.get_board()
        compacted_state = self.compact_state(state)
        compacted_actions = [self.compact_move(action) for action in actions]
        
        q_values = np.array([self.get_q_value(compacted_state, action) for action in compacted_actions])
        max_q_value = np.max(q_values)
        best_actions = [compacted_actions[i] for i in range(len(compacted_actions)) if q_values[i] == max_q_value]
        
        return self.decode_move(random.choice(best_actions))
