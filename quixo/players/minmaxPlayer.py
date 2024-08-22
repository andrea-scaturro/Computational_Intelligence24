from copy import deepcopy
from game import Game, Move, Player
import numpy as np

class MinMaxPlayer(Player):
    def __init__(self, player: int = 0) -> None:
        super().__init__()
        self.player = player % 2  # Ensure the player is either 0 or 1

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        # Create a deep copy of the game to explore moves
        temp_game = deepcopy(game)
        move = self.minmax(temp_game)
        print(move)
        return (move[0], move[1]), move[2]  # Return position and direction of the move

    def minmax(self, game: 'Game', depth: int = 0, alpha: float = -np.inf, beta: float = np.inf) -> tuple[tuple[int, int], Move, int]:
        player_id = game.current_player_idx
        possible_moves = game.available_moves(player_id)

        # Initialize the best score information for the current player
        best_score_info = [-1, -1, -1, -np.inf] if player_id == self.player else [-1, -1, -1, np.inf]

        # Termination conditions
        if game.check_winner() != -1 or depth >= 2 or not possible_moves:
            return [-1, -1, -1, self.calculate_score(game)]

        for move in possible_moves:
            position, slide = move
            # Save the current board state
            temp_board_state = deepcopy(game._board[position[1], :] if slide in {Move.LEFT, Move.RIGHT} else game._board[:, position[0]])

            # Execute the move
            game.execute_move(position, slide, player_id)
            reward = self.minmax(game, depth + 1, alpha, beta)

            # Restore the board state
            if slide in {Move.LEFT, Move.RIGHT}:
                game._board[position[1]] = temp_board_state
            else:
                game._board[:, position[0]] = temp_board_state

            reward[0], reward[1], reward[2] = position[0], position[1], slide  # Update position and slide

            # Update the best score for Max or Min player
            if player_id == self.player:
                if reward[3] > best_score_info[3]:  # Max player
                    best_score_info = reward
                    alpha = reward[3]
            else:
                if reward[3] < best_score_info[3]:  # Min player
                    best_score_info = reward
                    beta = reward[3]

            # Pruning
            if alpha >= beta:
                break

        return best_score_info

    def calculate_score(self, game: 'Game') -> int:
        winner = game.check_winner()
        if winner == self.player:
            return 1  # Win
        elif winner == -1:
            return 0  # Draw
        else:
            return -1  # Lose
