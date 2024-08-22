import random
from game import Game, Move, Player
import copy  

class MonteCarloPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        return self.monte_carlo_move(game)

    def monte_carlo_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        num_simulations = 1000  
        num_selected_moves = min(len(game.available_moves(game.get_current_player())), 40)
        best_move = None
        best_score = float('-inf')
        best_from_pos = None  

        for move in random.sample(game.available_moves(game.get_current_player()), num_selected_moves):
            total_score = 0
                        
            for _ in range(num_simulations):
                cloned_game = game
                cloned_game.execute_move(move[0], move[1], game.get_current_player())
                
                while True:
                    winner = cloned_game.check_winner()
                    if winner != -1:
                        break
                    
                    random_move = random.choice(cloned_game.available_moves(cloned_game.get_current_player()))
                    cloned_game.execute_move(random_move[0], random_move[1], cloned_game.get_current_player())
                
                if winner == 0:
                    total_score += 1
                elif winner == 1:
                    total_score -= 1

            if total_score > best_score:
                best_score = total_score
                best_move = move[1]
                best_from_pos = move[0]  

        if best_from_pos is None:
            best_from_pos = (0,0)
        return best_from_pos, best_move  