import random
from game import Game, Move, Player
from copy import deepcopy

# Il giocatore utilizza la tecnica di simulazione Monte Carlo per decidere le sue mosse
class MonteCarloPlayer(Player):
    def __init__(self, num_simulations=500, max_selected_moves=40) -> None:
        super().__init__()
        self.num_simulations = num_simulations  # Numero di simulazioni Monte Carlo
        self.max_selected_moves = max_selected_moves  # Numero massimo di mosse selezionate

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        return self.monte_carlo_move(game)

    def monte_carlo_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        available_moves = game.available_moves(game.get_current_player())
        num_selected_moves = min(len(available_moves), self.max_selected_moves)
        best_move = None
        best_score = float('-inf')

        # Iterare su un sottoinsieme casuale di mosse disponibili
        for move in random.sample(available_moves, num_selected_moves):
            total_score = self.simulate_move(game, move)
            
            # Aggiorna la migliore mossa in base al punteggio totale
            if total_score > best_score:
                best_score = total_score
                best_move = move

        return best_move

    def simulate_move(self, game: 'Game', move: tuple[int, int]) -> int:
        """Simula il gioco dopo una mossa e restituisce il punteggio totale."""
        total_score = 0

        for _ in range(self.num_simulations):
            cloned_game = deepcopy(game)
            cloned_game.execute_move(move[0], move[1], cloned_game.get_current_player())

            winner = -1
            # Simula il gioco fino a quando non c'è un vincitore
            while winner == -1:
                random_move = random.choice(cloned_game.available_moves(cloned_game.get_current_player()))
                cloned_game.execute_move(random_move[0], random_move[1], cloned_game.get_current_player())
                winner = cloned_game.check_winner()

            # Aggiorna il punteggio totale in base al vincitore
            total_score += self.evaluate_winner(winner)

        return total_score

    def evaluate_winner(self, winner: int) -> int:
        """Restituisce un punteggio basato sul vincitore."""
        if winner == 0:
            return 1  # Punteggio positivo per il giocatore corrente
        elif winner == 1:
            return -1  # Punteggio negativo per il giocatore corrente
        return 0  # In caso di pareggio o stato non determinato
