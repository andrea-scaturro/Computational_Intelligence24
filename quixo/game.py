from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
import numpy as np

# Rules on PDF


class Move(Enum):
    '''
    Selects where you want to place the taken piece. The rest of the pieces are shifted
    '''
    TOP = 0
    BOTTOM = 1
    LEFT = 2
    RIGHT = 3


class Player(ABC):
    def __init__(self) -> None:
        '''You can change this for your player if you need to handle state/have memory'''
        pass

    @abstractmethod
    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        '''
        The game accepts coordinates of the type (X, Y). X goes from left to right, while Y goes from top to bottom, as in 2D graphics.
        Thus, the coordinates that this method returns shall be in the (X, Y) format.

        game: the Quixo game. You can use it to override the current game with yours, but everything is evaluated by the main game
        return values: this method shall return a tuple of X,Y positions and a move among TOP, BOTTOM, LEFT and RIGHT
        '''
        pass


class Game(object):
    def __init__(self,showPrint: bool = True) -> None:
        self._board = np.ones((5, 5), dtype=np.uint8) * -1
        self.current_player_idx = 1
        self.showPrint = showPrint
        self.num_playes=0

        

    def get_board(self) -> np.ndarray:
        '''
        Returns the board
        '''
        return deepcopy(self._board)

    def get_current_player(self) -> int:
        '''
        Returns the current player
        '''
        return deepcopy(self.current_player_idx)

    def print(self):
         if self.showPrint:
            """
            Prints the current player and the board in a more readable way
            - ⬜ are neutral pieces
            - ❌ are pieces of player 0
            - 🔴 are pieces of player 1
            """

            # 1. Print the board
            print("\n*****************\n")
            for row in self._board:
                for cell in row:
                    if cell == -1:
                        print("⬜", end=" ")
                    elif cell == 0:
                        print("❌", end=" ")
                    elif cell == 1:
                        print("🔴", end=" ")
                print()
            print()

      
    def check_winner(self) -> int:
        '''Check the winner. Returns the player ID of the winner if any, otherwise returns -1'''
        # for each row
        for x in range(self._board.shape[0]):
            # if a player has completed an entire row
            if self._board[x, 0] != -1 and all(self._board[x, :] == self._board[x, 0]):
                # return the relative id
                return self._board[x, 0]
        # for each column
        for y in range(self._board.shape[1]):
            # if a player has completed an entire column
            if self._board[0, y] != -1 and all(self._board[:, y] == self._board[0, y]):
                # return the relative id
                return self._board[0, y]
        # if a player has completed the principal diagonal
        if self._board[0, 0] != -1 and all(
            [self._board[x, x]
                for x in range(self._board.shape[0])] == self._board[0, 0]
        ):
            # return the relative id
            return self._board[0, 0]
        # if a player has completed the secondary diagonal
        if self._board[0, -1] != -1 and all(
            [self._board[x, -(x + 1)]
             for x in range(self._board.shape[0])] == self._board[0, -1]
        ):
            # return the relative id
            return self._board[0, -1]
        return -1
    
    
    def play(self, player1: Player, player2: Player) -> int:
        '''Play the game. Returns the winning player'''
        players = [player1, player2]
        winner = -1
        while winner < 0:
            self.current_player_idx += 1
            self.current_player_idx %= len(players)
            ok = False
            attempts = 0  # Limite di tentativi
            while not ok and attempts < 10:  # Limite a 10 tentativi
                from_pos, slide = players[self.current_player_idx].make_move(self)
                ok = self.__move(from_pos, slide, self.current_player_idx)
                attempts += 1
                
            winner = self.check_winner()
        return winner



    
    

    def __move(self, from_pos: tuple[int, int], slide: Move, player_id: int) -> bool:
        '''Perform a move'''
        if player_id > 2:
            return False
        # Oh God, Numpy arrays
        
        prev_value = deepcopy(self._board[(from_pos[1], from_pos[0])])

        acceptable = self.__take((from_pos[1], from_pos[0]), player_id)
        if acceptable:
            acceptable = self.__slide((from_pos[1], from_pos[0]), slide)
            if not acceptable:
                self._board[(from_pos[1], from_pos[0])] = deepcopy(prev_value)
        return acceptable
    

    def move(self, from_pos: tuple[int, int], slide: Move, player_id: int) -> bool:
        '''Perform a move'''

        if player_id > 2:
            return False
        
        self.num_playes+=1

        prev_value = deepcopy(self._board[(from_pos[0], from_pos[1])])
        acceptable = self.__take((from_pos[0], from_pos[1]), player_id)
        if acceptable:
            acceptable = self.__slide((from_pos[0], from_pos[1]), slide)
            if not acceptable:
                self._board[(from_pos[0], from_pos[1])] = deepcopy(prev_value)
        return acceptable

    def __take(self, from_pos: tuple[int, int], player_id: int) -> bool:
        '''Take piece'''
        # acceptable only if in border
        acceptable: bool = (
            # check if it is in the first row
            (from_pos[0] == 0 and from_pos[1] < 5)
            # check if it is in the last row
            or (from_pos[0] == 4 and from_pos[1] < 5)
            # check if it is in the first column
            or (from_pos[1] == 0 and from_pos[0] < 5)
            # check if it is in the last column
            or (from_pos[1] == 4 and from_pos[0] < 5)
            # and check if the piece can be moved by the current player
        ) and (self._board[from_pos] < 0 or self._board[from_pos] == player_id)
        if acceptable:
            self._board[from_pos] = player_id
        return acceptable

    def __slide(self, from_pos: tuple[int, int], slide: Move) -> bool:
        '''Slide the other pieces'''
        # define the corners
        SIDES = [(0, 0), (0, 4), (4, 0), (4, 4)]
        # if the piece position is not in a corner
        if from_pos not in SIDES:
            # if it is at the TOP, it can be moved down, left or right
            acceptable_top: bool = from_pos[0] == 0 and (
                slide == Move.BOTTOM or slide == Move.LEFT or slide == Move.RIGHT
            )
            # if it is at the BOTTOM, it can be moved up, left or right
            acceptable_bottom: bool = from_pos[0] == 4 and (
                slide == Move.TOP or slide == Move.LEFT or slide == Move.RIGHT
            )
            # if it is on the LEFT, it can be moved up, down or right
            acceptable_left: bool = from_pos[1] == 0 and (
                slide == Move.BOTTOM or slide == Move.TOP or slide == Move.RIGHT
            )
            # if it is on the RIGHT, it can be moved up, down or left
            acceptable_right: bool = from_pos[1] == 4 and (
                slide == Move.BOTTOM or slide == Move.TOP or slide == Move.LEFT
            )
        # if the piece position is in a corner
        else:
            # if it is in the upper left corner, it can be moved to the right and down
            acceptable_top: bool = from_pos == (0, 0) and (
                slide == Move.BOTTOM or slide == Move.RIGHT)
            # if it is in the lower left corner, it can be moved to the right and up
            acceptable_left: bool = from_pos == (4, 0) and (
                slide == Move.TOP or slide == Move.RIGHT)
            # if it is in the upper right corner, it can be moved to the left and down
            acceptable_right: bool = from_pos == (0, 4) and (
                slide == Move.BOTTOM or slide == Move.LEFT)
            # if it is in the lower right corner, it can be moved to the left and up
            acceptable_bottom: bool = from_pos == (4, 4) and (
                slide == Move.TOP or slide == Move.LEFT)
        # check if the move is acceptable
        acceptable: bool = acceptable_top or acceptable_bottom or acceptable_left or acceptable_right
        # if it is
        if acceptable:
            # take the piece
            piece = self._board[from_pos]
            # if the player wants to slide it to the left
            if slide == Move.LEFT:
                # for each column starting from the column of the piece and moving to the left
                for i in range(from_pos[1], 0, -1):
                    # copy the value contained in the same row and the previous column
                    self._board[(from_pos[0], i)] = self._board[(
                        from_pos[0], i - 1)]
                # move the piece to the left
                self._board[(from_pos[0], 0)] = piece
            # if the player wants to slide it to the right
            elif slide == Move.RIGHT:
                # for each column starting from the column of the piece and moving to the right
                for i in range(from_pos[1], self._board.shape[1] - 1, 1):
                    # copy the value contained in the same row and the following column
                    self._board[(from_pos[0], i)] = self._board[(
                        from_pos[0], i + 1)]
                # move the piece to the right
                self._board[(from_pos[0], self._board.shape[1] - 1)] = piece
            # if the player wants to slide it upward
            elif slide == Move.TOP:
                # for each row starting from the row of the piece and going upward
                for i in range(from_pos[0], 0, -1):
                    # copy the value contained in the same column and the previous row
                    self._board[(i, from_pos[1])] = self._board[(
                        i - 1, from_pos[1])]
                # move the piece up
                self._board[(0, from_pos[1])] = piece
            # if the player wants to slide it downward
            elif slide == Move.BOTTOM:
                # for each row starting from the row of the piece and going downward
                for i in range(from_pos[0], self._board.shape[0] - 1, 1):
                    # copy the value contained in the same column and the following row
                    self._board[(i, from_pos[1])] = self._board[(
                        i + 1, from_pos[1])]
                # move the piece down
                self._board[(self._board.shape[0] - 1, from_pos[1])] = piece
        return acceptable
    





# Function for New Players

    def available_moves(self, player_idx) -> list:
        possible_moves = []

        # Definisci le posizioni valide per il movimento
        edge_positions = [(0, x) for x in range(5)] + [(4, x) for x in range(5)] + [(y, 0) for y in range(1, 4)] + [(y, 4) for y in range(1, 4)]

        for from_pos in edge_positions:
            for slide in Move:
                if self.check_move(from_pos, slide, player_idx):
                    possible_moves.append((from_pos, slide))

        return possible_moves

    def check_move(self, from_pos: tuple[int, int], slide: Move, player_id: int) -> bool:
        '''Controlla se il movimento è valido per il giocatore specificato.'''
        if player_id > 2:
            return False
        
        app = (from_pos[1], from_pos[0])  # Ruota la posizione per il controllo

        # Controlla il "take"
        is_edge_position = (app[0] in [0, 4]) and (app[1] < 5)  # Posizione su un bordo
        acceptable_take = is_edge_position and (self._board[app] < 0 or self._board[app] == player_id)

        # Controlla lo "slide"
        is_corner = app in [(0, 0), (0, 4), (4, 0), (4, 4)]
        acceptable_slide = False

        if is_corner:
            acceptable_slide = (app == (0, 0) and slide in {Move.BOTTOM, Move.RIGHT}) or \
                            (app == (4, 0) and slide in {Move.TOP, Move.RIGHT}) or \
                            (app == (0, 4) and slide in {Move.BOTTOM, Move.LEFT}) or \
                            (app == (4, 4) and slide in {Move.TOP, Move.LEFT})
        else:
            acceptable_slide = (app[0] == 0 and slide in {Move.BOTTOM, Move.LEFT, Move.RIGHT}) or \
                            (app[0] == 4 and slide in {Move.TOP, Move.LEFT, Move.RIGHT}) or \
                            (app[1] == 0 and slide in {Move.BOTTOM, Move.TOP, Move.RIGHT}) or \
                            (app[1] == 4 and slide in {Move.BOTTOM, Move.TOP, Move.LEFT})

        return acceptable_take and acceptable_slide

    def execute_move(self, from_pos: tuple[int, int], slide: Move, player_id: int) -> bool:
        '''Esegue un movimento e restituisce se è stato effettuato con successo.'''
        if player_id > 2:
            return False

        # Salva lo stato precedente del pezzo prima del movimento
        prev_value = self._board[from_pos[1], from_pos[0]].copy()

        # Funzione interna per eseguire il "take" e il "slide"
        def take_and_slide(pos: tuple[int, int], slide: Move) -> bool:
            take_ok = self.__take(pos, player_id)
            if take_ok:
                slide_ok = self.__slide(pos, slide)
                if not slide_ok:
                    self._board[pos] = prev_value  # Ripristina il pezzo se lo slide fallisce
                return slide_ok
            return False

        move_ok = take_and_slide((from_pos[1], from_pos[0]), slide)

        if move_ok:
            self.current_player_idx = (self.current_player_idx + 1) % 2  # Cambia giocatore dopo il movimento

        return move_ok


    def possible_moves(self, player_id: int) -> list[tuple[tuple[int, int], Move]]:
        '''Returns a list of possible moves for the player'''
        moves = []
        CORNER = [(0,0), (0,4), (4,4), (4,0)]
        STEP = [1, 1, -1, -1]
        MOVES = [Move.TOP, Move.RIGHT, Move.BOTTOM, Move.LEFT]
        for i in range(len(CORNER)):
            match i:
                case 0:
                    if self._board[CORNER[i]] == -1 or self._board[CORNER[i]] == player_id:
                        moves.append((CORNER[i], Move.RIGHT))
                        moves.append((CORNER[i], Move.BOTTOM))
                case 1:
                    if self._board[CORNER[i]] == -1 or self._board[CORNER[i]] == player_id:
                        moves.append((CORNER[i], Move.LEFT))
                        moves.append((CORNER[i], Move.BOTTOM))
                case 2:
                    if self._board[CORNER[i]] == -1 or self._board[CORNER[i]] == player_id:
                        moves.append((CORNER[i], Move.LEFT))
                        moves.append((CORNER[i], Move.TOP))
                case 3:
                    if self._board[CORNER[i]] == -1 or self._board[CORNER[i]] == player_id:
                        moves.append((CORNER[i], Move.RIGHT))
                        moves.append((CORNER[i], Move.TOP))
                
            for x in range(CORNER[i][0], CORNER[(i+1)%4][0]+STEP[i], STEP[i]):
                for y in range(CORNER[i][1], CORNER[(i+1)%4][1]+STEP[i], STEP[i]):
                    if (x,y) not in CORNER and (self._board[x,y] == -1 or self._board[x,y] == player_id):
                        for j in range(len(MOVES)):
                            if j!= i:
                                moves.append(((x,y), MOVES[j]))
        return moves
    

    def qlearning_move(self, from_pos: tuple[int, int], slide: Move, player_id: int) -> bool: #Takes a position, a move, and a player ID. It performs a move if it is valid
        '''Perform a move'''
        if player_id > 2:
            return False
        # Oh God, Numpy arrays
        prev_value = deepcopy(self._board[(from_pos[1], from_pos[0])])
        acceptable = self.__take((from_pos[1], from_pos[0]), player_id) #il metodo take controlla se la mossa è accettabile, sta qui sotto
        if acceptable:
            acceptable = self.__slide((from_pos[1], from_pos[0]), slide)
            if not acceptable:
                self._board[(from_pos[1], from_pos[0])] = deepcopy(prev_value)
        return acceptable