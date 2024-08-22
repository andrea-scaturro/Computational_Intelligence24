from tqdm import tqdm
from game import Game

from players.randomPlayer import RandomPlayer
from players.myPlayer import MyPlayer
from players.minmaxPlayer import MinMaxPlayer
from players.montecarloPlayer import MonteCarloPlayer
from players.qlearningPlayer import QLearningPlayer

if __name__ == '__main__':


    g = Game(showPrint=True)    
    g.print()
    
    
    player0 = MonteCarloPlayer()  # Giocatore 0
    player1 = RandomPlayer()       # Giocatore 1
    
    winner = g.play(player0, player1)  # Gioca una partita
    g.print()

    print(f"Winner: Player {winner} {'❌' if winner == 0 else '🔴'}")

    


    win0 =0
    win1 =0
    nGame = 20
    

    for game in tqdm(range(nGame)):
        g = Game(showPrint = False)

        player0 = MonteCarloPlayer()
        player1 = RandomPlayer()
        winner = g.play(player0, player1)
        
        if winner:
            win1+=1
        else: 
            win0+=1
        
    print("\nResult:")
    print("\tWin Rate Player0: {:.2f}%".format((win0 / nGame) * 100))  
    print("\tWin Rate Player1: {:.2f}%".format((win1 / nGame) * 100)) 