# Quixo AI Project

## Introduction

This project implements a series of intelligent agents to play **Quixo**, a strategic board game for two players. The implemented agents are designed to explore various artificial intelligence techniques, including search algorithms, machine learning, and case-based approaches.

## The Game of Quixo

Quixo is an abstract board game for two players, where the objective is to align five of your own symbols (❌ or 🔴) in a row, column, or diagonal. The board is a 5x5 grid, and each move involves taking a cube from the periphery of the board, rotating it (if necessary) to display your symbol, and then pushing it back into the board, shifting the other cubes in the row or column.

## Project Structure

The project is organized as follows:

/quixo  
│  
├── main.py                    # Script principale per eseguire partite tra agenti  
├── game.py                    # Implementazione del gioco Quixo  
├── players                    # Cartella contenente gli agenti implementati  
│   ├── randomPlayer.py        # Agente che gioca in modo casuale  
│   ├── myPlayer.py            # Agente "MyPlayer" che gioca con una strategia personalizzata  
│   ├── minmaxPlayer.py        # Agente basato su Minimax con potatura alpha-beta  
│   ├── montecarloPlayer.py    # Agente basato su Monte Carlo Tree Search  
│   └── qlearningPlayer.py     # Agente basato su Q-learning  
└── impl  
    ├── qlearning.py           # Script per l'addestramento del Q-learning  
    └── Qtable_Player*.txt     # Tabelle Q generate durante l'addestramento  



## Implemented Agents

### 1. **Random Player**
The `RandomPlayer` agent selects moves completely randomly, without any strategy or evaluation. This agent is useful as a benchmark to compare the performance of more advanced agents.

### 2. **MyPlayer**
The `MyPlayer` agent implements a custom strategy, which can combine predefined game rules and heuristics. This agent represents a manual approach to AI construction, where decisions are made based on user-defined insights or strategies.

### 3. **Minimax Player**
The `MinMaxPlayer` agent uses the Minimax algorithm, enhanced with alpha-beta pruning, to explore the move tree and decide on the optimal move. This approach aims to minimize potential loss in a game scenario, assuming that the opponent plays optimally.

### 4. **Monte Carlo Player**
The `MonteCarloPlayer` agent uses Monte Carlo Tree Search (MCTS) to select moves. This method combines random simulations with structured search to determine the most promising move. It is particularly effective in games with large state spaces.

### 5. **Q-learning Player**
The `QLearningPlayer` agent is an example of reinforcement learning. It uses the Q-learning algorithm to learn an optimal policy by playing thousands of games against itself or other opponents. The learned policies are stored in Q-tables, which are used to make decisions during gameplay.

