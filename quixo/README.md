# Quixo AI Project

## Introduction

This project implements a series of intelligent agents to play **Quixo**, a strategic board game for two players. The implemented agents are designed to explore various artificial intelligence techniques, including search algorithms, machine learning, and case-based approaches.

## The Game of Quixo

Quixo is an abstract board game for two players, where the objective is to align five of your own symbols (❌ or 🔴) in a row, column, or diagonal. The board is a 5x5 grid, and each move involves taking a cube from the periphery of the board, rotating it (if necessary) to display your symbol, and then pushing it back into the board, shifting the other cubes in the row or column.

## Project Structure

The project is organized as follows:

       /quixo  
        │  
        ├── main.py                    # Main script to run matches between agents  
        ├── game.py                    # Implementation of the Quixo game  
        ├── players                    # Folder containing the implemented agents  
        │   ├── randomPlayer.py        # Agent that plays randomly  
        │   ├── myPlayer.py            # "MyPlayer" agent that plays with a custom strategy  
        │   ├── minmaxPlayer.py        # Minimax-based agent with alpha-beta pruning  
        │   ├── montecarloPlayer.py    # Monte Carlo Tree Search-based agent  
        │   └── qlearningPlayer.py     # Q-learning-based agent  
        └── impl  
            ├── qlearning.py           # Script for training Q-learning  
            └── Q_table*.txt           # Q-tables generated during training  




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

