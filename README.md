# Lunar Lander Coach

An evolutionary training simulator that uses genetic algorithms to teach 100 neural network agents to land on the moon.

## Overview

This project simulates 100 lunar lander agents simultaneously, each controlled by a small neural network. Through evolutionary algorithms (selection, crossover, mutation), the agents learn to:
- Navigate to the landing pad
- Control velocity and orientation
- Collect green coins (+1 score) and avoid red coins (-1 score)
- Successfully land within the landing pad boundaries

## Features

- **Genetic Evolution**: Elite preservation (~10%), tournament selection, crossover, and mutation
- **Hall of Fame**: Top 10 champions across all generations (marked with blue borders)
- **Neural Networks**: 10 inputs (position, velocity, angle, coin distances) → 4 outputs (action distribution)
- **Interactive Coin Placement**: Place reward/penalty coins while paused
- **Performance Tracking**: Running totals and percentages of successful landings
- **Data Export**: CSV and JSON files saved for landed agents each generation

## Controls

- **P**: Pause/unpause simulation
- **Left Click** (while paused): Add green coin (+1 score)
- **Right Click** (while paused): Add red coin (-1 score)
- **Click existing coin**: Remove it

## Running the Simulator

1. Ensure you have Go installed (1.20+ recommended)
2. From the project directory run:

```bash
go build
./lunar-lander-coach
```

Or directly:
```bash
go run .
```

## Neural Network Architecture

Each agent has 4 independent neural network heads (one per action):
- **Inputs (10)**: horizontal offset, vertical offset, x velocity, y velocity, angle, sin(angle), cos(angle), speed, distance to closest green coin, distance to closest red coin
- **Actions (4)**: nop, main thrust, right thrust, left thrust
- **Output**: Softmax distribution over actions (currently using argmax for deterministic selection)

## Evolution Parameters

- Population: 100 agents
- Elite preservation: ~10% (exact clones, never mutated)
- Hall of Fame: Top 10 champions preserved across all generations
- Mutation rate: 8%
- Mutation scale: 0.05
- Selection: Tournament selection (size 3) from top 50%
- Episode length: 400 steps

## Scoring System

Agents are scored based on:
- Horizontal distance from landing pad center
- Height above ground
- Velocity magnitude
- Angle deviation from upright
- Landing bonus: +50 points
- Coin collection: +1 per green, -1 per red

Only agents landing within the landing pad boundaries count as successful landings.

## Data Output

Each generation saves to `data/`:
- `YYYY-MM-DD_HH-MM-SS.csv`: All agent states (position, velocity, landed/crashed/killed status, scores)
- `landed_weights_YYYY-MM-DD_HH-MM-SS_SCORE.json`: Neural network weights for all agents that successfully landed

## Technologies

- Go 1.20+
- [Ebiten v2](https://github.com/hajimehoshi/ebiten) for rendering and input
- Custom neural network package (`nn/`) with genetic operators

## File Structure

- `main.go`: Main game loop, rendering, evolution
- `game.go`: Game struct, physics simulation, coin collection
- `types.go`: Core types (Environment, Lander, Agent, Policy, Coin)
- `policy.go`: Neural network policy implementation
- `scoring.go`: Fitness/scoring functions
- `nn/nn.go`: Neural network module with mutation and crossover

## Notes

- Tweak evolution parameters in `evolvePopulation()` for different training dynamics
- Adjust scoring weights in `scoring.go` to prioritize different behaviors
- Landing pad width and physics constants defined in `game.go` and constants
- Champions (blue-bordered agents) are preserved across generations and never mutated
- Coins persist across generations but reset their collected state each episode

