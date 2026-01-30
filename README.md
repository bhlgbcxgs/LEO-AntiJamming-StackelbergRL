# Robust LEO Satellite Resource Allocation via Stackelberg Game and Lagrange-Parameterized DQN

This repository contains the official implementation of a hierarchical framework to counter intelligent jammers in LEO networks. The solution decomposes the problem into an outer-loop topology selection and an inner-loop resource allocation solved via L-PDQN.
## File Structure

- **`main.py`**: The main entry point of the training of the inner-loop L-PDQN architecture. It handles configuration loading, environment and agent initialization, the training loop, and data logging.
- **`agent.py`**: The implementation of the DRL agent. It includes the neural network architectures for the Actor, DQN, and Multiplier.
- **`environment.py`**: The satellite communication simulation environment. It calculates throughput of the system handles the dynamics of satellite and jammer positions.
- **`config.py`**: This is the template for the configuration file, containing placeholders for environmental physical constants and training hyperparameters.
- **`satellite_param.py`**: A utility library for calculating satellite orbital positions and antenna gains.
- **`rician_data_collect.py`**: Utilities for calculating BER and rewards under Rician fading channels.
- **`linksimulation.py`**: Network communication model.
- **`lpdqn_performance.py`**: A test for the performance of the proposed L-PDQN method under three attack modes via greedy.
- **`waterfilling_performance.py`**: A test for the performance of the water filling method under three attack modes via greedy.

## Requirements

This project is developed using Python 3. Please ensure the following dependencies are installed:

```bash
pip install numpy torch
```

## Configuration

All hyperparameters are managed within **`config.py`**, allowing you to adjust the experiment without modifying the source code:

- **Environment**: Controls physical parameters.
- **Training**: Controls RL training hyperparameters.
- **Network**: Controls the weight initialization distributions for the neural networks.


## Results

- **Models**: Models will be saved in the **`./results/model/`** directory.
