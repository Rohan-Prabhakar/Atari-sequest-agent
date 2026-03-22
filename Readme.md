# Seaquest Deep Q-Learning (DQN) Project

This repository contains a Deep Q-Network (DQN) implementation specifically designed and trained to master the Atari 2600 game **Seaquest**. The project utilizes Kaggle's GPU infrastructure to explore the impact of various hyperparameters and exploration policies on agent performance.

---

## 🚀 Execution Flow & Model Outputs

To reproduce the results, the notebooks must be run in the following chronological order. Each training stage generates specific **.pt (PyTorch)** model files that are required for subsequent analysis and testing.

### 1. `atari-seaquest-baseline-1.ipynb`
* **Running Order:** Run this first to establish the performance floor.
* **Execution:** 5,000 training episodes using standard baseline parameters ($\alpha = 0.00025, \gamma = 0.99$).
* **Primary Output:** `seaquest_baseline.pt` (Saved in `/kaggle/working/checkpoints/`).
* **Key Finding:** Achieved a Mean Reward of **307.4**. This model serves as the control group for evaluating all subsequent variations.

### 2. `atari-variations-2.ipynb`
* **Running Order:** Run this second to conduct a comprehensive hyperparameter grid search.
* **Execution:** Three separate training loops testing Alpha (Learning Rate), Gamma (Discount Factor), and Policy variations.
* **Primary Outputs (.pt files):**
    * **`seaquest_alpha_0001.pt`**: Result of the Learning Rate test (Mean Reward: **849**).
    * **`seaquest_gamma_095.pt`**: Result of the Discount Factor test (**Top Performer: 1,960**).
    * **`seaquest_boltzmann.pt`**: Result of the Exploration Policy test (Mean Reward: **122**).
* **Key Finding:** The **Gamma 0.95** variation is the most effective. It prioritizes immediate survival and combat efficiency in high-density enemy zones, leading to significantly higher scores.

### 3. `atari-test-3.ipynb`
* **Running Order:** Run this last for final validation and video generation.
* **Input Requirement:** You must load the **`seaquest_gamma_095.pt`** file generated in the previous notebook.
* **Execution:** 10 evaluation episodes with $\epsilon = 0$ (Pure Exploitation Mode).
* **Primary Output:** `seaquest_gameplay.mp4`.
* **Evaluation Results:** The agent demonstrated consistent high-tier performance, with test rewards frequently peaking at **1,280** and **1,780**.

---


## 🛠 Project Components
* **DQNConfig**: A centralized dataclass for managing all training and environment hyperparameters.
* **Frame Preprocessing**: Automated conversion of raw Atari frames to $84 \times 84$ grayscale images.
* **Frame Stacker**: Groups 4 consecutive frames to provide the model with temporal context (movement and velocity).
* **DQNAgent**: The core orchestrator implementing Experience Replay, Target Networks, and the Bellman Equation update logic.