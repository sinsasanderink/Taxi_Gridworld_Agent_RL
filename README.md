# 🚖 Taxi-v3 Reinforcement Learning Agent

This project implements and evaluates various **Tabular Q-Learning** strategies to solve the **[Taxi-v3](https://gymnasium.farama.org/environments/toy_text/taxi/)** environment from OpenAI Gymnasium. 

Through iterative experimentation with exploration strategies (Epsilon-Greedy vs. Softmax), hyperparameter tuning (Alpha, Gamma), and decay schedules, this project achieves a high-performance agent with an average reward of **9.19** (over 100 episodes), exceeding the standard "solved" threshold.

## 📖 Overview

The **Taxi-v3** problem is a classic Gridworld environment where an agent must:
1.  Navigate a 5x5 grid.
2.  Pick up a passenger at one of 4 locations (R, G, Y, B).
3.  Drop the passenger off at a specific destination.

### The Environment
* **State Space:** 500 discrete states (25 taxi positions × 5 passenger locations × 4 destinations).
* **Action Space:** 6 discrete actions (South, North, East, West, Pickup, Dropoff).
* **Rewards:** * `+20`: Successful dropoff.
    * `-1`: Per time step (encourages efficiency).
    * `-10`: Illegal pickup/dropoff.

## 🏆 Best Performing Approach

After testing 8 different variations, the most successful strategy was **Q-Learning with Step-Based Epsilon Decay**.

Unlike standard implementations that decay exploration ($\epsilon$) after every *episode*, this agent decays $\epsilon$ after every **time step**. This allows for a much smoother transition from exploration to exploitation, ensuring the agent thoroughly maps the state space before converging on a policy.

### The "Winner" Configuration (Score: 9.19)
* **Algorithm:** Tabular Q-Learning
* **Exploration:** Epsilon-Greedy with Step-Based Decay
* **Episodes:** 150,000
* **Alpha ($\alpha$):** 0.08 (Learning Rate)
* **Gamma ($\gamma$):** 0.99 (Discount Factor)
* **Epsilon Decay:** 0.99995 (per step)

## 📊 Experiment Results

Below is a summary of the different agents trained and their best average reward (over a sliding window of 100 episodes).

| Rank | Strategy / Variation | Best Avg Reward | Key Differentiator |
| :--- | :--- | :--- | :--- |
| 🥇 | **Step-Based Decay (Long Run)** | **9.19** | Decayed $\epsilon$ per step, ran for 150k episodes. |
| 🥈 | **Step-Based Decay (Short Run)** | **9.14** | Decayed $\epsilon$ per step, ran for 20k episodes. |
| 🥉 | **Novelty-Softmax** | **9.01** | Used path memory to penalize frequently visited states. |
| 4 | Optimistic Initialization | 8.84 | Init Q-table at 1.5 + Adaptive Learning Rate. |
| 5 | Tuned Alpha (0.12) | 8.76 | Standard Q-learning with slightly higher learning rate. |
| 6 | Tweaked Softmax | 8.74 | Softmax exploration with path decay. |
| 7 | Lower Gamma (Myopic) | 8.67 | Gamma = 0.95 (prioritized immediate rewards). |
| 8 | Baseline Q-Learning | ~8.37 | Standard Epsilon-Greedy (Episode-based decay). |

## 🧠 Methodology

### 1. The Core Q-Learning Update
All agents utilize the standard Bellman update equation:

$$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s,a)]$$

### 2. Exploration Strategies Tested
* **Episode-Based Epsilon Decay:** The standard approach. The agent explores heavily early on, but the drop in exploration is "steppy."
* **Step-Based Epsilon Decay (The Winner):** Epsilon is updated inside the inner loop (`env.step`). This creates a continuous, smooth curve from 100% exploration to 0.1% exploration, preventing the agent from settling into local optima too early.
* **Novelty-Based Exploration:** Inspired by *Count-Based Exploration*, we maintained a table of `path_counts[state][action]` and subtracted this "boredom" factor from the Q-values during the exploration phase to force the agent into unknown territory.

## 📈 Key Learnings

* **Granularity Matters**: Decaying exploration per step rather than per episode was the single most impactful change.

* **Patience Pays Off**: Extending training from 20,000 to 150,000 episodes yielded a marginal but record-breaking gain (9.14 -> 9.19).

* **Simplicity vs. Complexity**: While the "Novelty Search" (Algorithm 3) was mathematically interesting and performed well (9.01), a well-tuned simple Q-learner with the correct decay schedule outperformed it.
