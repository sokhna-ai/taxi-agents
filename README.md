# taxi-agents
Reinforcement Learning Agent on Taxi-v3 environment using Q-learning. Comparison of random agent vs trained agent with instrumentation and KPI evaluation.
# Taxi-v3 Reinforcement Learning Agent

## 📋 Project Overview

This project demonstrates the development and evaluation of a **Reinforcement Learning agent** trained on the Taxi-v3 environment using **Q-learning**. The goal is to compare a baseline random agent with an intelligent trained agent using structured KPI evaluation, following **Agent Ops** best practices.

### Environment: Taxi-v3
- **Grid size**: 5×5
- **Agent task**: Pick up a passenger at one location and drop them off at a destination
- **Rewards**: 
  - -1 for each step
  - -10 for illegal pickup/drop operations
  - +20 for successfully dropping off the passenger
- **States**: 500 (taxi position, passenger location, destination)
- **Actions**: 6 (North, South, East, West, Pickup, Dropoff)

---

## 🎯 Project Objectives

1. **Baseline Evaluation**: Instrument a random agent and measure key performance indicators (KPI)
2. **Agent Training**: Implement Q-learning to train an intelligent policy
3. **Comparison**: Compare the random baseline with the trained Q-learning agent using the same metrics
4. **Analysis**: Provide insights into agent behavior and performance
5. **Generalization**: Connect findings to real-world agent systems in enterprise environments

---

## 📊 Results Summary

### Baseline (Random Agent) - 1000 Episodes
| Metric              | Value      |
|---------------------|-----------|
| **Avg Reward**      | -772.7    |
| **Avg Steps**       | 197.0     |
| **Success Rate**    | 4%        |

### Trained Q-Learning Agent - 1000 Episodes
| Metric              | Value      |
|---------------------|-----------|
| **Avg Reward**      | +8.0      |
| **Avg Steps**       | 13.0      |
| **Success Rate**    | 100%      |

### Key Insight
The Q-learning trained agent achieves **100% success rate** with an average of only **13 steps per episode**, compared to the random agent's 4% success rate over **197 steps**. This demonstrates the power of reinforcement learning for optimizing decision-making in structured environments.

---

## 🔧 Implementation Details

### Technologies Used
- **Framework**: Python 3.10+
- **Libraries**: 
  - `gymnasium` (RL environment)
  - `numpy` (numerical computing)
  - `pandas` (data analysis)
  - `matplotlib` (visualization)

### Key Components

#### 1. **Baseline Random Agent**
- Takes random actions at each step
- Logged over 1000 episodes
- Metrics recorded: total reward, steps, success indicator

#### 2. **Q-Learning Agent**
- **Hyperparameters**:
  - Learning rate (α) = 0.1
  - Discount factor (γ) = 0.99
  - Initial exploration (ε) = 1.0
  - Exploration decay = 0.999
  - Training episodes = 20,000
- **Q-table**: 500 states × 6 actions
- **Update rule**: Standard Q-learning temporal difference
- Epsilon-greedy strategy for exploration-exploitation tradeoff

#### 3. **Evaluation Methodology**
- Both agents evaluated over 1000 episodes
- Same KPIs: reward, steps, success
- Deterministic policy during evaluation (argmax Q-values, no exploration)

---

## 📈 What This Teaches Us

### Single-Agent Scenario (Taxi-v3)
This project demonstrates foundational **Agent Ops** concepts:
- **Instrumentation**: How to log and track agent behavior
- **Metrics Definition**: Choosing meaningful KPIs (success rate, efficiency, reward)
- **Baseline Comparison**: Establishing performance benchmarks
- **Policy Evaluation**: Assessing learned policies fairly

### Real-World Applications
The same principles apply to enterprise agents:
- **Customer Support Agents**: Track resolution rate, response time, satisfaction
- **Logistics Agents**: Monitor delivery compliance, cost efficiency, schedule adherence
- **Recommendation Agents**: Measure click-through rate, conversion, user retention

These agents follow the same cycle: **perception → decision → action → feedback**, with structured monitoring and iterative improvement.

---

## 🚀 How to Run

### Prerequisites
