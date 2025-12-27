import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=== ANALYZING TAXI-V3 AGENTS ===\n")

env = gym.make("Taxi-v3")
num_episodes_eval = 1000

# BASELINE
print("1️⃣ Baseline aléatoire...")
random_rewards, random_steps, random_success = [], [], []
for ep in range(num_episodes_eval):
    state, info = env.reset()
    done = False
    total_reward, steps, success = 0, 0, 0
    while not done:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
        if terminated:
            success = 1
        state = next_state
    random_rewards.append(total_reward)
    random_steps.append(steps)
    random_success.append(success)

print(f"✅ Baseline done\n")

# Q-LEARNING TRAINING
print("2️⃣ Q-Learning training (20k episodes)...")
alpha, gamma, epsilon, epsilon_min, epsilon_decay = 0.1, 0.99, 1.0, 0.1, 0.999
Q = np.zeros((env.observation_space.n, env.action_space.n))
training_rewards = []

for ep in range(20000):
    state, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(Q[state])
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        td_target = reward + gamma * np.argmax(Q[next_state]) * (0 if done else 1)
        Q[state, action] += alpha * (td_target - Q[state, action])
        total_reward += reward
        state = next_state
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    training_rewards.append(total_reward)
    if (ep + 1) % 5000 == 0:
        print(f"  {ep+1}/20000")

print(f"✅ Training done\n")

# Q-LEARNING EVAL
print("3️⃣ Q-Learning evaluation...")
q_rewards, q_steps, q_success = [], [], []
for ep in range(num_episodes_eval):
    state, info = env.reset()
    done = False
    total_reward, steps, success = 0, 0, 0
    while not done:
        action = np.argmax(Q[state])
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
        if terminated:
            success = 1
        state = next_state
    q_rewards.append(total_reward)
    q_steps.append(steps)
    q_success.append(success)

print(f"✅ Evaluation done\n")

# VISUALIZATIONS - AMÉLIORÉES
print("4️⃣ Creating visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

window = 100
smoothed = np.convolve(training_rewards, np.ones(window)/window, mode="valid")
axes[0, 0].plot(smoothed, linewidth=2, color='blue')
axes[0, 0].set_title("Learning Curve (Q-Learning)", fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel("Episode")
axes[0, 0].set_ylabel("Reward moyen")
axes[0, 0].grid(True, alpha=0.3)

# AMÉLIORATION: Histogrammes séparés pour voir les deux
axes[0, 1].hist(random_rewards, bins=30, alpha=0.6, label='Random Agent', color='red', edgecolor='black')
axes[0, 1].hist(q_rewards, bins=30, alpha=0.6, label='Q-Learning Agent', color='green', edgecolor='black')
axes[0, 1].set_title("Rewards Distribution (Separated)", fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel("Reward per Episode")
axes[0, 1].set_ylabel("Frequency")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].hist(random_steps, bins=30, alpha=0.6, label='Random Agent', color='red', edgecolor='black')
axes[1, 0].hist(q_steps, bins=30, alpha=0.6, label='Q-Learning Agent', color='green', edgecolor='black')
axes[1, 0].set_title("Steps Distribution", fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel("Steps per Episode")
axes[1, 0].set_ylabel("Frequency")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

data = [random_rewards, q_rewards]
bp = axes[1, 1].boxplot(data, labels=['Random Agent', 'Q-Learning Agent'], patch_artist=True, tick_labels=['Random', 'Q-Learning'])
bp['boxes'][0].set_facecolor('lightcoral')
bp['boxes'][1].set_facecolor('lightgreen')
axes[1, 1].set_title("Rewards Comparison (BoxPlot)", fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel("Reward per Episode")
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('agent_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Saved: agent_comparison.png\n")
plt.show()

# COMPARISON TABLE
print("="*70)
print("BASELINE vs Q-LEARNING")
print("="*70)
comparison = pd.DataFrame({
    'Metric': ['Reward Mean', 'Reward Std', 'Steps Mean', 'Success Rate (%)'],
    'Random Agent': [
        f"{np.mean(random_rewards):.2f}",
        f"{np.std(random_rewards):.2f}",
        f"{np.mean(random_steps):.2f}",
        f"{100*np.mean(random_success):.1f}%"
    ],
    'Q-Learning Agent': [
        f"{np.mean(q_rewards):.2f}",
        f"{np.std(q_rewards):.2f}",
        f"{np.mean(q_steps):.2f}",
        f"{100*np.mean(q_success):.1f}%"
    ]
})
print(comparison.to_string(index=False))
print("="*70)
print("✨ Analysis complete!")
