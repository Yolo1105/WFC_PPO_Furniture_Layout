import os
import matplotlib.pyplot as plt
import numpy as np

def parse_rewards(log_path):
    rewards = []
    with open(log_path, "r") as f:
        for line in f:
            if "Total reward" in line:
                try:
                    reward = float(line.strip().split(":")[-1])
                    rewards.append(reward)
                except ValueError:
                    continue
    return rewards

def plot_all_logs(log_dir="logs"):
    all_rewards = {}
    max_len = 0

    for filename in os.listdir(log_dir):
        if filename.endswith(".txt"):
            label = filename.replace(".txt", "")
            path = os.path.join(log_dir, filename)
            rewards = parse_rewards(path)
            all_rewards[label] = rewards
            max_len = max(max_len, len(rewards))

    # 补齐不同长度的 reward 列表
    for label in all_rewards:
        rewards = all_rewards[label]
        all_rewards[label] = rewards + [None] * (max_len - len(rewards))

    # 转换为 NumPy 数组
    labels = list(all_rewards.keys())
    reward_matrix = np.array([all_rewards[label] for label in labels])
    episodes = np.arange(max_len)

    plt.figure(figsize=(12, 6))

    # 每条曲线画出来
    for i, label in enumerate(labels):
        rewards = reward_matrix[i]
        plt.plot(episodes, rewards, label=label, alpha=0.6)

    # 均值和标准差
    valid_mask = ~np.isnan(reward_matrix)
    reward_matrix_masked = np.where(valid_mask, reward_matrix, np.nan)
    mean_rewards = np.nanmean(reward_matrix_masked, axis=0)
    std_rewards = np.nanstd(reward_matrix_masked, axis=0)

    plt.plot(episodes, mean_rewards, color='black', linewidth=2.0, label="Mean Reward")
    plt.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards,
                     color='gray', alpha=0.3, label="±1 Std Dev")

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Ablation Study: Reward Curves (with Mean ± Std)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ablation_reward_curves.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_all_logs()
