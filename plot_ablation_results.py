import os
import matplotlib.pyplot as plt

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
    plt.figure(figsize=(10, 6))
    for filename in os.listdir(log_dir):
        if filename.endswith(".txt"):
            label = filename.replace(".txt", "")
            path = os.path.join(log_dir, filename)
            rewards = parse_rewards(path)
            plt.plot(rewards, label=label)
    
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Ablation Study: Reward Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ablation_reward_curves.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_all_logs()
