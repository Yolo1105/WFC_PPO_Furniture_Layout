import os
import pandas as pd

def parse_rewards_for_csv(log_path):
    rewards = []
    with open(log_path, "r") as f:
        for line in f:
            if "Total reward" in line:
                try:
                    reward = float(line.strip().split(":")[-1])
                    rewards.append(reward)
                except ValueError:
                    rewards.append(None)
    return rewards

def compile_reward_csv(log_dir="logs", output_csv="ablation_rewards.csv"):
    all_data = {}
    max_len = 0

    for filename in os.listdir(log_dir):
        if filename.endswith(".txt"):
            label = filename.replace(".txt", "")
            path = os.path.join(log_dir, filename)
            rewards = parse_rewards_for_csv(path)
            all_data[label] = rewards
            max_len = max(max_len, len(rewards))

    # Pad shorter logs
    for key in all_data:
        all_data[key] += [None] * (max_len - len(all_data[key]))

    df = pd.DataFrame(all_data)
    df.index.name = "Episode"
    df.to_csv(output_csv)
    print(f"✅ Saved CSV to {output_csv}")
    return df

if __name__ == "__main__":
    compile_reward_csv()
