import torch
import torch.optim as optim
import numpy as np
import os

from env import FurniturePlacementEnv
from model import FurniturePPOAgent
from constants import ROOM_WIDTH, ROOM_HEIGHT, FURNITURE_LIST
from plot import plot_layout
from wfc import generate_candidate_positions

NUM_EPISODES = 200
GAMMA = 0.99
CLIP_EPS = 0.2
LEARNING_RATE = 1e-3
UPDATE_INTERVAL = 5
VISUALIZE_EVERY = 25
BUFFER_MARGIN = 0.1

os.makedirs("output", exist_ok=True)

def train():
    env = FurniturePlacementEnv()
    state_dim = len(env.reset())
    action_dim = env.action_dim

    agent = FurniturePPOAgent(state_dim, action_dim)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)

    for episode in range(NUM_EPISODES):
        state = torch.tensor(env.reset(), dtype=torch.float32)
        rewards, log_probs, values, states, actions = [], [], [], [], []

        success = True
        for idx in range(len(FURNITURE_LIST)):
            spec = FURNITURE_LIST[idx]
            env.current_index = idx
            env.placed = env.placed[:idx]
            state = torch.tensor(env._get_state(), dtype=torch.float32)

            # Retry logic: sample until no buffer box intersects any previous
            valid_action_found = False
            candidate_list = generate_candidate_positions(spec)
            for attempt in range(len(candidate_list)):
                action, log_prob = agent.act(state)
                if action >= len(candidate_list):
                    continue

                x, y = candidate_list[action]
                w, h = spec.width, spec.height
                if not violates_buffer_box(x, y, w, h, env.placed, BUFFER_MARGIN):
                    valid_action_found = True
                    break

            if not valid_action_found:
                success = False
                break

            next_state_raw, reward, done, _ = env.step(action)
            if reward < 0:
                success = False
                break

            # Record data for training
            next_state = torch.tensor(next_state_raw, dtype=torch.float32)
            value = agent.forward(state)[1]
            states.append(state)
            actions.append(torch.tensor(action))
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor(reward, dtype=torch.float32))
            state = next_state

        if not success:
            continue

        returns = compute_returns(rewards, GAMMA)
        log_probs = torch.stack(log_probs)
        values = torch.stack(values).squeeze()
        actions = torch.stack(actions)
        returns = torch.stack(returns).detach()
        advantages = returns - values.detach()

        for _ in range(UPDATE_INTERVAL):
            log_probs_new, values_new, entropy = agent.evaluate(torch.stack(states), actions)
            ratios = torch.exp(log_probs_new - log_probs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns - values_new).pow(2).mean()
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_reward = sum([r.item() for r in rewards])
        print(f"Episode {episode+1}/{NUM_EPISODES} | Total reward: {total_reward:.2f}")

        # ✅ Save & preview layout
        if (episode + 1) % VISUALIZE_EVERY == 0:
            save_path = f"output/episode_{episode+1}.png"
            plot_layout(env.placed, ROOM_WIDTH, ROOM_HEIGHT,
                        save_path=save_path, autopreview=True)
            print(f"📸 Saved layout to {save_path}")

def violates_buffer_box(x, y, w, h, others, margin) -> bool:
    """检查该家具是否与已有家具 buffer box 相交"""
    x0, x1 = x - margin, x + w + margin
    y0, y1 = y - margin, y + h + margin

    for _, ox, oy, ow, oh in others:
        ox0, ox1 = ox - margin, ox + ow + margin
        oy0, oy1 = oy - margin, oy + oh + margin

        overlap = not (x1 <= ox0 or x0 >= ox1 or y1 <= oy0 or y0 >= oy1)
        if overlap:
            return True
    return False

def compute_returns(rewards, gamma):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

if __name__ == "__main__":
    train()
