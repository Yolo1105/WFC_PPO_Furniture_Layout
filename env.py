import numpy as np
from shapely.geometry import LineString, box
from constants import FurnitureSpec, FURNITURE_LIST, ROOM_WIDTH, ROOM_HEIGHT
from wfc import generate_candidate_positions
import yaml
import os

class FurniturePlacementEnv:
    def __init__(self):
        self.furniture_list = FURNITURE_LIST
        self.current_index = 0
        self.placed = []
        self.room_state = np.zeros((int(ROOM_WIDTH * 10), int(ROOM_HEIGHT * 10)))
        self.candidates = [generate_candidate_positions(spec) for spec in self.furniture_list]
        self.action_dim = max(len(cands) for cands in self.candidates)

        # ✅ 默认奖励配置
        default_scores = {
            "wall_bonus": 0.2,
            "path_clear_bonus": 0.5,
            "path_block_penalty": -1.0,
            "desk_window_weight": 1.0,
            "nightstand_near_bed_bonus": 0.5,
            "nightstand_far_penalty": -0.5,
            "wardrobe_near_penalty": -0.5,
            "wardrobe_far_bonus": 0.3,
            "inter_item_too_close_penalty": -1.0,
            "inter_item_close_penalty": -0.5,
        }
        self.rule_scores = {**default_scores, **load_reward_config()}
        print("[Reward Config] Loaded:", self.rule_scores)

    def reset(self):
        self.current_index = 0
        self.placed.clear()
        self.room_state.fill(0)
        return self._get_state()

    def step(self, action: int):
        spec = self.furniture_list[self.current_index]
        cands = self.candidates[self.current_index]
        if action >= len(cands):
            return self._get_state(), -1.0, True, {}

        x, y = cands[action]
        w, h = spec.width, spec.height

        i0, j0 = int(x * 10), int(y * 10)
        i1 = int((x + w) * 10)
        j1 = int((y + h) * 10)

        if np.any(self.room_state[i0:i1, j0:j1]):
            return self._get_state(), -1.0, True, {}

        self.room_state[i0:i1, j0:j1] = 1
        self.placed.append((spec.name, x, y, w, h))
        reward = 1.0

        # ✅ 增加规则奖励
        furniture_center = (x + w / 2, y + h / 2)
        door_center = (0.5, 0.5)

        if abs(x) < 0.1 or abs(x + w - ROOM_WIDTH) < 0.1 or abs(y) < 0.1 or abs(y + h - ROOM_HEIGHT) < 0.1:
            reward += self.rule_scores["wall_bonus"]

        if spec.name in ["BED", "DESK"]:
            if is_path_clear(door_center, furniture_center, self.placed):
                reward += self.rule_scores["path_clear_bonus"]
            else:
                reward -= self.rule_scores["path_block_penalty"]

        if spec.name == "DESK":
            window_center = (ROOM_WIDTH / 2, ROOM_HEIGHT - 0.25)
            dist = np.linalg.norm(np.array(furniture_center) - np.array(window_center))
            max_dist = np.linalg.norm(np.array([0, 0]) - np.array([ROOM_WIDTH, ROOM_HEIGHT]))
            reward += max(0, 1.0 - dist / max_dist) * self.rule_scores["desk_window_weight"]

        if spec.name == "NIGHTSTAND":
            bed = next((item for item in self.placed if item[0] == "BED"), None)
            if bed:
                _, bx, by, bw, bh = bed
                bed_center = (bx + bw / 2, by + bh / 2)
                dist = np.linalg.norm(np.array(furniture_center) - np.array(bed_center))
                if dist < 1.0:
                    reward += self.rule_scores["nightstand_near_bed_bonus"]
                else:
                    reward -= self.rule_scores["nightstand_far_penalty"]

        if spec.name == "WARDROBE":
            window_center = (ROOM_WIDTH / 2, ROOM_HEIGHT - 0.25)
            dist_door = np.linalg.norm(np.array(furniture_center) - np.array(door_center))
            dist_window = np.linalg.norm(np.array(furniture_center) - np.array(window_center))
            if dist_door < 1.5 or dist_window < 1.5:
                reward -= self.rule_scores["wardrobe_near_penalty"]
            elif dist_door > 2.5 and dist_window > 2.5:
                reward += self.rule_scores["wardrobe_far_bonus"]

        buffer_dist = 0.1
        curr_center = np.array(furniture_center)
        for name2, x2, y2, w2, h2 in self.placed[:-1]:
            other_center = np.array([x2 + w2 / 2, y2 + h2 / 2])
            dist = np.linalg.norm(curr_center - other_center)
            if dist < buffer_dist:
                reward -= self.rule_scores["inter_item_too_close_penalty"]
            elif dist < buffer_dist * 2:
                reward -= self.rule_scores["inter_item_close_penalty"]

        self.current_index += 1
        done = self.current_index >= len(self.furniture_list)
        return self._get_state(), reward, done, {}

    def _get_state(self):
        flat_occ = self.room_state.flatten()
        if self.current_index < len(self.furniture_list):
            spec = self.furniture_list[self.current_index]
            return np.concatenate([flat_occ, np.array([spec.width, spec.height])])
        return np.concatenate([flat_occ, np.zeros(2)])

def is_path_clear(start, end, placed):
    path = LineString([start, end])
    for _, x, y, w, h in placed:
        buffer = box(x - 0.1, y - 0.1, x + w + 0.1, y + h + 0.1)
        if path.intersects(buffer):
            return False
    return True

def load_reward_config(path="reward_config.yaml") -> dict:
    if not os.path.exists(path):
        print(f"[Warning] reward config file not found at: {path}")
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}
