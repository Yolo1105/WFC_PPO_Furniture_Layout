# 🛋️ Furniture Placement Optimization (WFC + PPO)

This project combines **Wave Function Collapse (WFC)** and **Proximal Policy Optimization (PPO)** to learn realistic furniture arrangements under spatial and semantic constraints.

---

## 🚀 Features

### 📐 Spatial Reasoning
- Grid-based room with wall-aware candidate generation (WFC)
- Furniture specs and wall/window constraints
- Collision handling via occupancy map

### 🧠 Reinforcement Learning
- PPO agent selects from legal placements
- Curriculum strategy: one furniture per step, replay others
- Rule-based rewards integrated during environment feedback

### 🎯 Real-World Inspired Reward System
All rewards are configurable in `reward_config.yaml`:
- ✅ Touch wall
- ✅ Path from door to BED / DESK clear
- ✅ DESK near window
- ✅ NIGHTSTAND near BED
- ✅ WARDROBE far from door/window
- ✅ Minimum spacing between all furniture

### 🔄 Reward Config Management
- All reward weights are centralized in `self.rule_scores`
- Easily swap rules for ablation via YAML
- Full auto-sweep via `ablation_runner.py`

---

## 📊 Evaluation and Visualization
- Reward curves plotted across training (`plot_ablation_results.py`)
- Episode rewards exported to CSV (`ablation_rewards.csv`)
- Compare configs using bar charts and heatmaps
- All layout snapshots saved to `/output`

---

## 📁 Directory Structure (Key Modules)

```bash
furniture_mvp/
├── env.py                    # PPO environment with rule-based rewards
├── model.py                  # PPO neural policy
├── wfc.py                    # Legal placement generator
├── train.py                  # Training pipeline (episodes, reward, reset)
├── plot.py                   # Matplotlib furniture layout visualizer
├── reward_config.yaml        # Rule score configuration
├── ablation_runner.py        # Auto-run sweep with multiple reward configs
├── compile_reward_csv.py     # Export per-episode reward to CSV
├── plot_ablation_results.py  # Visualize reward bar chart and heatmap
└── logs/, output/            # Training logs and layout snapshots
