from dataclasses import dataclass
from typing import List

# === Room Configuration ===
ROOM_WIDTH = 6.0
ROOM_HEIGHT = 5.0
GRID_SIZE = 0.5

# Door zone (x, y, w, h)
DOOR_ZONE = (0.0, 0.0, 1.0, 1.0)

# === Visualization Defaults ===
DEFAULT_DPI = 300                         # 图像保存清晰度
RENDER_EVERY_N_EPISODES = 25             # 每隔多少集保存一张布局图
RECORD_LAST_N_EPISODES = 10              # 最后多少集用于录制 GIF

# === Furniture Specification ===
@dataclass
class FurnitureSpec:
    name: str
    width: float
    height: float
    must_touch_wall: bool = False
    avoid_door_zone: bool = False

FURNITURE_LIST: List[FurnitureSpec] = [
    FurnitureSpec("BED", 2.0, 1.5, must_touch_wall=True),
    FurnitureSpec("WARDROBE", 1.0, 1.5, must_touch_wall=True),
    FurnitureSpec("DESK", 1.5, 0.75, must_touch_wall=True),
    FurnitureSpec("BOOKSHELF", 0.8, 1.2, must_touch_wall=True),
    FurnitureSpec("NIGHTSTAND", 0.7, 0.7, avoid_door_zone=True),
]
