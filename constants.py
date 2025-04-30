from dataclasses import dataclass
from typing import List

# 基本房间设置
ROOM_WIDTH = 6.0
ROOM_HEIGHT = 5.0
GRID_SIZE = 0.5

# 门口区域默认在左下角 (0,0) 到 (1,1)
DOOR_ZONE = (0.0, 0.0, 1.0, 1.0)


@dataclass
class FurnitureSpec:
    name: str
    width: float
    height: float
    must_touch_wall: bool = False
    avoid_door_zone: bool = False


# MVP家具规格列表
FURNITURE_LIST: List[FurnitureSpec] = [
    FurnitureSpec("BED", 2.0, 1.5, must_touch_wall=True),
    FurnitureSpec("WARDROBE", 1.0, 1.5, must_touch_wall=True),
    FurnitureSpec("DESK", 1.5, 0.75, must_touch_wall=True),
    FurnitureSpec("BOOKSHELF", 0.8, 1.2, must_touch_wall=True),
    FurnitureSpec("NIGHTSTAND", 0.7, 0.7, avoid_door_zone=True),
]
