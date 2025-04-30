import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple

def plot_layout(placed: List[Tuple[str, float, float, float, float]],
                room_width: float,
                room_height: float,
                margin: float = 0.1,
                show: bool = False,
                save_path: str = None,
                autopreview: bool = False):
    """
    绘制家具布局（支持保存 + 自动预览 + 碰撞 buffer）

    :param placed: [(name, x, y, w, h)]
    :param room_width: 房间宽度
    :param room_height: 房间高度
    :param margin: 缓冲边框宽度
    :param show: 是否阻塞显示
    :param save_path: 是否保存图像路径
    :param autopreview: 是否自动短暂弹出图像并关闭
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    room_rect = patches.Rectangle((0, 0), room_width, room_height,
                                  linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(room_rect)

    for item in placed:
        name, x, y, w, h = item

        # 主体
        main = patches.Rectangle((x, y), w, h,
                                 linewidth=1, edgecolor="#3399cc", facecolor="#cce6ff", alpha=0.8)
        ax.add_patch(main)

        # 边框
        border = patches.Rectangle((x, y), w, h,
                                   linewidth=2, edgecolor="#3399cc", facecolor='none')
        ax.add_patch(border)

        # buffer box 智能排除贴墙边
        show_left = x > 0
        show_right = x + w < room_width
        show_bottom = y > 0
        show_top = y + h < room_height

        if any([show_left, show_right, show_bottom, show_top]):
            bx = x - margin if show_left else x
            by = y - margin if show_bottom else y
            bw = w + (margin if show_left else 0) + (margin if show_right else 0)
            bh = h + (margin if show_bottom else 0) + (margin if show_top else 0)

            buffer = patches.Rectangle((bx, by), bw, bh,
                                       linewidth=1.2, edgecolor='gray', linestyle='--', facecolor='none')
            ax.add_patch(buffer)

        # 标签
        cx, cy = x + w / 2, y + h / 2
        ax.text(cx, cy, name, ha='center', va='center', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    ax.set_xlim(-0.5, room_width + 0.5)
    ax.set_ylim(-0.5, room_height + 0.5)
    ax.set_aspect('equal')
    ax.set_title("Furniture Layout")
    ax.set_xlabel("Width (m)")
    ax.set_ylabel("Height (m)")
    ax.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    if autopreview:
        plt.show(block=False)
        plt.pause(1.5)
        plt.close()
    else:
        plt.close()
