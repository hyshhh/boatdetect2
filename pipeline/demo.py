"""
Demo 模块 — 视频演示可视化

当 config.yaml 中 demo=true 时启用，使用 OpenCV 展示处理结果。
支持：
  - 实时窗口显示
  - 检测框 + 跟踪 ID + 识别结果叠加
  - FPS / 帧率 HUD
  - 按键交互（q=退出, d=切换提示词模式, p=暂停）
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# 中文字体路径（按优先级尝试）
_CJK_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
]


def _load_cjk_font(size: int) -> ImageFont.FreeTypeFont:
    """加载中文字体，找不到则回退到 PIL 默认字体。"""
    for path in _CJK_FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    logger.warning("未找到中文字体，将使用 PIL 默认字体（中文可能仍显示异常）")
    return ImageFont.load_default()


def _pil_put_text(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int] = (255, 255, 255),
) -> tuple[int, int]:
    """用 PIL 在 numpy 图像上绘制中文文字。返回 (text_width, text_height)。"""
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    bbox = draw.textbbox((x, y), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((x, y), text, font=font, fill=fill)
    img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return tw, th


class DemoRenderer:
    """
    Demo 渲染器 — 在视频帧上叠加可视化信息。

    用法：
        renderer = DemoRenderer()
        display_frame = renderer.render(frame, detections, tracks, fps_info)
    """

    def __init__(
        self,
        show_fps: bool = True,
        show_track_id: bool = True,
        show_confidence: bool = False,
        font_scale: float = 0.5,
    ):
        self._show_fps = show_fps
        self._show_track_id = show_track_id
        self._show_confidence = show_confidence
        self._font_scale = font_scale
        self._paused = False

        # 中文字体（PIL 渲染，解决 OpenCV 不支持中文的问题）
        pil_size = max(12, int(font_scale * 32))
        self._cjk_font = _load_cjk_font(pil_size)

    @property
    def paused(self) -> bool:
        return self._paused

    def handle_key(self, key: int) -> str | None:
        """
        处理键盘输入。

        Returns:
            动作字符串："quit", "toggle_prompt", "pause", None
        """
        if key == ord("q") or key == 27:
            return "quit"
        elif key == ord("d"):
            return "toggle_prompt"
        elif key == ord("p"):
            self._paused = not self._paused
            logger.info("Demo %s", "暂停" if self._paused else "继续")
            return "pause"
        elif key == ord("s"):
            return "screenshot"
        return None

    def render(
        self,
        frame: np.ndarray,
        detections: list[Any],
        tracks: dict[int, Any],
        fps_info: dict[str, float] | None = None,
        frame_id: int = 0,
        queue_depth: int = 0,
        max_queue: int = 0,
    ) -> np.ndarray:
        """
        在帧上渲染所有可视化信息。

        Args:
            frame: 原始 BGR 帧。
            detections: Detection 对象列表。
            tracks: {track_id: TrackInfo} 字典。
            fps_info: FPS 统计字典。
            frame_id: 当前帧号。
            queue_depth: 当前队列深度。
            max_queue: 最大队列深度。

        Returns:
            渲染后的帧。
        """
        canvas = frame.copy()

        # 渲染检测框
        for det in detections:
            self._render_detection(canvas, det, tracks.get(det.track_id))

        # 渲染 HUD
        if self._show_fps and fps_info:
            self._render_hud(canvas, fps_info, frame_id, queue_depth, max_queue)

        # 暂停提示
        if self._paused:
            h, w = canvas.shape[:2]
            cv2.putText(
                canvas, "PAUSED (press 'p' to resume)",
                (w // 2 - 150, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
            )

        return canvas

    def _render_detection(
        self,
        canvas: np.ndarray,
        det: Any,
        track_info: Any,
    ) -> None:
        """渲染单个检测框及其标签。"""
        x1, y1, x2, y2 = det.bbox

        # 颜色映射
        if track_info and track_info.db_matched:
            color = (0, 200, 0)       # 绿色：精确匹配库内弦号
        elif track_info and track_info.recognized and track_info.hull_number and track_info.semantic_match_ids:
            color = (0, 215, 255)     # 黄色：识别到弦号，有语义匹配候选
        elif track_info and track_info.recognized and track_info.hull_number:
            color = (0, 0, 255)       # 红色：识别到弦号但未匹配库内
        elif track_info and track_info.recognized and not track_info.hull_number and track_info.semantic_match_ids:
            color = (0, 0, 255)       # 红色：未识别到弦号，通过描述语义匹配到候选
        elif track_info and track_info.recognized and not track_info.hull_number:
            color = (0, 0, 255)       # 红色：未识别到弦号且无语义匹配（"无"）
        elif track_info and track_info.pending:
            color = (255, 255, 0)     # 青色：识别中
        else:
            color = (180, 180, 180)   # 灰色：等待

        # 绘制检测框
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        # Track ID
        if self._show_track_id:
            label = f"ID:{det.track_id}"
            if self._show_confidence:
                label += f" ({det.confidence:.2f})"
            cv2.putText(
                canvas, label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                self._font_scale, color, 1,
            )

        # 识别结果
        if track_info:
            text = self._get_display_text(track_info)
            if text:
                self._render_label(canvas, text, x1, y2, color)

        # 弦号虚线框（hull_box 相对坐标 → frame 坐标）
        if track_info and getattr(track_info, "hull_box", None):
            self._render_hull_box(canvas, det.bbox, track_info.hull_box, color)

    @staticmethod
    def _hull_box_to_frame(
        det_bbox: tuple[int, int, int, int],
        hull_box: list[float],
        frame_shape: tuple[int, int],
    ) -> tuple[int, int, int, int] | None:
        """
        将弦号框的相对坐标映射到 frame 坐标。

        映射链：相对坐标 → crop 像素坐标 → frame 坐标
        crop 创建时在 YOLO bbox 外扩了 pad=20px（detector.py），
        crop 可能被 resize 到 256~512px，但相对坐标不变。

        Args:
            det_bbox: YOLO 检测框 (x1, y1, x2, y2)，frame 坐标。
            hull_box: [rx1, ry1, rx2, ry2]，相对坐标 0.0~1.0。
            frame_shape: (frame_h, frame_w)。

        Returns:
            (fx1, fy1, fx2, fy2) frame 坐标，或 None（无效输入）。
        """
        if not hull_box or len(hull_box) != 4:
            return None

        rx1, ry1, rx2, ry2 = hull_box
        x1, y1, x2, y2 = det_bbox
        frame_h, frame_w = frame_shape[:2]

        # 与 detector.py 保持一致的 pad 值
        pad = 20
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(frame_w, x2 + pad)
        cy2 = min(frame_h, y2 + pad)

        crop_w = cx2 - cx1
        crop_h = cy2 - cy1

        if crop_w <= 0 or crop_h <= 0:
            return None

        # 相对坐标 → frame 坐标
        fx1 = int(cx1 + rx1 * crop_w)
        fy1 = int(cy1 + ry1 * crop_h)
        fx2 = int(cx1 + rx2 * crop_w)
        fy2 = int(cy1 + ry2 * crop_h)

        # 边界裁剪
        fx1 = max(0, min(fx1, frame_w - 1))
        fy1 = max(0, min(fy1, frame_h - 1))
        fx2 = max(0, min(fx2, frame_w - 1))
        fy2 = max(0, min(fy2, frame_h - 1))

        if fx2 <= fx1 or fy2 <= fy1:
            return None

        return (fx1, fy1, fx2, fy2)

    def _render_hull_box(
        self,
        canvas: np.ndarray,
        det_bbox: tuple[int, int, int, int],
        hull_box: list[float],
        color: tuple[int, int, int],
    ) -> None:
        """在帧上绘制弦号虚线框。"""
        frame_coords = self._hull_box_to_frame(det_bbox, hull_box, canvas.shape)
        if frame_coords is None:
            return

        fx1, fy1, fx2, fy2 = frame_coords

        # 绘制虚线矩形（OpenCV 没有原生虚线，用线段拼接）
        dash_len = 8
        gap_len = 5
        thickness = 2

        # 用稍亮的颜色画虚线框
        dash_color = (
            min(color[0] + 80, 255),
            min(color[1] + 80, 255),
            min(color[2] + 80, 255),
        )

        # 四条边分别画虚线
        for (sx, sy, ex, ey) in [
            (fx1, fy1, fx2, fy1),  # 上边
            (fx1, fy2, fx2, fy2),  # 下边
            (fx1, fy1, fx1, fy2),  # 左边
            (fx2, fy1, fx2, fy2),  # 右边
        ]:
            self._draw_dashed_line(canvas, sx, sy, ex, ey, dash_color, thickness, dash_len, gap_len)

    @staticmethod
    def _draw_dashed_line(
        img: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        color: tuple[int, int, int],
        thickness: int = 1,
        dash_len: int = 10,
        gap_len: int = 5,
    ) -> None:
        """在图像上绘制一条虚线。"""
        dx = x2 - x1
        dy = y2 - y1
        length = max(abs(dx), abs(dy))
        if length == 0:
            return

        step_x = dx / length
        step_y = dy / length

        dist = 0.0
        drawing = True
        start_x, start_y = float(x1), float(y1)

        while dist < length:
            if drawing:
                end_dist = min(dist + dash_len, length)
            else:
                end_dist = min(dist + gap_len, length)

            end_x = x1 + step_x * end_dist
            end_y = y1 + step_y * end_dist

            if drawing:
                cv2.line(
                    img,
                    (int(start_x), int(start_y)),
                    (int(end_x), int(end_y)),
                    color, thickness,
                )

            start_x, start_y = end_x, end_y
            dist = end_dist
            drawing = not drawing

    @staticmethod
    def _get_display_text(track_info: Any) -> str:
        """获取显示文本。"""
        if not getattr(track_info, "recognized", False):
            return "(识别中...)" if getattr(track_info, "pending", False) else ""

        # 绿色：精确匹配库内
        if getattr(track_info, "db_matched", False):
            return f"(库内确定id：{getattr(track_info, 'db_match_id', '')})"

        hull_number = getattr(track_info, "hull_number", "") or ""
        semantic_ids = getattr(track_info, "semantic_match_ids", []) or []
        desc = getattr(track_info, "description", "")[:15]

        # 黄色：识别到弦号 + 有语义匹配候选
        if hull_number and semantic_ids:
            candidates = "/".join(semantic_ids[:3])
            return f"(未知id：{hull_number} 可能：{candidates})"

        # 红色：识别到弦号但无匹配
        if hull_number:
            if desc:
                return f"(未知id：{hull_number} - {desc})"
            return f"(未知id：{hull_number})"

        # 红色：未识别到弦号，通过描述语义匹配
        if semantic_ids:
            candidates = "/".join(semantic_ids[:3])
            return f"(未知id：无 可能：{candidates})"

        # 红色：完全未识别
        return "(未知id：无)"

    def _render_label(
        self,
        canvas: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
    ) -> None:
        """在检测框下方渲染带背景的文字标签（PIL 渲染中文）。"""
        pil_img = Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8))
        draw = ImageDraw.Draw(pil_img)
        bbox = draw.textbbox((0, 0), text, font=self._cjk_font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        baseline = 4

        cv2.rectangle(
            canvas,
            (x, y + 2),
            (x + tw + 6, y + th + baseline + 6),
            color, -1,
        )
        _pil_put_text(
            canvas, text,
            x + 3, y + 4,
            self._cjk_font,
            fill=(255, 255, 255),
        )

    def _render_hud(
        self,
        canvas: np.ndarray,
        fps_info: dict[str, float],
        frame_id: int,
        queue_depth: int,
        max_queue: int,
    ) -> None:
        """渲染 HUD（帧率、帧号、队列深度）。"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        color = (0, 255, 0)
        thickness = 1
        y = 18

        lines = [f"Frame: {frame_id}"]
        for ch, fps in fps_info.items():
            lines.append(f"{ch}: {fps:.1f} FPS")
        if max_queue > 0:
            lines.append(f"Queue: {queue_depth}/{max_queue}")

        for line in lines:
            cv2.putText(canvas, line, (10, y), font, scale, color, thickness)
            y += 18
