from PyQt6.QtWidgets import QWidget, QApplication
from PyQt6.QtGui import QPainter, QPen, QBrush, QPolygon
from PyQt6.QtCore import Qt, QPoint

import numpy as np
import matplotlib.pyplot as plt


def draw_zigzag_line(painter, start_point, end_point, zigzag_width, num_segments, color, width):
    painter.setPen(QPen(color, width))

    x1, y1 = start_point.x(), start_point.y()
    x2, y2 = end_point.x(), end_point.y()

    xs = np.linspace(x1, x2, num_segments)
    ys = np.linspace(y1, y2, num_segments)

    tangent_vector = np.array([x2 - x1, y2 - y1]).astype(float)
    tangent_vector /= np.linalg.norm(tangent_vector)
    normal_vector = np.array([ -tangent_vector[1], tangent_vector[0] ])

    zigzag_points = [start_point]

    for i in range(1, num_segments - 1):
        new_coord = np.array([xs[i], ys[i]]) + zigzag_width * normal_vector * (-1)**i
        zigzag_points.append(QPoint(int(new_coord[0]), int(new_coord[1])))

    zigzag_points.append(end_point)
    
    painter.drawPolyline(zigzag_points)


def draw_arrow(painter, start_point, end_point, arrow_size, color, width):
    painter.setPen(QPen(color, width))
    painter.drawLine(start_point, end_point)


    # draw arrow head
    direction = np.array([end_point.x() - start_point.x(), end_point.y() - start_point.y()]).astype(float)
    direction /= np.linalg.norm(direction)

    normal = np.array([ -direction[1], direction[0] ])
    
    arrow_head = np.array([end_point.x(), end_point.y()]).astype(float)
    arrow_head -= arrow_size * direction
    arrow_head += arrow_size * normal

    painter.drawLine(end_point, QPoint(int(arrow_head[0]), int(arrow_head[1])))
    arrow_head -= 2 * arrow_size * normal
    painter.drawLine(end_point, QPoint(int(arrow_head[0]), int(arrow_head[1])))


