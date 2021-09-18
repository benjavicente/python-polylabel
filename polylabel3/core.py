"Core functions"

from __future__ import annotations

import math
import heapq
from collections import UserList
from typing import TypeVar, Optional
from dataclasses import dataclass, field, InitVar

T = TypeVar("T")
Point = list[float]
Polygon = list[list[Point]]


def polylabel(polygon: Polygon, precision: float = 1.0) -> PointWithDistance:
    """
    Finds the pole of inaccessibility, the most distant
    internal point from the polygon outline.
    """
    # find bounding box
    first_item = polygon[0][0]
    max_x, max_y = min_x, min_y = first_item

    for x, y in polygon[0]:
        if x < min_x:
            min_x = x
        elif x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        elif y > max_y:
            max_y = y

    width = max_x - min_x
    height = max_y - min_y
    cell_size = min(width, height)
    h = cell_size / 2.0

    if cell_size == 0:
        return PointWithDistance([min_x, min_y])

    # a priority queue of cells
    cell_queue: list[Cell] = []

    # cover polygon with initial cells
    x = min_x
    while x < max_x:
        y = min_y
        while y < max_y:
            heapq.heappush(cell_queue, Cell(x + h, y + h, h, polygon))
            y += cell_size
        x += cell_size

    # take centroid as the first best guess
    best_cell = _get_centroid_cell(polygon)

    # second guess: bounding box centroid
    bbox_cell = Cell(min_x + width / 2, min_y + height / 2, 0, polygon)
    if bbox_cell.d > best_cell.d:
        best_cell = bbox_cell

    num_of_probes = len(cell_queue)

    while len(cell_queue) != 0:
        # pick the most promising cell from the queue
        cell = heapq.heappop(cell_queue)

        # update the best cell if we found a better one
        if cell.d > best_cell.d:
            best_cell = cell

        # do not drill down further if there's no chance of a better solution
        if cell.max - best_cell.d <= precision:
            continue

        # split the cell into four cells
        h = cell.h / 2
        heapq.heappush(cell_queue, Cell(cell.x - h, cell.y - h, h, polygon))
        heapq.heappush(cell_queue, Cell(cell.x + h, cell.y - h, h, polygon))
        heapq.heappush(cell_queue, Cell(cell.x - h, cell.y + h, h, polygon))
        heapq.heappush(cell_queue, Cell(cell.x + h, cell.y + h, h, polygon))
        num_of_probes += 4

    return PointWithDistance([best_cell.x, best_cell.y], dist=best_cell.d)


class PointWithDistance(UserList[T]):
    "A list representing a point, but with a distance attribute"
    def __init__(self, _list: list[T], *, dist: Optional[float] = None) -> None:
        super().__init__(initlist=_list)
        self.distance = dist


@dataclass(order=True)
class Cell:
    x: float = field(compare=False)
    y: float = field(compare=False)
    h: float = field(compare=False)
    d: float = field(init=False, compare=False, repr=False)
    max: float = field(init=False, compare=False)
    polygon: InitVar[Polygon]
    preference: float = field(init=False, repr=False)

    def __post_init__(self, polygon: Polygon):
        self.d = _point_to_polygon_distance(self.x, self.y, polygon)
        self.max = self.d + self.h * math.sqrt(2)
        self.preference = -self.max


def _point_to_polygon_distance(x, y, polygon: Polygon):
    inside = False
    min_dist_sq = math.inf

    for ring in polygon:
        bx, by = ring[-1]
        for ax, ay in ring:
            if (ay > y) != (by > y) and (x < (bx - ax) * (y - ay) / (by - ay) + ax):
                inside = not inside

            min_dist_sq = min(min_dist_sq, _get_seg_dist_sq(x, y, ax, ay, bx, by))
            bx, by = ax, ay

    return math.sqrt(min_dist_sq) * (1 if inside else -1)


def _get_centroid_cell(polygon: Polygon):
    area = .0
    x = .0
    y = .0
    points = polygon[0]
    bx, by = points[-1]
    for ax, ay in points:
        f = ax * by - bx * ay
        x += (ax + bx) * f
        y += (ay + by) * f
        area += f * 3
        bx, by = ax, ay
    if area == 0:
        return Cell(points[0][0], points[0][1], 0, polygon)
    return Cell(x / area, y / area, 0, polygon)


def _get_seg_dist_sq(px, py, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay

    if dx != 0 or dy != 0:
        t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)

        if t > 1:
            ax = bx
            ay = by
        elif t > 0:
            ax += dx * t
            ay += dy * t

    dx = px - ax
    dy = py - ay
    return dx * dx + dy * dy
