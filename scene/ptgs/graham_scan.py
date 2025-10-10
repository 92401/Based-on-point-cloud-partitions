

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from shapely.geometry import Polygon, box
from scipy.spatial import ConvexHull


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def distance(a, b):
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def cross_product(a, b, c):  #ab和ac的叉乘，用来判断3个点的位置关系  如果结果大于0，说明三个点是逆时针方向排列。如果结果小于0，说明三个点是顺时针方向排列。如果结果等于0，说明三个点共线
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)


def compare_angles(pivot, p1, p2):  #用于计算p1，p2相对于第一个变量的排列顺序
    orientation = cross_product(pivot, p1, p2)
    if orientation == 0:
        return distance(pivot, p1) - distance(pivot, p2)
    return -1 if orientation > 0 else 1


def graham_scan(points):  #接收点返回该点集的凸包
    n = len(points)
    if n < 3:
        return "凸包需要至少3个点"

    pivot = min(points, key=lambda point: (point.y, point.x))  #根据y坐标排列，取出最小点
    points = sorted(points, key=lambda point: (np.arctan2(point.y - pivot.y, point.x - pivot.x), -point.y, point.x))   #对points进行排序

    stack = [points[0], points[1], points[2]]
    for i in range(3, n):
        while len(stack) > 1 and compare_angles(stack[-2], stack[-1], points[i]) > 0:
            stack.pop()
        stack.append(points[i])

    return stack   #stack 中保存的是凸包的顶点，按照逆时针顺序排列。


def plot_convex_hull(points, convex_hull, x, y):  #绘制凸包包围点的示意图
    plt.figure()
    plt.scatter([p.x for p in points], [p.y for p in points], color='b', label="所有点")

    # 绘制凸包
    plt.plot([p.x for p in convex_hull] + [convex_hull[0].x], [p.y for p in convex_hull] + [convex_hull[0].y],
             linestyle='-', color='g', label="篱笆边")

    for i in range(len(convex_hull)):
        plt.plot([convex_hull[i].x, convex_hull[(i + 1) % len(convex_hull)].x],
                 [convex_hull[i].y, convex_hull[(i + 1) % len(convex_hull)].y], linestyle='-', color='g')

    plt.plot(x, y)

    plt.show()


def run_graham_scan(points, W, H):
    """
    获取点集围成的区域的凸包
    :param points: 投影后的二维坐标点集
    :param W: 图像宽度
    :param H: 图像高度
    :return: 凸包的相关信息
    """
    points = np.array(points)

    # 确保点集是二维的，且至少有3个点
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        print(f"Invalid input: points shape is {points.shape}")
        return {
            "intersection_area": 0,
            "image_area": W * H,
            "intersection_rate": 0,
        }

    try:
        convex_hull = ConvexHull(points)
        convex_hull_list = points[convex_hull.vertices]
        convex_hull_polygon = Polygon(convex_hull_list)
        image_bounds = box(0, 0, W, H)

        intersection = convex_hull_polygon.intersection(image_bounds)
        image_area = W * H
        intersection_rate = intersection.area / image_area

        return {
            "intersection_area": intersection.area,
            "image_area": image_area,
            "intersection_rate": intersection_rate,
        }
    except Exception as e:
        print(f"Error in run_graham_scan: {e}")
        return {
            "intersection_area": 0,
            "image_area": W * H,
            "intersection_rate": 0,
        }

