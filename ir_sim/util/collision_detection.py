import numpy as np
import cvxpy as cp
from math import sqrt, pi, cos, sin


# collision detection (True is collision)

# circle:  x, y, r
# segment: [point1, point2]
# point: x, y
# point_set: 2*n, matrix
# rectangle:


def collision_cir_cir(circle1, circle2):
    """ Collision detection between two circles """
    dis = sqrt((circle2.x - circle1.x) ** 2 + (circle2.y - circle1.y) ** 2)

    # with itself the distance is equal to zero
    if 0 < dis <= circle1.r + circle2.r:
        return True

    return False


def collision_cir_matrix(circle, matrix, reso, offset=np.zeros(2,)):
    """ Check collision with map """
    if matrix is None:
        return False

    rad_step = 0.1
    cur_rad = 0

    while cur_rad <= 2 * pi:
        # the point on the edge of the circle
        crx = circle.x + circle.r * cos(cur_rad)
        cry = circle.y + circle.r * sin(cur_rad)
        cur_rad = cur_rad + rad_step
        index_x = int((crx - offset[0]) / reso)
        index_y = int((cry - offset[1]) / reso)
        if matrix[index_x, index_y]:
            return True


def collision_cir_seg(circle, segment):
    """ Collision detection between circle and segment """
    # reference: https://stackoverflow.com/questions/1073336/circle-line-segment-collision-detection-algorithm

    point = np.array([circle.x, circle.y])  # center of a circle
    sp = np.array([segment[0].x, segment[0].y])  # start_point of segment
    ep = np.array([segment[1].x, segment[1].y])  # end_point of segment

    l2 = (ep - sp) @ (ep - sp)  # vector dot product

    if l2 == 0.0:  # convert to a point
        distance = np.linalg.norm(point - sp)  # L2-norm
        if distance < circle.r:
            return True
    # cos(\theta)
    t = max(0, min(1, ((point - sp) @ (ep - sp)) / l2))

    # the project point
    projection = sp + t * (ep - sp)
    relative = projection - point

    distance = np.linalg.norm(relative)  # the distance between the center of circle and segment
    if distance < circle.r:
        return True


def collision_seg_matrix(segment, matrix, reso, offset=np.zeros(2,)):
    """ Check collision with map for segment """
    if matrix is None:
        return False

    init_point = segment[0]
    dif_x = segment[1].x - segment[0].x
    dif_y = segment[1].y - segment[0].y

    len_seg = sqrt(dif_x ** 2 + dif_y ** 2)

    slope_cos = dif_x / len_seg
    slope_sin = dif_y / len_seg

    point_step = 2 * reso
    cur_len = 0

    while cur_len <= len_seg:

        cur_point_x = init_point.x + cur_len * slope_cos
        cur_point_y = init_point.y + cur_len * slope_sin

        cur_len = cur_len + point_step

        index_x = int((cur_point_x - offset[0]) / reso)
        index_y = int((cur_point_y - offset[1]) / reso)

        # must in reason scope
        if index_x < 0 or index_x > matrix.shape[0] or index_y < 0 or index_y > matrix.shape[1]:
            return True

        if matrix[index_x, index_y]:
            return True


def collision_circle_point(circle, point_set):
    """ Collision detection between circle and point_set """
    assert point_set.shape[0] == 2  # the shape of point_set is (2, n)

    center = np.array([[circle.x], [circle.y]])  # 2*1
    temp = point_set - center

    dis_set = np.linalg.norm(temp, axis=0)

    min_dis = np.min(dis_set)

    return min_dis < circle.r


def collision_rect_point(rectangle, point_set):
    """ Collision detection between rectangle and point_set """
    polytope = rectangle
    if isinstance(rectangle, list):
        polytope = np.array(rectangle).T

    ver_num = polytope.shape[1]
    A, b = gen_matrix(ver_num, polytope)  # polytope in size (2, n)

    assert point_set.shape[0] == 2  # point_set in size (2, n)
    point_num = point_set.shape[1]
    for i in range(point_num):
        if A @ point_num[:, i] <= b:
            return True
    return False


# refer to https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
# refer to https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
def collision_seg_seg(segment1, segment2):
    """ Collision detection between two segments """
    p1, q1 = segment1[0], segment1[1]
    p2, q2 = segment2[0], segment2[1]

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # general case
    if o1 != o2 and o3 != o4:
        return True

    # special case
    # p1, q1 and p2 are collinear and p2 lies on segment p1q1
    if o1 == 0 and onSegment(p1, p2, q1):
        return True

    # p1, q1 and q2 are collinear and q2 lies on segment p1q1
    if o2 == 0 and onSegment(p1, q2, q1):
        return True

    # p2, q2 and p1 are collinear and p1 lies on segment p2q2
    if o3 == 0 and onSegment(p2, p1, q2):
        return True

    # p2, q2 and q1 are collinear and q1 lies on segment p2q2
    if o4 == 0 and onSegment(p2, q1, q2):
        return True

    return False


def onSegment(p, q, r):
    """ Given three collinear points p, q, r, the function checks if point q lies in line segment 'p-->r',True is in """
    if max(p.x, r.x) >= q.x >= min(p.x, r.x) and max(p.y, r.y) >= q.y >= min(p.y, r.y):
        return True

    return False


def orientation(p, q, r):
    """ To find orientation of ordered triplet (p, q, r), return following values 0 1 2 """
    # refer to https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
    # 0 collinear
    # 1 counterclockwise
    # 2 clockwise

    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))

    if val > 0:
        return 1
    elif val < 0:
        return 2
    else:
        return 0


def gen_matrix(ver_num, vertexes):
    """ Get the matrix Ax <= b for polytope, anti-clockwise """
    # vertexes given in (2, n), list in (n, 2)
    A = np.zeros((ver_num, 2))  # n * 2
    B = np.zeros((ver_num, 1))  # n * 1
    B_collision = np.zeros((ver_num, 1))  # n * 1

    for i in range(ver_num):
        if i + 1 < ver_num:
            pre_point = vertexes[:, i]
            next_point = vertexes[:, i + 1]
        else:
            pre_point = vertexes[:, i]
            next_point = vertexes[:, 0]

        diff = next_point - pre_point

        a = diff[1]
        b = -diff[0]
        c = a * pre_point[0] + b * pre_point[1]

        A[i, 0] = a
        A[i, 1] = b
        B[i, 0] = c

        if b != 0:
            B_collision[i, 0] = c + 0.01 * abs(b)
        else:
            B_collision[i, 0] = c + 0.01 * abs(a)

    return A, B


def collision_polygon_polygon(polygon1_vertexes, polygon2_vertexes, min_distance=1e-8):
    """ Check collision between polygons """
    # polygon1, convert vertex to (2, n) shape
    if isinstance(polygon1_vertexes, list):
        polygon1_vertexes = np.array(polygon1_vertexes, ndmin=2).T
    polygon1_num = polygon1_vertexes.shape[1]
    A1, b1 = gen_matrix(polygon1_num, polygon1_vertexes)

    # polygon2
    if isinstance(polygon2_vertexes, list):
        polygon2_vertexes = np.array(polygon2_vertexes, ndmin=2).T
    polygon2_num = polygon2_vertexes.shape[1]
    A2, b2 = gen_matrix(polygon2_num, polygon2_vertexes)

    # optimization variable
    x = cp.Variable((2, 1))
    y = cp.Variable((2, 1))

    # construct the optimization problem
    obj = cp.Minimize(cp.norm(y - x))
    cons = [A1 @ x <= b1, A2 @ y <= b2]
    prob = cp.Problem(obj, cons)

    # solve the problem
    optimal_value = prob.solve()
    if optimal_value <= min_distance:
        return True
    return False


def collision_polygon_circle(polygon_vertexes, circle, min_distance=0.001):
    """ Check collision between polygon and circle """
    if isinstance(polygon_vertexes, list):
        polygon_vertexes = np.array(polygon_vertexes, ndmin=2).T
    polygon_num = polygon_vertexes.shape[1]
    A, b = gen_matrix(polygon_num, polygon_vertexes)

    # optimization variable
    x = cp.Variable((2, 1))
    y = cp.Variable((2, 1))

    y_center = cp.Parameter((2, 1))
    y_center.value = np.array([circle.x, circle.y], ndmin=2).T
    radius = cp.Parameter(nonneg=True)
    radius.value = circle.r

    # construct the optimization
    obj = cp.Minimize(cp.norm(y - x))
    cons = [A @ x <= b, cp.norm(y - y_center) <= radius]
    prob = cp.Problem(obj, cons)

    # solve
    optimal_value = prob.solve()
    if optimal_value <= min_distance:
        return True
    return False
