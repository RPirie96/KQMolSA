"""
script of helper functions

"""
import numpy as np
from numpy import linalg as la
import math

import scipy
from scipy.optimize import minimize


def get_chain(no_atoms, level_mat, adjacency_matrix, sphere, level):
    """
    Inputs the level matrix, Intersection matrix and writes the path from base sphere
    @param no_atoms:
    @param level_mat:
    @param adjacency_matrix:
    @param sphere:
    @param level:
    @return: chain - path through molecule from base sphere
    """
    chain = np.zeros(
        level + 1, dtype=int
    )  # whatever the level is we will output a vector of length L+1
    i = level
    current_sphere = sphere
    chain[level] = sphere

    while i > 0:
        for k in range(0, no_atoms):
            # if there is a lower level non-t sphere and it meets the current sphere
            if level_mat[i - 1][k] == 1 and adjacency_matrix[current_sphere][k] == 1:
                current_sphere = k
                chain[i - 1] = k
        i = i - 1

    return chain


def get_m_rot(vector):
    """
    the element of SO(3) that carries out the rotation
    @param vector:
    @return: m_rot
    """
    m_rot = np.array(
        [
            [
                1 - ((vector[0] ** 2) / (1 + vector[2])),
                -vector[0] * vector[1] / (1 + vector[2]),
                vector[0],
            ],
            [
                -vector[0] * vector[1] / (1 + vector[2]),
                1 - ((vector[1] ** 2) / (1 + vector[2])),
                vector[1],
            ],
            [-vector[0], -vector[1], vector[2]],
        ]
    )

    return m_rot


def t_circle(alpha, beta, gamma, delta, c, r_rel):
    """
    function that computes the centre and radius of the image of a circle under the Mobius transformation
    z-->(az+b)/(cz+d)
    @param alpha:
    @param beta:
    @param gamma:
    @param delta:
    @param c:
    @param r_rel:
    @return: [cent, Radius]
    """
    cent = (
        ((beta + (c * alpha)) * np.conj(delta + (c * gamma)))
        - (r_rel * r_rel * alpha * np.conj(gamma))
    ) / ((abs(delta + (c * gamma)) ** 2) - (r_rel * abs(gamma)) ** 2)
    k = (((r_rel * abs(alpha)) ** 2) - (abs(beta + (c * alpha))) ** 2) / (
        (abs(delta + (c * gamma)) ** 2) - (r_rel * abs(gamma)) ** 2
    )
    radius = math.sqrt(abs(k + abs(cent) ** 2))

    return [cent, radius]


def new_coeff(a, b, c, alpha, beta, gamma, delta):
    """
    function to keep track of how the volume form changes under pullback by (az+b)/(cz+d)

    @param a:
    @param b:
    @param c:
    @param alpha:
    @param beta:
    @param gamma:
    @param delta:
    @return: updated coefficients post-mobius map
    """
    k_2 = abs(alpha - a * gamma) ** 2 + b * abs(gamma) ** 2
    k_1 = np.conj(
        (((alpha - a * gamma) * np.conj(beta - a * delta)) + b * gamma * np.conj(delta))
    )
    k_0 = abs(beta - a * delta) ** 2 + b * abs(delta) ** 2

    return [-k_1 / k_2, (k_0 / k_2) - abs(k_1 / k_2) ** 2, c / (k_2 ** 2)]


# functions to get distance between two hermitian matrices

def kob_dist(a_herm, b_herm):
    """
    #Function that computes the distance between A and B
    must input the inverse of the first entry
    @param a_herm:
    @param b_herm:
    @return:
    """
    c_herm = np.matmul(a_herm, b_herm)
    w = la.eigvals(c_herm)
    d = 0
    for i in range(0, len(w)):
        d += (np.log(w[i])) ** 2

    return d.real


def scalar_min(a_herm, b_herm):
    """
    function computes the optimal scaling for the pair of matrices
    @param a_herm:
    @param b_herm:
    @return:
    """
    c_herm = np.matmul(la.inv(a_herm), b_herm)
    w = la.eigvals(c_herm)
    d = 0
    for i in range(0, len(w)):
        d += (np.log(w[i]))
    l = -(d/len(w))

    return l.real


def conj_1(x, a_herm):
    """
    Hard coded conjugation by SL(2,C) for k=1
    @param x:
    @param a_herm:
    @return:
    """
    a = complex(x[0], x[1])
    b = complex(x[2], x[3])
    c = complex(x[4], x[5])
    d = (1 + (b * c)) / a

    a_b = complex(x[0], -x[1])
    b_b = complex(x[2], -x[3])
    c_b = complex(x[4], -x[5])
    d_b = (1 + (b_b * c_b)) / a_b

    s = np.matrix([[a ** 2, a * c, c ** 2], [2 * a * b, (a * d + b * c), 2 * c * d], [b ** 2, b * d, d ** 2]])
    s_ast = np.matrix([[a_b ** 2, 2 * a_b * b_b, b_b ** 2], [a_b * c_b, (a_b * d_b + b_b * c_b), b_b * d_b],
                       [c_b ** 2, 2 * c_b * d_b, d_b ** 2]])

    return np.matmul(np.matmul(s_ast, a_herm), s)


def conj_2(x, a_herm):
    """
    Hard coded conjugation by SL(2,C) for k=2
    @param x:
    @param a_herm:
    @return:
    """
    a = complex(x[0], x[1])
    b = complex(x[2], x[3])
    c = complex(x[4], x[5])
    d = (1 + (b * c)) / a

    a_b = complex(x[0], -x[1])
    b_b = complex(x[2], -x[3])
    c_b = complex(x[4], -x[5])
    d_b = (1 + (b_b * c_b)) / a_b

    s = np.matrix([
        [a ** 4, a ** 3 * c, a ** 2 * c ** 2, a * c ** 3, c ** 4],
        [4 * a ** 3 * b, (a ** 3 * d + 3 * a ** 2 * b * c), (2 * a ** 2 * c * d + 2 * a * b * c ** 2),
         (3 * a * c ** 2 * d + b * c ** 3), 4 * c ** 3 * d],
        [6 * a ** 2 * b ** 2, (3 * a ** 2 * b * d + 3 * a * b ** 2 * c),
         (4 * a * b * c * d + a ** 2 * d ** 2 + b ** 2 * c ** 2), (3 * a * c * d ** 2 + 3 * b * c ** 2 * d),
         6 * c ** 2 * d ** 2],
        [4 * a * b ** 3, (b ** 3 * c + 3 * a * b ** 2 * d), (2 * b ** 2 * c * d + 2 * a * b * d ** 2),
         (3 * b * c * d ** 2 + a * d ** 3), 4 * c * d ** 3],
        [b ** 4, b ** 3 * d, b ** 2 * d ** 2, b * d ** 3, d ** 4]
    ])

    s_ast = np.matrix([
        [a_b ** 4, 4 * a_b ** 3 * b_b, 6 * a_b ** 2 * b_b ** 2, 4 * a_b * b_b ** 3, b_b ** 4],
        [a_b ** 3 * c_b, (a_b ** 3 * d_b + 3 * a_b ** 2 * b_b * c_b),
         (3 * a_b ** 2 * b_b * d_b + 3 * a_b * b_b ** 2 * c_b), (b_b ** 3 * c_b + 3 * a_b * b_b ** 2 * d_b),
         b_b ** 3 * d_b],
        [a_b ** 2 * c_b ** 2, (2 * a_b ** 2 * c_b * d_b + 2 * a_b * b_b * c_b ** 2),
         (4 * a_b * b_b * c_b * d_b + a_b ** 2 * d_b ** 2 + b_b ** 2 * c_b ** 2),
         (2 * b_b ** 2 * c_b * d_b + 2 * a_b * b_b * d_b ** 2), b_b ** 2 * d_b ** 2],
        [a_b * c_b ** 3, (3 * a_b * c_b ** 2 * d_b + b_b * c_b ** 3),
         (3 * a_b * c_b * d_b ** 2 + 3 * b_b * c_b ** 2 * d_b), (3 * b_b * c_b * d_b ** 2 + a_b * d_b ** 3),
         b_b * d_b ** 3],
        [c_b ** 4, 4 * c_b ** 3 * d_b, 6 * c_b ** 2 * d_b ** 2, 4 * c_b * d_b ** 3, d_b ** 4]
    ])

    return np.matmul(np.matmul(s_ast, a_herm), s)


def diff_fun(x, a_herm, b_herm):
    """
    function that finds the distance between the 'rotation' of A and B k=1
    @param x:
    @param a_herm:
    @param b_herm:
    @return:
    """
    a = complex(x[0], x[1])
    b = complex(x[2], x[3])
    c = complex(x[4], x[5])
    d = (1 + (b * c)) / a

    a_b = complex(x[0], -x[1])
    b_b = complex(x[2], -x[3])
    c_b = complex(x[4], -x[5])
    d_b = (1 + (b_b * c_b)) / a_b

    s = np.matrix([[a ** 2, a * c, c ** 2], [2 * a * b, (a * d + b * c), 2 * c * d], [b ** 2, b * d, d ** 2]])
    s_ast = np.matrix([[a_b ** 2, 2 * a_b * b_b, b_b ** 2], [a_b * c_b, (a_b * d_b + b_b * c_b), b_b * d_b],
                       [c_b ** 2, 2 * c_b * d_b, d_b ** 2]])

    c_herm = np.matmul(a_herm, np.matmul(np.matmul(s_ast, b_herm), s))

    w = la.eigvals(c_herm)
    dist = 0
    for i in range(0, len(w)):
        dist += (np.log(w[i])) ** 2

    return dist.real


def conj_2(x, a_herm):  # Hard coded conjugation by SL(2,C) for k=2
    a = complex(x[0], x[1])
    b = complex(x[2], x[3])
    c = complex(x[4], x[5])
    d = (1 + (b * c)) / a

    a_b = complex(x[0], -x[1])
    b_b = complex(x[2], -x[3])
    c_b = complex(x[4], -x[5])
    d_b = (1 + (b_b * c_b)) / a_b

    s = np.matrix([
        [a ** 4, a ** 3 * c, a ** 2 * c ** 2, a * c ** 3, c ** 4],
        [4 * a ** 3 * b, (a ** 3 * d + 3 * a ** 2 * b * c), (2 * a ** 2 * c * d + 2 * a * b * c ** 2),
         (3 * a * c ** 2 * d + b * c ** 3), 4 * c ** 3 * d],
        [6 * a ** 2 * b ** 2, (3 * a ** 2 * b * d + 3 * a * b ** 2 * c),
         (4 * a * b * c * d + a ** 2 * d ** 2 + b ** 2 * c ** 2), (3 * a * c * d ** 2 + 3 * b * c ** 2 * d),
         6 * c ** 2 * d ** 2],
        [4 * a * b ** 3, (b ** 3 * c + 3 * a * b ** 2 * d), (2 * b ** 2 * c * d + 2 * a * b * d ** 2),
         (3 * b * c * d ** 2 + a * d ** 3), 4 * c * d ** 3],
        [b ** 4, b ** 3 * d, b ** 2 * d ** 2, b * d ** 3, d ** 4]
    ])

    s_ast = np.matrix([
        [a_b ** 4, 4 * a_b ** 3 * b_b, 6 * a_b ** 2 * b_b ** 2, 4 * a_b * b_b ** 3, b_b ** 4],
        [a_b ** 3 * c_b, (a_b ** 3 * d_b + 3 * a_b ** 2 * b_b * c_b),
         (3 * a_b ** 2 * b_b * d_b + 3 * a_b * b_b ** 2 * c_b), (b_b ** 3 * c_b + 3 * a_b * b_b ** 2 * d_b),
         b_b ** 3 * d_b],
        [a_b ** 2 * c_b ** 2, (2 * a_b ** 2 * c_b * d_b + 2 * a_b * b_b * c_b ** 2),
         (4 * a_b * b_b * c_b * d_b + a_b ** 2 * d_b ** 2 + b_b ** 2 * c_b ** 2),
         (2 * b_b ** 2 * c_b * d_b + 2 * a_b * b_b * d_b ** 2), b_b ** 2 * d_b ** 2],
        [a_b * c_b ** 3, (3 * a_b * c_b ** 2 * d_b + b_b * c_b ** 3),
         (3 * a_b * c_b * d_b ** 2 + 3 * b_b * c_b ** 2 * d_b), (3 * b_b * c_b * d_b ** 2 + a_b * d_b ** 3),
         b_b * d_b ** 3],
        [c_b ** 4, 4 * c_b ** 3 * d_b, 6 * c_b ** 2 * d_b ** 2, 4 * c_b * d_b ** 3, d_b ** 4]
    ])

    return np.matmul(np.matmul(s_ast, a_herm), s)


def diff_fun_2(x, a_herm, b_herm):  # function that finds the distance between the 'rotation' of A and B k=2

    a = complex(x[0], x[1])
    b = complex(x[2], x[3])
    c = complex(x[4], x[5])
    d = (1 + (b * c)) / a

    a_b = complex(x[0], -x[1])
    b_b = complex(x[2], -x[3])
    c_b = complex(x[4], -x[5])
    d_b = (1 + (b_b * c_b)) / a_b

    s = np.matrix([
        [a ** 4, a ** 3 * c, a ** 2 * c ** 2, a * c ** 3, c ** 4],
        [4 * a ** 3 * b, (a ** 3 * d + 3 * a ** 2 * b * c), (2 * a ** 2 * c * d + 2 * a * b * c ** 2),
         (3 * a * c ** 2 * d + b * c ** 3), 4 * c ** 3 * d],
        [6 * a ** 2 * b ** 2, (3 * a ** 2 * b * d + 3 * a * b ** 2 * c),
         (4 * a * b * c * d + a ** 2 * d ** 2 + b ** 2 * c ** 2), (3 * a * c * d ** 2 + 3 * b * c ** 2 * d),
         6 * c ** 2 * d ** 2],
        [4 * a * b ** 3, (b ** 3 * c + 3 * a * b ** 2 * d), (2 * b ** 2 * c * d + 2 * a * b * d ** 2),
         (3 * b * c * d ** 2 + a * d ** 3), 4 * c * d ** 3],
        [b ** 4, b ** 3 * d, b ** 2 * d ** 2, b * d ** 3, d ** 4]
    ])

    s_ast = np.matrix([
        [a_b ** 4, 4 * a_b ** 3 * b_b, 6 * a_b ** 2 * b_b ** 2, 4 * a_b * b_b ** 3, b_b ** 4],
        [a_b ** 3 * c_b, (a_b ** 3 * d_b + 3 * a_b ** 2 * b_b * c_b),
         (3 * a_b ** 2 * b_b * d_b + 3 * a_b * b_b ** 2 * c_b), (b_b ** 3 * c_b + 3 * a_b * b_b ** 2 * d_b),
         b_b ** 3 * d_b],
        [a_b ** 2 * c_b ** 2, (2 * a_b ** 2 * c_b * d_b + 2 * a_b * b_b * c_b ** 2),
         (4 * a_b * b_b * c_b * d_b + a_b ** 2 * d_b ** 2 + b_b ** 2 * c_b ** 2),
         (2 * b_b ** 2 * c_b * d_b + 2 * a_b * b_b * d_b ** 2), b_b ** 2 * d_b ** 2],
        [a_b * c_b ** 3, (3 * a_b * c_b ** 2 * d_b + b_b * c_b ** 3),
         (3 * a_b * c_b * d_b ** 2 + 3 * b_b * c_b ** 2 * d_b), (3 * b_b * c_b * d_b ** 2 + a_b * d_b ** 3),
         b_b * d_b ** 3],
        [c_b ** 4, 4 * c_b ** 3 * d_b, 6 * c_b ** 2 * d_b ** 2, 4 * c_b * d_b ** 3, d_b ** 4]
    ])

    c_herm = np.matmul(a_herm, np.matmul(np.matmul(s_ast, b_herm), s))

    w = la.eigvals(c_herm)
    dist = 0
    for i in range(0, len(w)):
        dist += (np.log(w[i])) ** 2

    return dist.real


def dist_k1(query_des, test_des):
    x0 = np.array([1, 0, 0, 0, 0, 0, 0])  # identity rotation array
    k_quant = 1
    query_des = la.inv(query_des)  # use inverse of first molecule

    # set scale factor
    fac = k_quant ** (-3 / 2)

    scal = scalar_min(la.inv(query_des), test_des)

    res = minimize(diff_fun, x0, method='BFGS', args=(query_des, test_des))
    x0 = res.x

    res = minimize(diff_fun, x0, method='COBYLA', args=(query_des, test_des))
    x0 = res.x

    dist = fac * np.sqrt(kob_dist(query_des, np.exp(scal) * conj_1(x0, test_des)))

    return dist, x0


def dist_k2(query_des, test_des, x0):
    k_quant = 2
    query_des = la.inv(query_des)  # use inverse of first molecule

    # set scale factor
    fac = k_quant ** (-3 / 2)
    scal = scalar_min(la.inv(query_des), test_des)

    res = minimize(diff_fun_2, x0, method='BFGS', args=(query_des, test_des))
    x0 = res.x

    res = minimize(diff_fun_2, x0, method='Powell', args=(query_des, test_des))
    x0 = res.x

    dist = fac * np.sqrt(kob_dist(query_des, np.exp(scal) * conj_2(x0, test_des)))

    return dist, x0


def get_score(query_des, test_des, query_area, test_area, k_quant, query_id=None, test_id=None, x0=None):

    """
    find distances using optimize toolbox, return score between 0 and 1 (by normalising distance)
    @param x0:
    @param k_quant:
    @param test_area:
    @param query_area:
    @param test_des:
    @param query_des:
    @param query_id: optional
    @param test_id: optional
    @return:
    """
    # get area contribution
    if query_area >= test_area:
        area_diff = test_area / query_area
    else:
        area_diff = query_area / test_area

    # if query id provided, check for self comparison
    if query_id is not None and test_id is not None and query_id == test_id:
        return "self", "self", "self"  # marker for self comparison

    elif k_quant == 1:
        try:
            dist, x0 = dist_k1(query_des, test_des)
            shape_diff = (1 / (1 + dist))
            sim_score = round(((0.3 * area_diff) + (0.7 * shape_diff)), 3)

            return round(dist, 3), sim_score, x0

        except np.linalg.LinAlgError:
            try:
                test_des_10 = test_des * 10
                dist, x0 = dist_k1(query_des, test_des_10)
                shape_diff = (1 / (1 + dist))
                sim_score = round(((0.3 * area_diff) + (0.7 * shape_diff)), 3)

                return round(dist, 3), sim_score, x0

            except np.linalg.LinAlgError:
                try:
                    test_des_100 = test_des * 100
                    dist, x0 = dist_k1(query_des, test_des_100)
                    shape_diff = (1 / (1 + dist))
                    sim_score = round(((0.3 * area_diff) + (0.7 * shape_diff)), 3)

                    return round(dist, 3), sim_score, x0
                except np.linalg.LinAlgError:
                    try:
                        test_des_1000 = test_des * 1000
                        dist, x0 = dist_k1(query_des, test_des_1000)
                        shape_diff = (1 / (1 + dist))
                        sim_score = round(((0.3 * area_diff) + (0.7 * shape_diff)), 3)

                        return round(dist, 3), sim_score, x0
                    except np.linalg.LinAlgError:
                        return "LinAlgError", "LinAlgError", "LinAlgError"

    elif k_quant == 2:
        try:
            dist, x0 = dist_k2(query_des, test_des, x0)
            shape_diff = (1 / (1 + dist))
            sim_score = round(((0.3 * area_diff) + (0.7 * shape_diff)), 3)

            return round(dist, 3), sim_score, x0
        except np.linalg.LinAlgError:
            try:
                test_des_10 = test_des * 10
                dist, x0 = dist_k2(query_des, test_des_10, x0)
                shape_diff = (1 / (1 + dist))
                sim_score = round(((0.3 * area_diff) + (0.7 * shape_diff)), 3)

                return round(dist, 3), sim_score, x0

            except np.linalg.LinAlgError:
                try:
                    test_des_100 = test_des * 100
                    dist, x0 = dist_k2(query_des, test_des_100, x0)
                    shape_diff = (1 / (1 + dist))
                    sim_score = round(((0.3 * area_diff) + (0.7 * shape_diff)), 3)

                    return round(dist, 3), sim_score, x0
                except np.linalg.LinAlgError:
                    try:
                        test_des_1000 = test_des * 1000
                        dist, x0 = dist_k2(query_des, test_des_1000, x0)
                        shape_diff = (1 / (1 + dist))
                        sim_score = round(((0.3 * area_diff) + (0.7 * shape_diff)), 3)

                        return round(dist, 3), sim_score, x0
                    except np.linalg.LinAlgError:
                        return "LinAlgError", "LinAlgError", "LinAlgError"
