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

def distance(a_herm, b_herm):  # This
    """
    function to compute the symmetric space distance between two Hermitian matrices a_herm and b_herm
    @param a_herm:
    @param b_herm:
    @return:
    """
    q = np.asmatrix(la.cholesky(b_herm))
    q_t = q.getH()

    q_inv = la.inv(q)
    q_inv_t = la.inv(q_t)

    c = np.matmul(np.matmul(q_inv, a_herm), q_inv_t)

    w, v = la.eig(c)

    d = 0
    for i in range(0, len(w)):
        d += (np.log(w[i])) ** 2

    return np.sqrt(d).real


def induced_sl_n(n, a, b, c, d):
    """
    compute the induced N representation of SL_2 [[a,b],[c,d]]
    @param n:
    @param a:
    @param b:
    @param c:
    @param d:
    @return:
    """
    m = np.zeros((n, n), dtype=complex)

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            coeff = 0
            for k in range(0, i):
                if 0 <= i - 1 - k <= j - 1 and k <= n - j:
                    coeff = coeff + (scipy.special.comb(n - j, k) * scipy.special.comb(j - 1, i - 1 - k) * (b ** k) * (
                                a ** (n - j - k)) * (c ** (j - i + k)) * (d ** (i - 1 - k)))

            m[i - 1, j - 1] = coeff
    return m


def conj(x, a_herm):
    """
    function 'Rotates' the shape descriptor by the element of SL(N,C) induced by SL(2,C)
    @param x:
    @param a_herm:
    @return:
    """
    n = len(a_herm)
    a = complex(x[0], x[1])
    b = complex(x[2], x[3])
    c = complex(x[4], x[5])
    d = (1 + (b * c)) / a  # Once we know a,b,c we know d as the matrix [[a,b],[c,d]] has determinant 1

    s = np.matrix(induced_sl_n(n, a, b, c, d))
    s_ast = s.getH()  # conjugate transpose

    # we also allow for multiplication by our scale factor exp(x[6])
    return np.exp(x[6]) * np.matmul(np.matmul(s_ast, a_herm), s)


def diff_fun_2(x, a_herm, b_herm):
    """
    function that finds the distance between the 'rotation' of A and B
    @param x:
    @param a_herm:
    @param b_herm:
    @return:
    """
    return distance(conj(x, a_herm), b_herm)


def get_score(query_vals, test_vals, query_id=None, test_id=None, k_quant=None):
    """
    find distances using optimize toolbox, return score between 0 and 1 (by normalising distance)
    @param k_quant:
    @param query_vals:
    @param test_vals:
    @param query_id: optional
    @param test_id: optional
    @return:
    """

    # unpack named tuples
    query = query_vals.kq_shape
    query_area = query_vals.surface_area
    test = test_vals.kq_shape
    test_area = test_vals.surface_area

    # set default value for k_quant
    if k_quant is None:
        k_quant = 1

    # set scale factor
    fac = k_quant ** (-3 / 2)

    # get area contribution
    if query_area >= test_area:
        area_diff = test_area / query_area
    else:
        area_diff = query_area / test_area

    # if query id provided, check for self comparison
    if query_id is not None:
        if query_id == test_id:
            return "self"  # marker for self comparison
        x0 = np.array([1, 0, 0, 0, 0, 0, 1])  # identity rotation array
        res = minimize(diff_fun_2, x0, method="BFGS", args=(query, test))
        x0 = res.x
        # get score between matrices
        dist = (fac * distance(conj(x0, query), test))
        shape_diff = (1 / (1 + dist))
    else:
        x0 = np.array([1, 0, 0, 0, 0, 0, 1])  # identity rotation array
        res = minimize(diff_fun_2, x0, method="BFGS", args=(query, test))
        x0 = res.x
        # get score between matrices
        dist = (fac * distance(conj(x0, query), test))
        shape_diff = (1 / (1 + dist))

    sim_score = round(((0.3 * area_diff) + (0.7 * shape_diff)), 3)

    return dist, sim_score
