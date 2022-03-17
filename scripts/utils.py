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


# functions to return distance between two hermitian matrices
def distance(
    a_herm, b_herm
):  # This function computes the symmetric space distance between two Hermitian matrices A,B
    """
    Function to compute the symmetric space distance between two hermitian matrices a_herm and b_herm
    @param a_herm:
    @param b_herm:
    @return:
    """
    w, v = la.eig(
        a_herm
    )  # Find the eigenvalues w, and a NxN matrix of Id-orthonormal eigenvctors (columns are the vectors)
    for i in range(0, len(a_herm)):
        for j in range(0, len(a_herm)):
            v[i, j] = (
                np.sqrt(1 / w[j]) * v[i, j]
            )  # The matrix v now has as rows an A-ONB

    b_tilde = np.matmul(
        np.matmul(v.getH(), b_herm), v
    )  # this is inner product defined by B in the A-ONB
    w_tilde, v_tilde = la.eig(b_tilde)  # now find eigenvalues of B_tilde

    d = 0
    for i in range(0, len(a_herm)):
        d += (
            np.log(w_tilde[i].real)
        ) ** 2  # distance is sum of squares of log of eigenvalues of B_tilde

    return np.sqrt(d)


def induced_sl_n(n, a, b, c, d):
    """
    This code computes the induced N representation of SL_2 [[a,b],[c,d]]
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
                    coeff = coeff + (
                        scipy.special.comb(n - j, k)
                        * scipy.special.comb(j - 1, i - 1 - k)
                        * (b ** k)
                        * (a ** (n - j - k))
                        * (c ** (j - i + k))
                        * (d ** (i - 1 - k))
                    )

            m[i - 1, j - 1] = coeff
    return m


def conj(x, a_herm):  # This function
    """
    Function to 'Rotate' the shape descriptor by the element of SL(N,C) induced by SL(2,C)
    @param x:
    @param a_herm:
    @return:
    """
    n = len(a_herm)
    a = complex(x[0], x[1])
    b = complex(x[2], x[3])
    c = complex(x[4], x[5])
    d = (
        1 + (b * c)
    ) / a  # Once we know a,b,c we know d as the matrix [[a,b],[c,d]] has determinant 1

    s = np.matrix(induced_sl_n(n, a, b, c, d))
    s_ast = s.getH()  # conjugate transpose

    return x[6] * np.matmul(
        np.matmul(s_ast, a_herm), s
    )  # we also allow for multiplication by our scale factor x[6]


def diff_fun_2(x, a_herm, b_herm):
    """
    function that finds the distance between the 'rotation' of a_herm and b_herm
    @param x:
    @param a_herm:
    @param b_herm:
    @return:
    """
    return distance(conj(x, a_herm), b_herm)


def get_score(query, test, query_id=None, test_id=None):
    """
    find distances using optimize toolbox, return score between 0 and 1 (by normalising distance)
    @param query:
    @param test:
    @param query_id: optional
    @param test_id: optional
    @return:
    """

    if query_id is not None:
        if query_id == test_id:
            return "self"  # marker for self comparison
        x0 = np.array([1, 0, 0, 0, 0, 0, 1])  # identity rotation array
        res = minimize(diff_fun_2, x0, method="BFGS", args=(query, test))
        x0 = res.x
        return round(
            (1 / (1 + distance(conj(x0, query), test))), 3
        )  # get score between matrices
    else:
        x0 = np.array([1, 0, 0, 0, 0, 0, 1])  # identity rotation array
        res = minimize(Diff_fun_2, x0, method="BFGS", args=(query, test[i]))
        x0 = res.x
        return round(
            (1 / (1 + distance(conj(x0, query), test))), 3
        )  # get score between matrices
