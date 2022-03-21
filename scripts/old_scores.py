"""
Module to retain old scoring functions
"""

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
        res = minimize(diff_fun_2, x0, method="BFGS", args=(query, test))
        x0 = res.x
        return round(
            (1 / (1 + distance(conj(x0, query), test))), 3
        )  # get score between matrices
