"""
Module to return the Hermitian matrix descriptor of a molecule

Functions:
- phi_core: helper function to solve the Poisson equation
- get_k_mat: function to compute the matrix descriptor of a molecule
"""

import numpy as np


def phi_core(z, d_0, d_1, d_2):
    """
    solves Poisson equation

    @param z:
    @param d_0:
    @param d_1:
    @param d_2:
    @return:
    """
    p_core = (d_0 / d_2) * np.log(abs(z - d_1) ** 2 + d_2)

    return p_core


def get_k_mat(no_atoms, sgp, sphere_levels_vec, fingerprint, no_levels, level_list, k_quant=None):
    """
    Function to compute the Hermitian matrix representation of a molecule

    @param no_atoms:
    @param sgp:
    @param sphere_levels_vec:
    @param fingerprint:
    @param no_levels:
    @param level_list:
    @param k_quant: optional, defined in code as k_quant=2 unless specified
    """

    # unpack tuples
    base_to_unit_maps = sgp.base_to_unit_maps
    internal_corr = sgp.internal_corr
    external_corr = sgp.external_corr
    a_coeff = sgp.a_coeff
    b_coeff = sgp.b_coeff
    c_coeff = sgp.c_coeff
    avoid_cent = sgp.avoid_cent
    avoid_rad = sgp.avoid_rad

    # If the Kahler potential is of the form (C/B log(|z-A|^2))+K_ijlog(|alpha_ijz+beta_ij|^2)+Const_ij
    k_mat = np.zeros((no_atoms, no_atoms), dtype=float)
    alpha_mat = np.zeros((no_atoms, no_atoms), dtype=complex)
    beta_mat = np.zeros((no_atoms, no_atoms), dtype=complex)
    const = np.zeros((no_atoms, no_atoms), dtype=float)

    for sphere_r in range(0, no_atoms):
        level_s = sphere_levels_vec[sphere_r]
        inside = []
        in_s = level_s
        while in_s > -1:
            inside.append(fingerprint[in_s][sphere_r])
            in_s += -1

        for sphere_c in range(0, no_atoms):
            # outside corrections
            k_mat[sphere_r][sphere_c] = external_corr[sphere_c]
            alpha_mat[sphere_r][sphere_c] = base_to_unit_maps[sphere_c][0][0]
            beta_mat[sphere_r][sphere_c] = base_to_unit_maps[sphere_c][0][1]

            if sphere_c in inside:
                # inside corrections
                k_mat[sphere_r][sphere_c] = -internal_corr[sphere_c]
                alpha_mat[sphere_r][sphere_c] = base_to_unit_maps[sphere_c][1][0]
                beta_mat[sphere_r][sphere_c] = base_to_unit_maps[sphere_c][1][1]

    # find the constant matrix - this is the final slightly unsatisfactory part of the Kahler potential
    for level in range(1, no_levels + 1):
        for sphere in level_list[level]:
            w = 1
            m_t = base_to_unit_maps[sphere]
            p = (m_t[1][1] * w - m_t[0][1]) / (-m_t[1][0] * w + m_t[0][0])
            correct = 0

            for s_c in range(0, no_atoms):
                correct += k_mat[sphere][s_c] * np.log(
                    abs(alpha_mat[sphere][s_c] * p + beta_mat[sphere][s_c]) ** 2
                )
            phi_1 = (
                phi_core(w, c_coeff[sphere], a_coeff[sphere], b_coeff[sphere])
                + correct
                - (c_coeff[sphere] / b_coeff[sphere])
                * np.log(abs(-m_t[1][0] * w + m_t[0][0]) ** 2)
            )

            level_s = sphere_levels_vec[sphere]
            pr_s = fingerprint[level_s - 1][sphere]
            m_t_p = base_to_unit_maps[pr_s]
            w_pr = (m_t_p[0][0] * p + m_t_p[0][1]) / (m_t_p[1][0] * p + m_t_p[1][1])
            correct = 0
            for s_c in range(0, no_atoms):
                correct += k_mat[pr_s][s_c] * np.log(
                    abs(alpha_mat[pr_s][s_c] * p + beta_mat[pr_s][s_c]) ** 2
                )

            phi_2 = (
                phi_core(w_pr, c_coeff[pr_s], a_coeff[pr_s], b_coeff[pr_s])
                + correct
                - (c_coeff[pr_s] / b_coeff[pr_s])
                * np.log(abs(-m_t_p[1][0] * w_pr + m_t_p[0][0]) ** 2)
            )
            const[sphere][sphere] = phi_2 - phi_1

    area = 0
    n_rad = 60  # no radial points
    n_theta = 30  # no angular points
    dtheta = 2 * np.pi / n_theta

    # if user hasn't specified k_quant, set value
    if k_quant is None:
        k_quant = 2

    shape_descriptor = np.zeros((2 * k_quant + 1, 2 * k_quant + 1), dtype=complex)
    for sphere in range(0, no_atoms):
        radius = 1
        dr = radius / n_rad
        k_rad = 1  # integrand always vanishes at r=0
        m_t = base_to_unit_maps[sphere]  # Mobius transform for sphere
        n_n_l = len(avoid_rad[sphere])  # number of higher-level spheres

        while k_rad < n_rad:
            rad = k_rad * dr
            sphe_cont = np.asmatrix(
                np.zeros((2 * k_quant + 1, 2 * k_quant + 1), dtype=complex)
            )
            k_theta = 1
            area_t = 0  # theta contribution to area

            # end point contributions theta=0,2pi
            z = rad

            # Next bit of code checks to remove contributions from higher level spheres
            test = 1
            for avoid in range(0, n_n_l):
                if abs(z - avoid_cent[sphere][avoid]) <= avoid_rad[sphere][avoid]:
                    test = 0

            # Now (if necessary) compute quantities
            if test == 1:
                w = (m_t[1][1] * z - m_t[0][1]) / (-m_t[1][0] * z + m_t[0][0])
                w_bar = np.conj(w)
                correct = 0
                for s_c in range(0, no_atoms):
                    correct += k_mat[sphere][s_c] * np.log(
                        abs((alpha_mat[sphere][s_c] * w) + beta_mat[sphere][s_c]) ** 2
                    )

                phi = (
                    phi_core(z, c_coeff[sphere], a_coeff[sphere], b_coeff[sphere])
                    - (c_coeff[sphere] / b_coeff[sphere])
                    * np.log(abs(-m_t[1][0] * z + m_t[0][0]) ** 2)
                    + correct
                    + const[sphere][sphere]
                )

                vol = (
                    2
                    * rad
                    * c_coeff[sphere]
                    / (abs(z - a_coeff[sphere]) ** 2 + b_coeff[sphere]) ** 2
                ) * dtheta

                herm = np.exp(-k_quant * phi)

                area_t += vol

                w_power = 1
                for mat_i in range(0, 2 * k_quant + 1):
                    w_power_conj = 1
                    for mat_j in range(0, mat_i + 1):
                        sphe_cont[mat_i, mat_j] += w_power * w_power_conj * herm * vol
                        w_power_conj *= w_bar
                    w_power *= w
            # Here computed end point

            # Now rest of circle
            while k_theta < n_theta:
                theta = k_theta * dtheta
                z = rad * (np.cos(theta) + 1j * np.sin(theta))
                # Next bit of code checks to remove contributions from higher level spheres
                test = 1
                for avoid in range(0, n_n_l):
                    if abs(z - avoid_cent[sphere][avoid]) <= avoid_rad[sphere][avoid]:
                        test = 0
                # Now (if necessary) compute quantities
                if test == 1:
                    w = (m_t[1][1] * z - m_t[0][1]) / (-m_t[1][0] * z + m_t[0][0])
                    w_bar = np.conj(w)

                    correct = 0
                    for s_c in range(0, no_atoms):
                        correct += k_mat[sphere][s_c] * np.log(
                            abs((alpha_mat[sphere][s_c] * w) + beta_mat[sphere][s_c])
                            ** 2
                        )

                    phi = (
                        phi_core(z, c_coeff[sphere], a_coeff[sphere], b_coeff[sphere])
                        - (c_coeff[sphere] / b_coeff[sphere])
                        * np.log(abs(-m_t[1][0] * z + m_t[0][0]) ** 2)
                        + correct
                        + const[sphere][sphere]
                    )

                    vol = (
                        2
                        * rad
                        * c_coeff[sphere]
                        / (abs(z - a_coeff[sphere]) ** 2 + b_coeff[sphere]) ** 2
                    ) * dtheta

                    herm = np.exp(-k_quant * phi)

                    area_t += vol

                    w_power = 1
                    for mat_i in range(0, 2 * k_quant + 1):
                        w_power_conj = 1
                        for mat_j in range(0, mat_i + 1):
                            sphe_cont[mat_i, mat_j] += (
                                w_power * w_power_conj * herm * vol
                            )
                            w_power_conj *= w_bar
                        w_power *= w
                k_theta += 1
            # THETA CALCS DONE HERE

            area += area_t * dr
            for mat_i in range(0, 2 * k_quant + 1):
                for mat_j in range(0, mat_i + 1):
                    shape_descriptor[mat_i][mat_j] += sphe_cont[mat_i, mat_j] * dr

            k_rad += 1

        # Radial calcs done here except end point
        # RADIAL TRAPEZIUM RULE Add on last bit
        sphe_cont = np.asmatrix(
            np.zeros((2 * k_quant + 1, 2 * k_quant + 1), dtype=complex)
        )
        k_theta = 1
        area_t = 0
        # theta end point contributions
        z = radius
        # Next bit of code checks to remove contributions from higher level spheres
        test = 1
        for avoid in range(0, n_n_l):
            if abs(z - avoid_cent[sphere][avoid]) <= avoid_rad[sphere][avoid]:
                test = 0
        # Now (if necessary) compute quantities
        if test == 1:
            w = (m_t[1][1] * z - m_t[0][1]) / (-m_t[1][0] * z + m_t[0][0])
            w_bar = np.conj(w)
            correct = 0
            for s_c in range(0, no_atoms):
                correct += k_mat[sphere][s_c] * np.log(
                    abs((alpha_mat[sphere][s_c] * w) + beta_mat[sphere][s_c]) ** 2
                )

            phi = (
                phi_core(z, c_coeff[sphere], a_coeff[sphere], b_coeff[sphere])
                - (c_coeff[sphere] / b_coeff[sphere])
                * np.log(abs(-m_t[1][0] * z + m_t[0][0]) ** 2)
                + correct
                + const[sphere][sphere]
            )
            vol = (
                2
                * radius
                * c_coeff[sphere]
                / (abs(z - a_coeff[sphere]) ** 2 + b_coeff[sphere]) ** 2
            ) * dtheta

            area_t += vol

            herm = np.exp(-k_quant * phi)
            w_power = 1
            for mat_i in range(0, 2 * k_quant + 1):
                w_power_conj = 1
                for mat_j in range(0, mat_i + 1):
                    sphe_cont[mat_i, mat_j] += w_power * w_power_conj * herm * vol
                    w_power_conj *= w_bar
                w_power *= w

        # Here computed end point
        # Now rest of circle
        while k_theta < n_theta:
            theta = k_theta * dtheta
            z = radius * (np.cos(theta) + 1j * np.sin(theta))
            # Next bit of code checks to remove contributions from higher level spheres
            test = 1
            for avoid in range(0, n_n_l):
                if abs(z - avoid_cent[sphere][avoid]) <= avoid_rad[sphere][avoid]:
                    test = 0
            # Now (if necessary) compute quantities
            if test == 1:
                w = (m_t[1][1] * z - m_t[0][1]) / (-m_t[1][0] * z + m_t[0][0])
                w_bar = np.conj(w)
                correct = 0
                for s_c in range(0, no_atoms):
                    correct += k_mat[sphere][s_c] * np.log(
                        abs((alpha_mat[sphere][s_c] * w) + beta_mat[sphere][s_c]) ** 2
                    )

                phi = (
                    phi_core(z, c_coeff[sphere], a_coeff[sphere], b_coeff[sphere])
                    - (c_coeff[sphere] / b_coeff[sphere])
                    * np.log(abs(-m_t[1][0] * z + m_t[0][0]) ** 2)
                    + correct
                    + const[sphere][sphere]
                )

                vol = (
                    2
                    * radius
                    * c_coeff[sphere]
                    / (abs(z - a_coeff[sphere]) ** 2 + b_coeff[sphere]) ** 2
                ) * dtheta

                area_t += vol

                herm = np.exp(-k_quant * phi)
                w_power = 1
                for mat_i in range(0, 2 * k_quant + 1):
                    w_power_conj = 1
                    for mat_j in range(0, mat_i + 1):
                        sphe_cont[mat_i, mat_j] += w_power * w_power_conj * herm * vol
                        w_power_conj *= w_bar
                    w_power *= w
            k_theta += 1
        # THETA CALCS DONE HERE

        area += area_t * dr
        for mat_i in range(0, 2 * k_quant + 1):
            for mat_j in range(0, mat_i + 1):
                shape_descriptor[mat_i][mat_j] += sphe_cont[mat_i, mat_j] * dr

    for mat_i in range(0, 2 * k_quant + 1):
        for mat_j in range(mat_i + 1, 2 * k_quant + 1):
            shape_descriptor[mat_i][mat_j] = np.conj(shape_descriptor[mat_j][mat_i])

    return shape_descriptor
