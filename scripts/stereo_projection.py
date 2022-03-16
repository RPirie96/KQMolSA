"""
Module to complete the mapping of the molecule from real to complex projective space.

Functions:
- get_stereographic_projection: piecewise stereographic projection of the molecule into CP^n
- alpha_coefficient: helper function to get alpha value
- beta_coefficient: helper function to get beta value
"""

import numpy as np
from numpy import linalg as la
from collections import namedtuple

from utils import get_chain, get_m_rot, t_circle, new_coeff


# helper functions for performing piecewise stereographic projection
# if we rotate (0,0,1) onto (v_1,v_2,v_3) this induces an element of PSU(2)  [[alpha, beta],[-conj(beta), alpha]]
def alpha_coefficient(vector):  # alpha coefficient
    """
    function to get alpha coefficient
    @param vector:
    @return: alpha
    """
    if (vector[2] + 1) ** 2 > 10 ** (-9):
        return np.sqrt((1 + vector[2]) / 2)
    else:
        return 0


def beta_coefficient(vector):  # beta coefficient
    """
    function to get beta coefficient
    @param vector:
    @return: beta
    """
    if (vector[2] + 1) ** 2 > 10 ** (-9):
        return -np.sqrt(1 / (2 * (1 + vector[2]))) * complex(vector[0], vector[1])
    else:
        return 1j


def get_stereographic_projection(inputs, base_sphere, levels, level_list, next_vector, rescaled, fingerprint):
    """
    Function to return the piecewise stereographic projection of the molecule into CP^n

    @param inputs:
    @param base_sphere:
    @param levels:
    @param level_list:
    @param next_vector:
    @param rescaled:
    @param fingerprint:
    @return:
    """
    # unpack tuples
    no_atoms = inputs.no_atoms
    adjacency_matrix = inputs.adjacency_matrix
    no_levels = levels.no_levels
    level_mat = levels.level_mat
    centres_r = rescaled.centres_r
    radii_r = rescaled.radii_r
    lam_r = rescaled.lam_r
    sphere_levels_vec = next_vector.sphere_levels_vec
    next_level = next_vector.next_level

    # Attempt to !Automatically! describe surface by 'slices' through each atom outside = C
    # previous levels inside a disc centred at 0
    # next levels inside spheres
    slice_maps = []  # these map a unit disc in C to the corresponding disc in the piecewise stereographic projection
    rotation_maps = []  # These map (0,0,1) onto the corresponding vector normal to plane of intersection
    scale_factors = []  # Scale factor starting at level-(k-1) to get level-k coordinate
    disc_radii = []  # radii of discs containing previous sphere

    integration_maps = []  # maps used to normalise everything to a unit disc

    complex_plane_centres = np.zeros(no_atoms, dtype=complex)
    complex_plane_radii = np.zeros(no_atoms, dtype=float)

    # initialise with empty maps
    for i in range(0, no_atoms):
        m_t = np.zeros((2, 2), dtype=complex)
        i_t = np.zeros((2, 2), dtype=complex)
        rot_t = np.zeros((3, 3), dtype=float)
        slice_maps.append(m_t)
        rotation_maps.append(rot_t)
        integration_maps.append(i_t)
        scale_factors.append(0)
        disc_radii.append(0)

    slice_maps[base_sphere][0][0] = 1
    slice_maps[base_sphere][1][1] = 1

    rotation_maps[base_sphere][0][0] = 1
    rotation_maps[base_sphere][1][1] = 1
    rotation_maps[base_sphere][2][2] = 1

    scale_factors[base_sphere] = 1

    for level in range(0, no_levels):
        c_s_l = len(level_list[level])  # number of current spheres
        for c_s in range(0, c_s_l):
            current_sphere = level_list[level][c_s]
            n_s_l = len(next_level[current_sphere])  # number of spheres at next level
            for n_s in range(0, n_s_l):
                assign_sphere = next_level[current_sphere][n_s]
                chain_s = get_chain(no_atoms, level_mat, adjacency_matrix, assign_sphere, level + 1)

                # generate rotation matrix
                rel_cent = np.zeros((level + 1, 3), float)  # set up vectors of relative centres
                for q in range(0, level + 1):
                    rel_cent[q] = centres_r[chain_s[q + 1]] - centres_r[chain_s[q]]
                    norm = la.norm(rel_cent[q])
                    rel_cent[q] = rel_cent[q] / norm

                for q in range(1, level + 1):
                    p_s = chain_s[q]
                    m_rel = la.inv(rotation_maps[p_s])
                    rel_cent[level] = (np.dot(m_rel, np.array(
                        [[rel_cent[level][0]], [rel_cent[level][1]], [rel_cent[level][2]]]))).reshape(1, 3)

                rotation_maps[assign_sphere] = get_m_rot([rel_cent[level][0], rel_cent[level][1], rel_cent[level][2]])

                # Now do scale factors
                h_ght = 2 * radii_r[current_sphere] - abs(radii_r[current_sphere] - lam_r[current_sphere][assign_sphere])
                r_l = np.sqrt(h_ght / ((2 * radii_r[current_sphere]) - h_ght))
                h_ght_next = 2 * radii_r[assign_sphere] - abs(
                    radii_r[assign_sphere] - lam_r[assign_sphere][current_sphere])
                r_next = np.sqrt((2 * radii_r[assign_sphere] - h_ght_next) / h_ght_next)

                scale_factors[assign_sphere] = r_next / r_l
                disc_radii[assign_sphere] = r_next

                # Now do Mobius maps
                alpha = alpha_coefficient(rel_cent[level])
                beta = beta_coefficient(rel_cent[level])
                gamma = -np.conj(beta)
                delta = alpha

                slice_maps[assign_sphere][0][0] = alpha
                slice_maps[assign_sphere][0][1] = beta
                slice_maps[assign_sphere][1][0] = gamma
                slice_maps[assign_sphere][1][1] = delta

                [complex_plane_centres[assign_sphere], complex_plane_radii[assign_sphere]] = t_circle(alpha, beta,
                                                                                                      gamma, delta, 0,
                                                                                                      r_l)

                # Integration maps stuff
                i_resc = np.sqrt(r_next)

                integration_maps[assign_sphere][0][0] = 0
                integration_maps[assign_sphere][0][1] = -1j * i_resc
                integration_maps[assign_sphere][1][0] = -1j / i_resc
                integration_maps[assign_sphere][1][1] = 0

    # Contribution of base sphere
    f_l_1 = level_list[1][0]

    integration_maps[base_sphere][0][0] = slice_maps[f_l_1][0][0] / np.sqrt(scale_factors[f_l_1] / disc_radii[f_l_1])
    integration_maps[base_sphere][0][1] = slice_maps[f_l_1][0][1] * np.sqrt(scale_factors[f_l_1] / disc_radii[f_l_1])
    integration_maps[base_sphere][1][0] = slice_maps[f_l_1][1][0] / np.sqrt(scale_factors[f_l_1] / disc_radii[f_l_1])
    integration_maps[base_sphere][1][1] = slice_maps[f_l_1][1][1] * np.sqrt(scale_factors[f_l_1] / disc_radii[f_l_1])

    # Reproducing old plot of discs within discs using the local data
    d_in_d_centre = []
    d_in_d_radii = []

    for sphere in range(0, no_atoms):

        level = sphere_levels_vec[sphere]

        q = level - 1

        [cent, rad] = [0, disc_radii[sphere] / (scale_factors[sphere])]
        alpha = slice_maps[sphere][0][0]
        beta = slice_maps[sphere][0][1]
        gamma = slice_maps[sphere][1][0]
        delta = slice_maps[sphere][1][1]
        [cent, rad] = t_circle(alpha, beta, gamma, delta, cent, rad)

        while q > 0:
            sph_c = fingerprint[q][sphere]
            alpha = slice_maps[sph_c][0][0]
            beta = slice_maps[sph_c][0][1]
            gamma = slice_maps[sph_c][1][0]
            delta = slice_maps[sph_c][1][1]
            lambda_r = 1 / np.sqrt(scale_factors[sph_c])
            m_t = (np.array([[lambda_r * alpha, beta / lambda_r], [lambda_r * gamma, delta / lambda_r]]))
            [cent, rad] = t_circle(m_t[0][0], m_t[0][1], m_t[1][0], m_t[1][1], cent, rad)

            q = q - 1

        d_in_d_centre.append(cent)
        d_in_d_radii.append(rad)

    # Finding the maps that take Base_Sphere picture to unit disc for each higher level sphere

    base_to_unit_maps = []
    test_1 = []
    test_2 = []

    for sphere in range(0, no_atoms):
        level = sphere_levels_vec[sphere]
        q = level
        mat = np.array([[1, 0], [0, 1]])
        l_c = 1
        while q > 0:
            sp_c = fingerprint[l_c][sphere]
            s_f = np.sqrt(scale_factors[sp_c])
            mat = np.matmul(la.inv(slice_maps[sp_c]), mat)
            mat = np.matmul(np.array([[s_f, 0], [0, 1 / s_f]]), mat)
            l_c = l_c + 1
            q = q - 1
        mat = np.matmul(la.inv(integration_maps[sphere]), mat)
        test_1.append(mat[1][1] / mat[1][0])
        test_2.append(mat[0][1] / mat[0][0])
        base_to_unit_maps.append(mat)

    # when integrating over a standard unit disc need to avoid the discs corresponding to higher level spheres

    avoid_cent = []
    avoid_rad = []
    a_coeff = []  # A coefficients in rescaled volume form
    b_coeff = []  # B coefficients in rescaled volume form
    c_coeff = []  # C coefficients in rescaled volume form

    a_coeff_2 = []  # A coefficients needed in phi function
    b_coeff_2 = []  # B coefficients needed in phi function
    c_coeff_2 = []  # C coefficients needed in phi function

    internal_corr = np.zeros(no_atoms)
    external_corr = np.zeros(no_atoms)

    for sphere in range(0, no_atoms):
        mob_mat = la.inv(integration_maps[sphere])

        s_f = np.sqrt(scale_factors[sphere])
        mob_mat_2 = np.matmul(np.array([[s_f, 0], [0, 1 / s_f]]), la.inv(slice_maps[sphere]))
        mob_mat_2 = np.matmul(la.inv(integration_maps[sphere]), mob_mat_2)

        [a_c, b_c, c_c] = new_coeff(0, 1, 2 * (radii_r[sphere] ** 2), mob_mat[1][1], -mob_mat[0][1], -mob_mat[1][0],
                                    mob_mat[0][0])
        a_coeff.append(a_c)
        b_coeff.append(b_c)
        c_coeff.append(c_c)

        [a_c_2, b_c_2, c_c_2] = new_coeff(a_coeff[sphere], b_coeff[sphere], c_coeff[sphere],
                                          base_to_unit_maps[sphere][0][0], base_to_unit_maps[sphere][0][1],
                                          base_to_unit_maps[sphere][1][0], base_to_unit_maps[sphere][1][1])
        a_coeff_2.append(a_c_2)
        b_coeff_2.append(b_c_2)
        c_coeff_2.append(c_c_2)

        n_n_l = len(next_level[sphere])  # number of next level spheres
        # will need to skip first level 1 sphere if level=0
        off_set = 0
        if sphere_levels_vec[sphere] == 0:
            off_set = 1
        if sphere_levels_vec[sphere] != 0:
            s_f = np.sqrt(scale_factors[sphere])
            pr_sphere = fingerprint[sphere_levels_vec[sphere] - 1][sphere]

            # Previous Coefficients
            [a_p, b_p, c_p] = new_coeff(0, 1, 2 * (radii_r[pr_sphere] ** 2), mob_mat_2[1][1], -mob_mat_2[0][1],
                                        -mob_mat_2[1][0], mob_mat_2[0][0])
            internal_corr[sphere] = (c_c / (1 + b_c)) - (c_p / (1 + b_p))
            external_corr[sphere] = ((c_c / b_c) / (1 + b_c)) - ((c_p / b_p) / (1 + b_p))

        # Now find the centres+radius of the next level spheres need to avoid ##
        next_l_avoid_cent = []
        next_l_avoid_rad = []

        for avoid in range(off_set, n_n_l):
            n_sphere = next_level[sphere][avoid]
            [cent, rad] = t_circle(mob_mat[0][0], mob_mat[0][1], mob_mat[1][0], mob_mat[1][1],
                                   complex_plane_centres[n_sphere], complex_plane_radii[n_sphere])
            next_l_avoid_cent.append(cent)
            next_l_avoid_rad.append(rad)

        avoid_cent.append(next_l_avoid_cent)
        avoid_rad.append(next_l_avoid_rad)

    sgp = namedtuple('sgp', ['base_to_unit_maps', 'internal_corr', 'external_corr', 'd_in_d_centre', 'd_in_d_radii',
                             'a_coeff', 'b_coeff', 'c_coeff', 'a_coeff_2', 'b_coeff_2', 'c_coeff_2',
                             'avoid_cent', 'avoid_rad'])

    return sgp(base_to_unit_maps=base_to_unit_maps, internal_corr=internal_corr, external_corr=external_corr,
               d_in_d_centre=d_in_d_centre, d_in_d_radii=d_in_d_radii, a_coeff=a_coeff, b_coeff=b_coeff, c_coeff=c_coeff,
               a_coeff_2=a_coeff_2, b_coeff_2=b_coeff_2, c_coeff_2=c_coeff_2, avoid_cent=avoid_cent, avoid_rad=avoid_rad)
