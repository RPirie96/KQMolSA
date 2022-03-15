"""
module to obtain info surrounding the 'base-sphere'

Functions:
- get_base_sphere: function to find the base sphere (centroid) of the molecule
- get_levels: finds path through molecule from base sphere
- get_area: finds surface area of the molecule\
- rescale_inputs: rescale matrix constructs to have area equal to 4pi
- get_fingerprint: Fingerprint Matrix that tells you how to navigate through molecule
- get_next_level_vec: Code to produce vector of next level spheres
- get_level_list: list of the level-k sphere

Exceptions:
- ArithmeticError: raised when molecule has negative surface area (typically bridged bicyclics)
"""

from collections import namedtuple
import numpy as np
from scipy.spatial import distance_matrix

from utils import get_chain


def get_base_sphere(centres):

    """
    Function which selects the starting atom (base-sphere). This is taken as the atom closest to the centroid

    @param centres:
    @return: centres, base_sphere
    """

    # Find the centroid
    centroid = [
        np.sum(centres[:, 0]) / len(centres[:, 0]),
        np.sum(centres[:, 1]) / len(centres[:, 1]),
        np.sum(centres[:, 2]) / len(centres[:, 2]),
    ]

    # Find the index of the minimum Euclidean distance and set this as the base sphere
    base_sphere = np.argmin(np.sqrt(np.sum(np.square(centres - centroid), axis=1)))

    # re-centre so the base sphere site on the origin
    c_rel = centres - centres[base_sphere]
    centres = c_rel[:]

    base = namedtuple("base", ["centres", "base_sphere"])

    return base(centres=centres, base_sphere=base_sphere)


def get_levels(adjacency_matrix, no_atoms, base_sphere):

    """
    Function to generate matrix of levels starting from base sphere. produce a matrix of integers row = level;
    1 = non-terminal at this level, 2 = terminal at this level

    @param adjacency_matrix:
    @param no_atoms:
    @param base_sphere:
    @return: level_mat, no_levels
    """

    r_sum = adjacency_matrix.sum(axis=1)
    to_do = no_atoms - 1  # how may remaining spheres need to be assigned
    assigned, level_mat = (
        np.zeros((1, no_atoms), dtype=int),
        np.zeros((1, no_atoms), dtype=int),
    )
    assigned[0, base_sphere] = 1
    level_mat[0, base_sphere] = 1

    current_level = 0
    while to_do > 0 and current_level < 500:
        next_level = np.zeros((1, no_atoms), dtype=int)

        for j in range(0, no_atoms):
            if level_mat[current_level, j] == 1:
                current_sphere = j

                for i in range(0, no_atoms):
                    if (
                        adjacency_matrix[current_sphere, i] == 1
                        and r_sum[i] == 1
                        and assigned[0, i] == 0
                    ):
                        next_level[0, i] = 2
                        assigned[0, i] = 1
                        to_do += -1
                    if (
                        adjacency_matrix[current_sphere, i] == 1
                        and r_sum[i] > 1
                        and assigned[0, i] == 0
                    ):
                        next_level[0, i] = 1
                        assigned[0, i] = 1
                        to_do += -1

        level_mat = np.vstack((level_mat, next_level))
        current_level += 1

    no_levels = len(level_mat) - 1  # number of levels

    levels = namedtuple("levels", ["level_mat", "no_levels"])

    return levels(level_mat=level_mat, no_levels=no_levels)


def get_area(adjacency_matrix, centres, no_atoms, radii):

    """
    Function to return the surface area of the molecule, and the matrix of lambda values

    If the area is negative (usually for bridged bicyclic compounds with >2 intersecting rings) a
    ValueError is raised. As the area is computed as the area of a sphere - the bit where two spheres
    intersect, multiple large spheres intersecting leads to a negative value, and thus the surface of the
    molecule cannot be approximated.

    @param adjacency_matrix:
    @param centres:
    @param no_atoms:
    @param radii:
    @return: area and matrix of lambda values
    """

    # matrix of distances between intersecting atoms
    distances = adjacency_matrix * distance_matrix(centres, centres)

    # matrix of lambdas
    lam = np.zeros((no_atoms, no_atoms))
    for i in range(0, no_atoms):
        for j in range(0, no_atoms):
            if adjacency_matrix[i, j] == 1:
                lam[i, j] = (radii[i] ** 2 - radii[j] ** 2 + distances[i, j] ** 2) / (
                    2 * distances[i, j]
                )
            else:
                lam[i, j] = 0

    # surface area of the molecule
    area = 0
    for i in range(0, no_atoms):
        sphere_i = 4 * np.pi * radii[i] ** 2
        for j in range(0, no_atoms):
            if adjacency_matrix[i, j] == 1:
                sphere_i = sphere_i - 2 * radii[i] * np.pi * abs(radii[i] - lam[i, j])
        area += sphere_i

    if area < 0:
        raise ArithmeticError("Negative Surface Area, cannot approximate surface")

    mol_area = namedtuple("mol_area", ["lam", "area"])

    return mol_area(lam=lam, area=area)


def rescale_inputs(area, centres, radii, lam):

    """
    Function to rescale all inputs to give total surface area equal to 4pi

    @param area:
    @param centres:
    @param radii:
    @param lam:
    @return: inputs rescaled to have surface area 4pi
    """

    centres_r = centres * np.sqrt(4 * np.pi / area)
    radii_r = radii * np.sqrt(4 * np.pi / area)
    lam_r = lam * np.sqrt(4 * np.pi / area)

    rescaled = namedtuple("rescaled", ["centres_r", "radii_r", "lam_r"])

    return rescaled(centres_r=centres_r, radii_r=radii_r, lam_r=lam_r)


def get_fingerprint(levels, inputs):
    """
    Fingerprint Matrix that tells you how to navigate through molecule
    :param levels:
    :param inputs:
    :return: fingerprint matrix
    """

    # unpack tuples
    no_levels = levels.no_levels
    level_mat = levels.level_mat
    no_atoms = inputs.no_atoms
    adjacency_matrix = inputs.adjacency_matrix

    fingerprint = np.tile(-1, (no_levels + 1, no_atoms))

    for level in range(0, no_levels + 1):
        for sphere in range(0, no_atoms):
            if level_mat[level][sphere] > 0:
                s_list = get_chain(no_atoms, level_mat, adjacency_matrix, sphere, level)
                for k in range(0, len(s_list)):
                    fingerprint[k][sphere] = s_list[k]

    return fingerprint


def get_next_level_vec(no_atoms, fingerprint, no_levels):
    """
    Code to produce vector of next level spheres
    :param no_atoms:
    :param fingerprint:
    :param no_levels:
    :return: named tuple of sphere_levels_vec and next_level
    """

    sphere_levels_vec = []  # Sphere_levels_vec[i] = level of sphere i
    next_level = []  # Next_Level[i] = list of spheres of level above that of sphere i
    for s in range(0, no_atoms):
        stop = 0
        l = 0
        while stop < 1:
            if fingerprint[l][s] == s:
                stop = 1
            else:
                l = l + 1
        sphere_levels_vec.append(l)
        next_level.append(
            [
                s_n
                for s_n in range(0, no_atoms)
                if l < no_levels
                and fingerprint[l][s_n] == s
                and fingerprint[l + 1][s_n] == s_n
            ]
        )

    next_vector = namedtuple("next_level_vector", ["sphere_levels_vec", "next_level"])

    return next_vector(sphere_levels_vec=sphere_levels_vec, next_level=next_level)


def get_level_list(no_levels, no_atoms, sphere_levels_vec):
    """
    list of the level-k sphere
    :param no_levels:
    :param no_atoms:
    :param sphere_levels_vec:
    :return: list of levels within molecule
    """
    return [
        [j for j in range(0, no_atoms) if sphere_levels_vec[j] == i]
        for i in range(0, no_levels + 1)
    ]
