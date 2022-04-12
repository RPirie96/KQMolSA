"""
main script to return the descriptor for a single molecule
"""
from collections import namedtuple

from get_inputs import get_mol_info
import basesphere
from stereo_projection import get_stereographic_projection
from k_matrix import get_k_mat


def get_descriptor(mol, k_quant=None):
    """
    Function to generate shape descriptor for a single molecule
    @param mol:
    @param k_quant:
    @return:
    """
    # get centres, radii, adjacency matrix and no. atoms
    inputs = get_mol_info(mol)

    # get base sphere and re-centre on origin
    base = basesphere.get_base_sphere(inputs.centres)

    # get levels within molecule
    levels = basesphere.get_levels(
        inputs.adjacency_matrix, inputs.no_atoms, base.base_sphere
    )

    # get molecule area
    mol_area = basesphere.get_area(
        inputs.adjacency_matrix, base.centres, inputs.no_atoms, inputs.radii
    )

    # rescale inputs so molecule has surface area equivalent to a unit sphere
    rescaled = basesphere.rescale_inputs(
        mol_area.area, base.centres, inputs.radii, mol_area.lam
    )

    # get fingerprint (tells you how to navigate through molecule)
    fingerprint = basesphere.get_fingerprint(levels, inputs)

    # get next level vector
    next_vector = basesphere.get_next_level_vec(
        inputs.no_atoms, fingerprint, levels.no_levels
    )

    # error handling to account for cases where there is an atom over the north pole
    centres_r = basesphere.base_error(base, rescaled, next_vector)

    # get level list
    level_list = basesphere.get_level_list(
        levels.no_levels, inputs.no_atoms, next_vector.sphere_levels_vec
    )

    # perform 'piecewise stereographic projection' to move molecule into CP^n
    stereo_proj = get_stereographic_projection(
        inputs, base.base_sphere, levels, level_list, next_vector, rescaled, fingerprint, centres_r
    )

    # get shape descriptor
    kq_shape, area_check = get_k_mat(
        inputs.no_atoms,
        stereo_proj,
        next_vector.sphere_levels_vec,
        fingerprint,
        levels.no_levels,
        level_list,
        k_quant,
    )

    descriptor = namedtuple("descriptor", ["surface_area", "kq_shape", "area_check"])

    return descriptor(surface_area=mol_area.area, kq_shape=kq_shape, area_check=area_check)
