# script containing functions to plot the 3D structure and map of the molecule in CP^1
import numpy as np
import plotly.graph_objects as go

import matplotlib.pyplot as plt
from rdkit import Chem


def spheres_plot(mol=None, inputs=None, filename):
    """Function to return a basic plot of the molecule with the atoms represented as spheres"""

    if inputs is not None:
        # unpack tuple for molecule with rings replaced
        no_atoms = inputs.no_atoms
        centres = inputs.centres
        radii = inputs.radii

    if mol is not None:
        # get inputs if rings not replaced
        no_atoms = mol.GetNumAtoms()
        pos = [mol.GetConformer().GetAtomPosition(i) for i in range(0, no_atoms)]
        centres = np.array([[p.x, p.y, p.z] for p in pos])
        radii = np.array(
        [
            0.6 * Chem.GetPeriodicTable().GetRvdw(a.GetAtomicNum())
            for a in mol.GetAtoms()
        ]
    )

    theta_1, theta_2 = np.mgrid[0:2*np.pi:50j, -np.pi:np.pi:50j]
    data = []

    for i in range(no_atoms):
        x = centres[i][0] + (radii[i]*np.cos(theta_1)*np.sin(theta_2))
        y = centres[i][1] + (radii[i]*np.sin(theta_1)*np.sin(theta_2))
        z = centres[i][2] + (radii[i]*np.cos(theta_2))
        data.append(go.Surface(x=x, y=y, z=z))

    fig = go.Figure(data)

    fig.write_image(filename)


def cp_plot(levels, no_atoms, sgp, filename):
    """ Function to return plot of the molecule in complex projective space """

    # unpack tuples
    n_levels = levels.no_levels
    level_mat = levels.level_mat
    com_plan_cent = sgp.com_plan_cent
    com_plan_rad = sgp.com_plan_rad

    theta = np.linspace(start=0., stop=2*np.pi, num=200)

    for i in range(1, n_levels + 1):
        for k in range(0, no_atoms):
            if level_mat[i][k] > 0 and i < 5:
                x = com_plan_cent[i][k].real + com_plan_rad[i][k] * np.cos(theta)
                y = com_plan_cent[i][k].imag + com_plan_rad[i][k] * np.sin(theta)

                if i == 1:
                    plt.plot(x, y, color="red")
                if i == 2:
                    plt.plot(x, y, color="blue")
                if i == 3:
                    plt.plot(x, y, color="green")
                if i == 4:
                    plt.plot(x, y, color="orange")
                if i == 5:
                    plt.plot(x, y, color="pink")

    plt.savefig(filename, dpi=300)