#!/usr/bin/python
"""
Example script for running full RGMolSA method including conformer generation and data filtering
"""

import sys
import pandas as pd
from rdkit.Chem import PandasTools

from data_filters import filter_dataset
from conf_gen import embed_multi_3d
from get_descriptor import get_descriptor
from utils import get_score


def get_cl_inputs():
    """
    helper function to get the filenames for the initial set and to write confs too from command line + csv filename
    @return:
    """
    mol_set = sys.argv[1]
    conf_filename = sys.argv[2]
    scores_name = sys.argv[3]
    k_vals = sys.argv[4][1:len(sys.argv[4])-1].split(',')

    return mol_set, conf_filename, scores_name, k_vals


def run_descriptors():

    # get the dataset and conformer filenames
    mol_set, conf_filename, scores_name, k_vals = get_cl_inputs()

    k_vals = [int(i) for i in k_vals]

    # load dataset
    data = PandasTools.LoadSDF(mol_set)
    mols = list(data['ROMol'])
    ids = list(data['ID'])

    # filter the data
    filtered = filter_dataset(mols, ids)  # add filename to save, ro5/pains = True for additional filters
    mols = filtered.mols_new
    ids = filtered.ids_new

    # generate conformers
    embed_multi_3d(mols, ids, conf_filename, no_confs=None, energy_sorted=False)

    # load conformers
    data = PandasTools.LoadSDF(conf_filename)
    mols = list(data['ROMol'])
    ids = list(data['ID'])
    cids = list(data['CID'])

    # get descriptor for each molecule
    descriptors = [get_descriptor(mol, k_vals) for mol in mols]

    # get scores, treating first molecule as query for k=1 only
    if len(k_vals) == 1:
        query_des = descriptors[0].kq_shape[0]
        query_area = descriptors[0].surface_area
        qid = ids[0]
        q_cid = cids[0]

        scores = []
        for i, des in enumerate(descriptors):
            dist, sim_score, x0 = get_score(query_des, des.kq_shape[0], query_area,
                                            des.surface_area, k_vals[0], q_cid, cids[i])
            scores.append(sim_score)

        # create dataframe of ID vs score, sort with highest first and save to csv
        scores_df = pd.DataFrame(list(zip(ids, scores)), columns=['ID', 'Score'])
        scores_df = scores_df[scores_df['Score'] != 'self']
        scores_df = scores_df.sort_values('Score', ascending=False)
        scores_df = scores_df.reset_index(drop=True)

        scores_df.to_csv(scores_name)

    # get scores, treating first molecule as query for k=1 and k=2
    elif len(k_vals) == 2:
        query_des_1 = descriptors[0].kq_shape[0]
        query_des_2 = descriptors[0].kq_shape[1]
        query_area = descriptors[0].surface_area
        qid = ids[0]
        q_cid = cids[0]

        scores_1, scores_2 = [], []
        for i, des in enumerate(descriptors):
            dist1, sim_score1, x0 = get_score(query_des_1, des.kq_shape[0], query_area, des.surface_area, k_vals[0], q_cid, cids[i])
            dist2, sim_score2, x0 = get_score(query_des_2, des.kq_shape[1], query_area, des.surface_area, k_vals[1], q_cid, cids[i], x0)
            scores_1.append(sim_score1)
            scores_2.append(sim_score2)

        # create dataframe of ID vs score, sort with highest first and save to csv
        scores_df = pd.DataFrame(list(zip(ids, scores_1, scores_2)), columns=['ID', 'Score k=1', 'Score k=2'])
        scores_df = scores_df[scores_df['Score k=1'] != 'self']
        scores_df = scores_df.sort_values('Score k=1', ascending=False)
        scores_df = scores_df.reset_index(drop=True)

        scores_df.to_csv(scores_name)


run_descriptors()
