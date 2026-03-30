"""
Apply model converted ipynb script
"""

import os
import sys
from glob import glob
import argparse
import torch as pt
from tqdm import tqdm
import pandas as pd


from pesto.dataset import StructuresDataset, collate_batch_features
from pesto.data_encoding import encode_structure, encode_features, extract_topology
from pesto.structure import concatenate_chains

def get_arguments()-> argparse.Namespace:
    """
    CL argument parser
    """
    parser = argparse.ArgumentParser(
        prog='PeSTo',
        description='Apply PeSTo: Krapp, L.F., Abriata, L.A., Cortés Rodriguez, F. et al. PeSTo: parameter-free geometric deep learning for accurate prediction of protein binding interfaces. Nat Commun 14, 2175 (2023). https://doi.org/10.1038/s41467-023-37701-8',
        epilog='---------------------'
    )
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='path/to/pdbs')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='path/to/output_folder')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='path/to/saved/model')
    
    return parser.parse_args()


def run_model(model, dataset, out_path, device="cpu"):
    # run model on all subunits

    pdb_res = {}

    with pt.no_grad():
        for subunits, filepath in tqdm(dataset):
            pdb_name = os.path.basename(filepath)[:-4]
            # concatenate all chains together
            structure = concatenate_chains(subunits)
            # encode structure and features
            X, M = encode_structure(structure)
            #q = pt.cat(encode_features(structure), dim=1)
            q = encode_features(structure)[0]
            # extract topology
            ids_topk, _, _, _, _ = extract_topology(X, 64)
            # pack data and setup sink (IMPORTANT)
            X, ids_topk, q, M = collate_batch_features([[X, ids_topk, q, M]])
            # run model
            z = model(X.to(device), ids_topk.to(device), q.to(device), M.float().to(device))
            # for all predictions
            i_dict = {}
            for i, cat in zip(range(z.shape[1]), ["protein", "DNA-RNA", "ion", "ligand", "lipid"]):
                # prediction
                p = pt.sigmoid(z[:,i])
                p = p.cpu().numpy()
                # encode result
                pd.DataFrame({"res_num":[n+1 for n in range(len(p))],"prob":p}).to_csv(os.path.join(out_path,f'{pdb_name}_{cat}.tsv'), sep='\t', index=False)


def main():
    args = get_arguments()

    save_path = args.model
    # select saved model
    model_filepath = os.path.join(save_path, 'model_ckpt.pt')   
    # add module to path
    if save_path not in sys.path:
        sys.path.insert(0, save_path)
    from config import config_model, config_data
    from data_handler import Dataset
    from model import Model

    # load model
    device = pt.device("cpu")
    model = Model(config_model)
    model.load_state_dict(pt.load(model_filepath, map_location=pt.device("cpu")))
    # set model to inference
    model = model.eval().to(device)
    # find pdb files and ignore already predicted oins
    pdb_filepaths = glob(os.path.join(args.input, "*.pdb"), recursive=True)
    # create dataset loader with preprocessing
    dataset = StructuresDataset(pdb_filepaths, with_preprocessing=True)
    # debug print
    print(f'Number of PDBs to process: {len(dataset)}')
    run_model(model, dataset, args.output)

if __name__ == "__main__":
    main()