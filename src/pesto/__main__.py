"""
Apply model converted ipynb script
"""

import os
import sys
from glob import glob
import argparse
import re

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

def calculate_combined_effects(intermediates, output_file, threshold=0.2):

    pattern = re.compile(
    r"^(?P<uniprot>[A-Za-z0-9]+)"
    r"(?:_(?P<mut>[A-Z][0-9]+[A-Z]))?"
    r"_(?P<interaction>[A-Za-z0-9\-]+)\.tsv$"
    )


    files = os.listdir(intermediates)

    # Structure:
    # dict[(uniprot, mutation)] = {interaction: delta}
    all_results = {}

    # Map WT and mutant files
    file_map = {}  # (uniprot, interaction) → {"WT": file, "mutants": {mut: file}}

    for fname in files:
        m = pattern.match(fname)
        if not m:
            continue
        
        uniprot = m.group("uniprot")
        mut = m.group("mut")           # None for WT
        interaction = m.group("interaction")

        key = (uniprot, interaction)
        file_map.setdefault(key, {"WT": None, "mutants": {}})
        
        if mut is None:
            file_map[key]["WT"] = fname
        else:
            file_map[key]["mutants"][mut] = fname
            
    # Compute deltas
    for (uniprot, interaction), entry in file_map.items():
        wt_file = entry["WT"]
        if wt_file is None:
            continue  # no WT to compare to
        
        wt_df = pd.read_csv(os.path.join(intermediates, wt_file), sep="\t")

        for mut, mut_file in entry["mutants"].items():
            mut_df = pd.read_csv(os.path.join(intermediates, mut_file), sep="\t")
            
            merged = wt_df.merge(mut_df, on="res_num", suffixes=("_wt", "_mut"))
            
            #calc from cut-off
            mask = (merged["prob_wt"] > threshold) | (merged["prob_mut"] > threshold)
            high_conf = merged[mask].copy()

            if len(high_conf) == 0:
                delta = 0
            else:
                high_conf["diff"] = (high_conf["prob_mut"] - high_conf["prob_wt"])
                delta = high_conf["diff"].mean()
                        
            key = (uniprot, mut)
            if key not in all_results:
                all_results[key] = {}
            all_results[key][f'pesto_delta_interface_prob_{interaction}'] = delta


    # Convert to a table (mutants × interactions)
    df = pd.DataFrame([
        {"UniProt": uniprot, "UniProt_AAchange": mutation, **interaction_dict}
        for (uniprot, mutation), interaction_dict in all_results.items()
    ])

    # Fill missing interactions with 0 or NaN (choose 0 for convenience)
    df = df.fillna(0)

    df.to_csv(output_file, index=False, sep='\t')
    print("Done. Matrix written to", output_file)


def run_model(model, dataset, out_path, device="cpu"):
    # run model on all subunits
    out_folder = os.path.join(out_path,'intermediates')
    os.makedirs(out_folder, exist_ok=True)
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
            for i, cat in zip(range(z.shape[1]), ["protein", "DNA-RNA", "ion", "ligand", "lipid"]):
                # prediction
                p = pt.sigmoid(z[:,i])
                p = p.cpu().numpy()
                # encode result
                pd.DataFrame({"res_num":[n+1 for n in range(len(p))],"prob":p}).to_csv(os.path.join(out_folder,f'{pdb_name}_{cat}.tsv'), sep='\t', index=False)
    
    calculate_combined_effects(out_folder, os.path.join(out_path,'combined_results.tsv'))


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