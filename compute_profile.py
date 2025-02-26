import argparse
import torch
import os
import numpy as np
from chroma import Protein

from src.plots import plot_SAXS_profile
from src.profile import compute_profile, fit_profile
from src.structure import FormFactorType
from src.utils import X_to_particles, read_exp_profile

def main(args):
    os.makedirs(args.outdir, exist_ok=True)

    gpu = torch.cuda.is_available()

    protein = Protein.from_PDB(args.pdb, device='cuda' if gpu else 'cpu')
    X_gt, C_gt, S_gt = protein.to_XCS(all_atom=False)  # we use X_gt to compute RMSD
    mask_gt = (C_gt == 1)[0]

    X_gt_full, C_gt_full, S_gt_full = protein.to_XCS(all_atom=True)
    particles = X_to_particles(X_gt_full, C_gt_full, S_gt_full)

    print("Computing the SAXS Profile")
    exp_profile = read_exp_profile(args.dat)
    if exp_profile is None:
        print("No experimental profile found")
    qmin = exp_profile.min_q_
    qmax = exp_profile.max_q_
    delta_q = exp_profile.delta_q_
    ff_type = FormFactorType.ALL_ATOMS    
    profile = compute_profile(particles=particles, min_q=qmin, max_q=qmax, delta_q=delta_q, ff_type=ff_type, gpu=gpu)
    model_profile, chi_square, fitted_params = fit_profile(exp_profile, profile, min_c1=0.99, max_c1=1.05, min_c2=0.0, max_c2=0.0, use_offset=False)
    file_name = f"{args.pdb.split('/')[-1].split('.')[0]}_profile.png"
    plot_SAXS_profile(model_profile, exp_profile, chi_square, fitted_params, os.path.join(args.outdir, file_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # required parameters
    parser.add_argument("--outdir", type=str, required=True, help="Path to output directory.")
    parser.add_argument("--dat", type=str, required=True, help="Path to the SAXS data file.")
    parser.add_argument("--pdb", type=str, required=True, help="Path to deposited PDB file.")
    
    args = parser.parse_args()
    main(args)
