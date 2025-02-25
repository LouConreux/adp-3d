import argparse
import torch
import os
import numpy as np
from chroma import Protein

from src.plots import plot_SAXS_profile
from src.profile import compute_profile
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
    profile = compute_profile(particles=particles, min_q=qmin, max_q=qmax, delta_q=delta_q, ff_type=ff_type, hydration_layer=not args.explicit_water, reciprocal=args.reciprocal, ab_initio=args.ab_initio, vacuum=args.vacuum, gpu=gpu)
    file_name = f"{args.pdb.split('/')[-1].split('.')[0]}_profile.png"
    plot_SAXS_profile(profile, exp_profile, os.path.join(args.outdir, file_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # required parameters
    parser.add_argument("--outdir", type=str, required=True, help="Path to output directory.")
    parser.add_argument("--dat", type=str, required=True, help="Path to the SAXS data file.")
    parser.add_argument("--pdb", type=str, required=True, help="Path to deposited PDB file.")
    parser.add_argument("--explicit_water", help="use waters from input PDB (default = False)", action="store_true")
    parser.add_argument("--reciprocal", help="compute profile in reciprocal space (default = False)", action="store_true")
    parser.add_argument("--ab_initio", help="compute profile for a bead model with constant form factor (default = False)", action="store_true")
    parser.add_argument("--vacuum", help="compute profile in vacuum (default = False)", action="store_true")
    
    args = parser.parse_args()
    main(args)
