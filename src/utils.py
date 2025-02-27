import os
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import numpy as np
import torch
from src.profile import Profile
from src.structure import Atom, Residue, Chain, get_atom_type_exists, add_atom_type
from chroma import constants
import chroma.utility.polyseq as polyseq

def ma_cif_to_X(path, n_residues):
    dict = MMCIF2Dict(path)
    label_seq_id = np.array(dict['_atom_site.label_seq_id'])
    label_atom_id = np.array(dict['_atom_site.label_atom_id'])
    xs = np.array(dict['_atom_site.Cartn_x'])
    ys = np.array(dict['_atom_site.Cartn_y'])
    zs = np.array(dict['_atom_site.Cartn_z'])

    X = torch.zeros(1, n_residues, 4, 3).float()
    mask = torch.zeros(n_residues).float()

    for idx, element, x, y, z in zip(label_seq_id, label_atom_id, xs, ys, zs):
        if element == 'N':
            X[0, int(idx) - 1, 0] = torch.tensor([float(x), float(y), float(z)]).float()
        if element == 'CA':
            X[0, int(idx) - 1, 1] = torch.tensor([float(x), float(y), float(z)]).float()
        if element == 'C':
            X[0, int(idx) - 1, 2] = torch.tensor([float(x), float(y), float(z)]).float()
        if element == 'O':
            X[0, int(idx) - 1, 3] = torch.tensor([float(x), float(y), float(z)]).float()
        mask[int(idx) - 1] = 1.

    return X, mask.reshape(1, -1, 1, 1)

def read_exp_profile(dat_file):
    if not os.path.exists(dat_file):
        print("Can't open file " + dat_file)
        return
    exp_profile = Profile(file_name=dat_file, fit_file=False, constructor=1)
    if exp_profile.size() == 0:
        print("Can't parse input file " + dat_file)
        return
    else:
        print("Profile read from file " + dat_file + " size = " + str(exp_profile.size()))        
    return exp_profile

def X_to_particles(X, C, S, alternate_alphabet=None):
    particles = []
    chains = {}
    alphabet = constants.AA20 if alternate_alphabet is None else alternate_alphabet
    all_atom = X.shape[2] == 14
    print(f"All-atom ? {all_atom}")

    X, C, S = [T.squeeze(0).cpu().data.numpy() for T in [X, C, S]]

    chain_ids = np.abs(C)

    for i, chain_id in enumerate(np.unique(chain_ids)):
        if chain_id == 0:
            continue
        print(f"Chain ID: {chain_id}")
        chain_bool = chain_ids == chain_id
        X_chain = X[chain_bool, :, :].tolist()
        C_chain = C[chain_bool].tolist()
        S_chain = S[chain_bool].tolist()

        chain = Chain(chain_id=chr(65 + i))
        chains[chain_id] = chain

        for res_ix, (X_i, C_i, S_i) in enumerate(zip(X_chain, C_chain, S_chain)):
            resname = polyseq.to_triple(alphabet[int(S_i)])
            print(f"Residue {res_ix}: {resname}")
            residue = Residue(residue_type=resname, index=res_ix + 1)
            chain.residues.append(residue)

            if C_i > 0:
                atom_names = constants.ATOMS_BB

                if all_atom and resname in constants.AA_GEOMETRY:
                    atom_names = (
                        atom_names + constants.AA_GEOMETRY[resname]["atoms"]
                    )

                for atom_ix, atom_name in enumerate(atom_names):
                    print(f"Atom {atom_ix}: {atom_name}")
                    x, y, z = X_i[atom_ix]
                    print(f"Coordinates {atom_name}: {(x,y,z)}")
                    atom = Atom(name=atom_name, type=atom_name[0], atom_index=atom_ix, coord=(x, y, z))
                    residue.add_child(atom)
                    atom.residue = residue
                    particles.append(atom)

    return particles