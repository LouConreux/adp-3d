"""
Differentiable SAXS profile computation in PyTorch.

Implements the Debye formula for computing Small-Angle X-ray Scattering (SAXS)
profiles from atomic coordinates, fully compatible with PyTorch autograd.

The computation follows the pipeline:
    atomic coordinates → form factors → pairwise distances → Debye formula → I(q)

References:
    - FoXS: Schneidman-Duhovny et al., Nucleic Acids Res. 2010
    - Debye formula: P. Debye, Ann. Phys. 1915
"""

import torch
import torch.nn.functional as F
import numpy as np
from chroma import constants


# =============================================================================
# Form Factor Tables (from pyFoXS / IMP SAXS module)
# =============================================================================

# Residue-level form factors keyed by 3-letter amino acid code
# Values: (effective_ff, vacuum_ff, dummy_ff)
_RESIDUE_FF_DICT = {
    "ALA": (9.037, 37.991, 28.954),
    "ARG": (23.289, 84.972, 61.683),
    "ASN": (19.938, 59.985, 40.047),
    "ASP": (20.165, 58.989, 38.824),
    "CYS": (18.403, 53.991, 35.588),
    "GLN": (19.006, 67.984, 48.978),
    "GLU": (19.233, 66.989, 47.755),
    "GLY": (10.689, 28.992, 18.303),
    "HIS": (21.235, 78.977, 57.742),
    "ILE": (6.241, 61.989, 55.748),
    "LEU": (6.241, 61.989, 55.748),
    "LYS": (10.963, 70.983, 60.020),
    "MET": (16.539, 69.989, 53.450),
    "PHE": (9.206, 77.986, 68.781),
    "PRO": (8.613, 51.990, 43.377),
    "SER": (13.987, 45.991, 32.004),
    "THR": (13.055, 53.990, 40.935),
    "TRP": (14.945, 98.979, 84.034),
    "TYR": (14.156, 85.986, 71.830),
    "VAL": (7.173, 53.990, 46.817),
}

# Single-letter to 3-letter mapping
_AA1_TO_AA3 = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
}


def _build_residue_ff_tensors():
    """
    Build residue form factor tensors indexed by Chroma's S encoding.
    Reads constants.AA20 at runtime to match Chroma's amino acid ordering.
    """
    n_aa = len(constants.AA20)
    ff = torch.zeros(n_aa, dtype=torch.float32)
    vac_ff = torch.zeros(n_aa, dtype=torch.float32)
    dum_ff = torch.zeros(n_aa, dtype=torch.float32)

    for idx, letter in enumerate(constants.AA20):
        aa3 = _AA1_TO_AA3.get(letter, None)
        if aa3 is not None and aa3 in _RESIDUE_FF_DICT:
            ff[idx], vac_ff[idx], dum_ff[idx] = _RESIDUE_FF_DICT[aa3]

    return ff, vac_ff, dum_ff


# Build at import time
RESIDUE_FORM_FACTORS, RESIDUE_VACUUM_FORM_FACTORS, RESIDUE_DUMMY_FORM_FACTORS = (
    _build_residue_ff_tensors()
)

# Heavy atom form factors at q=0 (from pyFoXS zero_form_factors_)
# Indexed by FormFactorAtomType enum value
HEAVY_ATOM_FORM_FACTORS = torch.tensor([
    -0.720147,  # 0: H
    -0.720228,  # 1: He
    1.591,      # 2: Li
    2.591,      # 3: Be
    3.591,      # 4: B
    0.50824,    # 5: C
    6.16294,    # 6: N
    4.94998,    # 7: O
    7.591,      # 8: F
    6.993,      # 9: Ne
    7.9864,     # 10: Na
    8.9805,     # 11: Mg
    9.984,      # 12: Al
    10.984,     # 13: Si
    13.0855,    # 14: P
    9.36656,    # 15: S
    13.984,     # 16: Cl
    16.591,     # 17: Ar
    15.984,     # 18: K
    14.9965,    # 19: Ca
    20.984,     # 20: Cr
    21.984,     # 21: Mn
    20.9946,    # 22: Fe
    23.984,     # 23: Co
    24.984,     # 24: Ni
    25.984,     # 25: Cu
    24.9936,    # 26: Zn
    30.9825,    # 27: Se
    31.984,     # 28: Br
    43.984,     # 29: Ag
    49.16,      # 30: I
    70.35676,   # 31: Ir
    71.35676,   # 32: Pt
    72.324,     # 33: Au
    73.35676,   # 34: Hg
    # Heavy atom groups (with implicit hydrogens):
    -0.211907,  # 35: CH
    -0.932054,  # 36: CH2
    -1.6522,    # 37: CH3
    5.44279,    # 38: NH
    4.72265,    # 39: NH2
    4.0025,     # 40: NH3
    4.22983,    # 41: OH
    3.50968,    # 42: OH2
    8.64641,    # 43: SH
], dtype=torch.float32)

# Enum indices for heavy atom types
FF_C = 5
FF_N = 6
FF_O = 7
FF_S = 15
FF_CH = 35
FF_CH2 = 36
FF_CH3 = 37
FF_NH = 38
FF_NH2 = 39
FF_NH3 = 40
FF_OH = 41
FF_SH = 43

# =============================================================================
# Atom-level form factor assignment for Chroma's 14-atom representation
# =============================================================================

# Chroma atom ordering per residue: [N, CA, C, O, ...side chain atoms...]
# Maximum 14 atoms per residue. Atom names per amino acid type:
CHROMA_ATOM_NAMES = {
    # residue_index: [atom_names for positions 0..13]
    "GLY": ["N", "CA", "C", "O"],
    "ALA": ["N", "CA", "C", "O", "CB"],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "SER": ["N", "CA", "C", "O", "CB", "OG"],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
    "CYS": ["N", "CA", "C", "O", "CB", "SG"],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
}


def _get_ff_type_for_atom(atom_name, residue_name):
    """
    Determine the heavy-atom form factor type index for a given atom
    in a given residue context. Mirrors pyFoXS logic.
    """
    # Carbon atoms
    if atom_name in ("C",):
        return FF_C
    if atom_name == "CA":
        return FF_CH2 if residue_name == "GLY" else FF_CH
    if atom_name == "CB":
        if residue_name in ("ILE", "THR", "VAL"):
            return FF_CH
        if residue_name == "ALA":
            return FF_CH3
        return FF_CH2
    if atom_name == "CG":
        if residue_name in ("ASN", "ASP", "HIS", "PHE", "TRP", "TYR"):
            return FF_C
        if residue_name == "LEU":
            return FF_CH
        return FF_CH2
    if atom_name == "CG1":
        if residue_name == "ILE":
            return FF_CH2
        if residue_name == "VAL":
            return FF_CH3
        return FF_CH2
    if atom_name == "CG2":
        return FF_CH3
    if atom_name == "CD":
        if residue_name in ("GLU", "GLN"):
            return FF_C
        return FF_CH2
    if atom_name == "CD1":
        if residue_name in ("LEU", "ILE"):
            return FF_CH3
        if residue_name in ("PHE", "TRP", "TYR"):
            return FF_CH
        return FF_C
    if atom_name == "CD2":
        if residue_name == "LEU":
            return FF_CH3
        if residue_name in ("PHE", "HIS", "TYR"):
            return FF_CH
        return FF_C
    if atom_name == "CE":
        if residue_name == "LYS":
            return FF_CH2
        if residue_name == "MET":
            return FF_CH3
        return FF_C
    if atom_name == "CE1":
        if residue_name in ("PHE", "HIS", "TYR"):
            return FF_CH
        return FF_C
    if atom_name == "CE2":
        if residue_name in ("PHE", "TYR"):
            return FF_CH
        return FF_C
    if atom_name in ("CE3", "CZ2", "CZ3", "CH2"):
        if residue_name == "TRP":
            return FF_CH
        return FF_C
    if atom_name == "CZ":
        if residue_name == "PHE":
            return FF_CH
        return FF_C

    # Nitrogen atoms
    if atom_name == "N":
        return FF_N if residue_name == "PRO" else FF_NH
    if atom_name == "ND1":
        return FF_NH if residue_name == "HIS" else FF_N
    if atom_name == "ND2":
        return FF_NH2 if residue_name == "ASN" else FF_N
    if atom_name in ("NH1", "NH2"):
        return FF_NH2 if residue_name == "ARG" else FF_N
    if atom_name == "NE":
        return FF_NH if residue_name == "ARG" else FF_N
    if atom_name == "NE1":
        return FF_NH if residue_name == "TRP" else FF_N
    if atom_name == "NE2":
        return FF_NH2 if residue_name == "GLN" else FF_N
    if atom_name == "NZ":
        return FF_NH3 if residue_name == "LYS" else FF_N

    # Oxygen atoms
    if atom_name in ("O", "OE1", "OE2", "OD1", "OD2", "OXT"):
        return FF_O
    if atom_name == "OG":
        return FF_OH if residue_name == "SER" else FF_O
    if atom_name == "OG1":
        return FF_OH if residue_name == "THR" else FF_O
    if atom_name == "OH":
        return FF_OH if residue_name == "TYR" else FF_O

    # Sulfur atoms
    if atom_name == "SD":
        return FF_S
    if atom_name == "SG":
        return FF_SH if residue_name == "CYS" else FF_S

    # Default: nitrogen
    return FF_N


def build_form_factor_table(device='cuda'):
    """
    Build a lookup table of form factors: shape (n_aa, 14)
    Indexed by [amino_acid_type, atom_position] matching Chroma's representation.

    Returns:
        ff_table: (n_aa, 14) tensor of form factor values
    """
    n_aa = len(constants.AA20)
    ff_table = torch.zeros(n_aa, 14, dtype=torch.float32)

    for aa_idx, letter in enumerate(constants.AA20):
        aa_name = _AA1_TO_AA3.get(letter, None)
        if aa_name is None:
            continue
        atom_names = CHROMA_ATOM_NAMES.get(aa_name, [])
        for atom_idx, atom_name in enumerate(atom_names):
            ff_type_idx = _get_ff_type_for_atom(atom_name, aa_name)
            ff_table[aa_idx, atom_idx] = HEAVY_ATOM_FORM_FACTORS[ff_type_idx]

    return ff_table.to(device)


# =============================================================================
# Differentiable SAXS Profile Computation
# =============================================================================

def debye_formula(coords, form_factors, q_values, mask=None):
    """
    Compute SAXS profile I(q) using the Debye formula.

    I(q) = sum_i sum_j f_i * f_j * sin(q * r_ij) / (q * r_ij)

    Args:
        coords: (N, 3) atom coordinates
        form_factors: (N,) form factors for each atom
        q_values: (n_q,) scattering vector magnitudes
        mask: (N,) boolean mask for valid atoms (optional)

    Returns:
        intensity: (n_q,) SAXS profile I(q)
    """
    if mask is not None:
        coords = coords[mask]
        form_factors = form_factors[mask]

    N = coords.shape[0]

    # Pairwise distance matrix: (N, N)
    diff = coords.unsqueeze(0) - coords.unsqueeze(1)  # (N, N, 3)
    dist_sq = torch.sum(diff ** 2, dim=-1)  # (N, N)
    dist = torch.sqrt(dist_sq + 1e-10)  # (N, N), small eps for numerical stability

    # Form factor product matrix: f_i * f_j, shape (N, N)
    ff_outer = form_factors.unsqueeze(0) * form_factors.unsqueeze(1)  # (N, N)

    # Upper triangle (avoid double counting + diagonal separately)
    triu_mask = torch.triu(torch.ones(N, N, device=coords.device, dtype=torch.bool), diagonal=1)
    dist_upper = dist[triu_mask]  # (N*(N-1)/2,)
    ff_upper = ff_outer[triu_mask]  # (N*(N-1)/2,)

    # Diagonal contribution: sum_i f_i^2
    diag_contribution = torch.sum(form_factors ** 2)  # scalar

    # Compute I(q) for each q value
    # q_values: (n_q,), dist_upper: (n_pairs,)
    qr = q_values.unsqueeze(1) * dist_upper.unsqueeze(0)  # (n_q, n_pairs)

    # sinc(x) = sin(x)/x, handle x=0
    sinc_qr = torch.where(
        qr.abs() > 1e-7,
        torch.sin(qr) / qr,
        torch.ones_like(qr)
    )

    # I(q) = diag + 2 * sum_{i<j} f_i*f_j * sinc(q*r_ij)
    intensity = diag_contribution + 2.0 * torch.sum(ff_upper.unsqueeze(0) * sinc_qr, dim=1)

    return intensity


def debye_formula_chunked(coords, form_factors, q_values, mask=None, chunk_size=2048):
    """
    Memory-efficient Debye formula using chunked pairwise computation.
    For large proteins where O(N^2) memory is prohibitive.

    Args:
        coords: (N, 3) atom coordinates
        form_factors: (N,) form factors for each atom
        q_values: (n_q,) scattering vector magnitudes
        mask: (N,) boolean mask for valid atoms (optional)
        chunk_size: number of atom pairs to process at once

    Returns:
        intensity: (n_q,) SAXS profile I(q)
    """
    if mask is not None:
        coords = coords[mask]
        form_factors = form_factors[mask]

    N = coords.shape[0]
    n_q = q_values.shape[0]
    device = coords.device

    # Diagonal contribution
    intensity = torch.sum(form_factors ** 2).expand(n_q).clone()

    # Process pairs in chunks of rows
    for i in range(0, N - 1, chunk_size):
        i_end = min(i + chunk_size, N - 1)
        coords_i = coords[i:i_end]  # (chunk, 3)
        ff_i = form_factors[i:i_end]  # (chunk,)

        # Distances from chunk to all atoms with index > i
        for j_start in range(i, N, chunk_size):
            j_end = min(j_start + chunk_size, N)

            # Only compute upper-triangle pairs
            coords_j = coords[j_start:j_end]  # (chunk_j, 3)
            ff_j = form_factors[j_start:j_end]  # (chunk_j,)

            # Pairwise distances for this chunk pair
            diff = coords_i.unsqueeze(1) - coords_j.unsqueeze(0)  # (ci, cj, 3)
            dist = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-10)  # (ci, cj)

            # Form factor products
            ff_prod = ff_i.unsqueeze(1) * ff_j.unsqueeze(0)  # (ci, cj)

            # Mask to only include upper triangle pairs (j > i)
            i_indices = torch.arange(i, i_end, device=device).unsqueeze(1)
            j_indices = torch.arange(j_start, j_end, device=device).unsqueeze(0)
            upper_mask = j_indices > i_indices

            dist_valid = dist[upper_mask]
            ff_valid = ff_prod[upper_mask]

            if dist_valid.numel() == 0:
                continue

            # Debye: sinc(qr) for all q values
            qr = q_values.unsqueeze(1) * dist_valid.unsqueeze(0)  # (n_q, n_pairs)
            sinc_qr = torch.where(
                qr.abs() > 1e-7,
                torch.sin(qr) / qr,
                torch.ones_like(qr)
            )

            intensity += 2.0 * torch.sum(ff_valid.unsqueeze(0) * sinc_qr, dim=1)

    return intensity


# =============================================================================
# High-level SAXS Solver for integration with model_refinement.py
# =============================================================================

class SAXSSolver:
    """
    Differentiable SAXS profile solver for use in the ADP-3D refinement loop.

    Computes a theoretical SAXS profile from Chroma's all-atom representation
    and compares it with an experimental profile.
    """

    def __init__(
        self,
        S,
        q_min=0.0,
        q_max=0.5,
        n_q=500,
        level='residue',
        device='cuda',
    ):
        """
        Args:
            S: (1, n_residues) sequence tensor (Chroma encoding)
            q_min: minimum scattering vector (Å⁻¹)
            q_max: maximum scattering vector (Å⁻¹)
            n_q: number of q points
            level: 'residue' for CA-only with residue form factors,
                   'atom' for all-atom with heavy atom form factors
            device: torch device
        """
        self.device = device
        self.level = level
        self.n_q = n_q

        # q values
        self.q_values = torch.linspace(q_min, q_max, n_q, device=device)

        # Build form factor lookup
        if level == 'residue':
            # Use residue-level form factors indexed by sequence
            ff_table = RESIDUE_FORM_FACTORS.to(device)
            # Precompute per-residue form factors from sequence
            self.form_factors = ff_table[S[0].long()]  # (n_residues,)
        elif level == 'atom':
            # Use all-atom form factors
            self.ff_table = build_form_factor_table(device)
            self.S = S
        else:
            raise ValueError(f"Unknown level: {level}. Use 'residue' or 'atom'.")

    def compute_profile(self, X, C=None, mask=None, chunked=False, chunk_size=2048):
        """
        Compute SAXS profile from coordinates.

        Args:
            X: coordinates tensor
                - If level='residue': (batch, n_residues, 4, 3) backbone coords
                  → uses CA atoms (index 1)
                - If level='atom': (batch, n_residues, 14, 3) all-atom coords
            C: (batch, n_residues) chain mask (optional, for masking valid residues)
            mask: (n_residues,) or (n_residues, 14) boolean mask for valid positions
            chunked: use memory-efficient chunked computation
            chunk_size: chunk size for chunked computation

        Returns:
            intensity: (batch, n_q) SAXS profiles
        """
        batch_size = X.shape[0]
        intensities = []

        for b in range(batch_size):
            if self.level == 'residue':
                # Extract CA coordinates: position 1 in the 4-atom backbone
                coords = X[b, :, 1, :]  # (n_residues, 3)
                ff = self.form_factors

                # Mask invalid residues
                if C is not None:
                    valid = (C[b] == 1)
                elif mask is not None:
                    valid = mask
                else:
                    valid = None

                if chunked:
                    I_q = debye_formula_chunked(coords, ff, self.q_values, mask=valid, chunk_size=chunk_size)
                else:
                    I_q = debye_formula(coords, ff, self.q_values, mask=valid)

            elif self.level == 'atom':
                # Flatten all atoms and apply form factors
                n_res = X.shape[1]
                coords_flat = X[b].reshape(-1, 3)  # (n_res * 14, 3)

                # Get form factors for each atom position
                S_expanded = self.S[0].long()  # (n_residues,)
                ff_per_atom = self.ff_table[S_expanded]  # (n_residues, 14)
                ff_flat = ff_per_atom.reshape(-1)  # (n_res * 14,)

                # Mask: only valid atoms (non-zero form factor and valid residue)
                valid_atoms = ff_flat.abs() > 1e-8
                if C is not None:
                    residue_valid = (C[b] == 1)  # (n_residues,)
                    atom_valid = residue_valid.unsqueeze(1).expand(-1, 14).reshape(-1)
                    valid_atoms = valid_atoms & atom_valid

                if chunked:
                    I_q = debye_formula_chunked(coords_flat, ff_flat, self.q_values, mask=valid_atoms, chunk_size=chunk_size)
                else:
                    I_q = debye_formula(coords_flat, ff_flat, self.q_values, mask=valid_atoms)

            intensities.append(I_q)

        return torch.stack(intensities, dim=0)  # (batch, n_q)

    def compute_loss(self, I_computed, I_experimental, error=None):
        """
        Compute fitting loss between computed and experimental SAXS profiles.

        Args:
            I_computed: (batch, n_q) computed profiles
            I_experimental: (n_q,) experimental profile
            error: (n_q,) experimental errors (optional)
            mode: 'chi2' for chi-squared, 'log' for log-intensity MSE,
                  'correlation' for negative correlation

        Returns:
            loss: scalar loss (summed over batch)
            loss_per_sample: (batch,) loss per sample
        """
        I_exp = I_experimental.unsqueeze(0)  # (1, n_q)

        # Scale computed to match experimental (linear least squares)
        if error is not None:
            w = 1.0 / (error.unsqueeze(0) ** 2 + 1e-10)
        else:
            w = torch.ones_like(I_exp)

        # Optimal scale factor c = sum(w * I_comp * I_exp) / sum(w * I_comp^2)
        c = torch.sum(w * I_computed * I_exp, dim=1, keepdim=True) / (
            torch.sum(w * I_computed ** 2, dim=1, keepdim=True) + 1e-10
        )
        residuals = I_exp - c * I_computed
        loss_per_sample = torch.sum(w * residuals ** 2, dim=1) / self.n_q
        loss = loss_per_sample.sum()

        return loss, loss_per_sample.detach()


def load_saxs_profile(filepath, device='cuda'):
    """
    Load an experimental SAXS profile from a .dat file.

    Expected format: whitespace-separated columns [q, I(q), error]

    Args:
        filepath: path to .dat file
        device: torch device

    Returns:
        q: (n_q,) scattering vector values
        intensity: (n_q,) intensity values
        error: (n_q,) error values (or None if not present)
    """
    data = np.loadtxt(filepath, comments='#')
    q = torch.tensor(data[:, 0], dtype=torch.float32, device=device)
    intensity = torch.tensor(data[:, 1], dtype=torch.float32, device=device)
    error = None
    if data.shape[1] >= 3:
        error = torch.tensor(data[:, 2], dtype=torch.float32, device=device)
    return q, intensity, error
