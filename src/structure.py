"""
Structure modules contains classes and functions to represent and manipulate molecular structures.

Copyright 2007-2022 IMP Inventors. All rights reserved.
"""

import numpy as np
import enum
from enum import IntEnum
from scipy.spatial import distance
import math

atom_entry_type_field_ = 0
atom_number_field_ = 6
atom_type_field_ = 12
atom_alt_loc_field_ = 16
atom_res_name_field_ = 17
atom_chain_id_field_ = 21
atom_res_number_field_ = 22
atom_res_insertion_field_ = 26
atom_xcoord_field_ = 30
atom_ycoord_field_ = 38
atom_zcoord_field_ = 46
atom_occupancy_field_ = 54
atom_temp_factor_field_ = 60
atom_element_field_ = 76
model_index_field_ = 6

class Element(IntEnum):
    UNKNOWN_ELEMENT = 0
    OH = -1
    H2O = -2
    H = 1
    He = 2
    Li = 3
    Be = 4
    B = 5
    C = 6
    N = 7
    O = 8
    F = 9
    Ne = 10
    Na = 11
    Mg = 12
    Al = 13
    Si = 14
    P = 15
    S = 16
    Cl = 17
    Ar = 18
    K = 19
    Ca = 20
    Sc = 21
    Ti = 22
    V = 23
    Cr = 24
    Mn = 25
    Fe = 26
    Co = 27
    Ni = 28
    Cu = 29
    Zn = 30
    Ga = 31
    Ge = 32
    As = 33
    Se = 34
    Br = 35
    Kr = 36
    Rb = 37
    Sr = 38
    Y = 39
    Zr = 40
    Nb = 41
    Mo = 42
    Tc = 43
    Ru = 44
    Rh = 45
    Pd = 46
    Ag = 47
    Cd = 48
    In = 49
    Sn = 50
    Sb = 51
    Te = 52
    I = 53
    Xe = 54
    Cs = 55
    Ba = 56
    La = 57
    Ce = 58
    Pr = 59
    Nd = 60
    Pm = 61
    Sm = 62
    Eu = 63
    Gd = 64
    Tb = 65
    Dy = 66
    Ho = 67
    Er = 68
    Tm = 69
    Yb = 70
    Lu = 71
    Hf = 72
    Ta = 73
    W = 74
    Re = 75
    Os = 76
    Ir = 77
    Pt = 78
    Au = 79
    Hg = 80
    Tl = 81
    Pb = 82
    Bi = 83
    Po = 84
    At = 85
    Rn = 86
    Fr = 87
    Ra = 88
    Ac = 89
    Th = 90
    Pa = 91
    U = 92
    Np = 93
    Pu = 94
    Am = 95
    Cm = 96
    Bk = 97
    Cf = 98
    Es = 99
    Fm = 100
    Md = 101
    No = 102
    Lr = 103
    Db = 104
    Jl = 105
    Rf = 106

ElementType = {Element.UNKNOWN_ELEMENT : 'UNKNOWN',
                Element.H : 'H',
                Element.C : 'C',
                Element.N : 'N',
                Element.O : 'O',
                Element.S : 'S',
                Element.Se : 'SE',
                Element.P : 'P',
                Element.Mg: 'MG',
                }

AtomType = {'N' : Element.N,
            'H' : Element.H,
            '1H' : Element.H,
            'H1' : Element.H,
            '2H' : Element.H,
            'H2' : Element.H,
            '3H' : Element.H,
            'H3' : Element.H,
            'C' : Element.C,
            'O' : Element.O,
            'OXT' : Element.O,
            'OT1' : Element.O,
            'OT2' : Element.O,
            'CH3' : Element.C,
            'CH' : Element.C,

            'S' : Element.S,
            'SE' : Element.Se,
            'MG' : Element.Mg,

            'CA' : Element.C,
            'HA' : Element.H,
            'HA1' : Element.H,
            'HA2' : Element.H,
            'HA3' : Element.H,

            'CB' : Element.C,
            'HB' : Element.H,
            'HB1' : Element.H,
            'HB2' : Element.H,
            'HB3' : Element.H,

            'CG' : Element.C,
            'CG1' : Element.C,
            'CG2' : Element.C,
            'HG' : Element.H,
            'HG1' : Element.H,
            'HG2' : Element.H,
            'HG3' : Element.H,
            'HG11' : Element.H,
            'HG21' : Element.H,
            'HG31' : Element.H,
            'HG12' : Element.H,
            'HG13' : Element.H,
            'HG22' : Element.H,
            'HG23' : Element.H,
            'HG32' : Element.H,
            'OG' : Element.O,
            'OG1' : Element.O,
            'SG' : Element.S,

            'CD' : Element.C,
            'CD1' : Element.C,
            'CD2' : Element.C,
            'HD' : Element.H,
            'HD1' : Element.H,
            'HD2' : Element.H,
            'HD3' : Element.H,
            'HD11' : Element.H,
            'HD21' : Element.H,
            'HD31' : Element.H,
            'HD12' : Element.H,
            'HD13' : Element.H,
            'HD22' : Element.H,
            'HD23' : Element.H,
            'HD32' : Element.H,
            'SD' : Element.S,
            'OD1' : Element.O,
            'OD2' : Element.O,
            'ND1' : Element.N,
            'ND2' : Element.N,

            'CE' : Element.C,
            'CE1' : Element.C,
            'CE2' : Element.C,
            'CE3' : Element.C,
            'HE' : Element.H,
            'HE1' : Element.H,
            'HE2' : Element.H,
            'HE3' : Element.H,
            'HE21' : Element.H,
            'HE22' : Element.H,
            'OE1' : Element.O,
            'OE2' : Element.O,
            'NE' : Element.N,
            'NE1' : Element.N,
            'NE2' : Element.N,

            'CZ' : Element.C,
            'CZ2' : Element.C,
            'CZ3' : Element.C,
            'NZ' : Element.N,
            'HZ' : Element.H,
            'HZ1' : Element.H,
            'HZ2' : Element.H,
            'HZ3' : Element.H,

            'CH2' : Element.C,
            'NH1' : Element.N,
            'NH2' : Element.N,
            'OH' : Element.O,
            'HH' : Element.H,

            'HH11' : Element.H,
            'HH21' : Element.H,
            'HH2' : Element.H,
            'HH12' : Element.H,
            'HH22' : Element.H,
            'HH23' : Element.H,
            'HH33' : Element.H,
            'HH13' : Element.H,

            'P' : Element.P,
            'OP1' : Element.O,
            'OP2' : Element.O,
            'OP3' : Element.O,
            'O5p' : Element.O,
            "O5'" : Element.O,
            'C5p' : Element.C,
            "C5'" : Element.C,
            'H5pp' : Element.H,
            "H5''" : Element.H,
            'C4p' : Element.C,
            "C4'" : Element.C,
            'H4p' : Element.H,
            "H4'" : Element.H,
            'H5p' : Element.H,
            "H5'" : Element.H,
            'O4p' : Element.O,
            "O4'" : Element.O,
            'C1p' : Element.C,
            "C1'" : Element.C,
            'H1p' : Element.H,
            "H1'" : Element.H,
            'C3p' : Element.C,
            "C3'" : Element.C,
            'H3p' : Element.H,
            "H3'" : Element.H,
            'O3p' : Element.O,
            "O3'" : Element.O,
            'C2p' : Element.C,
            "C2'" : Element.C,
            'H2p' : Element.H,
            "H2'" : Element.H,
            'H2pp' : Element.H,
            "H2''" : Element.H,
            'O2p' : Element.O,
            "O2'" : Element.O,
            'HO2p' : Element.H,
            "HO2'" : Element.H,
            'N9' : Element.N,
            'C8' : Element.C,
            'H8' : Element.H,
            'N7' : Element.N,
            'C5' : Element.C,
            'C4' : Element.C,
            'N3' : Element.N,
            'C2' : Element.C,
            'N1' : Element.N,
            'C6' : Element.C,
            'N6' : Element.N,
            'H61' : Element.H,
            'H62' : Element.H,
            'O6' : Element.O,

            'N2' : Element.N,
            'NT' : Element.N,
            'H21' : Element.H,
            'H22' : Element.H,

            'H6' : Element.H,
            'H5' : Element.H,
            'O2' : Element.O,
            'N4' : Element.N,
            'H41' : Element.H,
            'H42' : Element.H,

            'O4' : Element.O,
            'C7' : Element.C,
            'H71' : Element.H,
            'H72' : Element.H,
            'H73' : Element.H,
            'O1A' : Element.O,
            'O2A' : Element.O,
            'O3A' : Element.O,
            'O1B' : Element.O,
            'O2B' : Element.O,
            'O3B' : Element.O,
            'CAY' : Element.C,
            'CY' : Element.C,
            'OY' : Element.O,
            'CAT' : Element.C,

            'NO2' : Element.N,

            'UNKNOWN' : Element.UNKNOWN_ELEMENT}

def add_atom_type(name, e):
    if name in AtomType:
        raise Exception("An AtomType with that name already exists: " + name)
    try:
        AtomType[name] = AtomType[e]
    except KeyError:
        AtomType[name] = getattr(Element, e)
    return name

def get_element_for_atom_type(at):
    if at not in AtomType:
        raise Exception("Invalid AtomType index: " + str(at))
    return ElementType[AtomType[at]]

def get_atom_type_exists(name):
    return name in AtomType

def get_residue(d, nothrow=False):
    mhd = d
    while mhd is not None:
        mhd = mhd.parent
        if isinstance(mhd, Residue):
            return mhd
    if nothrow:
        return Residue()
    else:
        print("Atom is not the child of a residue: " + str(d))

class Atom:
    def __init__(self, name='', type='', element='', atom_index=0, coord=(0,0,0), occupancy=None, temp_factor=None):
        self.atom_type = type
        # self.element = element
        self.atom_index = atom_index
        self.coordinates = coord
        self.occupancy = occupancy
        self.temp_factor = temp_factor
        self.atom_name = name
        self.parent = None
        self.radius = None
        self.residue = None
        self.cache_attributes = {}

    def setup_particle(self, radius):
        self.radius = radius

    @staticmethod
    def get_atom_type_key():
        return "atom_type"

    @staticmethod
    def get_element_key():
        return "element"

    @staticmethod
    def get_input_index_key():
        return "pdb_atom_index"

    @staticmethod
    def get_occupancy_key():
        return "occupancy"

    @staticmethod
    def get_temperature_factor_key():
        return "tempFactor"

class Chain:
    def __init__(self, chain_id):
        self.name = "Chain"
        self.chain_id = chain_id
        self.parent = None
        self.residues = []

def compute_max_distance(particles):
    max_dist2 = 0.0
    coordinates = [particle.coordinates for particle in particles]

    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            dist2 = distance.sqeuclidean(coordinates[i], coordinates[j])
            if dist2 > max_dist2:
                max_dist2 = dist2

    return np.sqrt(max_dist2)

def compute_max_distance_between_particles(particles1, particles2):
    max_dist2 = 0.0
    coordinates1 = [particle.coordinates for particle in particles1]
    coordinates2 = [particle.coordinates for particle in particles2]

    for i in range(len(coordinates1)):
        for j in range(len(coordinates2)):
            dist2 = distance.sqeuclidean(coordinates1[i], coordinates2[j])
            if dist2 > max_dist2:
                max_dist2 = dist2

    return np.sqrt(max_dist2)

class ResidueType(IntEnum):
    UNK = 0
    GLY = 1
    ALA = 2
    VAL = 3
    LEU = 4
    ILE = 5
    SER = 6
    THR = 7
    CYS = 8
    MET = 9
    PRO = 10
    ASP = 11
    ASN = 12
    GLU = 13
    GLN = 14
    LYS = 15
    ARG = 16
    HIS = 17
    PHE = 18
    TYR = 19
    TRP = 20
    ACE = 21
    NH2 = 22
    MSE = 23
    ADE = 24
    URA = 25
    CYT = 26
    GUA = 27
    THY = 28
    DADE = 29
    DURA = 30
    DCYT = 31
    DGUA = 32
    DTHY = 33
    HOH = 34
    HEME = 35
    POP = 36

residue_types = {
        "UNK": ResidueType.UNK,
        "GLY": ResidueType.GLY,
        "ALA": ResidueType.ALA,
        "VAL": ResidueType.VAL,
        "LEU": ResidueType.LEU,
        "ILE": ResidueType.ILE,
        "SER": ResidueType.SER,
        "THR": ResidueType.THR,
        "CYS": ResidueType.CYS,
        "MET": ResidueType.MET,
        "PRO": ResidueType.PRO,
        "ASP": ResidueType.ASP,
        "ASN": ResidueType.ASN,
        "GLU": ResidueType.GLU,
        "GLN": ResidueType.GLN,
        "LYS": ResidueType.LYS,
        "ARG": ResidueType.ARG,
        "HIS": ResidueType.HIS,
        "PHE": ResidueType.PHE,
        "TYR": ResidueType.TYR,
        "TRP": ResidueType.TRP,
        "ACE": ResidueType.ACE,
        "NH2": ResidueType.NH2,
        "MSE": ResidueType.MSE,
        "ADE": ResidueType.ADE,
        "URA": ResidueType.URA,
        "CYT": ResidueType.CYT,
        "GUA": ResidueType.GUA,
        "THY": ResidueType.THY,
        "DADE": ResidueType.DADE,
        "DURA": ResidueType.DURA,
        "DCYT": ResidueType.DCYT,
        "DGUA": ResidueType.DGUA,
        "DTHY": ResidueType.DTHY,
        "A" : ResidueType.ADE,
        "U" : ResidueType.URA,
        "C" : ResidueType.CYT,
        "G" : ResidueType.GUA,
        "T" : ResidueType.THY,
        "DA": ResidueType.DADE,
        "DU": ResidueType.DURA,
        "DC": ResidueType.DCYT,
        "DG": ResidueType.DGUA,
        "DT": ResidueType.DTHY,
        "HOH": ResidueType.HOH,
        "HEME": ResidueType.HEME,
        "POP": ResidueType.POP
    }
def get_residue_type(code):
    return residue_types[code]

residue_string_types = {
        ResidueType.UNK : "UNK",
        ResidueType.GLY : "GLY",
        ResidueType.ALA : "ALA",
        ResidueType.VAL : "VAL",
        ResidueType.LEU : "LEU",
        ResidueType.ILE : "ILE",
        ResidueType.SER : "SER",
        ResidueType.THR : "THR",
        ResidueType.CYS : "CYS",
        ResidueType.MET : "MET",
        ResidueType.PRO : "PRO",
        ResidueType.ASP : "ASP",
        ResidueType.ASN : "ASN",
        ResidueType.GLU : "GLU",
        ResidueType.GLN : "GLN",
        ResidueType.LYS : "LYS",
        ResidueType.ARG : "ARG",
        ResidueType.HIS : "HIS",
        ResidueType.PHE : "PHE",
        ResidueType.TYR : "TYR",
        ResidueType.TRP : "TRP",
        ResidueType.ACE : "ACE",
        ResidueType.NH2 : "NH2",
        ResidueType.MSE : "MSE",
        ResidueType.ADE : "ADE",
        ResidueType.URA : "URA",
        ResidueType.CYT : "CYT",
        ResidueType.GUA : "GUA",
        ResidueType.THY : "THY",
        ResidueType.DADE : "DADE",
        ResidueType.DURA : "DURA",
        ResidueType.DCYT : "DCYT",
        ResidueType.DGUA : "DGUA",
        ResidueType.DTHY : "DTHY",
        ResidueType.HOH : "HOH",
        ResidueType.HEME : "HEME",
        ResidueType.POP : "POP"
    }
def residue_to_string(code):
    return residue_string_types[code]

one_letter_codes = {
        ResidueType.UNK: "X",
        ResidueType.GLY: "G",
        ResidueType.ALA: "A",
        ResidueType.VAL: "V",
        ResidueType.LEU: "L",
        ResidueType.ILE: "I",
        ResidueType.SER: "S",
        ResidueType.THR: "T",
        ResidueType.CYS: "C",
        ResidueType.MET: "M",
        ResidueType.PRO: "P",
        ResidueType.ASP: "D",
        ResidueType.ASN: "N",
        ResidueType.GLU: "E",
        ResidueType.GLN: "Q",
        ResidueType.LYS: "K",
        ResidueType.ARG: "R",
        ResidueType.HIS: "H",
        ResidueType.PHE: "F",
        ResidueType.TYR: "Y",
        ResidueType.TRP: "W",
        ResidueType.ACE: "ACE",
        ResidueType.NH2: "NH2",
        ResidueType.MSE: "MSE",
        ResidueType.ADE: "A",
        ResidueType.URA: "U",
        ResidueType.CYT: "C",
        ResidueType.GUA: "G",
        ResidueType.THY: "T",
        ResidueType.DADE: "DA",
        ResidueType.DURA: "DU",
        ResidueType.DCYT: "DC",
        ResidueType.DGUA: "DG",
        ResidueType.DTHY: "DT",
        ResidueType.HOH: "HOH",
        ResidueType.HEME: "HEME",
        ResidueType.POP: "POP"
    }

def get_one_letter_code(residue_type):
    return one_letter_codes.get(residue_type)

residue_masses = {
        ResidueType.ALA: 71.079,
        ResidueType.ARG: 156.188,
        ResidueType.ASP: 115.089,
        ResidueType.ASN: 114.104,
        ResidueType.CYS: 103.144,
        ResidueType.GLN: 128.131,
        ResidueType.GLU: 129.116,
        ResidueType.GLY: 57.052,
        ResidueType.HIS: 137.142,
        ResidueType.ILE: 113.160,
        ResidueType.LEU: 113.160,
        ResidueType.LYS: 128.174,
        ResidueType.MET: 131.198,
        ResidueType.PHE: 147.177,
        ResidueType.PRO: 97.117,
        ResidueType.SER: 87.078,
        ResidueType.THR: 101.105,
        ResidueType.TYR: 163.170,
        ResidueType.TRP: 186.213,
        ResidueType.VAL: 99.133,
        ResidueType.UNK: 113.160,
        ResidueType.ADE: 507.2,
        ResidueType.URA: 484.2,
        ResidueType.CYT: 483.2,
        ResidueType.GUA: 523.2,
        ResidueType.DADE: 491.2,
        ResidueType.DTHY: 482.2,
        ResidueType.DCYT: 467.2,
        ResidueType.DGUA: 507.2,
        ResidueType.POP: 686.97
    }

def get_mass(residue_type):
    return residue_masses.get(residue_type, math.nan)

class Residue:
    def __init__(self, residue_type, index=-1, insertion_code=" "):
        self.name = "Residue" + str(index)
        self.residue_type = residue_type
        self.index = index
        self.insertion_code = insertion_code
        self.parent = None
        self.atoms = []

    def add_child(self, atom):
        self.atoms.append(atom)

    @property
    def residue_type(self):
        return self._residue_type

    @residue_type.setter
    def residue_type(self, value):
        if isinstance(value, ResidueType):
            self._residue_type = value
        elif isinstance(value, str):
            try:
                self._residue_type = get_residue_type(value)
            except KeyError:
                residue_types[value] = value
                residue_string_types[value] = value
                self._residue_type = get_residue_type(value)

        else:
            raise ValueError("Invalid residue type.")

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        self._index = int(value)


class FormFactorType(enum.Enum):
    ALL_ATOMS = 0
    HEAVY_ATOMS = 1
    CA_ATOMS = 2
    RESIDUES = 3

class FormFactor:
    def __init__(self, ff=0, vacuum_ff=0, dummy_ff=0):
        self.ff_ = ff
        self.vacuum_ff_ = vacuum_ff
        self.dummy_ff_ = dummy_ff

class FormFactorTable:
    # electron density of solvent - default=0.334 e/A^3 (H2O)
    rho_ = 0.334

    class FormFactorAtomType(enum.Enum):
        H = 0
        He = 1
        Li = 2
        Be = 3
        B = 4
        C = 5
        N = 6
        O = 7
        F = 8
        Ne = 9
        Na = 10
        Mg = 11
        Al = 12
        Si = 13
        P = 14
        S = 15
        Cl = 16
        Ar = 17
        K = 18
        Ca = 19
        Cr = 20
        Mn = 21
        Fe = 22
        Co = 23
        Ni = 24
        Cu = 25
        Zn = 26
        Se = 27
        Br = 28
        Ag = 29
        I = 30
        Ir = 31
        Pt = 32
        Au = 33
        Hg = 34
        ALL_ATOM_SIZE = 35
        CH = 35
        CH2 = 36
        CH3 = 37
        NH = 38
        NH2 = 39
        NH3 = 40
        OH = 41
        OH2 = 42
        SH = 43
        HEAVY_ATOM_SIZE = 44
        UNK = 45

    element_ff_type_map_ = {member.name.upper(): member for member in FormFactorAtomType}
    residue_type_form_factor_map_ = {
        ResidueType.ALA : FormFactor(9.037, 37.991, 28.954),
        ResidueType.ARG : FormFactor(23.289, 84.972, 61.683),
        ResidueType.ASP : FormFactor(20.165, 58.989, 38.824),
        ResidueType.ASN : FormFactor(19.938, 59.985, 40.047),
        ResidueType.CYS : FormFactor(18.403, 53.991, 35.588),
        ResidueType.GLN : FormFactor(19.006, 67.984, 48.978),
        ResidueType.GLU : FormFactor(19.233, 66.989, 47.755),
        ResidueType.GLY : FormFactor(10.689, 28.992, 18.303),
        ResidueType.HIS : FormFactor(21.235, 78.977, 57.742),
        ResidueType.ILE : FormFactor(6.241, 61.989, 55.748),
        ResidueType.LEU : FormFactor(6.241, 61.989, 55.748),
        ResidueType.LYS : FormFactor(10.963, 70.983, 60.020),
        ResidueType.MET : FormFactor(16.539, 69.989, 53.450),
        ResidueType.PHE : FormFactor(9.206, 77.986, 68.7806),
        ResidueType.PRO : FormFactor(8.613, 51.9897, 43.377),
        ResidueType.SER : FormFactor(13.987, 45.991, 32.004),
        ResidueType.THR : FormFactor(13.055, 53.99, 40.935),
        ResidueType.TYR : FormFactor(14.156, 85.986, 71.83),
        ResidueType.TRP : FormFactor(14.945, 98.979, 84.034),
        ResidueType.VAL : FormFactor(7.173, 53.9896, 46.817),
        ResidueType.POP : FormFactor(45.616, 365.99, 320.41),
        ResidueType.UNK : FormFactor(9.037, 37.991, 28.954)
    }

    zero_form_factors_ = [
        -0.720147, -0.720228,
        #   H       He - periodic table line 1
        1.591, 2.591, 3.591, 0.50824, 6.16294, 4.94998, 7.591, 6.993,
        # Li     Be      B     C       N        O       F      Ne - line 2
        7.9864, 8.9805, 9.984, 10.984, 13.0855, 9.36656, 13.984, 16.591,
        #  Na      Mg        Al       Si        P        S       Cl    Ar - line 3
        15.984, 14.9965, 20.984, 21.984, 20.9946, 23.984,
        # K       Ca2+       Cr      Mn      Fe2+      Co - line 4
        24.984, 25.984, 24.9936, 30.9825, 31.984, 43.984, 49.16,
        # Ni     Cu          Zn2+      Se       Br       Ag      I
        70.35676, 71.35676, 72.324, 73.35676,
        # Ir         Pt      Au      Hg
        -0.211907, -0.932054, -1.6522, 5.44279, 4.72265, 4.0025, 4.22983, 3.50968,
        8.64641
        #  CH        CH2        CH3     NH       NH2       NH3     OH       OH2
        # SH
    ]

    vacuum_zero_form_factors_ = [
        #   H       He - periodic table line 1
        0.999953, 0.999872,
        # Li  Be    B     C       N       O       F     Ne - line 2
        2.99, 3.99, 4.99, 5.9992, 6.9946, 7.9994, 8.99, 9.999,
        #  Na     Mg     Al     Si      P        S        Cl     Ar - line 3
        10.9924, 11.9865, 12.99, 13.99, 14.9993, 15.9998, 16.99, 17.99,
        # K    Ca2+     Cr     Mn     Fe2+     Co - line 4
        18.99, 18.0025, 23.99, 24.99, 24.0006, 26.99,
        # Ni   Cu      Zn2+     Se     Br - line 4 cont.
        27.99, 28.99, 27.9996, 33.99, 34.99,
        # Ag    I       Ir     Pt      Au     Hg - some elements from lines 5, 6
        46.99, 52.99, 76.99, 77.99, 78.9572, 79.99,
        # CH      CH2     CH3     NH       NH2       NH3     OH      OH2      SH
        6.99915, 7.99911, 8.99906, 7.99455, 8.99451, 9.99446, 8.99935, 9.9993,
        16.9998
    ]

    dummy_zero_form_factors_ = [
        1.7201, 1.7201, 1.399, 1.399, 1.399, 5.49096, 0.83166, 3.04942,
        1.399, 3.006,
        #  H     He     Li?    Be?    B?       C        N        O      F?     Ne
        3.006, 3.006, 3.006, 3.006, 1.91382, 6.63324, 3.006, 1.399,
        # Na     Mg    Al?    Si?      P        S      Cl?    Ar?
        3.006, 3.006, 3.006, 3.006, 3.006, 3.006,
        # K?   Ca2+    Cr?    Mn?   Fe2+   Co?
        3.006, 3.006, 3.006, 3.006, 3.006,
        # Ni?   Cu?   Zn2+    Se     Br?
        3.006, 3.83, 6.63324, 6.63324, 6.63324, 6.63324,
        # Ag?   I?       Ir?      Pt?       Au      Hg
        7.21106, 8.93116, 10.6513, 2.55176, 4.27186, 5.99196, 4.76952, 6.48962,
        8.35334
        #  CH       CH2      CH3     NH       NH2       NH3     OH       OH2   SH
    ]

    # form_factor_type_key_ = IntKey()

    form_factors_coefficients_ = []
    form_factors_ = []
    vacuum_form_factors_ = []
    dummy_form_factors_ = []
    min_q_ = 0.0
    max_q_ = 0.0
    delta_q_ = 0.0
    # warn_context_ = WarningContext()

    class AtomFactorCoefficients:
        def __init__(self, atom_type, a, c, b, excl_vol):
            self.atom_type_ = atom_type
            self.a_ = a
            self.c_ = c
            self.b_ = b
            self.excl_vol_ = excl_vol

        def __str__(self):
            return (
                f"Atom Type: {self.atom_type_}\n"
                f"a values: {self.a_}\n"
                f"c value: {self.c_}\n"
                f"b values: {self.b_}\n"
                f"Excluded Volume: {self.excl_vol_}"
            )

    def __init__(self, table_name="", min_q=0.0, max_q=0.0, delta_q=0.0):
        if table_name:
            self.read_form_factor_table(table_name)
        self.min_q_ = min_q
        self.max_q_ = max_q
        self.delta_q_ = delta_q
        self.dummy_form_factors_ = self.dummy_zero_form_factors_
        self.vacuum_form_factors_ = self.vacuum_zero_form_factors_
        self.zero_form_factors_ = self.zero_form_factors_
        self.form_factor_type_key_ = "form_factor_type_key"


    def get_form_factor(self, p, ff_type=FormFactorType.HEAVY_ATOMS):
        if ff_type in (FormFactorType.CA_ATOMS, FormFactorType.RESIDUES):  # residue level form factors
            residue_type = p.residue.residue_type
            return self.get_form_factor_r(residue_type)

        # atomic form factor, initialization by request
        if self.form_factor_type_key_ in p.cache_attributes:
            return self.zero_form_factors_[p.cache_attributes[self.form_factor_type_key_].value]

        ff_atom_type = self.get_form_factor_atom_type(p, ff_type)
        if ff_atom_type.value >= self.FormFactorAtomType.HEAVY_ATOM_SIZE.value:
            print("Can't find form factor for particle",
                  p.atom_type,
                  "using default")
            ff_atom_type = self.FormFactorAtomType.N

        form_factor = self.zero_form_factors_[ff_atom_type.value]
        p.cache_attributes[self.form_factor_type_key_] = ff_atom_type
        return form_factor

    def get_form_factor_r(self, rt):
        if rt in self.residue_type_form_factor_map_:
            return self.residue_type_form_factor_map_[rt].ff_
        else:
            print("Can't find form factor for residue", rt,
                  "using default value of ALA")
            return self.residue_type_form_factor_map_[self.FormFactorAtomType.UNK].ff_

    def get_vacuum_form_factor(self, p, ff_type):
        if ff_type == FormFactorType.CA_ATOMS:  # residue level form factors
            residue_type = p.residue.residue_type if p.residue else None
            if residue_type:
                return self.get_vacuum_form_factor_r(residue_type)

        if ff_type == FormFactorType.RESIDUES:  # residue level form factors
            residue_type = p.residue.residue_type if p.residue else None
            if residue_type:
                return self.get_form_factor_r(residue_type)

        if self.form_factor_type_key_ in p.cache_attributes:
            return self.vacuum_zero_form_factors_[p.cache_attributes[self.form_factor_type_key_].value]

        ff_atom_type = self.get_form_factor_atom_type(p, ff_type)
        form_factor = self.vacuum_zero_form_factors_[ff_atom_type.value]
        p.cache_attributes[self.form_factor_type_key_] = ff_atom_type
        return form_factor

    def get_vacuum_form_factor_r(self, rt):
        if rt in self.residue_type_form_factor_map_:
            return self.residue_type_form_factor_map_[rt].vacuum_ff_
        else:
            print("Can't find form factor for residue", rt, "using default value of ALA")
            return self.residue_type_form_factor_map_[ResidueType.UNK].vacuum_ff_

    def get_dummy_form_factor(self, p, ff_type):
        if ff_type == FormFactorType.CA_ATOMS:
            # Residue level form factors
            residue_type = p.residue.residue_type if p.residue else None
            if residue_type:
                return self.get_dummy_form_factor_r(residue_type)

        if ff_type == FormFactorType.RESIDUES:
            # Residue level form factors
            residue_type = p.residue.residue_type if p.residue else None
            if residue_type:
                return self.get_form_factor_r(residue_type)

        if self.form_factor_type_key_ in p.cache_attributes:
            return self.dummy_zero_form_factors_[p.cache_attributes[self.form_factor_type_key_].value]

        ff_atom_type = self.get_form_factor_atom_type(p, ff_type)
        form_factor = self.dummy_zero_form_factors_[ff_atom_type.value]
        p.cache_attributes[self.form_factor_type_key_] = ff_atom_type
        return form_factor

    def get_dummy_form_factor_r(self, rt):
        if rt in self.residue_type_form_factor_map_:
            return self.residue_type_form_factor_map_[rt].dummy_ff_
        else:
            print("Can't find form factor for residue ", rt,
                    " using default value of ALA")
            return self.residue_type_form_factor_map_[ResidueType.UNK].dummy_ff_

    def get_form_factor_atom_type(self, p, ff_type):
        ad = p
        residue_type = ad.residue.residue_type if ad.residue is not None else ""
        residue_type = residue_to_string(residue_type)
        atom_type = ad.atom_name
        # Find FormFactorAtomType
        ret_type = self.element_ff_type_map_[ad.atom_type] if ad.atom_type in self.element_ff_type_map_ else self.FormFactorAtomType.UNK

        if ff_type == FormFactorType.HEAVY_ATOMS:
            if ret_type == self.FormFactorAtomType.C:
                ret_type = self.get_carbon_atom_type(atom_type, residue_type)
            elif ret_type == self.FormFactorAtomType.N:
                ret_type = self.get_nitrogen_atom_type(atom_type, residue_type)
            elif ret_type == self.FormFactorAtomType.O:
                ret_type = self.get_oxygen_atom_type(atom_type, residue_type)
            elif ret_type == self.FormFactorAtomType.S:
                ret_type = self.get_sulfur_atom_type(atom_type, residue_type)

        if ret_type.value >= self.FormFactorAtomType.HEAVY_ATOM_SIZE.value:
            print("Can't find form factor for particle "
                    + ad.atom_type
                    + " using default value of nitrogen\n")
            ret_type = self.FormFactorAtomType.N

        return ret_type

    def get_water_form_factor(self):
        return self.zero_form_factors_[self.FormFactorAtomType.OH2.value]

    def get_vacuum_water_form_factor(self):
        return self.vacuum_zero_form_factors_[self.FormFactorAtomType.OH2.value]

    def get_dummy_water_form_factor(self):
        return self.dummy_zero_form_factors_[self.FormFactorAtomType.OH2.value]

    def get_form_factors(self, p, ff_type=FormFactorType.HEAVY_ATOMS):
        return self.get_form_factors(p, ff_type)

    def get_vacuum_form_factors(self, p, ff_type=FormFactorType.HEAVY_ATOMS):
        return self.get_vacuum_form_factors(p, ff_type)

    def get_dummy_form_factors(self, p, ff_type=FormFactorType.HEAVY_ATOMS):
        return self.get_dummy_form_factors(p, ff_type)

    def get_water_form_factors(self):
        return self.form_factors_[self.FormFactorAtomType.OH2.value]

    def get_water_vacuum_form_factors(self):
        return self.vacuum_form_factors_[self.FormFactorAtomType.OH2.value]

    def get_water_dummy_form_factors(self):
        return self.dummy_form_factors_[self.FormFactorAtomType.OH2.value]

    def get_radius(self, p, ff_type):
        # dummy_zero_form_factor = volume * rho
        # volume = 4/3 * pi * r^3
        # r^3 = 3*dummy_zero_form_factor / 4*pi*rho
        one_third = 1.0 / 3
        c = 3.0 / (4 * math.pi * self.rho_)
        form_factor = self.get_dummy_form_factor(p, ff_type)
        return math.pow(c * form_factor, one_third)

    def get_volume(self, p, ff_type):
        # dummy_zero_form_factor = volume * rho
        form_factor = self.get_dummy_form_factor(p, ff_type)
        return form_factor / self.rho_

    def show(self, out=print, prefix=""):
        self.show(out, prefix)

    def get_carbon_atom_type(self, atom_type, residue_type):
        # protein atoms
        # CH
        if atom_type == "CH":
            return self.FormFactorAtomType.CH
        # CH2
        if atom_type == "CH2":
            return self.FormFactorAtomType.CH2
        # CH3
        if atom_type == "CH3":
            return self.FormFactorAtomType.CH3
        # C
        if atom_type == "C":
            return self.FormFactorAtomType.C

        # CA
        if atom_type == "CA":
            if residue_type == "GLY":
                return self.FormFactorAtomType.CH2  # Glycine has 2 hydrogens
            return self.FormFactorAtomType.CH
        # CB
        if atom_type == "CB":
            if (residue_type == "ILE" or residue_type == "THR" or
                    residue_type == "VAL"):
                return self.FormFactorAtomType.CH
            if residue_type == "ALA":
                return self.FormFactorAtomType.CH3
            return self.FormFactorAtomType.CH2
        # CG1
        if atom_type == "CG":
            if (residue_type == "ASN" or residue_type == "ASP" or
                    residue_type == "HIS" or residue_type == "PHE" or
                    residue_type == "TRP" or residue_type == "TYR"):
                return self.FormFactorAtomType.C
            if residue_type == "LEU":
                return self.FormFactorAtomType.CH
            return self.FormFactorAtomType.CH2
        # CG1
        if atom_type == "CG1":
            if residue_type == "ILE":
                return self.FormFactorAtomType.CH2
            if residue_type == "VAL":
                return self.FormFactorAtomType.CH3
        # CG2 - only VAL, ILE, and THR
        if atom_type == "CG2":
            return self.FormFactorAtomType.CH3
        # CD
        if atom_type == "CD":
            if residue_type == "GLU" or residue_type == "GLN":
                return self.FormFactorAtomType.C
            return self.FormFactorAtomType.CH2
        # CD1
        if atom_type == "CD1":
            if residue_type == "LEU" or residue_type == "ILE":
                return self.FormFactorAtomType.CH3
            if (residue_type == "PHE" or residue_type == "TRP" or
                    residue_type == "TYR"):
                return self.FormFactorAtomType.CH
            return self.FormFactorAtomType.C
        # CD2
        if atom_type == "CD2":
            if residue_type == "LEU":
                return self.FormFactorAtomType.CH3
            if (residue_type == "PHE" or residue_type == "HIS" or
                    residue_type == "TYR"):
                return self.FormFactorAtomType.CH
            return self.FormFactorAtomType.C
        # CE
        if atom_type == "CE":
            if residue_type == "LYS":
                return self.FormFactorAtomType.CH2
            if residue_type == "MET":
                return self.FormFactorAtomType.CH3
            return self.FormFactorAtomType.C
        # CE1
        if atom_type == "CE1":
            if (residue_type == "PHE" or residue_type == "HIS" or
                    residue_type == "TYR"):
                return self.FormFactorAtomType.CH
            return self.FormFactorAtomType.C
        # CE2
        if atom_type == "CE2":
            if residue_type == "PHE" or residue_type == "TYR":
                return self.FormFactorAtomType.CH
            return self.FormFactorAtomType.C
        # CZ
        if atom_type == "CZ":
            if residue_type == "PHE":
                return self.FormFactorAtomType.CH
            return self.FormFactorAtomType.C
        # CZ2, CZ3, CE3
        if (atom_type == "CZ2" or atom_type == "CZ3" or
                atom_type == "CE3"):
            if residue_type == "TRP":
                return self.FormFactorAtomType.CH
            return self.FormFactorAtomType.C

        # DNA/RNA atoms
        # C5'
        if atom_type == "C5p" or atom_type == "C5'":
            return self.FormFactorAtomType.CH2
        # C1', C2', C3', C4'
        if (atom_type == "C4p" or atom_type == "C3p" or
            atom_type == "C2p" or atom_type == "C1p"
            or atom_type == "C4'" or atom_type == "C3'"
            or atom_type == "C2'" or atom_type == "C1'"):
            return self.FormFactorAtomType.CH
        # C2
        if atom_type == "C2":
            if (residue_type == "DADE" or residue_type == "ADE"):
                return self.FormFactorAtomType.CH
            return self.FormFactorAtomType.C
        # C4
        if atom_type == "C4":
            return self.FormFactorAtomType.C
        # C5
        if atom_type == "C5":
            if (residue_type == "DCYT" or residue_type == "CYT" or
                    residue_type == "DURA" or residue_type == "URA"):
                return self.FormFactorAtomType.CH
            return self.FormFactorAtomType.C
        # C6
        if atom_type == "C6":
            if (residue_type == "DCYT" or residue_type == "CYT" or
                    residue_type == "DURA" or residue_type == "URA" or
                    residue_type == "DTHY" or residue_type == "THY"):
                return self.FormFactorAtomType.CH
            return self.FormFactorAtomType.C
        # C7
        if atom_type == "C7":
            return self.FormFactorAtomType.CH3
        # C8
        if atom_type == "C8":
            return self.FormFactorAtomType.CH

        print("Carbon atom not found, using default C form factor for "
                    + atom_type + " " + residue_type)
        return self.FormFactorAtomType.C

    def get_nitrogen_atom_type(self, atom_type, residue_type):
        # protein atoms
        # N
        if atom_type == "N":
            if residue_type == "PRO":
                return self.FormFactorAtomType.N
            return self.FormFactorAtomType.NH
        # ND1
        if atom_type == "ND1":
            if residue_type == "HIS":
                return self.FormFactorAtomType.NH
            return self.FormFactorAtomType.N
        # ND2
        if atom_type == "ND2":
            if residue_type == "ASN":
                return self.FormFactorAtomType.NH2
            return self.FormFactorAtomType.N
        # NH1, NH2
        if atom_type == "NH1" or atom_type == "NH2":
            if residue_type == "ARG":
                return self.FormFactorAtomType.NH2
            return self.FormFactorAtomType.N
        # NE
        if atom_type == "NE":
            if residue_type == "ARG":
                return self.FormFactorAtomType.NH
            return self.FormFactorAtomType.N
        # NE1
        if atom_type == "NE1":
            if residue_type == "TRP":
                return self.FormFactorAtomType.NH
            return self.FormFactorAtomType.N
        # NE2
        if atom_type == "NE2":
            if residue_type == "GLN":
                return self.FormFactorAtomType.NH2
            return self.FormFactorAtomType.N
        # NZ
        if atom_type == "NZ":
            if residue_type == "LYS":
                return self.FormFactorAtomType.NH3
            return self.FormFactorAtomType.N

        # DNA/RNA atoms
        # N1
        if atom_type == "N1":
            if residue_type == "DGUA" or residue_type == "GUA":
                return self.FormFactorAtomType.NH
            return self.FormFactorAtomType.N
        # N2, N4, N6
        if atom_type == "N2" or atom_type == "N4" or atom_type == "N6":
            return self.FormFactorAtomType.NH2
        # N3
        if atom_type == "N3":
            if residue_type == "DURA" or residue_type == "URA":
                return self.FormFactorAtomType.NH
            return self.FormFactorAtomType.N
        # N7, N9
        if atom_type == "N7" or atom_type == "N9":
            return self.FormFactorAtomType.N

        print(f"Nitrogen atom not found, using default N form factor for {atom_type} {residue_type}")

        return self.FormFactorAtomType.N

    def get_oxygen_atom_type(self, atom_type, residue_type):
        # O OE1 OE2 OD1 OD2 O1A O2A OXT OT1 OT2
        if atom_type == "O" or atom_type == "OE1" or \
                atom_type == "OE2" or atom_type == "OD1" or \
                atom_type == "OD2" or atom_type == "OXT":
            return self.FormFactorAtomType.O
        # OG
        if atom_type == "OG":
            if residue_type == "SER":
                return self.FormFactorAtomType.OH
            return self.FormFactorAtomType.O
        # OG1
        if atom_type == "OG1":
            if residue_type == "THR":
                return self.FormFactorAtomType.OH
            return self.FormFactorAtomType.O
        # OH
        if atom_type == "OH":
            if residue_type == "TYR":
                return self.FormFactorAtomType.OH
            return self.FormFactorAtomType.O

        # DNA/RNA atoms
        # O1P, O3', O2P, O2',O4',05', O2,O4,O6
        if (atom_type == "OP1" or atom_type == "O3p "or
            atom_type == "OP2" or atom_type == "O4p" or
            atom_type == "O5p" or atom_type == "O2" or
            atom_type == "O4" or atom_type == "O6"
            or atom_type == "O3'" or atom_type == "O4'"
            or atom_type == "O5'"):
            return self.FormFactorAtomType.O
        # O2'
        if atom_type == "O2p" or atom_type =="O2'":
            return self.FormFactorAtomType.OH

        # water molecule
        if residue_type == "HOH":
            return self.FormFactorAtomType.OH2

        print(f"Oxygen atom not found, using default O form factor for {atom_type} {residue_type}")

        return self.FormFactorAtomType.O


    def get_sulfur_atom_type(self, atom_type, residue_type):
        # SD
        if atom_type == "SD":
            return self.FormFactorAtomType.S
        # SG
        if atom_type == "SG":
            if residue_type == "CYS":
                return self.FormFactorAtomType.SH
            return self.FormFactorAtomType.S

        print(f"Sulfur atom not found, using default S form factor for {atom_type} {residue_type}")

        return self.FormFactorAtomType.S


def get_default_form_factor_table():
    return FormFactorTable()

