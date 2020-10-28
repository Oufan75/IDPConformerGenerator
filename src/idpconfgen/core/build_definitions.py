"""Definitions for the building process."""
from math import pi
from pathlib import Path
from statistics import fmean, stdev

import numpy as np

from idpconfgen.libs.libstructure import Structure

_filepath = Path(__file__).resolve().parent  # folder

# amino-acids atom labels
# from: http://www.bmrb.wisc.edu/ref_info/atom_nom.tbl
# PDB column
# Taken from PDB entry 6I1B REVDAT 15-OCT-92.
atom_labels = {
    'A': ('N', 'CA', 'C', 'O', 'CB', 'H', 'HA', 'HB1', 'HB2', 'HB3'),  # noqa: E501
    'C': ('N', 'CA', 'C', 'O', 'CB', 'SG', 'H', 'HA', '1HB', '2HB', 'HG'),  # noqa: E501
    'D': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2', 'H', 'HA', '1HB', '2HB'),  # noqa: E501
    'E': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2', 'H', 'HA', '1HB', '2HB', '1HG', '2HG'),  # noqa: E501
    'F': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'H', 'HA', '1HB', '2HB', 'HD1', 'HD2', 'HE1', 'HE2', 'HZ'),  # noqa: E501
    'G': ('N', 'CA', 'C', 'O', 'H', '1HA', '2HA'),  # noqa: E501
    'H': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2', 'H', 'HA', '1HB', '2HB', 'HD1', 'HD2', 'HE1', 'HE2'),  # noqa: E501
    'I': ('N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', 'H', 'HA', 'HB', '1HG1', '2HG1', '1HG2', '2HG2', '3HG2', '1HD1', '2HD1', '3HD1'),  # noqa: E501
    'K': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', 'H', 'HA', '1HB', '2HB', '1HG', '2HG', '1HD', '2HD', '1HE', '2HE', '1HZ', '2HZ', '3HZ'),  # noqa: E501
    'L': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'H', 'HA', '1HB', '2HB', 'HG', '1HD1', '2HD1', '3HD1', '1HD2', '2HD2', '3HD2'),  # noqa: E501
    'M': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE', 'H', 'HA', '1HB', '2HB', '1HG', '2HG', '1HE', '2HE', '3HE'),  # noqa: E501
    'N': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2', 'H', 'HA', '1HB', '2HB', '1HD2', '2HD2'),  # noqa: E501
    'P': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'H2', 'H1', 'HA', '1HB', '2HB', '1HG', '2HG', '1HD', '2HD'),  # noqa: E501
    'Q': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2', 'H', 'HA', '1HB', '2HB', '1HG', '2HG', '1HE2', '2HE2'),  # noqa: E501
    'R': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2', 'H', 'HA', '1HB', '2HB', '1HG', '2HG', '1HD', '2HD', 'HE', '1HH1', '2HH1', '1HH2', '2HH2'),  # noqa: E501
    'S': ('N', 'CA', 'C', 'O', 'CB', 'OG', 'H', 'HA', '1HB', '2HB', 'HG'),  # noqa: E501
    'T': ('N', 'CA', 'C', 'O', 'CB', 'CG2', 'OG1', 'H', 'HA', 'HB', 'HG1', '1HG2', '2HG2', '3HG2'),  # noqa: E501
    'V': ('N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'H', 'HA', 'HB', '1HG1', '2HG1', '3HG1', '1HG2', '2HG2', '3HG2'),  # noqa: E501
    'W': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2', 'NE1', 'H', 'HA', '1HB', '2HB', 'HD1', 'HE1', 'HE3', 'HZ2', 'HZ3', 'HH2'),  # noqa: E501
    'Y': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', 'H', 'HA', '1HB', '2HB', 'HD1', 'HD2', 'HE1', 'HE2', 'HH'),  # noqa: E501
    }

# bend angles are in radians
# bend angle for the CA-C-O bond was virtually the same, so will be computed
# as a single value. Changes in CA-C-Np1 create variations in Np1-C-O according
# beacuse CA-C-O is fixed
bend_angles_N_CA_C = {
    'A': 4365251754739323 / 2251799813685248,  # 1.939 0.036
    'R': 8719677895929259 / 4503599627370496,  # 1.936 0.039
    'N': 4379328358183431 / 2251799813685248,  # 1.945 0.046
    'D': 8725333487606339 / 4503599627370496,  # 1.937 0.045
    'C': 272188429672133 / 140737488355328,    # 1.934 0.043
    'E': 4367135841953607 / 2251799813685248,  # 1.939 0.038
    'Q': 2182067275023153 / 1125899906842624,  # 1.938 0.039
    'G': 4450758630407041 / 2251799813685248,  # 1.977 0.045
    'H': 8728063705254555 / 4503599627370496,  # 1.938 0.044
    'I': 1077205649192195 / 562949953421312,   # 1.914 0.041
    'L': 8709813928320865 / 4503599627370496,  # 1.934 0.04
    # 'K': 273144179869987 / 140737488355328, # Example by THG templates
    'K': 8725282730666569 / 4503599627370496,  # 1.937 0.039
    'M': 4357918721408073 / 2251799813685248,  # 1.935 0.039
    'F': 4350621570448131 / 2251799813685248,  # 1.932 0.042
    'P': 4431159883507209 / 2251799813685248,  # 1.968 0.039
    'S': 2185976912710845 / 1125899906842624,  # 1.942 0.041
    'T': 8701671394739109 / 4503599627370496,  # 1.932 0.042
    'W': 4357706976096223 / 2251799813685248,  # 1.935 0.041
    'Y': 2177418951092023 / 1125899906842624,  # 1.934 0.042
    'V': 4310842035620741 / 2251799813685248,  # 1.914 0.04
    }

build_bend_angles_N_CA_C = {
    key: (pi - v) / 2
    for key, v in bend_angles_N_CA_C.items()
    }

bend_angles_CA_C_Np1 = {
    'A': 4588994859787807 / 2251799813685248,  # 2.038 0.022
    'R': 4588049090836895 / 2251799813685248,  # 2.038 0.022
    'N': 4589163477109627 / 2251799813685248,  # 2.038 0.024
    'D': 4591476087195707 / 2251799813685248,  # 2.039 0.023
    'C': 4584298882571901 / 2251799813685248,  # 2.036 0.024
    'E': 4591360788509533 / 2251799813685248,  # 2.039 0.022
    'Q': 2295079292392017 / 1125899906842624,  # 2.038 0.022
    'G': 2292570018001035 / 1125899906842624,  # 2.036 0.026
    'H': 2293539098323251 / 1125899906842624,  # 2.037 0.024
    'I': 1146740789727809 / 562949953421312,   # 2.037 0.021
    'L': 2295421027511699 / 1125899906842624,  # 2.039 0.021
    'K': 2294523703162747 / 1125899906842624,  # 2.038 0.022
    'M': 71693445620547 / 35184372088832,      # 2.038 0.023
    'F': 4583159124932161 / 2251799813685248,  # 2.035 0.023
    'P': 4587486592844193 / 2251799813685248,  # 2.037 0.027
    'S': 2293301456194919 / 1125899906842624,  # 2.037 0.024
    'T': 1146400995781303 / 562949953421312,   # 2.036 0.023
    'W': 1146548728446729 / 562949953421312,   # 2.037 0.024
    'Y': 4582697136860539 / 2251799813685248,  # 2.035 0.023
    'V': 4584841207534447 / 2251799813685248,  # 2.036 0.021
    }

build_bend_angles_CA_C_Np1 = {
    key: (pi - v) / 2
    for key, v in bend_angles_CA_C_Np1.items()
    }

bend_angles_Cm1_N_CA = {
    'A': 4768579151967919 / 2251799813685248,  # 2.118 0.028
    'R': 1192445900065887 / 562949953421312,   # 2.118 0.028
    'N': 1193332907551887 / 562949953421312,   # 2.12 0.03
    'D': 4771817124476497 / 2251799813685248,  # 2.119 0.029
    'C': 4773336800981679 / 2251799813685248,  # 2.12 0.03
    'E': 596106910867665 / 281474976710656,    # 2.118 0.028
    'Q': 2384417995863009 / 1125899906842624,  # 2.118 0.029
    'G': 1194549700095835 / 562949953421312,   # 2.122 0.028
    'H': 4771903805828599 / 2251799813685248,  # 2.119 0.03
    'I': 4770277841981895 / 2251799813685248,  # 2.118 0.029
    'L': 2383227038060725 / 1125899906842624,  # 2.117 0.029
    'K': 1193021713030775 / 562949953421312,   # 2.119 0.029
    'M': 4767963234571985 / 2251799813685248,  # 2.117 0.029
    'F': 4771480556811017 / 2251799813685248,  # 2.119 0.03
    'P': 2386863444043781 / 1125899906842624,  # 2.12 0.03
    'S': 4772350172472667 / 2251799813685248,  # 2.119 0.03
    'T': 4772846148285813 / 2251799813685248,  # 2.12 0.03
    'W': 4773546458813579 / 2251799813685248,  # 2.12 0.032
    'Y': 4773380997634081 / 2251799813685248,  # 2.12 0.031
    'V': 596711317736503 / 281474976710656,    # 2.12 0.029
    }

build_bend_angles_Cm1_N_CA = {
    key: (pi - v) / 2
    for key, v in bend_angles_Cm1_N_CA.items()
    }

# distances are in angstroms
distances_N_CA = {
    'A': 6579089706805643 / 4503599627370496,  # 1.461 0.012
    'R': 822248550821425 / 562949953421312,    # 1.461 0.012
    'N': 3288466758786951 / 2251799813685248,  # 1.46 0.012
    'D': 6581461030414551 / 4503599627370496,  # 1.461 0.012
    'C': 6573024845758137 / 4503599627370496,  # 1.46 0.012
    'E': 6578019054232101 / 4503599627370496,  # 1.461 0.012
    'Q': 6577391610142879 / 4503599627370496,  # 1.46 0.013
    'G': 6551432980914649 / 4503599627370496,  # 1.455 0.013
    'H': 6576300419750417 / 4503599627370496,  # 1.46 0.013
    'I': 1644634584789097 / 1125899906842624,  # 1.461 0.012
    'L': 6576809707492807 / 4503599627370496,  # 1.46 0.012
    'K': 1644714268825877 / 1125899906842624,  # 1.461 0.015
    'M': 6579479788753381 / 4503599627370496,  # 1.461 0.013
    'F': 6574810553972951 / 4503599627370496,  # 1.46 0.012
    'P': 6602907863490325 / 4503599627370496,  # 1.466 0.011
    'S': 6575297426758937 / 4503599627370496,  # 1.46 0.012
    'T': 6573005843763805 / 4503599627370496,  # 1.46 0.012
    'W': 1643809779698609 / 1125899906842624,  # 1.46 0.012
    'Y': 3287342856106863 / 2251799813685248,  # 1.46 0.012
    'V': 6577083632702971 / 4503599627370496,  # 1.46 0.012
    }

average_distance_N_CA = fmean(distances_N_CA.values())
std_distance_N_CA = stdev(distances_N_CA.values())

distances_CA_C = {
    'A': 6864424746753997 / 4503599627370496,  # 1.524 0.012
    'R': 6864915463757277 / 4503599627370496,  # 1.524 0.012
    'N': 1716505987820907 / 1125899906842624,  # 1.525 0.012
    'D': 3436265496864801 / 2251799813685248,  # 1.526 0.012
    'C': 1714936829082835 / 1125899906842624,  # 1.523 0.012
    'E': 3433630145140679 / 2251799813685248,  # 1.525 0.012
    'Q': 6864676438206105 / 4503599627370496,  # 1.524 0.012
    'G': 3412585135594759 / 2251799813685248,  # 1.515 0.011
    'H': 6858488656304525 / 4503599627370496,  # 1.523 0.013
    'I': 858607386435583 / 562949953421312,    # 1.525 0.012
    'L': 3431008760147005 / 2251799813685248,  # 1.524 0.012
    'K': 858360898168883 / 562949953421312,    # 1.525 0.015
    'M': 3430952602650511 / 2251799813685248,  # 1.524 0.012
    'F': 1714968063284791 / 1125899906842624,  # 1.523 0.012
    'P': 6865394859536305 / 4503599627370496,  # 1.524 0.012
    'S': 3431691276368069 / 2251799813685248,  # 1.524 0.012
    'T': 6865564932278851 / 4503599627370496,  # 1.524 0.012
    'W': 6859935246968493 / 4503599627370496,  # 1.523 0.013
    'Y': 6859340134216747 / 4503599627370496,  # 1.523 0.012
    'V': 6867874764281747 / 4503599627370496,  # 1.525 0.012
    }

average_distance_CA_C = fmean(distances_CA_C.values())
std_distance_CA_C = stdev(distances_CA_C.values())

distances_C_Np1 = {
    'A': 5993805121385571 / 4503599627370496,  # 1.331 0.01
    'R': 2996218862401521 / 2251799813685248,  # 1.331 0.009
    'N': 5993831729486957 / 4503599627370496,  # 1.331 0.01
    'D': 1498633271940567 / 1125899906842624,  # 1.331 0.01
    'C': 2996353095777321 / 2251799813685248,  # 1.331 0.01
    'E': 2996409892006783 / 2251799813685248,  # 1.331 0.01
    'Q': 5992661913157337 / 4503599627370496,  # 1.331 0.009
    'G': 5991694366764827 / 4503599627370496,  # 1.33 0.01
    'H': 5993608389065211 / 4503599627370496,  # 1.331 0.01
    'I': 5994057580226143 / 4503599627370496,  # 1.331 0.009
    'L': 2996685407441657 / 2251799813685248,  # 1.331 0.009
    'K': 5992148258082287 / 4503599627370496,  # 1.331 0.01
    'M': 5993036042256113 / 4503599627370496,  # 1.331 0.01
    'F': 5992325017189537 / 4503599627370496,  # 1.331 0.009
    'P': 2994612945437999 / 2251799813685248,  # 1.33 0.01
    'S': 5992949209364285 / 4503599627370496,  # 1.331 0.01
    'T': 187268519548059 / 140737488355328,    # 1.331 0.01
    'W': 2996589405775763 / 2251799813685248,  # 1.331 0.009
    'Y': 2996393655695127 / 2251799813685248,  # 1.331 0.009
    'V': 2996598670315555 / 2251799813685248,  # 1.331 0.009
    }

average_distance_C_Np1 = fmean(distances_C_Np1.values())
std_distance_C_Np1 = stdev(distances_C_Np1.values())

build_bend_CA_C_OXT = (pi - (2 * pi / 3)) / 2
build_bend_CA_C_O = 2.102 / 2
distance_C_OXT = 1.27
distance_C_O = 5556993099130213 / 4503599627370496

# NH atom:
distance_H_N = 0.9
build_bend_H_N_C = np.radians(114) / 2


# side chain template coordinates


def _get_structure_coords(path_):
    s = Structure(path_)
    s.build()
    return s.coords.astype(np.float64)


sidechain_templates = {
    pdb.stem.upper(): _get_structure_coords(pdb)
    for pdb in _filepath.joinpath('sidechain_templates').glob('*.pdb')
    }
