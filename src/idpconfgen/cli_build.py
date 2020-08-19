"""
Builds.

USAGE:
    $ idpconfgen build DB

"""
import argparse
import re
import sys
from random import choice as RC
from functools import partial
from itertools import cycle
from math import pi

import numpy as np

from idpconfgen import log
from idpconfgen.libs import libcli
from idpconfgen.libs.libcalc import make_coord_Q, make_coord_Q_CO, make_coord_Q_COO
from idpconfgen.libs.libio import read_dictionary_from_disk
from idpconfgen.libs.libfilter import (
    aligndb,
    regex_search,
    )
from idpconfgen.libs.libtimer import timeme, ProgressWatcher
from idpconfgen.core.definitions import (
    build_bend_CA_C_Np1,
    build_bend_Cm1_N_CA,
    build_bend_N_CA_C,
    build_bend_CA_C_O,
    distance_N_CA,
    distance_CA_C,
    distance_C_Np1,
    distance_C_O,
    atom_labels,
    aa1to3,
    )
from idpconfgen.core.exceptions import IDPConfGenException
from idpconfgen.libs.libpdb import atom_line_formatter, format_atom_name
from idpconfgen.libs.libvalidate import validate_conformer_for_builder


_name = 'build'
_help = 'Builds conformers from database.'


_prog, _des, _us = libcli.parse_doc_params(__doc__)

ap = libcli.CustomParser(
    prog=_prog,
    description=libcli.detailed.format(_des),
    usage=_us,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    )
# https://stackoverflow.com/questions/24180527

ap.add_argument(
    '-db',
    '--database',
    help='The IDPConfGen database.',
    required=True,
    )

ap.add_argument(
    '-seq',
    '--input_seq',
    help='The Conformer residue sequence.',
    required=True
    )

ap.add_argument(
    '-nc',
    '--nconfs',
    help='Number of conformers to build',
    default=1,
    type=int,
    )

ap.add_argument(
    '-dr',
    '--dssp-regexes',
    help='Regexes used to search in DSSP',
    default='(?=(L{2,6}))',
    nargs='+',
    )



def main(
        input_seq,
        database,
        func=None,
        dssp_regexes=r'(?=(L{2,6}))',
        nconfs=1,
        conformer_name='conformer',
        conf_n=1,
        ROUND=np.round,
        ):
    """."""
    # bring global to local
    MAKE_COORD_Q_LOCAL = make_coord_Q
    MAKE_COORD_Q_CO_LOCAL = make_coord_Q_CO
    MAKE_COORD_Q_COO_LOCAL = make_coord_Q_COO
    VALIDATE_CONF_LOCAL = validate_conformer_for_builder


    db = read_dictionary_from_disk(database)
    ldb = len(db)
    log.info(f'Read DB with {ldb} entries')

    # reads and aligns IDPConfGen data base
    timed = partial(timeme, aligndb)
    pdbs, angles, dssp, resseq = timed(db)

    # seachs for slices in secondary structure
    timed = partial(timeme, regex_search)
    slices = []
    if isinstance(dssp_regexes, str):
        dssp_regexes = [dssp_regexes]

    for dssp_regex_string in dssp_regexes:
        slices.extend(timed(dssp, dssp_regex_string))

    log.info(f'Found {len(slices)} indexes for {dssp_regexes}')


    # building

    # prepares data based on the input sequence
    #len_conf = len(input_seq)  # number of residues
    atom_labels = np.array(generate_atom_labels(input_seq))  # considers sidechain all-atoms
    num_atoms = len(atom_labels)
    residue_numbers = np.array(generate_residue_numbers(atom_labels))
    coords = np.ones((num_atoms, 3), dtype=np.float32)

    # creates masks
    bb_mask = np.isin(atom_labels, ('N', 'CA', 'C'))
    carbonyl_mask = np.isin(atom_labels, ('O',))
    OXT_index = np.argwhere(atom_labels == 'OXT')[0][0]
    # replces the last TRUE value by false because that is a carboxyl
    # and nota carbonyl
    OXT1_index = np.argwhere(carbonyl_mask)[-1][0]
    carbonyl_mask[OXT1_index] = False

    # creates views
    bb = np.ones((np.sum(bb_mask), 3), dtype=np.float64)
    bb_CO = np.ones((np.sum(carbonyl_mask), 3), dtype=np.float64)

    # places seed coordinates
    # coordinates are created always from the parameters in the core
    # definitions of IDPConfGen

    # first atom (N-terminal) is at 0, 0, 0
    bb[0, :] = 0.0
    # second atom (CA of the firs residue) is at the x-axis
    bb[1, :] = (distance_N_CA, 0.0, 0.0)

    # third atom (C of the first residue) needs to be computed according
    # to the bond length parameters and bend angle.
    bb[2, :] = make_coord_Q(
        np.array((0.0, distance_N_CA, 0.0)),  # dummy coordinate used only here
        bb[0, :],
        bb[1, :],
        distance_CA_C,
        build_bend_N_CA_C,
        0,  # null torsion angle
        )
    seed_coords = np.ndarray.copy(bb[:3, :])


    bbi0_register = []
    bbi0_R_APPEND = bbi0_register.append
    bbi0_R_POP = bbi0_register.pop

    COi0_register = []
    COi0_R_APPEND = COi0_register.append
    COi0_R_POP = COi0_register.pop

    # prepares cycles for building process
    bond_lens = cycle((distance_C_Np1, distance_N_CA, distance_CA_C))
    bond_bend = cycle((
        build_bend_CA_C_Np1,
        build_bend_Cm1_N_CA,
        build_bend_N_CA_C,
        ))


    pw = ProgressWatcher(nconfs)
    pw.__enter__()

    from time import time
    start = time()
    # STARTS BUILDING
    for conf_n in range(nconfs):

        coords[:, :] = 1.0#np.nan
        bb[:, :] = 1.0#np.nan
        bb[:3, :] = seed_coords
        bb_CO[:, :] = 1.0#np.nan
        # SIDECHAINS HERE

        bbi = 3  # starts at 2 because the first 3 atoms are already placed
        bbi0_register.clear()
        bbi0_R_APPEND(bbi)

        # and needs to adjust with the += assignment inside the loop
        COi = 0  # carbonyl atoms
        COi0_register.clear()
        COi0_R_APPEND(COi)

        backbone_done = False
        number_of_trials = 0
        # run this loop until a specific BREAK is triggered
        while True:

            # the slice [1:-2] removes the first phi and the last psi and omega
            # from the group of angles. These angles are not needed because the
            # implementation always follows the order: psi-omega-phi(...)
            agls = angles[RC(slices), :].ravel()[1:-2]

            # index at the start of the current cycle
            bbi0 = bbi
            COi0 = COi
            try:
                for torsion in agls:
                    bb[bbi, :] = MAKE_COORD_Q_LOCAL(
                        bb[bbi - 3, :],
                        bb[bbi - 2, :],
                        bb[bbi - 1, :],
                        next(bond_lens),
                        next(bond_bend),
                        torsion,
                        )
                    bbi += 1
            except IndexError:
                # here bbi is the last index + 1
                backbone_done = True

            # this else performs if the for loop concludes properly
            #else:
            # when backbone completes,
            # adds carbonyl oxygen atoms
            # after this loop all COs are added for the portion of BB
            # added previously
            for k in range(bbi0, bbi, 3):
                bb_CO[COi, :] = MAKE_COORD_Q_CO_LOCAL(
                    bb[k - 2, :],
                    bb[k - 1, :],
                    bb[k, :],
                    )
                COi += 1

            if backbone_done:  # make terminal carboxyl coordinates
                coords[[OXT_index, OXT1_index]] = MAKE_COORD_Q_COO_LOCAL(
                    bb[-2, :],
                    bb[-1, :],
                    )

            # add sidechains here.....

            # validate conformer current state
            coords[bb_mask] = bb
            coords[carbonyl_mask] = bb_CO
            energy = VALIDATE_CONF_LOCAL(
                coords,
                atom_labels,
                residue_numbers,
                bb_mask,
                carbonyl_mask,
                )

            if energy > 0:  # not valid
                # reset coordinates to the original value
                # before the last chunk added

                # reset the same chunk maximum 5 times,
                # after that reset also the chunk before
                if number_of_trials > 5:
                    bbi0_R_POP()
                    COi0_R_POP()
                    number_of_trials = 0

                try:
                    _bbi0 = bbi0_register[-1]
                    _COi0 = COi0_register[-1]
                except IndexError:
                    # if this point is reached,
                    # we erased until the beginning of the conformer
                    # discard conformer, something went really wrong
                    sys.exit('total error')  # change this to a functional error

                bb[_bbi0:bbi, :] = 1.0
                bb_CO[_COi0:COi, :] = 1.0

                # do the same for sidechains
                # ...

                # reset also indexes
                bbi = _bbi0
                COi = _COi0

                # coords needs to be reset because size of protein next
                # chunks may not be equal
                coords[:, :] = 1.0

                backbone_done = False
                number_of_trials += 1
                continue

            number_of_trials = 0
            bbi0_R_APPEND(bbi)
            COi0_R_APPEND(COi)

            if backbone_done:
                # this point guarantees all protein atoms are built
                break
        # END of while loop

        coords[bb_mask] = bb
        coords[carbonyl_mask] = bb_CO

        sums = np.sum(coords, axis=1)
        relevant = np.logical_not(np.isclose(sums, 3))
        #relevant = np.logical_or(bb_mask, carbonyl_mask) 

        pdb_string = gen_PDB_from_conformer(
            input_seq,
            atom_labels[relevant],
            residue_numbers[relevant],
            ROUND(coords[relevant], decimals=3),
            )

        pw.increment()
        #with open(f'{conformer_name}_{conf_n}.pdb', 'w') as fout:
        #    fout.write(pdb_string)

    pw.__exit__()
    print(time() - start)
    return


    #save_conformer_to_disk(
    #    input_seq,
    #    atom_labels[relevant],
    #    residue_numbers[relevant],
    #    coords[relevant],
    #    )


#def save_conformer_to_disk(input_seq, atom_labels, residues, coords):

def gen_PDB_from_conformer(
        input_seq,
        atom_labels,
        residues,
        coords,
        ALF=atom_line_formatter,
        AA1TO3=aa1to3,
        ROUND=np.round,
        ):
    """."""
    lines = []
    LINES_APPEND = lines.append
    ALF_FORMAT = ALF.format
    resi = -1
    ATOM_LABEL_FMT = ' {: <3}'.format
    pdb_text = ''

    for i in range(len(atom_labels)):

        if atom_labels[i] == 'N':
            resi += 1
            current_residue = input_seq[resi]
            current_resnum = residues[i]

        atm = atom_labels[i].strip()
        ele = atm.lstrip('123')[0]

        if len(atm) < 4:
            atm = ATOM_LABEL_FMT(atm)

        LINES_APPEND(ALF_FORMAT(
            'ATOM',
            i,
            #format_atom_name(atom_labels[i], ele),
            atm,
            '',
            AA1TO3[current_residue],
            'A',
            current_resnum,
            '',
            coords[i, 0],
            coords[i, 1],
            coords[i, 2],
            0.0,
            0.0,
            '',
            ele,
            '',
            ))

    return '\n'.join(lines)


def generate_atom_labels(input_seq, AL=atom_labels):
    """."""
    labels = []
    LE = labels.extend

    for residue in input_seq:
        LE(AL[residue])

    labels.append('OXT')

    return labels


def generate_residue_numbers(atom_labels, start=1):
    """
    Create a list of residue numbers based on atom labels.

    Considers `N` to be the first atom of the residue.
    If this is not the case, the output can be meaningless.
    """
    if atom_labels[0] == 'N':
        start -= 1  # to compensate the +=1 implementation in the for loop

    residues = []
    RA = residues.append

    for al in atom_labels:
        if al == 'N':
            start += 1
        RA(start)

    return residues



if __name__ == "__main__":
    libcli.maincli(ap, main)
