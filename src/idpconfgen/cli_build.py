"""
Builds IDP conformers.

Build from a database of torsion angles and secondary structure
information. Database is as created by `idpconfgen torsions` CLI.

USAGE:
    $ idpconfgen build -db torsions.json -seq MMMMMMM...

"""
import argparse
import re
from collections import Counter
from functools import partial
from multiprocessing import Pool, Queue
# from numbers import Number
from random import choice as randchoice
from random import randint
from time import time

import numpy as np
from numba import njit

#from idpconfgen.cpp.faspr import faspr_sidechains as fsc
from idpconfgen import log, Path
from idpconfgen.core.build_definitions import (
    amber_pdbs,
    atom_names_amber,
    forcefields,
    backbone_atoms,
    bonds_equal_3_inter,
    bonds_le_2_inter,
    build_bend_H_N_C,
    distance_H_N,
    distances_CA_C,
    distances_C_Np1,
    distances_N_CA,
    expand_topology_bonds_apart,
    generate_residue_template_topology,
    inter_residue_connectivities,
    n_terminal_h_coords_at_origin,
    read_ff14SB_params,
    sidechain_templates,
    topology_3_bonds_apart,
    build_bend_CA_C_O,
    distance_C_O,
    )
from idpconfgen.core.definitions import aa1to3  # , vdW_radii_dict
from idpconfgen.core.exceptions import IDPConfGenException
from idpconfgen.libs import libcli
from idpconfgen.libs.libbuild import (
    compute_sidechains,
    init_conflabels,
    init_confmasks,
    get_cycle_distances_backbone,
    get_cycle_bond_type,
    translate_seq_to_3l,
    )
from idpconfgen.libs.libcalc import (
    # calc_all_vs_all_dists_square,
    calc_all_vs_all_dists,
    calc_residue_num_from_index,
    calc_torsion_angles,
    make_coord_Q,
    make_coord_Q_COO,
    make_coord_Q_planar,
    place_sidechain_template,
    rotate_coordinates_Q_njit,
    rrd10_njit,
    energycalculator_ij,
    init_lennard_jones_calculator,
    init_coulomb_calculator,
    )
from idpconfgen.libs.libfilter import aligndb, regex_search
from idpconfgen.libs.libhigherlevel import bgeo_reduce
from idpconfgen.libs.libio import read_dictionary_from_disk
from idpconfgen.libs.libparse import get_trimer_seq_njit, remap_sequence
from idpconfgen.libs.libpdb import atom_line_formatter
from idpconfgen.libs.libtimer import timeme


_file = Path(__file__).myparents()

# Global variables needed to build conformers.
# Why are global variables needed?
# I use global variables to facilitate distributing conformer creation
# processes across multiple cores. In this way cores can read global variables
# fast and with non-significant overhead.

# Bond Geometry library variables
# if __name__ == '__main__', these will be populated in main()
# else will be populated in conformer_generator
# populate_globals() populates these variables once called.
BGEO_path = Path(_file, 'core', 'data', 'bgeo.tar')
BGEO_full = {}
BGEO_trimer = {}
BGEO_res = {}

# SLICES and ANGLES will be populated in main() with the torsion angles.
# it is not expected SLICES or ANGLES to be populated anywhere else.
# The slice objects from where the builder will feed to extract torsion
# chunks from ANGLES.
SLICES = []
ANGLES = None

# keeps a record of the conformer numbers written to disk across the different
# cores
CONF_NUMBER = Queue()

# The conformer building process needs data structures for two different
# identities: the all-atom representation of the input sequence, and the
# corresponding Ala/Gly/Pro template uppon which the coordinates will be built.
# These variables are defined at the module level so they serve as global
# variables to be read by the different process during multiprocessing. Reading
# from global variables is performant in Python multiprocessing. This is the
# same strategy as applied for SLICES and ANGLES.
ALL_ATOM_LABELS = None
ALL_ATOM_MASKS = None
ALL_ATOM_EFUNC = None
TEMPLATE_LABELS = None
TEMPLATE_MASKS = None
TEMPLATE_EFUNC = None


def are_globals():
    """Assess if global variables needed for building are populated."""
    return all((
        ALL_ATOM_LABELS,
        ALL_ATOM_MASKS,
        ALL_ATOM_EFUNC,
        TEMPLATE_LABELS,
        TEMPLATE_MASKS,
        TEMPLATE_EFUNC,
        BGEO_full,
        BGEO_trimer,
        BGEO_res,
        ))

# CLI argument parser parameters
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
    help='Number of conformers to build.',
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

ap.add_argument(
    '-dsd',
    '--disable-sidechains',
    help='Whether or not to compute sidechais. Defaults to True.',
    action='store_true',
    )

_ffchoice = list(forcefields.keys())
ap.add_argument(
    '-ff',
    '--forcefield',
    help=(
        'Forcefield parameters and atom labels. '
        f'Defaults to {_ffchoice[0]}.'
        ),
    choices=_ffchoice,
    default=_ffchoice[0],
    )

ap.add_argument(
    '-bgeo',
    '--bgeo',
    help=(
        'Path to the bond geometry database as generated by `bgeo` CLI .'
        'Defaults to `None`, uses the internal library.'
        ),
    default=None,
    )

libcli.add_argument_ncores(ap)


def main(
        input_seq,
        database,
        dssp_regexes=r'(?=(L{2,6}))',
        func=None,
        forcefield=None,
        bgeo_path=None,
        nconfs=1,
        ncores=1,
        **kwargs,  # other kwargs target energy function, for example.
        ):
    """
    Execute main client logic.

    Distributes over processors.
    """
    # Calculates how many conformers are built per core
    if nconfs < ncores:
        ncores = 1
        core_chunks = nconfs
        remaining_chunks = 0
    else:
        core_chunks = nconfs // ncores
        # in case nconfs is not multiple of ncores, builds the remaining confs
        # at the end
        remaining_chunks = nconfs % ncores

    # populates globals
    global ANGLES, SLICES
    _slices, ANGLES = read_db_to_slices(database, dssp_regexes, ncores=ncores)
    SLICES.extend(_slices)

    populate_globals(
        bgeo_path=bgeo_path or BGEO_path,
        forcefield=forcefields[forcefield],
        **kwargs)

    # creates a queue of numbers that will serve all subprocesses.
    # Used to name the output files, conformer_1, conformer_2, ...
    for i in range(1, nconfs + 1):
        CONF_NUMBER.put(i)

    # prepars execution function
    execute = partial(
        _build_conformers,
        input_seq=input_seq,  # string
        nconfs=core_chunks,  # int
        **kwargs,
        )

    start = time()
    with Pool(ncores) as pool:
        imap = pool.imap(execute, range(ncores))
        for _ in imap:
            pass

    if remaining_chunks:
        execute(core_chunks * ncores, nconfs=remaining_chunks)

    log.info(f'{nconfs} conformers built in {time() - start:.3f} seconds')


def populate_globals(*, bgeo_path=BGEO_path, forcefield=None, **efunc_kwargs):
    """
    Populate global variables needed for building.

    Currently, global variables include:

    BGEO_full
    BGEO_trimer
    BGEO_res
    ALL_ATOM_LABELS, ALL_ATOM_MASKS, ALL_ATOM_EFUNC
    TEMPLATE_LABELS, TEMPLATE_MASKS, TEMPLATE_EFUNC

    Parameters
    ----------
    bgeo_path : str or Path
        The path pointing to a bond geometry library as created by the
        `bgeo` CLI.

    forcefield : str
        A key in the `core.build_definitions.forcefields` dictionary.
    """
    global BGEO_full, BGEO_trimer, BGEO_res

    BGEO_full.update(read_dictionary_from_disk(bgeo_path))
    _1, _2 = bgeo_reduce(BGEO_full)
    BGEO_trimer.update(_1)
    BGEO_res.update(_2)
    del _1, _2
    assert BGEO_full
    assert BGEO_trimer
    assert BGEO_res
    # this asserts only the first layer of keys
    assert list(BGEO_full.keys()) == list(BGEO_trimer.keys()) == list(BGEO_res.keys())  # noqa: E501

    # populates the labels
    global ALL_ATOM_LABELS, ALL_ATOM_MASKS, ALL_ATOM_EFUNC
    global TEMPLATE_LABELS, TEMPLATE_MASKS, TEMPLATE_EFUNC

    topobj = forcefield(add_OXT=True, add_Nterminal_H=True)

    ALL_ATOM_LABELS = init_conflabels(input_seq, topobj.atom_names)
    TEMPLATE_LABELS = init_conflabels(remap_sequence(input_seq, topobj.atom_names))

    ALL_ATOM_MASKS = init_confmasks(ALL_ATOM_LABELS.atom_labels)
    TEMPLATE_MASKS = init_confmasks(TEMPLATE_LABELS.atom_labels)

    ALL_ATOM_EFUNC = prepare_energy_func(
        ALL_ATOM_LABELS.atom_labels,
        ALL_ATOM_LABELS.res_num,
        ALL_ATOM_LABELS.res_labels,
        topobj,
        **efunc_kwargs)

    TEMPLATE_EFUNC = prepare_energy_func(
        TEMPLATE_LABELS.atom_labels,
        TEMPLATE_LABELS.res_num,
        TEMPLATE_LABELS.res_labels,
        topobj,
        **efunc_kwargs)

    del topobj
    return


# private function because it depends on the global `CONF_NUMBER`
# which is assembled in `main()`
def _build_conformers(
        *args,
        input_seq=None,
        conformer_name='conformer',
        nconfs=1,
        **kwargs):
    """
    Arrange building of conformers and saves them to PDB files.
    """
    ROUND = np.round

    # TODO: this has to be parametrized for the different HIS types
    input_seq_3_letters = translate_seq_to_3l(input_seq)

    builder = conformer_generator(input_seq=input_seq, **kwargs)

    atom_labels, residue_numbers, residue_labels = next(builder)

    for _ in range(nconfs):

        coords = next(builder)

        pdb_string = gen_PDB_from_conformer(
            input_seq_3_letters,
            atom_labels,
            residue_numbers,
            ROUND(coords, decimals=3),
            )

        fname = f'{conformer_name}_{CONF_NUMBER.get()}.pdb'

        with open(fname, 'w') as fout:
            fout.write(pdb_string)

    del builder
    return


# the name of this function is likely to change in the future
def conformer_generator(
        *,
        input_seq=None,
        generative_function=None,
        disable_sidechains=True,
        sidechain_method='faspr',
        bgeo_path=None,
        forcefield=None,
        lj_term=True,
        coulomb_term=False,
        ):
    """
    Build conformers.

    `conformer_generator` is actually a Python generator. Examples on
    how it works:

    Note that all arguments are **named** arguments.

    >>> builder = conformer_generator(
    >>>    input_seq='MGAETTWSCAAA'  # the primary sequence of the protein
    >>>    )

    `conformer_generator` is a generator, you can instantiate it simply
    providing the residue sequence of your protein of interest.

    The **very first** iteration will return the labels of the protein
    being built. Labels are sorted by all atom models. Likewise,
    `residue_number` and `residue_labels` sample **all atoms**. These
    three are numpy arrays and can be used to index the actual coordinates.

    >>> atom_labels, residue_numbers, residue_labels = next(builder)

    After this point, each iteraction `next(builder)` yields the coordinates
    for a new conformer. There is no limit in the generator.

    >>> new_coords = next(builder)

    `new_coords` is a (N, 3) np.float64 array where N is the number of
    atoms. As expected, atom coordinates are aligned with the labels
    previously generated.

    When no longer needed,

    >>> del builder

    Should delete the builder generator.

    You can gather the coordinates of several conformers in a single
    multi dimensional array with the following:

    >>> builder = conformer_generator(
    >>>     input_seq='MGGGGG...',
    >>>     generative_function=your_function)
    >>>
    >>> atoms, res3letter, resnums = next(builder)
    >>>
    >>> num_of_conformers = 10_000
    >>> shape = (num_of_conformers, len(atoms), 3)
    >>> all_coords = np.empty(shape, dtype=float64)
    >>>
    >>> for i in range(num_of_conformers):
    >>>     all_coords[i, :, :] = next(builder)
    >>>

    Parameters
    ----------
    input_seq : str, mandatory
        The primary sequence of the protein being built in FASTA format.
        `input_seq` will be used to generate the whole conformers' and
        labels arrangement.
        Example: "MAGERDDAPL".

    generative_function : callable, optional
        The generative function used by the builder to retrieve torsion
        angles during the building process.

        The builder expects this function to receive two parameters:
            - `nres`, the residue chunk size to get angles from
            - `cres`, the next residue being built. For example,
                with cres=10, the builder will expect a minimum of three
                torsion angles (phi, psi, omega) for residue 10.

        Depending on the nature of the `generative function` the two
        pameters may be ignored by the function itself (use **kwargs
        for that purpose).

        If `None` provided, the builder will use the internal `SLIDES`
        and `ANGLES` variables and will assume the `cli_build.main` was
        executed priorly, or that ANGLES and SLICES were populated
        properly.

    disable_sidechains : bool
        Disables sidechain creation. Defaults to `False`, computes
        sidechains.

    nconfs : int
        The number of conformers to build.

    sidechain_method : str
        The method used to build/pack sidechains over the backbone
        structure. Defaults to `faspr`.
        Expects a key in `libs.libbuild.compute_sidechains`.

    bgeo_path : str of Path
        Path to a bond geometry library as created by `bgeo` CLI.

    lj_term : bool
        Whether to compute the Lennard-Jones term during building and
        validation. If false, expect a physically meaningless result.
    """
    if not isinstance(input_seq, str):
        raise ValueError(f'`input_seq` must be given! {input_seq}')
    if forcefield not in forcefields:
        raise ValueError(
            f'{forcefield} not in `forcefields`. '
            f'Expected {list(forcefields.keys())}.'
            )
    if sidechain_method not in compute_sidechains:
        raise ValueError(
            f'{sidechain_method} not in `compute_sidechains`. '
            f'Expected {list(compute_sidechains.keys())}.'
            )

    all_atom_input_seq = input_seq
    template_input_seq = remap_sequence(all_atom_input_seq)
    template_seq_3l = translate_seq_to_3l(template_input_seq)
    del input_seq

    BUILD_BEND_H_N_C = build_bend_H_N_C
    CALC_TORSION_ANGLES = calc_torsion_angles
    DISTANCE_NH = distance_H_N
    DISTANCE_C_O = distance_C_O
    ISNAN = np.isnan
    GET_TRIMER_SEQ = get_trimer_seq_njit
    MAKE_COORD_Q_COO_LOCAL = make_coord_Q_COO
    MAKE_COORD_Q_PLANAR = make_coord_Q_planar
    MAKE_COORD_Q_LOCAL = make_coord_Q
    NAN = np.nan
    NORM = np.linalg.norm
    N_TERMINAL_H = n_terminal_h_coords_at_origin
    PI2 = np.pi * 2
    PLACE_SIDECHAIN_TEMPLATE = place_sidechain_template
    RAD_60 = np.radians(60)
    RC = randchoice
    RINT = randint
    ROT_COORDINATES = rotate_coordinates_Q_njit
    RRD10 = rrd10_njit
    SIDECHAIN_TEMPLATES = sidechain_templates
    angles = ANGLES
    slices = SLICES
    global BGEO_full
    global BGEO_trimer
    global BGEO_res
    global ALL_ATOM_LABELS
    global ALL_ATOM_MASKS
    global ALL_ATOM_EFUNC
    global TEMPLATE_LABELS
    global TEMPLATE_MASKS
    global TEMPLATE_EFUNC

    # these flags exist to populate the global variables in case they were not
    # populated yet. Global variables are populated through the main() function
    # if the script runs as CLI. Otherwise, if conformer_generator() is imported
    # and used directly, the global variables need to be configured here.
    if not are_globals():
        populate_globals(
            bgeo_path=bgeo_path or BGEO_path,
            forcefield=forcefields[forcefield],
            lj_term=lj_term,
            coulomb_term=coulomb_term,
            )

    # semantic exchange for speed al readibility
    with_sidechains = not(disable_sidechains)

    if with_sidechains:
        build_sidechains = compute_sidechains[sidechain_method](all_atom_input_seq)  # noqa: E501

    # tests generative function complies with implementation requirements
    if generative_function:
        try:
            generative_function(nres=1, cres=0)
        except Exception as err:  # this is generic Exception on purpose
            errmsg = (
                'The `generative_function` provided is not compatible with '
                'the building process. Please read `build_conformers` docstring'
                ' for more details.'
                )
            raise IDPConfGenException(errmsg) from err

    # yields atom labels
    # all conformers generated will share these labels
    yield (
        ALL_ATOM_LABELS.atom_labels,
        ALL_ATOM_LABELS.res_num,
        ALL_ATOM_LABELS.res_labels,
        )

    all_atom_num_atoms = len(ALL_ATOM_LABELS.atom_labels)
    template_num_atoms = len(TEMPLATE_LABELS.atom_labels)

    all_atom_coords = np.full((all_atom_num_atoms, 3), NAN, dtype=np.float64)
    template_coords = np.full((template_num_atoms, 3), NAN, dtype=np.float64)

    # +2 because of the dummy coordinates required to start building.
    # see later adding dummy coordinates to the structure seed
    bb = np.full((TEMPLATE_MASKS.bb3.size + 2, 3), NAN, dtype=np.float64)
    bb_real = bb[2:, :]  # backbone coordinates without the dummies

    # coordinates for the carbonyl oxigen atoms
    bb_CO = np.full((TEMPLATE_MASKS.COs.size, 3), NAN, dtype=np.float64)

    # notice that NHydrogen_mask does not see Prolines
    bb_NH = np.full((TEMPLATE_MASKS.NHs.size, 3), NAN, dtype=np.float64)
    bb_NH_idx = np.arange(len(bb_NH))
    # Creates masks and indexes for the `for` loop used to place NHs.
    # The first residue has no NH, prolines have no NH.
    non_pro = np.array(list(template_input_seq)[1:]) != 'P'
    # NHs index numbers in bb_real
    bb_NH_nums = np.arange(3, (len(template_input_seq) - 1) * 3 + 1, 3)[non_pro]
    bb_NH_nums_p1 = bb_NH_nums + 1
    assert bb_NH.shape[0] == bb_NH_nums.size == bb_NH_idx.size

    # sidechain masks
    # this is sidechain agnostic, works for every sidechain, yet here we
    # use only ALA, PRO, GLY - Mon Feb 15 17:29:20 2021
    ss_masks = create_sidechains_masks_per_residue(
        TEMPLATE_LABELS.res_nums,
        TEMPLATE_LABELS.atom_labels,
        backbone_atoms,
        )
    # ?

    # /
    # creates seed coordinates:
    # because the first torsion angle of a residue is the omega, we need
    # to prepare 2 dummy atoms to simulate the residue -1, so that the
    # first omega can be placed. There is no need to setup specific
    # positions, just to create a place upon which the build atom
    # routine can create a new atom from a torsion.
    dummy_CA_m1_coord = np.array((0.0, 1.0, 1.0))
    dummy_C_m1_coord = np.array((0.0, 1.0, 0.0))
    n_terminal_N_coord = np.array((0.0, 0.0, 0.0))

    # seed coordinates array
    seed_coords = np.array((
        dummy_CA_m1_coord,
        dummy_C_m1_coord,
        n_terminal_N_coord,
        ))
    # ?

    # /
    # prepares method binding
    bbi0_register = []
    bbi0_R_APPEND = bbi0_register.append
    bbi0_R_POP = bbi0_register.pop
    bbi0_R_CLEAR = bbi0_register.clear

    COi0_register = []
    COi0_R_APPEND = COi0_register.append
    COi0_R_POP = COi0_register.pop
    COi0_R_CLEAR = COi0_register.clear

    res_R = []  # residue number register
    res_R_APPEND = res_R.append
    res_R_POP = res_R.pop
    res_R_CLEAR = res_R.clear
    # ?

    # /
    # required inits
    broke_on_start_attempt = False
    start_attempts = 0
    max_start_attempts = 500  # maximum attempts to start a conformer
    # because we are building from a experimental database there can be
    # some angle combinations that fail on our validation process from start
    # if this happens more than `max_start_attemps` the production is canceled.
    # ?

    # /
    # STARTS BUILDING
    conf_n = 1
    while 1:
        # prepares cycles for building process
        bond_lens = get_cycle_distances_backbone()
        bond_type = get_cycle_bond_type()

        # in the first run of the loop this is unnecessary, but is better to
        # just do it once than flag it the whole time
        template_coords[:, :] = NAN
        bb[:, :] = NAN
        bb_CO[:, :] = NAN
        bb_NH[:, :] = NAN
        for _mask, _coords in ss_masks:
            _coords[:, :] = NAN

        bb[:3, :] = seed_coords  # this contains a dummy coord at position 0

        # add N-terminal hydrogens to the origin

        bbi = 1  # starts at 1 because there are two dummy atoms
        bbi0_R_CLEAR()
        bbi0_R_APPEND(bbi)

        COi = 0  # carbonyl atoms
        COi0_R_CLEAR()
        COi0_R_APPEND(COi)

        # residue integer number
        current_res_number = 0
        res_R_CLEAR()
        res_R_APPEND(current_res_number)

        backbone_done = False
        number_of_trials = 0
        # TODO: use or not to use number_of_trials2? To evaluate in future.
        number_of_trials2 = 0
        number_of_trials3 = 0
        # run this loop until a specific BREAK is triggered
        while 1:  # 1 is faster than True :-)

            # I decided to use an if-statement here instead of polymorph
            # the else clause to a `generative_function` variable because
            # the resulting overhead from the extra function call and
            # **kwargs handling was greater then the if-statement processing
            # https://pythonicthoughtssnippets.github.io/2020/10/21/PTS14-quick-in-if-vs-polymorphism.html
            if generative_function:
                agls = generative_function(
                    nres=RINT(1, 6),
                    cres=calc_residue_num_from_index(bbi)
                    )

            else:
                # following `aligndb` function,
                # `angls` will always be cyclic with:
                # omega - phi - psi - omega - phi - psi - (...)
                agls = angles[RC(slices), :].ravel()
                #agls = angles[:, :].ravel()

            # index at the start of the current cycle
            try:
                for (omg, phi, psi) in zip(agls[0::3], agls[1::3], agls[2::3]):

                    current_res_number = calc_residue_num_from_index(bbi - 1)
                    curr_res, tpair = GET_TRIMER_SEQ(
                        all_atom_input_seq,
                        current_res_number,
                        )
                    torpair = f'{RRD10(phi)},{RRD10(psi)}'

                    for torsion_angle in (omg, phi, psi):

                        _bt = next(bond_type)

                        try:
                            _bend_angle = RC(BGEO_full[_bt][curr_res][tpair][torpair])  # noqa: E501
                        except KeyError:
                            try:
                                _bend_angle = RC(BGEO_trimer[_bt][curr_res][tpair])  # noqa: E501
                            except KeyError:
                                _bend_angle = RC(BGEO_res[_bt][curr_res])

                        _bond_lens = next(bond_lens)[curr_res]

                        bb_real[bbi, :] = MAKE_COORD_Q_LOCAL(
                            bb[bbi - 1, :],
                            bb[bbi, :],
                            bb[bbi + 1, :],
                            _bond_lens,
                            _bend_angle,
                            torsion_angle,
                            )
                        bbi += 1

                    try:
                        co_bend = RC(BGEO_full['Ca_C_O'][curr_res][tpair][torpair])  # noqa: E501
                    except KeyError:
                        try:
                            co_bend = RC(BGEO_trimer['Ca_C_O'][curr_res][tpair])
                        except KeyError:
                            co_bend = RC(BGEO_res['Ca_C_O'][curr_res])

                    bb_CO[COi, :] = MAKE_COORD_Q_PLANAR(
                        bb_real[bbi - 3, :],
                        bb_real[bbi - 2, :],
                        bb_real[bbi - 1, :],
                        distance=DISTANCE_C_O,
                        bend=co_bend
                        )
                    COi += 1

            except IndexError:
                # IndexError happens when the backbone is complete
                # in this protocol the last atom build was a carbonyl C
                # bbi is the last index of bb + 1, and the last index of
                # bb_real + 2

                # activate flag to finish loop at the end
                backbone_done = True

                # add the carboxyls
                coords[TEMPLATE_MASKS.cterm] = \
                    MAKE_COORD_Q_COO_LOCAL(bb[-2, :], bb[-1, :])

            # Adds N-H Hydrogens
            # Not a perfect loop. It repeats for Hs already placed.
            # However, was a simpler solution than matching the indexes
            # and the time cost is not a bottle neck.
            _ = ~ISNAN(bb_real[bb_NH_nums_p1, 0])
            for k, j in zip(bb_NH_nums[_], bb_NH_idx[_]):

                bb_NH[j, :] = MAKE_COORD_Q_PLANAR(
                    bb_real[k + 1, :],
                    bb_real[k, :],
                    bb_real[k - 1, :],
                    distance=DISTANCE_NH,
                    bend=BUILD_BEND_H_N_C,
                    )

            # Adds sidechain template structures
            for res_i in range(res_R[-1], current_res_number + 1):  # noqa: E501

                _sstemplate, _sidechain_idxs = \
                    SIDECHAIN_TEMPLATES[template_seq_3l[res_i]]

                sscoords = PLACE_SIDECHAIN_TEMPLATE(
                    bb_real[res_i * 3:res_i * 3 + 3, :],  # from N to C
                    _sstemplate,
                    )

                ss_masks[res_i][1][:, :] = sscoords[_sidechain_idxs]

            # Transfers coords to the main coord array
            for _smask, _sidecoords in ss_masks[:current_res_number + 1]:
                template_coords[_smask] = _sidecoords

            # / Place coordinates for energy calculation
            #
            # use `bb_real` to do not consider the initial dummy atom
            template_coords[TEMPLATE_MASKS.bb3] = bb_real
            template_coords[TEMPLATE_MASKS.COs] = bb_CO
            template_coords[TEMPLATE_MASKS.NHs] = bb_NH

            if len(bbi0_register) == 1:
                # places the N-terminal Hs only if it is the first
                # chunk being built
                _ = PLACE_SIDECHAIN_TEMPLATE(bb_real[0:3, :], N_TERMINAL_H)
                template_coords[TEMPLATE_MASKS.Hterm, :] = _[3:, :]
                current_Hterm_coords = _[3:, :]
                del _

                if template_input_seq[0] != 'G':
                    # rotates only if the first residue is not an
                    # alanie

                    # measure torsion angle reference H1 - HA
                    _h1_ha_angle = CALC_TORSION_ANGLES(
                        template_coords[TEMPLATE_MASKS.H1_N_CA_CB, :]
                            )[0]

                    ## given any angle calculated along an axis, calculate how
                    ## much to rotate along that axis to place the
                    ## angle at 60 degrees
                    _rot_angle = _h1_ha_angle % PI2 - RAD_60

                    current_Hterm_coords = ROT_COORDINATES(
                        template_coords[TEMPLATE_MASKS.Hterm, :],
                        template_coords[1] / NORM(template_coords[1]),
                        _rot_angle,
                        )

                    template_coords[TEMPLATE_MASKS.Hterm, :] = current_Hterm_coords
            # ?

            # /
            # calc energy
            total_energy = TEMPLATE_ENERGYFUNC(coords)
            #TODO:
            # * separate create_energy_func_params
            # * cambiar la mask de connections a calcular para 0 en las que hay que evitar calcular
            # * homogeneizar energy function terms for ij.

            if total_energy > 10:
                # reset coordinates to the original value
                # before the last chunk added

                # reset the same chunk maximum 5 times,
                # after that reset also the chunk before
                if number_of_trials > 50:
                    bbi0_R_POP()
                    COi0_R_POP()
                    res_R_POP()
                    number_of_trials = 0
                    number_of_trials2 += 1

                if number_of_trials2 > 5:
                    bbi0_R_POP()
                    COi0_R_POP()
                    res_R_POP()
                    number_of_trials2 = 0
                    number_of_trials3 += 1

                if number_of_trials3 > 5:
                    bbi0_R_POP()
                    COi0_R_POP()
                    res_R_POP()
                    number_of_trials3 = 0

                try:
                    _bbi0 = bbi0_register[-1]
                    _COi0 = COi0_register[-1]
                    _resi0 = res_R[-1]
                except IndexError:
                    # if this point is reached,
                    # we erased until the beginning of the conformer
                    # discard conformer, something went really wrong
                    broke_on_start_attempt = True
                    break # conformer while loop, starts conformer from scratch

                # clean previously built protein chunk
                bb_real[_bbi0:bbi, :] = NAN
                bb_CO[_COi0:COi, :] = NAN

                # reset also indexes
                bbi = _bbi0
                COi = _COi0
                current_res_number = _resi0

                # coords needs to be reset because size of protein next
                # chunks may not be equal
                template_coords[:, :] = NAN
                template_coords[TEMPLATE_MASKS.Hterm, :] = current_Hterm_coords

                # prepares cycles for building process
                # this is required because the last chunk created may have been
                # the final part of the conformer
                if backbone_done:
                    bond_lens = get_cycle_distances_backbone()
                    bond_type = get_cycle_bond_type()

                # we do not know if the next chunk will finish the protein
                # or not
                backbone_done = False
                number_of_trials += 1
                continue  # send back to the CHUNK while loop

            # if the conformer is valid
            number_of_trials = 0
            bbi0_R_APPEND(bbi)
            COi0_R_APPEND(COi)
            # the residue where the build process stopped
            res_R_APPEND(current_res_number)

            if backbone_done:
                # this point guarantees all protein atoms are built
                break  # CHUNK while loop
        # END of CHUNK while loop, go up and build the next CHUNK

        if broke_on_start_attempt:
            start_attempts += 1
            if start_attempts > max_start_attempts:
                log.error(
                    'Reached maximum amount of re-starts. Canceling... '
                    f'Built a total of {conf_n} conformers.'
                    )
                return
            broke_on_start_attempt = False
            continue  # send back to the CHUNK while loop

        # we do not want sidechains at this point
        all_atoms_coords[ALL_ATOM_MASKS.bb4] = template_coords[TEMPLATE_MASKS.bb4]  # noqa: E501
        all_atoms_coords[ALL_ATOM_MASKS.NHs] = template_coords[TEMPLATE_MASKS.NHs]  # noqa: E501
        all_atoms_coords[ALL_ATOM_MASKS.Hterm] = template_coords[TEMPLATE_MASKS.Hterm]  # noqa: E501
        all_atoms_coords[ALL_ATOM_MASKS.cterm, :] = template_coords[TEMPLATE_MASKS.cterm, :]  # noqa: E501

        if with_sidechains:

            all_atoms_coords[ALL_ATOM_MASKS.non_Hs_non_OXT] = build_sidechains(
                    template_coords[TEMPLATE_MASKS.bb4],
                    )

            total_energy = ALL_ATOM_ENERGYFUNC(all_atoms_coords)

            if total_energy > 0:
                print('Conformer with WORST energy', total_energy)
                continue
            else:
                print(conf_n, total_energy)

        yield all_atoms_coords
        conf_n += 1






#def calc_LJ_energy(acoeff, bcoeff, dists_ij, to_eval_mask, NANSUM=np.nansum):
#    """Calculates Lennard-Jones Energy."""
#    # assert dists_ij.size == acoeff.size == bcoeff.size, (dists_ij.size, acoeff.size)
#    # assert dists_ij.size == to_eval_mask.size
#
#    ar = acoeff / (dists_ij ** 12)
#    br = bcoeff / (dists_ij ** 6)
#    energy_ij = ar - br
#
#    # nansum is used because some values in dists_ij are expected to be nan
#    # return energy_ij, NANSUM(energy_ij[to_eval_mask])
#    return NANSUM(energy_ij[to_eval_mask])
#
#
#njit_calc_LJ_energy = njit(calc_LJ_energy)










def gen_PDB_from_conformer(
        input_seq_3_letters,
        atom_labels,
        residues,
        coords,
        ALF=atom_line_formatter,
        ):
    """."""
    lines = []
    LINES_APPEND = lines.append
    ALF_FORMAT = ALF.format
    resi = -1

    # this is possible ONLY because there are no DOUBLE CHARS atoms
    # in the atoms that constitute a protein chain
    ATOM_LABEL_FMT = ' {: <3}'.format

    assert len(atom_labels) == coords.shape[0]

    atom_i = 1
    for i in range(len(atom_labels)):

        if np.isnan(coords[i, 0]):
            continue

        if atom_labels[i] == 'N':
            resi += 1
            current_residue = input_seq_3_letters[resi]
            current_resnum = residues[i]

        atm = atom_labels[i].strip()
        ele = atm.lstrip('123')[0]

        if len(atm) < 4:
            atm = ATOM_LABEL_FMT(atm)

        LINES_APPEND(ALF_FORMAT(
            'ATOM',
            atom_i,
            atm,
            '',
            current_residue,
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

        atom_i += 1

    return '\n'.join(lines)






#def populate_ff_parameters_in_structure(
#        atom_labels,
#        residue_numbers,
#        residue_labels,
#        force_field,
#        ):
#    """
#    Creates a list of SIGMAS, EPSILONS, and CHARGES.
#
#    Matches label information in the conformer to the respective values
#    in the force field.
#    """
#    assert len(atom_labels) == len(residue_numbers) == len(residue_labels)
#
#    sigmas_l = []
#    epsilons_l = []
#    charges_l = []
#
#    sigmas_append = sigmas_l.append
#    epsilons_append = epsilons_l.append
#    charges_append = charges_l.append
#
#    zipit = zip(atom_labels, residue_numbers, residue_labels)
#    for atom_name, res_num, res_label in zipit:
#
#        # adds C to the terminal residues
#        if res_num == residue_numbers[-1]:
#            res = 'C' + res_label
#            was_in_C_terminal = True
#            assert res.isupper() and len(res) == 4, res
#
#        elif res_num == residue_numbers[0]:
#            res = 'N' + res_label
#            was_in_N_terminal = True
#            assert res.isupper() and len(res) == 4, res
#
#        else:
#            res = res_label
#
#        # TODO:
#        # define protonation state in parameters
#        if res_label.endswith('HIS'):
#            res_label = res_label[:-3] + 'HIP'
#
#        try:
#            # force field atom type
#            atype = force_field[res][atom_name]['type']
#
#        # TODO:
#        # try/catch is here to avoid problems with His...
#        # for this purpose we are only using side-chains
#        except KeyError:
#            raise KeyError(tuple(force_field[res].keys()))
#
#        ep = float(force_field[atype]['epsilon'])
#        # epsilons by order of atom
#        epsilons_append(ep)
#
#        sig = float(force_field[atype]['sigma'])
#        # sigmas by order of atom
#        sigmas_append(sig)
#
#        charge = float(force_field[res][atom_name]['charge'])
#        charges_append(charge)
#
#    assert len(epsilons_l) == len(sigmas_l) == len(charges_l), 'Lengths differ.'
#    assert len(epsilons_l) == len(atom_labels),\
#        'Lengths differ. There should be one epsilon per atom.'
#    assert was_in_C_terminal, \
#        'The C terminal residue was never computed. It should have.'
#    assert was_in_N_terminal, \
#        'The N terminal residue was never computed. It should have.'
#
#    # brief theoretical review:
#    # sigmas and epsilons are combined parameters self vs self (i vs i)
#    # charges refer only to the self alone (i)
#    sigmas_ii = np.array(sigmas_l)
#    epsilons_ii = np.array(epsilons_l)
#    charges_i = np.array(charges_l)
#
#    assert epsilons_ii.shape == (len(epsilons_l),)
#    assert sigmas_ii.shape == (len(sigmas_l),)
#    assert charges_i.shape == (len(charges_l),)
#
#    return sigmas_ii, epsilons_ii, charges_i



# NOT USED HERE
# kept to maintain compatibility with cli_validate.
# TODO: revisit cli_validate
#def generate_vdW_data(
#        atom_labels,
#        residue_numbers,
#        residue_labels,
#        vdW_radii,
#        bonds_apart=3,
#        tolerance=0.4,
#        ):
#    # {{{
#    """
#    Generate van der Waals related data structures.
#
#    Generated data structures are aligned to the output generated by
#    scipy.spatial.distances.pdist used to compute all pairs distances.
#
#    This function generates an (N,) array with the vdW thresholds aligned
#    with the `pdist` output. And, a (N,) boolean array where `False`
#    denotes pairs of covalently bonded atoms.
#
#    Input should satisfy:
#
#    >>> len(atom_labels) == len(residue_numbers) == len(residue_labels)
#
#    Parameters
#    ----------
#    atom_labels : array or list
#        The ordered list of atom labels of the target conformer upon
#        which the returned values of this function will be used.
#
#    residue_numbers : array or list
#        The list of residue numbers corresponding to the atoms of the
#        target conformer.
#
#    residue_labels : array or list
#        The list of residue 3-letter labels of each atom of the target
#        conformer.
#
#    vdW_radii : dict
#        A dictionary containing the van der Waals radii for each atom
#        type (element) present in `atom_labels`.
#
#    bonds_apart : int
#        The number of bonds apart to ignore vdW clash validation.
#        For example, 3 means vdW validation will only be computed for
#        atoms at least 4 bonds apart.
#
#    tolerance : float
#        The tolerance in Angstroms.
#
#    Returns
#    -------
#    nd.array, dtype=np.float
#        The vdW atom pairs thresholds. Equals to the sum of atom vdW
#        radius.
#
#    nd.array, dtype=boolean
#        `True` where the distance between pairs must be considered.
#        `False` where pairs are covalently bound.
#
#    Note
#    ----
#    This function is slow, in the order of 1 or 2 seconds, but it is
#    executed only once at the beginning of the building protocol.
#    """
#    # }}}
#    assert len(atom_labels) == len(residue_numbers)
#    assert len(atom_labels) == len(residue_labels)
#
#    # we need to use re because hydrogen atoms start with integers some times
#    atoms_char = re.compile(r'[CHNOPS]')
#    findall = atoms_char.findall
#    cov_topologies = generate_residue_template_topology(
#        amber_pdbs,
#        atom_labels_amber,
#        add_OXT=True,
#        add_Nterminal_H=True,
#        )
#    bond_structure_local = \
#        expand_topology_bonds_apart(cov_topologies, bonds_apart)
#    inter_connect_local = inter_residue_connectivities[bonds_apart]
#
#    # adds OXT to the bonds connectivity, so it is included in the
#    # final for loop creating masks
#    #add_OXT_to_residue(bond_structure_local[residue_labels[-1]])
#
#    # the following arrays/lists prepare the conformer label data in agreement
#    # with the function scipy.spatial.distances.pdist, in order for masks to be
#    # used properly
#    #
#    # creates atom label pairs
#    atoms = np.array([
#        (a, b)
#        for i, a in enumerate(atom_labels, start=1)
#        for b in atom_labels[i:]
#        ])
#
#    # creates vdW radii pairs according to atom labels
#    vdw_pairs = np.array([
#        (vdW_radii[findall(a)[0]], vdW_radii[findall(b)[0]])
#        for a, b in atoms
#        ])
#
#    # creats the vdW threshold according to vdW pairs
#    # the threshold is the sum of the vdW radii pair
#    vdW_sums = np.power(np.sum(vdw_pairs, axis=1) - tolerance, 2)
#
#    # creates pairs for residue numbers, so we know from which residue number
#    # is each atom of the confronted pair
#    res_nums_pairs = np.array([
#        (a, b)
#        for i, a in enumerate(residue_numbers, start=1)
#        for b in residue_numbers[i:]
#        ])
#
#    # does the same as above but for residue 3-letter labels
#    res_labels_pairs = (
#        (_a, _b)
#        for i, _a in enumerate(residue_labels, start=1)
#        for _b in residue_labels[i:]
#        )
#
#    # we want to compute clashes to all atoms that are not covalentely bond
#    # that that have not certain connectivity between them
#    # first we assum True to all cases, and we replace to False where we find
#    # a bond connection
#    vdW_non_bond = np.full(len(atoms), True)
#
#    counter = -1
#    zipit = zip(atoms, res_nums_pairs, res_labels_pairs)
#    for (a1, a2), (res1, res2), (l1, _) in zipit:
#        counter += 1
#        if res1 == res2 and a2 in bond_structure_local[l1][a1]:
#            # here we found a bond connection
#            vdW_non_bond[counter] = False
#
#        # handling the special case of a covalent bond between two atoms
#        # not part of the same residue
#        elif a1 in inter_connect_local \
#                and res2 == res1 + 1 \
#                and a2 in inter_connect_local[a1]:
#            vdW_non_bond[counter] = False
#
#    return vdW_sums, vdW_non_bond


#def calc_energy(coords, acoeff, bcoeff, charges_ij, bonds_ge_3_mask):
#    """Calculates energy."""
#    CALC_DISTS = calc_all_vs_all_dists
#    NANSUM = np.nansum
#    distances_ij = CALC_DISTS(coords)
#
#    energy_lj = njit_calc_LJ_energy(
#        acoeff,
#        bcoeff,
#        distances_ij,
#        bonds_ge_3_mask,
#        )
#
#    energy_elec_ij = charges_ij / distances_ij
#    energy_elec = NANSUM(energy_elec_ij[bonds_ge_3_mask])
#
#    # total energy
#    return energy_lj + energy_elec


#def calc_coulomb(distances_ij, charges_ij, NANSUM=np.nansum):
#    return NANSUM(charges_ij / distances_ij)
#
#calc_coulomb_njit = njit(calc_coulomb)










# NOT used
#def create_energy_func_params(atom_labels, residue_numbers, residue_labels):
#    """Create parameters for energy function."""
#    # /
#    # Prepares terms for the energy function
#    # TODO: parametrize this.
#    # the user should be able to chose different forcefields
#    ff14SB = read_ff14SB_params()
#    ambertopology = AmberTopology(add_OXT=True, add_Nterminal_H=True)
#    #res_topology = generate_residue_template_topology(
#    #    amber_pdbs,
#    #    atom_labels_amber,
#    #    add_OXT=True,
#    #    add_Nterminal_H=True,
#    #    )
#    #bonds_equal_3_intra = topology_3_bonds_apart(res_topology)
#    #bonds_le_2_intra = expand_topology_bonds_apart(res_topology, 2)
#
#    # units in:
#    # nm, kJ/mol, proton units
#    sigmas_ii, epsilons_ii, charges_i = populate_ff_parameters_in_structure(
#        atom_labels,
#        residue_numbers,
#        residue_labels,
#        ff14SB,  # the forcefield dictionary
#        )
#
#    # this mask will deactivate calculations in covalently bond atoms and
#    # atoms separated 2 bonds apart
#    _bonds_ge_3_mask = create_bonds_apart_mask_for_ij_pairs(
#        atom_labels,
#        residue_numbers,
#        residue_labels,
#        ambertopology.bonds_le2_intra,
#        bonds_le_2_inter,
#        base_bool=True,
#        )
#    # on lun 21 dic 2020 17:33:31 EST, I tested for 1M sized array
#    # the numeric indexing performed better than the boolean indexing
#    # 25 ns versus 31 ns.
#    bonds_ge_3_mask = np.where(_bonds_ge_3_mask)[0]
#
#    # /
#    # Prepares Coulomb and Lennard-Jones pre computed parameters:
#    # calculates ij combinations using raw njitted functions because using
#    # numpy outer variantes in very large systems overloads memory and
#    # reduces performance.
#    #
#    num_ij_pairs = len(atom_labels) * (len(atom_labels) - 1) // 2
#    # sigmas
#    sigmas_ij_pre = np.empty(num_ij_pairs, dtype=np.float64)
#    njit_calc_sum_upper_diagonal_raw(sigmas_ii, sigmas_ij_pre)
#    #
#    # epsilons
#    epsilons_ij_pre = np.empty(num_ij_pairs, dtype=np.float64)
#    njit_calc_multiplication_upper_diagonal_raw(epsilons_ii, epsilons_ij_pre)
#    #
#    # charges
#    charges_ij = np.empty(num_ij_pairs, dtype=np.float64)
#    njit_calc_multiplication_upper_diagonal_raw(charges_i, charges_ij)
#
#    # mixing rules
#    epsilons_ij = epsilons_ij_pre ** 0.5
#    sigmas_ij = sigmas_ij_pre * 5  # mixing + nm to Angstrom converstion
#
#    acoeff = 4 * epsilons_ij * (sigmas_ij ** 12)
#    bcoeff = 4 * epsilons_ij * (sigmas_ij ** 6)
#
#    charges_ij *= 0.25  # dielectic constant
#
#    # The mask to identify ij pairs exactly 3 bonds apart is needed for the
#    # special scaling factor of Coulomb and LJ equations
#    # This mask will be used only aftert the calculation of the CLJ params
#    bonds_exact_3_mask = create_bonds_apart_mask_for_ij_pairs(
#        atom_labels,
#        residue_numbers,
#        residue_labels,
#        ambertopology.bonds_eq3_intra,
#        bonds_equal_3_inter,
#        )
#
#    # this is the Lennard-Jones special case, where scaling factors are applied
#    # to atoms bonded 3 bonds apart, known as the '14' cases.
#    # 0.4 was calibrated manually, until I could find a conformer
#    # within 50 trials dom 20 dic 2020 13:16:50 EST
#    # I believe, because we are not doing local minimization here, we
#    # cannot be that strick with the 14 scaling factor, and a reduction
#    # factor of 2 is not enough
#    acoeff[bonds_exact_3_mask] *= float(ff14SB['lj14scale']) * 0.2  # was 0.4
#    bcoeff[bonds_exact_3_mask] *= float(ff14SB['lj14scale']) * 0.2
#    charges_ij[bonds_exact_3_mask] *= float(ff14SB['coulomb14scale'])
#
#    #del sigmas_ij_pre, epsilons_ij_pre, epsilons_ij, sigmas_ij
#    #del bonds_exact_3_mask, _bonds_ge_3_mask, ff14SB
#    assert len(acoeff) == len(bcoeff) == len(charges_ij)
#
#    return acoeff, bcoeff, charges_ij, bonds_ge_3_mask
#    # ?









if __name__ == "__main__":
    libcli.maincli(ap, main)
