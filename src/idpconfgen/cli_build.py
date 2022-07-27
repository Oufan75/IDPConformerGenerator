"""
Builds IDP conformers.

Build from a database of torsion angles and secondary structure
information. Database is as created by `idpconfgen torsions` CLI.

USAGE:
    $ idpconfgen build -db torsions.json -seq MMMMMMM...

"""
#import argparse
#from multiprocessing import Queue, Pool
import os
import numpy as np

from idpconfgen import log
#from idpconfgen.components.energy_threshold_type import add_et_type_arg
from idpconfgen.core.build_definitions import (
    backbone_atoms,
    build_bend_H_N_C,
    distance_C_O,
    distance_H_N,
    bend_CA_C_O,
    forcefields,
    n_terminal_h_coords_at_origin,
    sidechain_templates,
    )

#from idpconfgen.core import help_docs
#from idpconfgen.libs import libcli
from idpconfgen.libs.libbuild import (
    build_regex_substitutions,
    #prepare_slice_dict,
    #read_db_to_slices_given_secondary_structure,
    create_sidechains_masks_per_residue,
    #get_cycle_bond_type,
    get_cycle_bend_angles,
    get_cycle_distances_backbone,
    init_conflabels,
    init_confmasks,
    prepare_energy_function,
    rotate_sidechain
    #read_db_to_slices,
    )
from idpconfgen.libs.libcalc import (
    calc_residue_num_from_index,
    calc_torsion_angles,
    make_coord_Q,
    make_coord_Q_COO,
    make_coord_Q_planar,
    #make_seq_probabilities,
    place_sidechain_template,
    rotate_coordinates_Q_njit,
    )
from idpconfgen.libs.libparse import (
    fill_list,
    get_seq_chunk_njit,
    #get_seq_next_residue_njit,
    get_trimer_seq_njit,
    #remap_sequence,
    #remove_empty_keys,
    translate_seq_to_3l,
    )
from idpconfgen.libs.libpdb import atom_line_formatter
#from idpconfgen.logger import S, T, init_files, pre_msg, report_on_crash


# Global variables needed to build conformers.
# Why are global variables needed?
# I use global variables to facilitate distributing conformer creation
# processes across multiple cores. In this way cores can read global variables
# fast and with non-significant overhead.

# Bond Geometry library variables
# if __name__ == '__main__', these will be populated in main()
# else will be populated in conformer_generator
# populate_globals() populates these variables once called.
#BGEO_path = Path(_file, 'core', 'data', 'dunback_bgeo.tar')
#BGEO_full = {}
#BGEO_trimer = {}
#BGEO_res = {}



ALL_ATOM_LABELS = None
ALL_ATOM_MASKS = None
ALL_ATOM_EFUNC = None



def generate_pdb_checkpt(seq, atom_labels, res_nums, coords,
                         nstep, dir_name):
    pdb_string = gen_PDB_from_conformer(seq, atom_labels, res_nums, 
                                        np.round(coords, decimals=3))
    fname = os.path.join(dir_name, 'step_%i.pdb'%nstep)
    with open(fname, 'w') as fout:
        fout.write(pdb_string)


def are_globals():
    """Assess if global variables needed for building are populated."""
    return all((
        ALL_ATOM_LABELS,
        ALL_ATOM_MASKS,
        ALL_ATOM_EFUNC,
        #BGEO_full,
        #BGEO_trimer,
        #BGEO_res,
        ))



def populate_globals(
        *,
        input_seq=None,
        #bgeo_path=BGEO_path,
        forcefield=None,
        **efunc_kwargs):
    """
    Populate global variables needed for building.

    Currently, global variables include:

    BGEO_full
    BGEO_trimer
    BGEO_res
    ALL_ATOM_LABELS, ALL_ATOM_MASKS, ALL_ATOM_EFUNC

    Parameters
    ----------
    bgeo_path : str or Path
        The path pointing to a bond geometry library as created by the
        `bgeo` CLI.

    forcefield : str
        A key in the `core.build_definitions.forcefields` dictionary.
    """
    if not isinstance(input_seq, str):
        raise ValueError(
            '`input_seq` not valid. '
            f'Expected string found {type(input_seq)}'
            )

    #global BGEO_full, BGEO_trimer, BGEO_res

    #BGEO_full.update(read_dictionary_from_disk(bgeo_path))
    #_1, _2 = bgeo_reduce(BGEO_full)
    #BGEO_trimer.update(_1)
    #BGEO_res.update(_2)
    #del _1, _2
    #assert BGEO_full
    #assert BGEO_trimer
    #assert BGEO_res
    # this asserts only the first layer of keys
    #assert list(BGEO_full.keys()) == list(BGEO_trimer.keys()) == list(BGEO_res.keys())  # noqa: E501

    # populates the labels
    global ALL_ATOM_LABELS, ALL_ATOM_MASKS, ALL_ATOM_EFUNC

    topobj = forcefield(add_OXT=True, add_Nterminal_H=True)

    ALL_ATOM_LABELS = init_conflabels(input_seq, topobj.atom_names)

    ALL_ATOM_MASKS = init_confmasks(ALL_ATOM_LABELS.atom_labels)

    ALL_ATOM_EFUNC = prepare_energy_function(
        ALL_ATOM_LABELS.atom_labels,
        ALL_ATOM_LABELS.res_nums,
        ALL_ATOM_LABELS.res_labels,
        topobj,
        **efunc_kwargs)

    del topobj
    return




# the name of this function is likely to change in the future
def conformer_generator(
        *,
        input_seq=None,
        generative_function=None,
        energy_threshold=100,
        bgeo_path=None,
        forcefield='Amberff14SB',
        random_seed=0,
        clash_check=True,
        res_try_limit=10,
        **energy_funcs_kwargs,
        ):
    """
    Build conformers. (builds an entire residue before the next)

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


    bgeo_path : str of Path
        Path to a bond geometry library as created by `bgeo` CLI.

    Yields
    ------
    First yield: tuple (np.ndarray, np.ndarray, np.ndarray)
        The conformer label arrays.

    Other yields: tuple (float, np.ndarray)
        Energy of the conformer, conformer coordinates.
    """
    if not isinstance(input_seq, str):
        raise ValueError(f'`input_seq` must be given! {input_seq}')
    

    #log.info(f'random seed: {random_seed}')
    #np.random.seed(random_seed)
    #seed_report = pre_msg(f'seed {random_seed}', sep=' - ')

    # prepares protein sequences
    all_atom_input_seq = input_seq
    all_atom_seq_3l = translate_seq_to_3l(all_atom_input_seq)
    #print(input_seq)
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
    ROT_COORDINATES = rotate_coordinates_Q_njit
    SIDECHAIN_TEMPLATES = sidechain_templates

    #global BGEO_full
    #global BGEO_trimer
    #global BGEO_res
    global ALL_ATOM_LABELS
    global ALL_ATOM_MASKS
    global ALL_ATOM_EFUNC


    if not are_globals():
        if forcefield not in forcefields:
            raise ValueError(
                f'{forcefield} not in `forcefields`. '
                f'Expected {list(forcefields.keys())}.'
                )
        populate_globals(
            input_seq=all_atom_input_seq,
            #bgeo_path=bgeo_path or BGEO_path,
            forcefield=forcefields[forcefield],
            **energy_funcs_kwargs,
            )

    # yields atom labels
    # all conformers generated will share these labels
    yield (
        ALL_ATOM_LABELS.atom_labels,
        ALL_ATOM_LABELS.res_nums,
        ALL_ATOM_LABELS.res_labels,
        )

    all_atom_num_atoms = len(ALL_ATOM_LABELS.atom_labels)

    all_atom_coords = np.full((all_atom_num_atoms, 3), NAN, dtype=np.float64)

    # +2 because of the dummy coordinates required to start building.
    # see later adding dummy coordinates to the structure seed
    bb = np.full((ALL_ATOM_MASKS.bb3.size + 2, 3), NAN, dtype=np.float64)
    bb_real = bb[2:, :]  # backbone coordinates without the dummies

    # coordinates for the carbonyl oxigen atoms
    bb_CO = np.full((ALL_ATOM_MASKS.COs.size, 3), NAN, dtype=np.float64)

    # notice that NHydrogen_mask does not see Prolines
    bb_NH = np.full((ALL_ATOM_MASKS.NHs.size, 3), NAN, dtype=np.float64)
    bb_NH_idx = np.arange(len(bb_NH))
    # Creates masks and indexes for the `for` loop used to place NHs.
    # The first residue has no NH, prolines have no NH.
    non_pro = np.array(list(all_atom_input_seq)[1:]) != 'P'
    # NHs index numbers in bb_real
    bb_NH_nums = np.arange(3, (len(all_atom_input_seq) - 1) * 3 + 1, 3)[non_pro]
    bb_NH_nums_p1 = bb_NH_nums + 1
    assert bb_NH.shape[0] == bb_NH_nums.size == bb_NH_idx.size
    #print(np.sum(ALL_ATOM_LABELS.res_nums == 7))
    # sidechain masks
    # this is sidechain agnostic, works for every sidechain
    ss_masks = create_sidechains_masks_per_residue(
        ALL_ATOM_LABELS.res_nums,
        ALL_ATOM_LABELS.atom_labels,
        backbone_atoms,
        )
    dummy_CA_m1_coord = np.array((0.0, 1.0, 1.0))
    dummy_C_m1_coord = np.array((0.0, 1.0, 0.0))
    n_terminal_N_coord = np.array((0.0, 0.0, 0.0))

    # seed coordinates array
    seed_coords = np.array((
        dummy_CA_m1_coord,
        dummy_C_m1_coord,
        n_terminal_N_coord,
        ))

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

    # required inits
    broke_on_start_attempt = False
    start_attempts = 0
    max_start_attempts = 10  # maximum attempts to start a conformer
    # /
    # STARTS BUILDING
    conf_n = 0
    while 1:
        # prepares cycles for building process
        bond_lens = get_cycle_distances_backbone()
        #bond_type = get_cycle_bond_type()
        bend_angle = get_cycle_bend_angles()

        # in the first run of the loop this is unnecessary, but is better to
        # just do it once than flag it the whole time
        all_atom_coords[:, :] = NAN
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
        same_res_try = 0
        number_of_trials = 0
        # TODO: use or not to use number_of_trials2? To evaluate in future.
        number_of_trials2 = 0
        number_of_trials3 = 0
        nstep = 0
        # run this loop until a specific BREAK is triggered
        while 1:  # 1 is faster than True :-)
            nstep += 1
            if same_res_try == 0:
                agls = generative_function(
                    cres = calc_residue_num_from_index(bbi)
                    )
                agl_stored = agls.copy()
                #print(np.degrees(agl_stored[:3]))
            else:
                agls = agl_stored + np.random.uniform(np.radians(-3), np.radians(3), len(agls))
                for n in range(len(agls)):
                    if agls[n] > np.pi: agls[n] -= 2*np.pi
                    elif agls[n] < -np.pi: agls[n] += 2*np.pi
                #print(np.degrees(agls[:3]))  
            try:
                current_res_number = calc_residue_num_from_index(bbi - 1) # bbi - 1 = 0 atom Ca
                curr_res, tpair = GET_TRIMER_SEQ(
                        all_atom_input_seq,
                        current_res_number,
                        )
                #print('resnum:', current_res_number)
                for torsion_angle in (agls[0], agls[1], agls[2]): 

                    #_bt = next(bond_type)
                    
                    #try:
                    #    _bend_angle = RC(BGEO_full[_bt][curr_res][tpair][torpair])  # noqa: E501
                    #except KeyError:
                    #    try:
                    #        _bend_angle = RC(BGEO_trimer[_bt][curr_res][tpair])  # noqa: E501
                    #    except KeyError:
                    #        _bend_angle = RC(BGEO_res[_bt][curr_res])
                    _bend_angle = next(bend_angle)[curr_res]
                    #_bend_angle = np.random.normal(_bend_stats[0], _bend_stats[1])

                    _bond_lens = next(bond_lens)[curr_res] + np.random.normal(0., 0.01)
                    #_bond_lens = np.random.normal(_bond_stats[0], _bond_stats[1])
                    
                    bb_real[bbi, :] = MAKE_COORD_Q_LOCAL(
                        bb[bbi - 1, :],
                        bb[bbi, :],
                        bb[bbi + 1, :],
                        _bond_lens,
                        _bend_angle,
                        torsion_angle,
                        )
                    bbi += 1

                #try:
                #    co_bend = RC(BGEO_full['Ca_C_O'][curr_res][tpair][torpair])  # noqa: E501
                #except KeyError:
                #    try:
                #        co_bend = RC(BGEO_trimer['Ca_C_O'][curr_res][tpair])
                #    except KeyError:
                #        co_bend = RC(BGEO_res['Ca_C_O'][curr_res])
                _co_bend_stats = bend_CA_C_O[curr_res]
                co_bend = np.random.normal(_co_bend_stats[0], _co_bend_stats[1])

                bb_CO[COi, :] = MAKE_COORD_Q_PLANAR(
                    bb_real[bbi - 3, :],
                    bb_real[bbi - 2, :],
                    bb_real[bbi - 1, :],
                    distance=DISTANCE_C_O,
                    bend=co_bend
                    )
                COi += 1

            except IndexError:
                # activate flag to finish loop at the end
                backbone_done = True

                # add the carboxyls
                all_atom_coords[ALL_ATOM_MASKS.cterm] = \
                    MAKE_COORD_Q_COO_LOCAL(bb[-2, :], bb[-1, :])

            # Adds N-H Hydrogens
            _ = ~ISNAN(bb_real[bb_NH_nums_p1, 0])
            for k, j in zip(bb_NH_nums[_], bb_NH_idx[_]):

                bb_NH[j, :] = MAKE_COORD_Q_PLANAR(
                    bb_real[k + 1, :],
                    bb_real[k, :],
                    bb_real[k - 1, :],
                    distance=DISTANCE_NH,
                    bend=BUILD_BEND_H_N_C,
                    )
                
            res_type = all_atom_seq_3l[current_res_number]
            #res_type_1l = all_atom_input_seq[current_res_number]
            _sstemplate, _sidechain_idxs = SIDECHAIN_TEMPLATES[res_type]
            
            # make false for testing backbones
            if res_type not in ['ALA', 'GLY']:
                _sstemplate, _sidechain_idxs = rotate_sidechain(res_type, agls[3:])
                
            # Adds sidechain template structures        
            sscoords = PLACE_SIDECHAIN_TEMPLATE(
                    bb_real[current_res_number * 3:current_res_number* 3 + 3, :],  # from N to C
                    _sstemplate,
                    )
            #print('res:', res_type, current_res_number + 1)
            ss_masks[current_res_number][1][:, :] = sscoords[_sidechain_idxs]
            
            
            # Transfers coords to the main coord array
            for _smask, _sidecoords in ss_masks[:current_res_number + 1]:
                all_atom_coords[_smask] = _sidecoords

            # / Place coordinates for energy calculation
            #
            # use `bb_real` to do not consider the initial dummy atom
            all_atom_coords[ALL_ATOM_MASKS.bb3] = bb_real
            all_atom_coords[ALL_ATOM_MASKS.COs] = bb_CO
            all_atom_coords[ALL_ATOM_MASKS.NHs] = bb_NH

            if len(bbi0_register) == 1:
                # places the N-terminal Hs only if it is the first
                # chunk being built
                _ = PLACE_SIDECHAIN_TEMPLATE(bb_real[0:3, :], N_TERMINAL_H)
                all_atom_coords[ALL_ATOM_MASKS.Hterm, :] = _[3:, :]
                del _

                if all_atom_input_seq[0] != 'G':
                    # rotates only if the first residue is not an
                    # alanie

                    # measure torsion angle reference H1 - HA
                    _h1_ha_angle = CALC_TORSION_ANGLES(
                        all_atom_coords[ALL_ATOM_MASKS.H1_N_CA_CB, :])[0]

                    # given any angle calculated along an axis, calculate how
                    # much to rotate along that axis to place the
                    # angle at 60 degrees
                    _rot_angle = _h1_ha_angle % PI2 - RAD_60

                    current_Hterm_coords = ROT_COORDINATES(
                        all_atom_coords[ALL_ATOM_MASKS.Hterm, :],
                        all_atom_coords[1] / NORM(all_atom_coords[1]),
                        _rot_angle,
                        )
                    all_atom_coords[ALL_ATOM_MASKS.Hterm, :] = current_Hterm_coords

            # TODO:mark last built atom to avoid recalculating old energies
            total_energy = ALL_ATOM_EFUNC(all_atom_coords)        
            
            if np.any(total_energy>energy_threshold):
                try:
                    if same_res_try < res_try_limit:
                        same_res_try += 1
                    else:
                        number_of_trials += 1
                        same_res_try = 0
                        if number_of_trials > 4:
                            bbi0_R_POP()
                            COi0_R_POP()
                            res_R_POP()
                            number_of_trials = 0
                            number_of_trials2 += 1
    
                        if number_of_trials2 > 2:
                            bbi0_R_POP()
                            COi0_R_POP()
                            res_R_POP()
                            number_of_trials2 = 0
                            number_of_trials3 += 1
    
                        if number_of_trials3 > 2:
                            bbi0_R_POP()
                            COi0_R_POP()
                            res_R_POP()
                            number_of_trials3 = 0

                    _bbi0 = bbi0_register[-1]
                    _COi0 = COi0_register[-1]
                    _resi0 = res_R[-1]
                except IndexError:
                    # discard conformer, something went really wrong
                    broke_on_start_attempt = True
                    break  # conformer while loop, starts conformer from scratch

                # clean previously built protein chunk
                bb_real[_bbi0:bbi, :] = NAN
                bb_CO[_COi0:COi, :] = NAN

                # reset also indexes
                bbi = _bbi0
                COi = _COi0
                current_res_number = _resi0

                # coords needs to be reset because size of protein next
                # chunks may not be equal
                for _mask, _coords in ss_masks:
                    all_atom_coords[_mask, :] = NAN
                all_atom_coords[ALL_ATOM_MASKS.NHs, :] = NAN

                # prepares cycles for building process
                # this is required because the last chunk created may have been
                # the final part of the conformer
                if backbone_done:
                    bond_lens = get_cycle_distances_backbone()
                    #bond_type = get_cycle_bond_type()

                # we do not know if the next chunk will finish the protein
                # or not
                backbone_done = False
                continue  # send back to the CHUNK while loop

            # if the conformer is valid
            #print(np.degrees(agls[:3]))
            same_res_try = 0
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
                print(
                    'Reached maximum amount of re-starts. Canceling... '
                    f'Built a total of {conf_n} conformers.'
                    )
                return
            broke_on_start_attempt = False
            continue  # send back to the CHUNK while loop

        start_attempts = 0
        yield all_atom_coords
        conf_n += 1


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


def get_adjacent_angles(
        options,
        probs,
        seq,
        db,
        slice_dict,
        residue_replacements=None,
        RC=np.random.choice,
        ):
    """
    Get angles to build the next adjacent protein chunk.

    Parameters
    ----------
    options : list
        The length of the possible chunk sizes.

    probs : list
        A list with the relative probabilites to select from `options`.

    seq : str
        The conformer sequence.

    db : dict-like
        The angle omega/phi/psi database.

    slice_dict : dict-like
        A dictionary containing the chunks strings as keys and as values
        lists with slice objects.
    """
    residue_replacements = residue_replacements or {}
    probs = fill_list(probs, 0, len(options))
    

    def func(aidx):

        # calculates the current residue number from the atom index
        cr = calc_residue_num_from_index(aidx)

        # chooses the size of the chunk from pre-configured range of sizes
        plen = RC(options, p=probs)

        # defines the chunk identity accordingly
        primer_template = get_seq_chunk_njit(seq, cr, plen)
        next_residue = get_seq_chunk_njit(seq, cr + plen, 1)

        # recalculates the plen to avoid plen/template inconsistencies that
        # occur if the plen is higher then the number of
        # residues until the end of the protein.
        plen = len(primer_template)

        pt_sub = build_regex_substitutions(primer_template, residue_replacements)

        while plen > 0:
            if next_residue == 'P':
                pt_sub = f'{pt_sub}_P'
            try:
                #print('pt_sub', pt_sub)
                angles = db[RC(slice_dict[plen][pt_sub]), :].ravel()
            except KeyError:
                plen -= 1
                next_residue = primer_template[-1]
                primer_template = primer_template[:-1]
                pt_sub = build_regex_substitutions(
                    primer_template,
                    residue_replacements,
                    )
                if next_residue == 'P':
                    pt_sub = f'{pt_sub}_P'
            else:
                break
        else:
            log.debug(plen, primer_template, seq[cr:cr + plen])
            # raise AssertionError to avoid `python -o` silencing
            raise AssertionError('The code should not arrive here')

        if next_residue == 'P':
            # because angles have the proline information
            return primer_template + 'P', angles
        else:
            return primer_template, angles

    return func


def pdb_checkpoint(seq, atom_labels, res_nums, coords,
                   nstep, dirname):
    pdb_string = gen_PDB_from_conformer(seq, atom_labels,
                                        res_nums,
                                        np.round(coords, decimals=3))
    fname = os.path.join(dirname, 'step_%i.pdb'%nstep)
    with open(fname, 'w') as fout:
        fout.write(pdb_string)


#if __name__ == "__main__":
#    libcli.maincli(ap, main)
