"""
Higher level functions.

Function which operate with several libraries
and are defined here to avoid circular imports.
"""
import re
from contextlib import suppress
from functools import partial, reduce

import numpy as np

from idpconfgen import Path, log
from idpconfgen.core.definitions import aa3to1, blocked_ids
from idpconfgen.core.exceptions import IDPConfGenException, PDBFormatError
from idpconfgen.libs.libcalc import (
    calc_angle_njit,
    calc_torsion_angles,
    get_separate_torsions,
    rrd10_njit,
    validate_backbone_labels_for_torsion,
    )
from idpconfgen.libs.libio import (
    concatenate_entries,
    read_PDBID_from_source,
    save_pairs_to_disk,
    )
from idpconfgen.libs.libmulticore import (
    consume_iterable_in_list,
    flat_results_from_chunk,
    pool_function_in_chunks,
    )
from idpconfgen.libs.libparse import group_by
from idpconfgen.libs.libpdb import PDBList, atom_name, atom_resSeq
from idpconfgen.libs.libstructure import (
    Structure,
    col_name,
    col_resName,
    cols_coords,
    )
from idpconfgen.logger import S, T, init_files, report_on_crash


# USED OKAY!
def download_pipeline(func, logfilename='.download'):
    """
    Context pipeline to download PDB/mmCIF files.

    Exists because fetching and download filtered PDBs shared the same
    operational contextm, only differing on the function which
    orchestrates the download process.

    Parameters
    ----------
    func : function
        The actual function that orchestrates the download operation.

    logfilename : str, optional
        The common stem name of the log files.
    """
    LOGFILESNAME = logfilename

    def main(
            pdbids,
            chunks=5_000,
            destination=None,
            ncores=1,
            update=False,
            ):
        """Run main script logic."""
        init_files(log, LOGFILESNAME)

        #
        log.info(T('reading input PDB list'))

        pdblist = PDBList(concatenate_entries(pdbids))

        log.info(
            f"{S(str(pdblist))}\n"
            f"{S('done')}\n"
            )

        #
        log.info(T('Filtering input'))
        destination = destination or Path.cwd()
        log.info(
            f"{S(f'from destination: {destination}')}\n"
            f"{S('and other sources...')}"
            )

        # comparison block
        def diff(first, other):
            return first.difference(other)

        remove_from_input = [
            read_PDBID_from_source(destination),
            PDBList(blocked_ids),
            ]

        # yes, there are just two items in remove_from_input, why use reduce?
        # what if more are added in the future? :-P
        # the engine is already created
        pdblist_comparison = reduce(diff, remove_from_input, pdblist)
        log.info(S(f'Found {str(pdblist_comparison)} to download'))
        #

        something_to_download = len(pdblist_comparison) > 0
        if something_to_download and update:

            # the function to be used in multiprocessing
            consume_func = partial(consume_iterable_in_list, func)

            # context to perform a dedicated report in case function fails
            # partial is mandatory because decorators won't pickle
            # in this way, crashes are reported for files, crashed files
            # ignored, and the process continues
            execute = partial(
                report_on_crash,
                consume_func,
                ROC_exception=Exception,
                ROC_prefix='download_pipeline',
                )

            # convinient partial
            execute_pool = partial(
                pool_function_in_chunks,
                execute,
                list(pdblist_comparison.name_chains_dict.items()),
                ncores=ncores,
                chunks=chunks,
                )

            flat_results_from_chunk(
                execute_pool,
                save_pairs_to_disk,
                destination=destination,
                )

            log.info(T('Reading UPDATED destination'))
            pdblist_updated = read_PDBID_from_source(destination)
            pdblist_up_comparison = pdblist.difference(pdblist_updated)
            log.info(S(f'{str(pdblist_up_comparison)}'))
            if len(pdblist_up_comparison) > 0:
                log.info(S(
                    'There are PDBIDs not downloaded\n.'
                    'Those IDs have been registered in the '
                    f'{LOGFILESNAME}.debug file.'
                    ))
                log.debug('\n'.join(str(_id) for _id in pdblist_up_comparison))

        elif not something_to_download and update:
            log.info(S('There is nothing to download.'))
            log.info(S(
                'All requested IDs are already at '
                'the destination folder.'
                ))

        log.info(T('PDB Downloader finished'))
        return

    return main


# USED OKAY
def extract_secondary_structure(
        pdbid,
        ssdata,
        atoms='all',
        minimum=0,
        structure='all',
        ):
    """
    Extract secondary structure elements from PDB data.

    Parameters
    ----------
    pdbid : tuple
        Where index 0 is the PDB id code, for example `12AS.pdb`, or
        `12AS_A`, or `12AS_A_seg1.pdb`.
        And, index 1 is the PDB data itself in bytes.

    ssdata : dict
        Dictionary containing the DSSP information. Must contain a key
        equal to `Path(pdbid).stem, where `dssp` key contains the DSSP
        information for that PDB.

    atoms : str or list of str or bytes, optional
        The atom names to keep.
        Defaults to `all`.

    minimum : int, optional
        The minimum size a segment must have in order to be considered.
        Defaults to 0.

    structure : str or list of chars
        The secondary structure character to separate.
        Multiple can be given in the form of a list.
    """
    # caused problems in the past
    assert atoms != ['all']
    assert structure != ['all']

    pdbname = Path(pdbid[0]).stem
    pdbdata = pdbid[1].split(b'\n')

    # gets PDB computed data from dictionary
    try:
        pdbdd = ssdata[pdbname]
    except KeyError:
        pdbdd = ssdata[f'{pdbname}.pdb']

    if structure == 'all':
        ss_to_isolate = set(pdbdd['dssp'])
    else:
        ss_to_isolate = set(structure)

    # general lines filters
    line_filters = []
    LF_append = line_filters.append
    LF_pop = line_filters.pop

    # in atoms filter
    if atoms != 'all':
        with suppress(AttributeError):  # it is more common to receive str
            atoms = [c.encode() for c in atoms]
        line_filters.append(lambda x: x[atom_name].strip() in atoms)

    dssp_slices = group_by(pdbdd['dssp'])
    # DR stands for dssp residues
    DR = [c for c in pdbdd['resids'].encode().split(b',')]

    for ss in ss_to_isolate:

        ssfilter = (slice_ for char, slice_ in dssp_slices if char == ss)
        minimum_size = (s for s in ssfilter if s.stop - s.start >= minimum)

        for counter, seg_slice in enumerate(minimum_size):

            LF_append(lambda x: x[atom_resSeq].strip() in DR[seg_slice])
            pdb = b'\n'.join(
                line for line in pdbdata if all(f(line) for f in line_filters)
                )
            LF_pop()

            yield f'{pdbname}_{ss}_{counter}.pdb', pdb

            counter += 1


def get_torsionsJ(
        fdata,
        decimals=5,
        degrees=False,
        hn_terminal=True,
        ):
    """
    Calculate HN-CaHA torsion angles from a PDB/mmCIF file path.

    Needs atom labels: H or H1, N, CA, HA or HA2 (Glycine).

    Parameters
    ----------
    decimals : int
        The decimal number to round the result.

    degrees : bool
        Whether or not to return values as degrees. If `False` returns
        radians.

    hn_terminal : bool
        If the N-terminal has no hydrogens, flag `hn_terminal` should be
    provided as `False`, and the first residue will be discarded.
    If `True` expects N-terminal to have `H` or `H1`.

    Returns
    -------
    np.ndarray
        The NH-CaHA torsion angles for the whole protein.
        Array has the same length of the protein if N-terminal has H,
        otherwise has length of protein minus 1.

    Notes
    -----
    Not optimized for speed. Not slow either.
    """
    # reads the structure file
    structure = Structure(fdata)
    structure.build()
    data = structure.data_array

    # to adjust data to calc_torsion_angles(), we consider the CD of Prolines
    # later we will DELETE those entries
    protons_and_proline = np.logical_or(
        np.isin(data[:, col_name], ('H', 'H1')),
        np.logical_and(data[:, col_resName] == 'PRO', data[:, col_name] == 'CD')
        )

    print(hn_terminal)
    hn_idx = 0 if hn_terminal else 1

    # some PDBs may not be sorted, this part sorts atoms properly before
    # performing calculation
    hs = data[protons_and_proline, :]

    n = data[data[:, col_name] == 'N', :][hn_idx:, :]
    ca = data[data[:, col_name] == 'CA', :][hn_idx:, :]
    ha = data[np.isin(data[:, col_name], ('HA', 'HA2')), :][hn_idx:, :]

    # expects N-terminal to have `H` or `H1`
    assert hs.shape == n.shape == ca.shape == ha.shape, (
        'We expected shapes to be equal. '
        'A possible reason is that the presence/absence of protons in '
        'the N-terminal does not match the flag `hn_terminal`. '
        f'Shapes found are as follow: {hs.shape, n.shape, ca.shape, ha.shape}'
        )

    n_data = np.hstack([hs, n, ca, ha]).reshape(hs.shape[0] * 4, hs.shape[1])

    coords = (n_data[:, cols_coords].astype(np.float64) * 1000).astype(int)

    # notice that calc_torsion_angles() is designed to accepted sequential
    # atoms for which torsions can be calculated. In this particular case
    # because of the nature of `n_data`, the only torsion angles that will have
    # physical meaning are the ones referrent to HN-CaHA, which are at indexes
    # 0::4
    torsions = calc_torsion_angles(coords)[0::4]

    # not the fastest approach
    # increase performance when needed
    tfasta = structure.fasta

    # assumes there is a single chain
    fasta = list(tfasta.values())[0][hn_idx:]
    pro_idx = [m.start() for m in re.finditer('P', fasta)]

    # assigns nan to proline positions
    torsions[pro_idx] = np.nan

    if degrees:
        torsions = np.degrees(torsions)

    return np.round(torsions, decimals)


def get_torsions(fdata, degrees=False, decimals=3):
    """
    Calculate torsion angles from structure.

    Parameters
    ----------
    fdata : str, bytes or Path
        A path to the structure file, or the string representing
        the file. Actually, Any type acceptable by :class:`libstructure.Structure`.

    degrees : bool, optional
        Whether to return torsion angles in degrees or radians.

    decimals : int, optional
        The number of decimals to return.
        Defaults to 3.

    Returns
    -------
    dict
        key: `fname`
        value: -> dict, `phi`, `phi`, `omega` -> list of floats
    """
    structure = Structure(fdata)
    structure.build()
    structure.add_filter_backbone(minimal=True)

    data = structure.filtered_atoms

    # validates structure data
    # rare are the PDBs that produce errors, still they exist.
    # errors can be from a panoply of sources, that is why I decided
    # not to attempt correcting them and instead ignore and report.
    validation_error = validate_backbone_labels_for_torsion(data[:, col_name])
    if validation_error:
        errmsg = (
            'Found errors on backbone label consistency: '
            f'{validation_error}\n'
            )
        err = IDPConfGenException(errmsg)
        # if running through cli_torsions, `err` will be catched and reported
        # by logger.report_on_crash
        raise err

    #coords = (data[:, cols_coords].astype(np.float64) * 1000).astype(int)
    coords = structure.coords
    #print(coords)
    torsions = calc_torsion_angles(coords)

    if degrees:
        torsions = np.degrees(torsions)

    return np.round(torsions, decimals)


def cli_helper_calc_torsions(fname, fdata, **kwargs):
    """Help `cli_torsion` to operate."""
    torsions = get_torsions(fdata, **kwargs)
    CA_C, C_N, N_CA = get_separate_torsions(torsions)
    return fname, {'phi': N_CA, 'psi': CA_C, 'omega': C_N}


def cli_helper_calc_torsionsJ(fdata_tuple, **kwargs):
    """Help cli_torsionsJ.py."""
    return fdata_tuple[0], get_torsionsJ(fdata_tuple[1], **kwargs)


def read_trimer_torsion_planar_angles(pdb, bond_geometry):
    """
    Create a trimer/torsion library of bend/planar angles.

    Given a PDB file:

    1) reads each of its trimers, and for the middle residue:
    2) Calculates phi/psi and rounds them to the closest 10 degree bin
    3) assigns the planar angles found for that residue to the
        trimer/torsion key.
    4) the planar angles are converted to the format needed by cli_build,
        which is that of (pi - angle) / 2.
    5) updates that information in `bond_gemetry`.

    Created key:values have the following form in `bond_geometry` dict:

    {
        'AAA:10,-30': {
            'Cm1_N_Ca': [],
            'N_Ca_C': [],
            'Ca_C_Np1': [],
            'Ca_C_O': [],
            }
        }


    Parameters
    ----------
    pdb : any input of `libstructure.Structure`
        The PDB/mmCIF file data.

    bond_geometry : dict
        The library dictionary to update.

    Returns
    -------
    None
    """
    ALL = np.all
    CEQ = np.char.equal
    TORSION_LABELS = np.array(['CA', 'C', 'N', 'CA', 'C', 'N', 'CA'])
    CO_LABELS = np.array(['CA', 'C', 'O', 'CA', 'C', 'O', 'CA'])
    aa3to1['MSE'] = 'M'  # seleno methionine

    s = Structure(pdb)
    s.build()
    s.add_filter_backbone(minimal=True)

    if s.data_array[0, col_name] != 'N':
        raise PDBFormatError(
            'PDB does not start with N. '
            f'{s.data_array[0, col_name]} instead.'
            )

    bb_minimal_names = s.filtered_atoms[:, col_name]
    bb_residue_names = s.filtered_atoms[:, col_resName]

    N_CA_C_coords = s.coords
    s.clear_filters()

    s.add_filter(lambda x: x[col_name] in ('CA', 'C', 'O'))
    CA_C_O_coords = s.coords
    co_minimal_names = s.filtered_atoms[:, col_name]

    # calc torsion angles
    for i in range(1, len(N_CA_C_coords) - 7, 3):

        idx = list(range(i, i + 7))

        _trimer = bb_residue_names[idx][0::3]

        try:
            trimer = ''.join(aa3to1[_t] for _t in _trimer)
        except KeyError:
            log.info(S(
                'trimer '
                f"{','.join(_trimer)}"
                ' not found. Skipping...'
                ))
            continue

        assert len(trimer) == 3
        del _trimer

        if not ALL(CEQ(bb_minimal_names[idx], TORSION_LABELS)):
            log.info(S(
                'Found non-matching labels: '
                f'{",".join(bb_minimal_names[idx])}'
                ))
            continue

        # selects omega, phi, and psi for the centra residue
        rad_tor = np.round(calc_torsion_angles(N_CA_C_coords[idx])[1:3], 10)
        ptorsions = [rrd10_njit(_) for _ in rad_tor]

        assert len(ptorsions) == 2
        for angle in ptorsions:
            assert -180 <= angle <= 180, 'Bin angle out of expected range.'

        # TODO: better key
        tuple_key = trimer + ':' + ','.join(str(_) for _ in ptorsions)

        # calc bend angles
        c = N_CA_C_coords[idx]
        Cm1_N = c[1] - c[2]
        Ca_N = c[3] - c[2]
        N_Ca = c[2] - c[3]
        C_Ca = c[4] - c[3]
        Ca_C = c[3] - c[4]
        Np1_C = c[5] - c[4]
        assert Cm1_N.shape == (3,)

        # the angles here are already corrected to the format needed by the
        # builder, which is (pi - a) / 2
        Cm1_N_Ca = (np.pi - calc_angle_njit(Cm1_N, Ca_N)) / 2
        N_Ca_C = (np.pi - calc_angle_njit(N_Ca, C_Ca)) / 2
        Ca_C_Np1 = (np.pi - calc_angle_njit(Ca_C, Np1_C)) / 2

        _ = bond_geometry[tuple_key].setdefault('Cm1_N_Ca', [])
        _.append(Cm1_N_Ca)

        _ = bond_geometry[tuple_key].setdefault('N_Ca_C', [])
        _.append(N_Ca_C)

        _ = bond_geometry[tuple_key].setdefault('Ca_C_Np1', [])
        _.append(Ca_C_Np1)

        co_idx = np.array(idx) - 1

        if not ALL(CEQ(co_minimal_names[co_idx], CO_LABELS)):
            log.info(S(
                'Found not matching labels '
                f'{",".join(co_minimal_names[co_idx])}'
                ))
            continue

        c = CA_C_O_coords[co_idx]
        Ca_C = c[3] - c[4]
        O_C = c[5] - c[4]

        Ca_C_O = calc_angle_njit(Ca_C, O_C)
        _ = bond_geometry[tuple_key].setdefault('Ca_C_O', [])
        _.append(Ca_C_O / 2)

    return
