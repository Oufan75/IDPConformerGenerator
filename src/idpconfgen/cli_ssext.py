"""
Extract secondary structure protein segments from PDBs.

Requires a *.dssp file as generated by `idpconfgen ssext` CLI.

USAGE:

idpcfongen segext PDBS_FOLDER DSSPFILE -d OUTPUTFOLDER -s [L/H/E/A]
"""
import argparse
from functools import partial

from idpconfgen import log
from idpconfgen.core.exceptions import IDPConfGenException
from idpconfgen.libs import libcli
from idpconfgen.libs.libhigherlevel import extract_secondary_structure
from idpconfgen.libs.libio import (
    FileReaderIterator,
    read_dictionary_from_disk,
    save_pairs_to_disk,
    )
from idpconfgen.libs.libmulticore import (
    consume_iterable_in_list,
    flat_results_from_chunk,
    pool_function_in_chunks,
    )
from idpconfgen.logger import S, T, init_files, report_on_crash


LOGFILESNAME = '.idpconfgen_ssext'

_name = 'ssext'
_help = 'Extract secondary structure elements from PDBs.'
_prog, _des, _us = libcli.parse_doc_params(__doc__)

ap = libcli.CustomParser(
    prog=_prog,
    description=libcli.detailed.format(_des),
    usage=_us,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    )

libcli.add_argument_pdb_files(ap)

ap.add_argument(
    'sscalc_file',
    help='The DSSP file as saved by IDPConfGen SSCALC CLI',
    )

libcli.add_argument_destination_folder(ap)

ap.add_argument(
    '-s',
    '--structure',
    help=(
        'The secondary structure type to extract. '
        'Defaults to \'all\'. '
        'A subfolder is created for each secondary structure type'
        ),
    default='all',
    action=libcli.AllParam,
    nargs='+',
    )

ap.add_argument(
    '-a',
    '--atoms',
    help=(
        'List of atom names to save in the selection.\n'
        'Defaults to `N`, `CA`, and `C`.'
        ),
    default='all',  # ('N', 'CA', 'C'),
    nargs='+',
    action=libcli.AllParam,
    )
libcli.add_argument_minimum(ap)
libcli.add_argument_ncores(ap)


def main(
        pdb_files,
        sscalc_file,
        atoms='all',
        chunks=5000,
        destination=None,
        func=None,
        minimum=4,
        ncores=1,
        structure='all',
        ):
    """
    Extract secondary structure segments from PDBs.

    Parameters
    ----------
    pdb_files : str or list
        Paths to PDB files.

    sscalc_file : str or Path
        Path to the `sscalc` file containing PDBID codes and `dssp`
        information.

    atoms : str or list
        Atoms to keep in the extracted segments.

    minimum : int
        The minimum residue length of the segments.
        Segments with less than minimum are discarded.

    destination : str or Path
        Where to save the new PDBs.

    structure : str or list
        Secondary structure characters to consider.

    chunks : int
        The size of the chunk to process in memory before saving to
        the disk.

    ncores : int
        The number of cores to use.
        Defaults to 1.
    """
    log.info(T('Extracting secondary structure elements.'))
    init_files(log, LOGFILESNAME)

    ssdata = read_dictionary_from_disk(sscalc_file)
    log.info(S('read sscalc file'))
    pdbs2operate = FileReaderIterator(pdb_files, ext='.pdb')
    log.info(S('read PDB files'))

    # function to be consumed by multiprocessing
    # this function receives each item of the iterable
    consume_func = partial(
        consume_iterable_in_list,
        extract_secondary_structure,
        atoms=atoms,
        minimum=minimum,
        ssdata=ssdata,
        structure=structure,
        )

    # logger context, special for multiprocessing
    # replacement for a decorator
    execute = partial(
        report_on_crash,
        consume_func,
        ROC_exception=IDPConfGenException,
        ROC_prefix=_name,
        )

    # multiprocessing execution
    # uses partial because `flat_results_from_chunk` expects a callable
    execute_pool = partial(
        pool_function_in_chunks,
        execute,
        pdbs2operate,
        ncores=ncores,
        chunks=chunks,
        )

    # flats the yields from results
    flat_results_from_chunk(
        execute_pool,
        save_pairs_to_disk,
        destination=destination,
        )

    return


if __name__ == '__main__':
    libcli.maincli()
