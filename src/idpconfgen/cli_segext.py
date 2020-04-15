"""
Extract secondary structure protein segments from PDBs.

Requires a *.dssp file as generated by `idpconfgen ssext` CLI.

USAGE:

idpcfongen segext PDBS_FOLDER DSSPFILE -d OUTPUTFOLDER -s [L/H/E/A]
"""
import argparse
from functools import partial

from idpconfgen import Path, log
from idpconfgen.libs import libcli, libio, libstructure, libparse, libpdb
from idpconfgen.logger import S, T, init_files


LOGFILESNAMES = '.idpconfgen_segext'

_prog, _des, _us = libcli.parse_doc_params(__doc__)

ap = libcli.CustomParser(
    prog=_prog,
    description=libcli.detailed.format(_des),
    usage=_us,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    )

libcli.add_parser_pdbs(ap)

ap.add_argument(
    'dssp',
    help='The DSSP file as saved by IDPConfGen SSEXT CLI',
    )

libcli.add_parser_destination_folder(ap)

ap.add_argument(
    '-s',
    '--structure',
    help=(
        'The secondary structure type to extract. '
        'Defaults to \'all\'. '
        'A subfolder is created for each secondary structure type'
        ),
    default='all',
    )


def filter_dssp_segments(seg, required='all'):
    if required == 'all':
        return True
    else:
        return seg == required


def _load_args():
    cmd = ap.parse_args()
    return cmd


def maincli():
    """
    Execute main client function.

    Reads command line arguments and executes logic.
    """
    cmd = _load_args()
    main(**vars(cmd))


def main(
        pdbs,
        dssp,
        destination=None,
        structure='all',
        func=None,
        ):

    dest = Path(destination)
    dest.mkdir(exist_ok=True, parents=True)

    assert structure == 'L'
    log.info(T('spliting PDBs into secondary structure components'))

    pdb_list = libio.read_path_bundle(pdbs)
    log.info(S('read pdb bundle'))

    codes, dssp_data = libparse.read_pipe_file(Path(dssp).read_text())
    log.info(S('read dssp'))

    for i, pdbid in enumerate(pdb_list):
        assert pdbid.stem == codes[i], \
            'PDBID {pdbid.stem} and CODE {codes[i]} do not match'

        log.info(S(f'working with {pdbid}'))

        dssp_segments = libparse.group_by(dssp_data[i])

        pdbdata = libstructure.Structure(pdbid.read_text())
        pdbdata.build()
        pdbdata.add_filter(lambda x: x[libpdb.PDBParams.acol.name] in ('N', 'CA', 'O', 'C'))

        counter = 1

        user_required_dssp_segments = filter(
            lambda x: x[0] == structure,
            dssp_segments,
            )

        for segtype, segslice in user_required_dssp_segments:

            res_slice = slice(
                segslice.start*4,
                segslice.stop*4,
                None,
                )
            to_write = list(pdbdata.filtered_atoms)[res_slice]

            if len(to_write) >= 4 * 3:
                libstructure.write_PDB(
                    libstructure.structure_to_pdb(to_write),
                    Path(dest, f'{pdbid.stem}_{segtype}_{counter}.pdb'),
                    )
                counter += 1




    return


if __name__ == '__main__':
    maincli()
