"""Functions and variables to download files and data."""
import time, sys
import urllib.request
from urllib.error import URLError

from idpconfgen import log
from idpconfgen.core.exceptions import DownloadFailedError
from idpconfgen.libs.libstructure import save_structure_by_chains
from idpconfgen.logger import S


PDB_WEB_LINK = "https://files.rcsb.org/download/{}.pdb"
CIF_WEB_LINK = "https://files.rcsb.org/download/{}.cif"
POSSIBLELINKS = [
    PDB_WEB_LINK,
    CIF_WEB_LINK,
    ]


def download_structure(pdbid, **kwargs):
    """
    Download a PDB/CIF structure chains.

    Parameters
    ----------
    pdbid : tuple of 2-elements
        0-indexed, the structure ID at RCSB.org;
        1-indexed, a list of the chains to download.

    **kwargs : as for :func:`save_structure_by_chains`.
    """
    pdbname = pdbid[0]
    chains = pdbid[1]

    downloaded_data = fetch_pdb_id_from_RCSB(pdbname)

    yield from save_structure_by_chains(
        downloaded_data,
        pdbname,
        chains=chains,
        **kwargs,
        )


def fetch_pdb_id_from_RCSB(pdbid, mmcif=False):
    """Fetch PDBID from RCSB."""
    if not mmcif:
        possible_links = (link.format(pdbid) for link in POSSIBLELINKS)
    else:
        POSSIBLELINKS.reverse()
        possible_links = (link.format(pdbid) for link in POSSIBLELINKS)

    attempts = 0
    while attempts < 10:
        try:
            for weblink in possible_links:
                try:
                    response = urllib.request.urlopen(weblink)
                    return response.read()
                except urllib.error.HTTPError:
                    continue
                except (AttributeError, UnboundLocalError):  # response is None
                    log.debug(S(f'Download {weblink} failed.'))
                    continue
            else:
                break
        except (TimeoutError, URLError) as err:
            log.error(
                f'failed download for {pdbid} because of {repr(err)}. '
                'Retrying...'
                )
            time.sleep(15)
            attempts += 1
    else:
        raise DownloadFailedError(f'Failed to download {pdbid}')


def fetch_raw_PDBs(pdbid, **kwargs):
    """Download raw PDBs without any filtering."""
    pdbname = pdbid[0]
    downloaded_data = fetch_pdb_id_from_RCSB(pdbname)
    yield f'{pdbname}.pdb', downloaded_data.decode('utf-8')

def fetch_raw_CIFs(pdbid, **kwargs):
    """Download raw mmCIFs without any filtering."""
    pdbname = pdbid[0]
    downloaded_data = fetch_pdb_id_from_RCSB(pdbname, True)
    yield f'{pdbname}.cif', downloaded_data.decode('utf-8')