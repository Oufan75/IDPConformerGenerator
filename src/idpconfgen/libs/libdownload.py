"""Functions and variables to download files and data."""
import time, sys
import urllib.request
from urllib.error import URLError
from functools import partial

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


def download_structure(pdbid, mmcif=False, **kwargs):
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

    downloaded_data = fetch_pdb_id_from_RCSB(pdbname, mmcif=mmcif)

    yield from save_structure_by_chains(
        downloaded_data,
        pdbname,
        chains=chains,
        **kwargs,
        )


def fetch_pdb_id_from_RCSB(pdbid, mmcif=False):
    """Fetch PDBID from RCSB."""
    # assumes the file to-download will have the intended format
    priority=True
    if mmcif:
        POSSIBLELINKS.reverse()

    possible_links = (link.format(pdbid) for link in POSSIBLELINKS)

    attempts = 0
    while attempts < 10:
        try:
            for weblink in possible_links:
                try:
                    response = urllib.request.urlopen(weblink)
                    # if the 2nd link goes through, then the format is not the one prioritized
                    if weblink == possible_links[1]:
                        priority=False
                    return response.read(), priority
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

def fetch_raw_structure(pdbid, ext, **kwargs):
    """Download raw structure from RCSB without any filtering."""
    pdbname = pdbid[0]
    mmcif=False
    if ext == 'cif': mmcif=True
    downloaded_data, priority = fetch_pdb_id_from_RCSB(pdbname, mmcif)
    
    if priority==True: # download file as intended
        yield f'{pdbname}.{ext}', downloaded_data.decode('utf-8') 
    else:
        if mmcif==False: # we want pdb but looks like there's only mmcif
            yield f'{pdbname}.cif', downloaded_data.decode('utf-8')
        else: # we want mmcif but looks like there's only pdb
            yield f'{pdbname}.pdb', downloaded_data.decode('utf-8')


fetch_raw_PDBs = partial(fetch_raw_structure, ext='pdb')
fetch_raw_CIFs = partial(fetch_raw_structure, ext='cif')
