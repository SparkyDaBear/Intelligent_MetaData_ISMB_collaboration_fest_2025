import sys,os
import numpy as np
import requests

############################################################
def has_sdrf_file(project_id: str) -> bool:
    """
    Return True if the given PRIDE project (e.g. "PXD012345") includes at least one SDRF file.

    This uses the PRIDE Archive v3 ‘files’ endpoint to list all files for the project,
    then checks for filenames ending in common SDRF extensions.
    
    References:
      - PRIDE Archive REST API v3 (projects & files resources) :contentReference[oaicite:0]{index=0}
    """
    url = f"https://www.ebi.ac.uk/pride/ws/archive/v3/files/sdrf/{project_id}"
    headers = {"accept": "application/json"}

    try:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to query PRIDE API: {e}")

    sdrf_files = resp.json()
    print(f'SDRF check response: {sdrf_files}')
    # API returns a JSON array of SDRF file metadata; non-empty means at least one SDRF present.
    return bool(sdrf_files)
############################################################

# get all files in ../../NLP_metadata_extraction/NLP_Trainingset_annotation/output/initial_GPTtraining_Top1000DownloadedPXD/clean_text/
files = os.listdir('../../NLP_metadata_extraction/NLP_Trainingset_annotation/output/initial_GPTtraining_Top1000DownloadedPXD/clean_text/')
print(len(files))

# get list of BiG Bio PXDs
BigBio_PXDs = np.loadtxt('../../NLP_metadata_extraction/NLP_Trainingset_annotation/data/BigBio_PXDs.txt', dtype=str)
print(BigBio_PXDs, BigBio_PXDs.shape)

# get the GPT files names ../../NLP_metadata_extraction/NLP_Trainingset_annotation/data/GPT/
GPT_files = os.listdir('../../NLP_metadata_extraction/NLP_Trainingset_annotation/data/GPT/')
print(len(GPT_files))
pmids = [f.split('.')[0] for f in GPT_files if f.endswith('.txt')]
print(pmids[:10])

valid_files = []
for f in files:
    if f.endswith('.txt'):
        pmid = f.split('_')[2].replace('.txt', '').replace('PMID', '')
        pxd = f.split('_')[0]
        print('\n', f, pmid, pxd)

        sdrf_file = has_sdrf_file(pxd)
        print(f'SDRF file present for {pxd}: {sdrf_file}')

        if pmid in pmids or pxd in BigBio_PXDs or sdrf_file:
            print('INVALID:', f)
        else:
            valid_files.append(f)

    if len(valid_files) == 100:
        break

print('Valid files:', len(valid_files))
print(valid_files[:10])

## copy the valid files to data/CleanText/Unseen/
import shutil
for f in valid_files[:100]:
    src = os.path.join('/storage/group/epo2/default/ims86/git_repos/Intelligent-metadata-compilation/NLP_metadata_extraction/NLP_Trainingset_annotation/output/initial_GPTtraining_Top1000DownloadedPXD/clean_text/', f)
    dst = os.path.join('/storage/group/epo2/default/ims86/git_repos/Intelligent-metadata-compilation/Hackathons_and_challenges/ISMB_collaboration_fest_2025/data/CleanText/Unseen/', f)
    shutil.copy(src, dst)
    print(f'Copied {f} to Unseen directory.')