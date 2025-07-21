import sys,os
import numpy as np
import requests
import glob

############################################################

CLEAN_TEXT_DIR = 'data/CleanText/Training/'
GoldStandard_SDRFs_DIR = 'data/GoldStandard_SDRFs/'

# Get a list of all clean text files
clean_text_files = glob.glob(os.path.join(CLEAN_TEXT_DIR, '*.txt'))
print(f'Found {len(clean_text_files)} clean text files in {CLEAN_TEXT_DIR}')

# Get a list of all Gold Standard SDRF files
gold_standard_sdrf_files = glob.glob(os.path.join(GoldStandard_SDRFs_DIR, '*_cleaned.sdrf.tsv'))
print(f'Found {len(gold_standard_sdrf_files)} Gold Standard SDRF files in {GoldStandard_SDRFs_DIR}')
# Remove those with a - in the name (these are two part PXDs and I do not want to merge them before the competition)
gold_standard_sdrf_files = [f for f in gold_standard_sdrf_files if '-' not in os.path.basename(f)]
print(f'After removing two-part PXDs, {len(gold_standard_sdrf_files)} Gold Standard SDRF files remain.')
PXDs = [os.path.basename(f).split('_')[0] for f in gold_standard_sdrf_files]
print(f'Extracted {len(PXDs)} unique PRIDE project IDs (PXDs) from Gold Standard SDRF files.')

## QC to check that for every BiG Bio PXD there is at least 1 clean text file
## There maybe more than 1 if there is more publications for this associated PXD
matched_clean_text_files = []
for PXD in PXDs:
    # make sure there is a clean text file present as well
    text_file = [f for f in clean_text_files if PXD in os.path.basename(f)]
    if len(text_file) == 0:
        raise ValueError(f'No clean text file found for PXD {PXD}. Skipping...')
    
    else:
        print(PXD, text_file)
        matched_clean_text_files += text_file

## QC to check that for any stray clean text file there is a Gold Standard SDRF file
# remove any clean_text_file that do not have a matching Gold Standard SDRF file
unmatched_clean_text_files = [f for f in clean_text_files if f not in matched_clean_text_files]
print(f'Found {len(unmatched_clean_text_files)} unmatched clean text files.')

if len(unmatched_clean_text_files) > 0:
    for f in unmatched_clean_text_files:
        print(f)
        os.system(f'git rm -r {f}')
