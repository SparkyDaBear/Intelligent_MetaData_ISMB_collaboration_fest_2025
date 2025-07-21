import sys,os
import pandas as pd
import numpy as np
import glob
from pathlib import Path
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import argparse
from Levenshtein import distance as lev
from rapidfuzz import fuzz

# Default truncation threshold is 1000 elements; lower it to trigger summarization:
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

#######################################################################################
### USER DEFINETED FUNCTIONS ##########################################################
#######################################################################################

#######################################################################################
def load_sdrf(SDRF_files:str, condensed_outfile:str, mapping_outfile:str) -> dict:
    """
    Loads the SDRF data
    """
    print(f'\n{"#"*50} Loading SDRF data {"#"*50}')

    # Load list of valid annotation terms
    ValidAnnTypes = np.loadtxt(ANNTYPES_PATH, dtype=str)
    print(f'Loaded {len(ValidAnnTypes)} records from ValidAnnotationTypes.csv')
    # Filter out the annotation types that are in the ignore list
    ValidAnnTypes = [ann_type for ann_type in ValidAnnTypes if ann_type not in ignore_ann_types]
    print(f'Filtered out {len(ValidAnnTypes)} annotation types from the ignore list')
    print(f'Valid Annotation Types: {ValidAnnTypes}')

    # load the dataframe which contains mapping of PXD to PaperID (PMID)
    AnnDatasetMeta = pd.read_csv(PRIDE_CSV_PATH, sep='|')
    # print(AnnDatasetMeta.head(10))
    # print(f'Loaded {len(AnnDatasetMeta)} records from Annotated_dataset_metadata.csv')

    # Load the SDRF data into a single dictionary 
    cond_sdrf_dict = {}
    sdrf_mapping = {'PXD':[], 'PMID':[], 'PMCID':[]}  # Dictionary to map PMID to PXD_ID
    for sdrf_file in SDRF_files:
        print(f'Loading SDRF file: {sdrf_file}')

        # Read the SDRF file into a dataframe
        sdrf_df = pd.read_csv(sdrf_file, sep='\t')
        print(f'sdrf_df:\n{sdrf_df.head(10)}')

        # get the columns that contain the AnnType
        df_columns = sdrf_df.columns
        # df_columns = [col.lower() for col in df_columns]  # Convert column names to lowercase
        # df_columns = [col.strip() for col in df_columns]
        print(f'Columns in SDRF file: {df_columns}')
        
        # get the PXD ID from the filename
        PXD_ID = os.path.basename(sdrf_file).split('.')[0].split('_')[0]
        if '-' in PXD_ID:  # handle cases where PXD_ID is split by '-'
            PXD_ID = PXD_ID.split('-')[0]
        print(f'PXD_ID: {PXD_ID}')
        
        # get the PaperID from the AnnDatasetMeta dataframe
        PXD_df = AnnDatasetMeta[AnnDatasetMeta['accession'] == PXD_ID]
        print(f'PXD_df:\n{PXD_df}')
        if PXD_df.empty:
            print(f'No records found for {PXD_ID} in AnnDatasetMeta.csv')
            continue

        PMID = PXD_df['pubmedID'].values[0]
        PMC = PXD_df['PMCID'].values[0]
        # print(f'PaperID: {PMID} | PMCID: {PMC}')
        if PMID not in cond_sdrf_dict: # populate the condensed SDRF dictionary with the paperID as the top level if it does not exist
            cond_sdrf_dict[PMID] = {}
        
        # add them to the mapping dictionary
        sdrf_mapping['PXD'].append(PXD_ID)
        sdrf_mapping['PMID'].append(PMID)
        sdrf_mapping['PMCID'].append(PMC)

        # Get the condensed SDRF data for this PaperID
        for AnnType in ValidAnnTypes:
            print(f'\nAnnType: {AnnType}')

            # check if the AnnType is in the SDRF dataframe
            if AnnType not in cond_sdrf_dict[PMID]:
                cond_sdrf_dict[PMID][AnnType] = []

            # Check if the AnnType is in the columns of the dataframe
            # Allow for identification of AnnType.1 or AnnType.2 etc.
            # Also allow for AnnType to be in square brackets, e.g. [AnnType]
            # This is to handle cases where the AnnType is in the form of [AnnType] or AnnType
            # e.g. [BiologicalReplicate] or BiologicalReplicate
            # This is to handle cases where the AnnType is in the form of [AnnType].1 or AnnType.1
            # e.g. [BiologicalReplicate].1 or BiologicalReplicate.1
            AnnType_cols = []
            for col in df_columns:
                if f'[{AnnType.lower()}]' in col.lower() or f'{AnnType.lower()}' == col.lower():
                    AnnType_cols.append(col)
                elif f'{AnnType.lower()}.' in col.lower():
                    AnnType_cols.append(col)
                elif f'{AnnType.lower()} ' in col.lower():
                    AnnType_cols.append(col)
                elif f'{AnnType.lower()}_' in col.lower():
                    AnnType_cols.append(col)
                elif f'[{AnnType.lower()}].' in col.lower():
                    AnnType_cols.append(col)
            print(f'AnnType_cols: {AnnType_cols}')
            

            # get the rows that contain the AnnType
            if len(AnnType_cols) == 0:
                # print(f'No columns found for {AnnType}')
                continue

            else:
                AnnType_rows = sdrf_df[AnnType_cols]
                AnnType_rows = AnnType_rows.astype(str)  # Convert all values to string type
                print(f'AnnType_rows:\n{AnnType_rows} {len(AnnType_rows)} rows')
                unique_rows = AnnType_rows.drop_duplicates().values
                #unique_rows = [str(row) for row in unique_rows]  # Convert each row to a string
                unique_rows = np.hstack(unique_rows)
                print(f'Unique AnnType_rows:\n{unique_rows} {len(unique_rows)} {unique_rows.shape} rows')

                # check if any string in the unique_rows as "NT=" in it
                cleaned_rows = []
                for row in unique_rows:
                    if row in mask_strings:  # Ignore the strings that are in the mask_strings list
                        continue

                    if 'NT=' in row:
                        # print(f'Found NT= in row: {row}')
                        row = row.split(';')  # Split the row by ';'
                        row = [r for r in row if 'NT=' in r]  # Keep only strings that contain 'NT='
                        row = row[0].replace('NT=', '')  # Remove 'NT=' from the first string
                        # print(f'Cleaned row: {row}')
                        cleaned_rows.append(row)
                    else:
                        cleaned_rows.append(row.strip())

                print(f'Cleaned rows: {cleaned_rows} {len(cleaned_rows)} rows')
                cond_sdrf_dict[PMID][AnnType] += cleaned_rows

    # Log data so we can quality check to see if the SDRF data is correct
    for paper in cond_sdrf_dict.keys():
        for AnnType in cond_sdrf_dict[paper].keys():
            #cond_sdrf_dict[paper][AnnType] = list(set(cond_sdrf_dict[paper][AnnType]))
            print(f'PaperID: {paper} | AnnType: {AnnType} | {cond_sdrf_dict[paper][AnnType]}')

    # save the condensed SDRF data to a pkl file
    # condensed_outfile = os.path.join(OUTPATH, 'condensed_sdrf_data.pkl')
    with open(condensed_outfile, 'wb') as f:
        np.save(f, cond_sdrf_dict, allow_pickle=True)
    print(f'Wrote {len(cond_sdrf_dict)} records to {condensed_outfile}')

    # save the mapping dictionary to a csv file
    # mapping_outfile = os.path.join(OUTPATH, 'sdrf_mapping.csv')
    mapping_df = pd.DataFrame(sdrf_mapping)
    print(f'mapping_df:\n{mapping_df}')
    mapping_df.to_csv(mapping_outfile, index=False)
    print(f'Wrote {len(sdrf_mapping)} records to {mapping_outfile}')

    return cond_sdrf_dict, sdrf_mapping
#######################################################################################

#######################################################################################
def load_annotations(ann_files, ann_outfile, mapping_outfile):
    """
    Loads all the annotations into a single dataframe for easy analysis
    """
    print(f'\n{"#"*50} Loading Annotations {"#"*50}')

    # Load list of valid annotation terms
    ValidAnnTypes = np.loadtxt(ANNTYPES_PATH, dtype=str)
    print(f'Loaded {len(ValidAnnTypes)} records from ValidAnnotationTypes.csv')
    # Filter out the annotation types that are in the ignore list
    ValidAnnTypes = [ann_type for ann_type in ValidAnnTypes if ann_type not in ignore_ann_types]
    print(f'Filtered out {len(ValidAnnTypes)} annotation types from the ignore list')
    print(f'Valid Annotation Types: {ValidAnnTypes}')

    # load the dataframe which contains mapping of PXD to PaperID (PMID)
    AnnDatasetMeta = pd.read_csv(PRIDE_CSV_PATH, sep='|')
    # print(AnnDatasetMeta.head(10))
    # print(f'Loaded {len(AnnDatasetMeta)} records from Annotated_dataset_metadata.csv')

    # For each ann_file load its contents into a dictionary
    ann_data_dict = {}
    ann_mapping = {'PXD':[], 'PMID':[], 'PMCID':[]}  # Dictionary to map PMID to PXD_ID
    for ann_file in ann_files:
        print(f'Loading annotation file: {ann_file}')

        
        PXD_ID = os.path.basename(ann_file).split('.')[0].split('_')[0]
        if '-' in PXD_ID:  # handle cases where PXD_ID is split by '-'
            PXD_ID = PXD_ID.split('-')[0]

        # get the PaperID from the AnnDatasetMeta dataframe
        PXD_df = AnnDatasetMeta[AnnDatasetMeta['accession'] == PXD_ID]
        # print(f'PXD_df:\n{PXD_df}')
        if PXD_df.empty:
            print(f'No records found for {PXD_ID} in AnnDatasetMeta.csv')
            continue

        PMID = os.path.basename(ann_file).split('.')[0].split('_')[1]
        if '-' in PMID:  # handle cases where PMID is split by '-'
            PMID = PMID.split('-')[0]
        PMID = PMID.replace('PMID', '')  # remove 'PMID' prefix if it exists

        PMID = PXD_df['pubmedID'].values[0]
        PMC = PXD_df['PMCID'].values[0]

        # add them to the mapping dictionary
        ann_mapping['PXD'].append(PXD_ID)
        ann_mapping['PMID'].append(PMID)
        ann_mapping['PMCID'].append(PMC)
        # print(f'PXD_ID: {PXD_ID} | PMID: {PMID}')

        if PMID not in ann_data_dict: # populate the condensed SDRF dictionary with the paperID as the top level if it does not exist
            ann_data_dict[PMID] = {}

        # Load the annotation file into a dictionary
        data = {}
        with open(ann_file, 'r') as f:
            for line in f:
                if ':' in line:  # ensure the line contains a key-value pair
                    # print(f'LINE: {line}')
                    key = line.split(': ')[0].strip()
                    value = line.split(': ')[1].strip()
                    data[key] = value


        # Get the condensed SDRF data for this PaperID
        for AnnType in ValidAnnTypes:
            # print(f'\nAnnType: {AnnType}')

            # check if the AnnType is in the SDRF dataframe
            if AnnType not in ann_data_dict[PMID]:
                ann_data_dict[PMID][AnnType] = [v for k,v in data.items() if AnnType.lower() == k.lower()]

    # Log data so we can quality check to see if the SDRF data is correct
    for paper in ann_data_dict.keys():
        for AnnType in ann_data_dict[paper].keys():
            #ann_data_dict[paper][AnnType] = list(set(ann_data_dict[paper][AnnType]))
            print(f'PaperID: {paper} | AnnType: {AnnType} | {ann_data_dict[paper][AnnType]}')

    # save the condensed annotation data to a pkl file
    with open(ann_outfile, 'wb') as f:
        np.save(f, ann_data_dict, allow_pickle=True)
    print(f'Wrote {len(ann_data_dict)} records to {ann_outfile}')

    # save the mapping dictionary to a csv file
    mapping_df = pd.DataFrame(ann_mapping)
    print(f'mapping_df:\n{mapping_df}')
    mapping_df.to_csv(mapping_outfile, index=False)
    print(f'Wrote {len(ann_mapping)} records to {mapping_outfile}')

    return ann_data_dict, ann_mapping 
#######################################################################################

##################################################################################
def Harmonize_and_Evaluate_datasets( 
    A: Dict[str, Dict[str, List[str]]],
    B: Dict[str, Dict[str, List[str]]],
    Aoutfile: str = 'harmonized_A.pkl',
    Boutfile: str = 'harmonized_B.pkl',
    eval_outfile: str = 'evaluation_metrics.csv',
    model_name: str = 'all-MiniLM-L6-v2',
    threshold: float = 0.80,
    method: str = 'RapidFuzz'  # 'Levenshtein' or 'Embedding' or 'RapidFuzz'
) -> Tuple[
    Dict[str, Dict[str, List[str]]],  # harmonized A
    Dict[str, Dict[str, List[str]]]  # harmonized B
]:
    model = SentenceTransformer(model_name)

    ## Get the common set of top level keys
    print(A.keys())
    print(B.keys())
    common_pubs = set(A.keys()).intersection(set(B.keys()))
    print(f'Found {len(common_pubs)} common publications between A and B datasets.')

    # raise SystemExit("Please run the harmonization process with the correct datasets before evaluating.")
    harmonized_A = {}
    harmonized_B = {}
    eval_metrics = {'publication': [], 'AnnotationType': [], 'precision': [], 'recall': [], 'f1': [], 'jacc': []}
    for pub in common_pubs:
        harmonized_A[pub] = {}
        harmonized_B[pub] = {}
        for category in A[pub].keys():
            vals_A = list(set(A[pub][category]))
            vals_B = list(set(B[pub][category]))
            all_vals = vals_A + [v for v in vals_B if v not in vals_A]
            print(f'\nProcessing {pub} - {category}: {len(all_vals)} unique values')
            print(f'vals_A: {vals_A} | vals_B: {vals_B} | all_vals: {all_vals}')

            # 1a. auto return empty if there is no annotations for set A or B 
            if len(vals_A) == 0 and len(vals_B) == 0:
                print(f'No values found for {pub} - {category}.')
                harmA = []
                harmB = []
                harmonized_A[pub][category] = harmA
                harmonized_B[pub][category] = harmB
                print(f'Harmonized A: {harmA}')
                print(f'Harmonized B: {harmB}')
                eval_metrics['publication'].append(pub)
                eval_metrics['AnnotationType'].append(category)
                eval_metrics['precision'].append(float('NaN'))
                eval_metrics['recall'].append(float('NaN'))
                eval_metrics['f1'].append(float('NaN'))
                eval_metrics['jacc'].append(float('NaN'))
                continue

            # handle case where there is only a single value in the all_vals list
            if len(all_vals) == 1:
                dist_mat = np.array([[0.0]])  # single element, distance to itself is 0
                labels = np.array([0])  # single cluster
                print(f'Only one unique value for {pub} - {category}. Skipping clustering.')

            else:
                
                if method == 'Embedding':
                    print(f'Using embeddings for {pub} - {category} with threshold {threshold}')
                    # if more than 1 value compute embeddings
                    embeddings = model.encode(all_vals, convert_to_numpy=True, normalize_embeddings=True)
                    print(f'embeddings shape: {embeddings.shape}')

                    # 1c. cosine similarity → distance matrix
                    sim_mat = cosine_similarity(embeddings)                # shape (N, N)
                    # print(f'sim_mat shape: {sim_mat.shape}')
                    # print(f'sim_mat:\n{sim_mat}')
                    dist_mat = 1.0 - sim_mat                        # distance in [0, 2]
                    print(f'dist_mat shape: {dist_mat.shape}')
                    print(f'dist_mat:\n{dist_mat}')

                elif method == 'Levenshtein':
                    print(f'Using Levenshtein distance for {pub} - {category} with threshold {threshold}')
                    # if more than 1 value compute Levenshtein distance
                    dist_mat = np.zeros((len(all_vals), len(all_vals)), dtype=float)
                    for i in range(len(all_vals)):
                        for j in range(i + 1, len(all_vals)):
                            dist = lev(all_vals[i], all_vals[j])
                            dist = dist / max(len(all_vals[i]), len(all_vals[j]))  # normalize distance
                            dist_mat[i, j] = dist
                            dist_mat[j, i] = dist
                    print(f'dist_mat shape: {dist_mat.shape}')
                    print(f'dist_mat:\n{dist_mat}')

                elif method == 'RapidFuzz':
                    print(f'Using RapidFuzz distance for {pub} - {category} with threshold {threshold}')
                    # if more than 1 value compute RapidFuzz distance
                    dist_mat = np.zeros((len(all_vals), len(all_vals)), dtype=float)
                    for i in range(len(all_vals)):
                        for j in range(i + 1, len(all_vals)):
                            dist = fuzz.ratio(all_vals[i], all_vals[j]) / 100.0  # normalize to [0, 1]
                            dist_mat[i, j] = 1.0 - dist  # convert to distance
                            dist_mat[j, i] = dist_mat[i, j]
                    print(f'dist_mat shape: {dist_mat.shape}')
                    print(f'dist_mat:\n{dist_mat}')

                else:
                    raise ValueError(f'Unknown method: {method}. Use "Embedding" or "Levenshtein".')

                # 1d. cluster
                clusterer = AgglomerativeClustering(
                    n_clusters=None,
                    metric='precomputed',
                    linkage='average',
                    distance_threshold=1.0 - threshold)
                labels = clusterer.fit_predict(dist_mat)        # array of length N

            # map each string → cluster
            str2cid = {s: int(labels[i]) for i, s in enumerate(all_vals)}
            print(f'str2cid: {str2cid}')

            # 1e. harmonize original lists
            harmA = [str2cid[s] for s in A[pub][category]]
            harmB = [str2cid[s] for s in B[pub][category]]
            print(f'Harmonized A: {harmA}')
            print(f'Harmonized B: {harmB}')

            harmonized_A[pub][category] = harmA
            harmonized_B[pub][category] = harmB

            # Else, proceed with evaluation
            print(f'\nEvaluating - {category}:')
            print(f'y_true: {harmA}')
            print(f'y_pred: {harmB}')

            unique_labels = set(harmA + harmB)
            print(f'Unique labels: {unique_labels}')

            y_true_p = []
            y_pred_p = []
            for label in unique_labels:
                if label in harmA:
                    y_true_p.append(1)
                else:
                    y_true_p.append(0)
                if label in harmB:
                    y_pred_p.append(1)
                else:
                    y_pred_p.append(0)
            print(f'y_true_p: {y_true_p}')
            print(f'y_pred_p: {y_pred_p}')

            precision = precision_score(y_true_p, y_pred_p, average='macro', zero_division=0)
            recall    = recall_score(y_true_p, y_pred_p, average='macro', zero_division=0)
            f1        = f1_score(y_true_p, y_pred_p, average='macro', zero_division=0)
            print(f'Precision: {precision}, Recall: {recall}, F1: {f1}')
        
            # consistency: Jaccard over the sets of cluster IDs
            set_A = set(harmonized_A[pub][category])
            set_B = set(harmonized_B[pub][category])
            if not set_A and not set_B:
                jacc = 1.0
            else:
                # compute |A ∩ B| / |A ∪ B|
                jacc = len(set_A & set_B) / len(set_A | set_B)
            print(f'Jaccard consistency for {pub} - {category}: {jacc}')

            # Store the evaluation metrics
            eval_metrics['publication'].append(pub)
            eval_metrics['AnnotationType'].append(category)
            eval_metrics['precision'].append(precision)
            eval_metrics['recall'].append(recall)
            eval_metrics['f1'].append(f1)
            eval_metrics['jacc'].append(jacc)

    # Convert the evaluation metrics to a DataFrame
    eval_df = pd.DataFrame(eval_metrics)
    print(f'\nEvaluation Metrics:\n{eval_df}')
    eval_df.to_csv(eval_outfile, index=False)
    print(f'Wrote evaluation metrics to {eval_outfile}')

    # save the harmonized datasets
    with open(Aoutfile, 'wb') as f:
        np.save(f, harmonized_A, allow_pickle=True)
    print(f'Wrote {len(harmonized_A)} records to {Aoutfile}')

    with open(Boutfile, 'wb') as f:
        np.save(f, harmonized_B, allow_pickle=True)
    print(f'Wrote {len(harmonized_B)} records to {Boutfile}')
    
    # return the harmonized datasets
    print(f'Harmonization complete. Returning harmonized datasets.')
    return harmonized_A, harmonized_B, eval_df
##################################################################################

##################################################################################
def PlotEvaluationMetrics(eval_df: pd.DataFrame, outpath: str = 'evaluation_metrics_plot.png'):
    """
    Plots the evaluation metrics from the evaluation DataFrame.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import bootstrap

    print(f'\n{"#"*50} Summarizing Evaluation Metrics {"#"*50}')
    plot_df = {'AnnotationType': [], 'precision': [], 'precision_lb': [], 'precision_ub':[], 
                                      'recall': [], 'recall_lb': [], 'recall_ub':[], 
                                      'f1': [], 'f1_lb': [], 'f1_ub':[], 
                                      'jacc': [], 'jacc_lb': [], 'jacc_ub':[], }
    for AnnType, AnnType_df in eval_df.groupby('AnnotationType'):
        print(f'Summarizing metrics for annotation type: {AnnType}')
        print(f'AnnType_df:\n{AnnType_df}')

        # remove rows with NaN values in the metrics
        AnnType_df = AnnType_df.dropna(subset=['precision', 'recall', 'f1', 'jacc'])
        print(f'AnnType_df after dropping NaN values:\n{AnnType_df}')

        # If there are no rows left after dropping NaN values, skip this annotation type
        if AnnType_df.empty:
            print(f'No valid data for {AnnType}. Skipping.')
            continue

        # Append the metrics to the plot DataFrame
        plot_df['AnnotationType'].append(AnnType)

        if len(AnnType_df) == 1:
            # If there is only one row, use the values directly
            plot_df['precision'].append(AnnType_df['precision'].values[0])
            plot_df['precision_lb'].append(AnnType_df['precision'].values[0])
            plot_df['precision_ub'].append(AnnType_df['precision'].values[0])
            plot_df['recall'].append(AnnType_df['recall'].values[0])
            plot_df['recall_lb'].append(AnnType_df['recall'].values[0])
            plot_df['recall_ub'].append(AnnType_df['recall'].values[0])
            plot_df['f1'].append(AnnType_df['f1'].values[0])
            plot_df['f1_lb'].append(AnnType_df['f1'].values[0])
            plot_df['f1_ub'].append(AnnType_df['f1'].values[0])
            plot_df['jacc'].append(AnnType_df['jacc'].values[0])
            plot_df['jacc_lb'].append(AnnType_df['jacc'].values[0])
            plot_df['jacc_ub'].append(AnnType_df['jacc'].values[0])
            print(f'Only one row for {AnnType}. Using the values directly.')
            continue
        
        # Calculate the mean and 95% confidence intervals for each metric
        precision_mean = AnnType_df['precision'].mean()
        precision_ci = bootstrap((AnnType_df['precision'].values,), np.mean, confidence_level=0.95, n_resamples=10000)
        precision_lb = precision_ci.confidence_interval.low
        precision_ub = precision_ci.confidence_interval.high

        recall_mean = AnnType_df['recall'].mean()
        recall_ci = bootstrap((AnnType_df['recall'].values,), np.mean, confidence_level=0.95, n_resamples=10000)
        recall_lb = recall_ci.confidence_interval.low
        recall_ub = recall_ci.confidence_interval.high

        f1_mean = AnnType_df['f1'].mean()
        f1_ci = bootstrap((AnnType_df['f1'].values,), np.mean, confidence_level=0.95, n_resamples=10000)
        f1_lb = f1_ci.confidence_interval.low
        f1_ub = f1_ci.confidence_interval.high

        jacc_mean = AnnType_df['jacc'].mean()
        jacc_ci = bootstrap((AnnType_df['jacc'].values,), np.mean, confidence_level=0.95, n_resamples=10000)
        jacc_lb = jacc_ci.confidence_interval.low
        jacc_ub = jacc_ci.confidence_interval.high

        # Append the metrics to the plot DataFrame
        plot_df['precision'].append(precision_mean)
        plot_df['precision_lb'].append(precision_lb)
        plot_df['precision_ub'].append(precision_ub)
        plot_df['recall'].append(recall_mean)
        plot_df['recall_lb'].append(recall_lb)
        plot_df['recall_ub'].append(recall_ub)
        plot_df['f1'].append(f1_mean)
        plot_df['f1_lb'].append(f1_lb)
        plot_df['f1_ub'].append(f1_ub)
        plot_df['jacc'].append(jacc_mean)
        plot_df['jacc_lb'].append(jacc_lb)
        plot_df['jacc_ub'].append(jacc_ub)

    plot_df = pd.DataFrame(plot_df)
    print(f'Plot DataFrame:\n{plot_df}')

    print(f'\n{"#"*50} Plotting Evaluation Metrics {"#"*50}')
    # assume your DataFrame is called df
    # set AnnotationType as the index
    df = plot_df.set_index('AnnotationType')

    # the metrics to plot
    metrics = ['precision', 'recall', 'f1', 'jacc']

    # build a matrix of annotation strings:
    # each row is [“val (lb, ub)” for each metric]
    annot = df.apply(
        lambda row: [
            f"{row[m]:.3f} ({row[f'{m}_lb']:.3f}, {row[f'{m}_ub']:.3f})"
            for m in metrics
        ],
        axis=1
    ).tolist()

    # convert to a NumPy array for sns.heatmap
    annot = np.array(annot)

    # Plot
    plt.figure(figsize=(12, len(df)*0.25))  # adjust height per row
    sns.set(style="whitegrid")
    ax = sns.heatmap(
        df[metrics].astype(float),
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=False,
        linewidths=0.5,
        linecolor="lightgray"
    )
    ax.set_ylabel("")
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.set_yticklabels(df.index, rotation=0)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    print(f'Saved evaluation metrics plot to {outpath}')
    quit()
    #plt.show()
#######################################################################################

#######################################################################################


##############################################################
### MAIN #####################################################
##############################################################

##############################################################
## STEP 0: Define some user variables
## CHANGE THESE TO YOUR OWN PATHS
ANN_PATH = f'/storage/group/epo2/default/ims86/git_repos/Intelligent-metadata-compilation/Hackathons_and_challenges/ISMB_collaboration_fest_2025/data/BenchmarkAnnotations/o4-mini-2025-04-16/'

SDRF_PATH = f'/storage/group/epo2/default/ims86/git_repos/Intelligent-metadata-compilation/Hackathons_and_challenges/ISMB_collaboration_fest_2025/data/GoldStandard_SDRFs/'

ANNTYPES_PATH = f'/storage/group/epo2/default/ims86/git_repos/Intelligent-metadata-compilation/Hackathons_and_challenges/ISMB_collaboration_fest_2025/data/AnnotatedTypes.txt'
PRIDE_CSV_PATH = f'/storage/group/epo2/default/ims86/git_repos/Intelligent-metadata-compilation/Hackathons_and_challenges/ISMB_collaboration_fest_2025/data/pride_projects_summary_20250630.csv'

OUTPATH = f'/storage/group/epo2/default/ims86/git_repos/Intelligent-metadata-compilation/Hackathons_and_challenges/ISMB_collaboration_fest_2025/data/Performance/BenchmarkAnnotations_test/'

# make output directory if it does not exist
if not os.path.exists(OUTPATH):
    os.makedirs(OUTPATH)


# Declare a list of mask strings to ignore when condensing the SDRF data
mask_strings = ['not applicable', 'not available',  'not applicable', 'notapplicable', 'n/a', 'na', 'n.a.', 'none', 'no data', 'no data available', 'no data available for this sample', 'no data available for this experiment', 'no data available for this study', 'no data available for this project']

# Declare a list of annotation types to ignore when condensing the SDRF data or annotation data
ignore_ann_types = ['BiologicalReplicate', 'TechnicalReplicate']
##############################################################


##############################################################################
### STEP 1: Load the SDRF data ###############################################
# Load the SDRF data
# This will create a dictionary with the following structure:
# sdrf_data[PMID][AnnType] = [<TextSpan>, <TextSpan>, ..., <TextSpan>]
# where AnnType is one of the allowed annotation types in the ANNTYPES_PATH.
# get the list of SDRF files
# Find the SDRF files in the SDRF_PATH
sdrf_files = [os.path.join(SDRF_PATH, f) for f in os.listdir(SDRF_PATH) if f.endswith('_cleaned.sdrf.tsv')]
print(f"Found {len(sdrf_files)} SDRF files in {SDRF_PATH}")

# Check if the condensed SDRF data already exists
condensed_outfile = os.path.join(OUTPATH, 'condensed_sdrf_data.pkl')
condensed_mapping_outfile = os.path.join(OUTPATH, 'sdrf_mapping.csv')
if os.path.exists(condensed_outfile) and os.path.exists(condensed_mapping_outfile):
    print(f'Condensed SDRF data already exists at {condensed_outfile}. Loading existing data.')

    with open(condensed_outfile, 'rb') as f:
        sdrf_data = np.load(f, allow_pickle=True).item()
    print(f'Loaded {len(sdrf_data)} records from {condensed_outfile}')

    sdrf_mapping = pd.read_csv(condensed_mapping_outfile)
    print(f'Loaded {len(sdrf_mapping)} records from {condensed_mapping_outfile}')

else:
    print(f'Condensed SDRF data does not exist. Creating new data at {condensed_outfile}.')
    sdrf_data, sdrf_mapping = load_sdrf(sdrf_files, condensed_outfile, condensed_mapping_outfile)
##############################################################################


##############################################################################
### STEP 2: Load the annotation data #########################################
# Load the annotations into a single dictionary
# This will create a dictionary with the following structure:
# ann_data_dict[PMID][AnnType] = [<TextSpan>, <TextSpan>, ..., <TextSpan>]
# where AnnType is one of the allowed annotation types in the ANNTYPES_PATH.
# The text spans are those present in the publication for that annotation type.
# get the list of annotation files
ann_files = glob.glob(os.path.join(ANN_PATH, '*.ann'))
print(f"Found {len(ann_files)} annotation files in {ANN_PATH}")

# Load the annotations into a single dictionary
ann_outfile = os.path.join(OUTPATH, 'condensed_ann_data.pkl')
ann_mapping_outfile = os.path.join(OUTPATH, 'ann_mapping.csv')
if os.path.exists(ann_outfile) and os.path.exists(ann_mapping_outfile):
    print(f'Condensed annotation data already exists at {ann_outfile}. Loading existing data.')

    with open(ann_outfile, 'rb') as f:
        ann_data = np.load(f, allow_pickle=True).item()
    print(f'Loaded {len(ann_data)} records from {ann_outfile}')

    ann_mapping = pd.read_csv(ann_mapping_outfile)
    print(f'Loaded {len(ann_mapping)} records from {ann_mapping_outfile}')

else:
    print(f'Condensed annotation data does not exist. Creating new data at {ann_outfile}.')
    ann_data, ann_mapping = load_annotations(ann_files, ann_outfile, ann_mapping_outfile)
##############################################################################


##############################################################################
## STEP 3: Harmonize the datasets and evaluate performance ###################
# Harmonize the datasets and evaluate performance
sdrf_Harm_outfile = os.path.join(OUTPATH, 'harmonized_sdrf.pkl')
ann_Harm_outfile = os.path.join(OUTPATH, 'harmonized_ann.pkl')
eval_outfile = os.path.join(OUTPATH, 'evaluation_metrics.csv')

if os.path.exists(sdrf_Harm_outfile) and os.path.exists(ann_Harm_outfile) and os.path.exists(eval_outfile):
    print(f'Harmonized SDRF data already exists at {sdrf_Harm_outfile}. Loading existing data.')
    print(f'Harmonized annotation data already exists at {ann_Harm_outfile}. Loading existing data.')
    print(f'Evaluation metrics already exist at {eval_outfile}. Loading existing data.')

    with open(sdrf_Harm_outfile, 'rb') as f:
        Harmonized_sdrf = np.load(f, allow_pickle=True).item()
    print(f'Loaded {len(Harmonized_sdrf)} records from {sdrf_Harm_outfile}')

    with open(ann_Harm_outfile, 'rb') as f:
        Harmonized_ann = np.load(f, allow_pickle=True).item()
    print(f'Loaded {len(Harmonized_ann)} records from {ann_Harm_outfile}')

    with open(eval_outfile, 'rb') as f:
        eval_metrics = pd.read_csv(f)
    print(f'Loaded {len(eval_metrics)} records from {eval_outfile}')

else:
    print(f'Harmonized SDRF data does not exist. Creating new data at {sdrf_Harm_outfile}.')
    print(f'Harmonized annotation data does not exist. Creating new data at {ann_Harm_outfile}.')
    print(f'Evaluation metrics do not exist. Creating new data at {eval_outfile}.')
    # instead of a big model like 'all-mpnet-base-v2', try:
    # model_name = 'sentence-transformers/paraphrase-MiniLM-L3-v2'
    # or even smaller:
    model_name = 'all-MiniLM-L6-v2'
    Harmonized_sdrf, Harmonized_ann, eval_metrics = Harmonize_and_Evaluate_datasets(A=sdrf_data, B=ann_data, Aoutfile=sdrf_Harm_outfile, Boutfile=ann_Harm_outfile, eval_outfile=eval_outfile, model_name=model_name, threshold=0.8)
PlotEvaluationMetrics(eval_metrics, outpath=os.path.join(OUTPATH, 'evaluation_metrics_plot.png'))
##############################################################################
