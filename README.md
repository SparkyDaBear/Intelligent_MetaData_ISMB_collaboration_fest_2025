# **Metadata Extraction from Mass Spetrometry Proteomics Publications**

## **Problem Statement**  
Reproducibility and data reusability in life-science research hinge on structured experimental metadata. However, most publications embed sample‐to‐parameter relationships in free-text methods and supplemental tables, making large-scale mining arduous. This challenge invites participants to build models or prompts that, given the plain-text of a scientific article, either reconstruct its Sample Data Relation File (SDRF) or at the very least identify the named entities present: a table whose rows represent individual samples and whose columns correspond to annotation classes (e.g., organism, instrument, strain, ect..). An effective solution will accelerate systematic meta-analysis across thousands of publications.

## **Hackathon Overview**  
There are two parts to this hackathon. In the first part you will attempt to reproduce the annotated metadata from a hand curated training set of 107 proteomics publications using any model/pipeline of your choice. In the second part you will apply your model to a set of publications lacking metadata annotations. 

## **Part #1 Train your model**
Build a custom model or pipeline that meets, or surpasses, the performance of the GPT-o4-mini benchmark on gold-standard SDRF annotations.  

- **Core Task**: Automatically identify every text span in a publication that corresponds to metadata annotations, and assign each span to one of our 71 predefined annotation categories.

- **Bonus Goal**: Go beyond span-level tagging and generate a fully populated SDRF file (tab-delimited format), ready for downstream analysis.

- **Evaluation**: Compute the precision, recall, F1-score for your model results compared to the SDRF. You can test the performance of your model using the interactive python notebooks located [src/data/Performance/](src/data/Performance/).   
   * To test the results of a model/algorithm that outputs a minimal-annotation files use the [src/data/Performance/MinimalAnnotation_Performance_relative2_SDRF.ipynb](src/data/Performance/MinimalAnnotation_Performance_relative2_SDRF.ipynb) Jupyter notebook.  
   * To test the results of a model/algorithm that outputs a full SDRF file use the [src/data/Performance/Predicted-SDRF_Performance_relative2_GS-SDRF.ipynb](src/data/Performance/Predicted-SDRF_Performance_relative2_GS-SDRF.ipynb) Jupyter notebook.  
        
- **Success Metric**: Your solution should match or outperform GPT-o4-mini on a held-out test set of annotated SDRF files.


**Data Provided**  
We have provided the following three key datasets 
1. [The cleaned abstract and methods sections texts from the 100 proteomics publications with gold standard metadata already annotated.](data/CleanText/Training/)  
2. [The Sample Data Relation File (SDRF) containing the gold standard metadata annotations.](data/GoldStandard_SDRFs/) 
3. [A benchmark set of annotations done by the GPT-o4-mini model.](data/BenchmarkAnnotations/o4-mini-2025-04-16/)


## **Part #2 Apply the trained model to unseen data**
Assess how well your trained model generalizes by applying it to completely unseen examples and benchmarking its performance against the training results.

- **Core Tasks**: 
   1. In [BRAT](https://brat.nlplab.org/), manually annotate the Abstract and Methods sections of 10 recently published proteomics papers using our 30-category schema. 
      - For more details about how BRAT works see their [introduction](https://brat.nlplab.org/introduction.html) and [manual](https://brat.nlplab.org/manual.html). 
      - You do not need to install anything and can access the Unseen publications for annotation: [ISMB Collaberation Fest Unseen Annotation Platform](https://cambridge-weak-low-keyword.trycloudflare.com/index.xhtml#/).
         - Read the tutorial popup and then click "OK".  
         - In the file browser that should be present navigate to ISMB/ and select one of the folders named Unseen_CleanText_Copy#. 
         - Each folder contains the same publications to annotate. 
         - If someone else has already started annotating the publications in a directory please do not over write their annotations and fine a clean directory. 
         - Hover your cursor over the brat logo in the upper right corner untill the login button apears. (Username: ISMB; password: 2025)
         - Once logged in you can annotate each document by highlighting text spans and choosing from one of the [annotation types](data/AnnotatedTypes.txt).  
         - You can download your annotations by hovering over the top left banner until the Data button appears and gives you the option to Export the collection with your annotations.  
   2.  Run your model on these new annotations to generate predicted spans and categories.

- **Bonus Goal**: Use both the [PRIDE](https://www.ebi.ac.uk/pride/)/[proteomexchange](https://www.proteomexchange.org/) with the publication to generate [lesSDRF](https://lessdrf.streamlit.app/) annotation files for the Unseen publications. 

- **Evaluation**: Compute the precision, recall, F1-score for your model results compared to the SDRF. You can test the performance of your model using the interactive python notebooks located [src/data/Performance/](src/data/Performance/).   
   * To test the results of a model/algorithm that outputs a minimal-annotation files use the [src/data/Performance/MinimalAnnotation_Performance_relative2_SDRF.ipynb](src/data/Performance/MinimalAnnotation_Performance_relative2_SDRF.ipynb) Jupyter notebook.  
   * To test the results of a model/algorithm that outputs a full SDRF file use the [src/data/Performance/Predicted-SDRF_Performance_relative2_GS-SDRF.ipynb](src/data/Performance/Predicted-SDRF_Performance_relative2_GS-SDRF.ipynb) Jupyter notebook.  

- **Success Metric**: Your model’s performance on these unseen papers should match—or exceed—its performance on the original training set.

**Data Provided**  
We have provided the following three key datasets 
1. [The cleaned abstract and methods sections texts from 100 PXDs with no SDRF present](data/CleanText/Unseen/)   

## How to Submit Your Results

When you’ve completed your analysis and are ready to share your work, please submit a **Pull Request (PR)** against this repository following the steps below:

1. **Fork this repository**  
   - Click “Fork” in the top-right corner to create your own copy under your GitHub account.

2. **Clone your fork locally**  
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```

3. **Create a new branch**  
   - Use a clear, descriptive name. We recommend:  
     ```
     git checkout -b results/<team-name>
     ```  
   - If you’re working solo, replace `<team-name>` with your GitHub username.

4. **Add your results**  
   - Place your result files (notebooks, figures, CSVs, etc.) into the `submissions/` directory.  
   - If your results span multiple files, create a subfolder named after your team:
     ```
     submissions/<team-name>/
       ├─ analysis.ipynb
       ├─ figures/
       └─ README.md        ← (brief explanation of your files)
     ```

5. **Commit your changes**  
   ```bash
   git add submissions/<team-name>/
   git commit -m "Add results for <team-name>: <one-line summary>"
   ```

6. **Push your branch to your fork**  
   ```bash
   git push origin results/<team-name>
   ```

7. **Open a Pull Request**  
   - Go to the original contest repo on GitHub.  
   - You should see a banner prompting you to open a PR from `results/<team-name>`.  
   - Click **Compare & pull request**.

8. **Fill out the PR template**  
   In your PR description, please include:
   - **Team name**  
   - **Primary contact GitHub handle**  
   - **Summary of your approach** (2–3 sentences)  
   - **List of files/folders** you’re submitting  
   - **Any special instructions** for running your code (e.g. environment, dependencies)

9. **Submit before the deadline**  
   - All PRs must be opened **by 11:59 PM UTC** on the contest end date.  
   - Late submissions may not be reviewed.

10. **After submission**  
    - Monitor your PR for any reviewer comments.  
    - You may push additional commits to the same branch; they will appear in your open PR.

If you encounter any issues with forking, branching, or PRs, please open an issue in this repo or ping one of the organizers on Slack. Good luck, and we look forward to your submissions! 🚀




## Other Technical Information  
### Structure of a Sample Data Relation File (SDRF)
An SDRF-Proteomics file is a single table where each row represents one sample–data file relationship. Columns can have the following headers:  
| Prefix                  | Indicates…                                                                |
| ----------------------- | ------------------------------------------------------------------------- |
| **Source Name**         | Unique ID for the *starting material* (e.g. organism)                     |
| **Characteristics\[…]** | Sample attributes (e.g. organism, tissue, disease)                        |
| **Factor Value\[…]**    | Study variables or experimental factors (e.g. treatment, time point)      |
| **Comment\[…]**         | Data-file or technical metadata (e.g. instrument, fraction, label)        |
| **Assay Name**          | Unique ID for the *assay* (i.e. the combination of sample + file)         |
| **Raw Data File**       | File name or path of the raw spectrum file                                |
| *(or)* **Data File**    | Processed output (e.g. mzML, mzTab)                                       |
  

### Characteristics — sample‐level metadata  
| Annotation Type                  | Definition                                                                                        |
| -------------------------------- | ------------------------------------------------------------------------------------------------- |
| **Age**                          | Age of the donor or developmental stage of the organism (e.g. “45 years”, “E14.5 embryo”).        |             
| **AnatomicSiteTumor**            | Anatomical location from which a tumor sample was taken (e.g. “left lung lobe”).                  |
| **AncestryCategory**             | Donor ancestry or ethnicity category (e.g. “European”, “East Asian”).                             |
| **Bait**                         | The protein or molecule used as bait in an affinity‐purification experiment.                      |
| **BMI**                          | Body‐Mass Index of the donor (kg/m²).                                                             |
| **BiologicalReplicate**          | Identifier for biological replicates (e.g. “bioRep1”, “bioRep2”).                                 |
| **CellLine**                     | Name of the immortalized cell line (e.g. “HEK293T”, “U2OS”).                                      |
| **CellPart**                     | Subcellular compartment or fraction (e.g. “nucleus”, “mitochondria”).                             |
| **CellType**                     | Primary cell type or lineage (e.g. “neurons”, “fibroblasts”).                                     |
| **CleavageAgent**                | Protease or chemical used to digest proteins (e.g. “trypsin”, “chymotrypsin”).                    |
| **Compound**                     | Chemical or small molecule added to the sample (e.g. drug, inhibitor)                             |
| **ConcentrationOfCompound**      | Concentration of the Compound used (e.g. “10 µM”).                                                |
| **Depletion**                    | Method used to remove high‐abundance proteins (e.g. “albumin depletion kit”).                     |
| **DevelopmentalStage**           | Stage of development for the sample source (e.g. “adult”, “P7 pup”).                              |
| **Disease**                      | Disease state or diagnosis (e.g. “breast cancer”, “Type 2 diabetes”).                             |
| **DiseaseTreatment**             | Pre‐treatment applied to diseased samples (e.g. “chemotherapy”, “radiation”).                     |
| **GeneticModification**          | Any genetic alteration in the source organism/cells (e.g. “GFP‐tagged”, “knockout of gene X”).    |
| **Genotype**                     | Genotypic background (e.g. “C57BL/6J”, “BRCA1-mutant”).                                           |
| **GrowthRate**                   | Doubling time or growth rate of cell cultures (e.g. “24 h doubling”).                             |
| **Label**                        | Isobaric or metabolic label applied (e.g. “TMT-126”, “SILAC heavy”).                              |
| **MaterialType**                 | Broad class of material (e.g. “tissue”, “cell line”, “biofluid”).                                 |
| **Modification**                 | Post‐translational modification enrichment or tagging (e.g. “phosphorylation”, “ubiquitination”). |
| **NumberOfBiologicalReplicates**\* | Total number of biological replicates in the study.                                               |
| **NumberOfSamples**\*              | Total number of samples processed.                                                                |
| **NumberOfTechnicalReplicates**\*  | Total number of technical replicates per sample.                                                  |
| **Organism**                     | Source species (NCBI Taxonomy ID and name, e.g. “9606 (Homo sapiens)”).                           |
| **OrganismPart**                 | Tissue or organ of origin (Uberon term, e.g. “UBERON:0002107 (liver)”).                           |
| **OriginSiteDisease**            | Anatomical site of disease origin (e.g. “colon”, “prostate”).                                     |
| **PooledSample**                 | Indicates if multiple samples were pooled (e.g. “pool1 of reps1–3”).                              |
| **ReductionReagent**             | Chemical used to reduce disulfide bonds (e.g. “DTT”, “TCEP”).                                     |
| **SamplingTime**                 | Time point of sample collection (e.g. “T0”, “24 h post‐treatment”).                               |
| **SampleTreatment**              | Any treatment applied to the sample before processing (e.g. “fixation”, “lysis buffer X”).        |
| **Sex**                          | Donor sex (e.g. “male”, “female”).                                                                |
| **Specimen**                     | Description of biological specimen (e.g. “biopsy”, “plasma”).                                     |
| **SpikedCompound**               | Exogenous standard or spike‐in added (e.g. “iRT peptides”).                                       |
| **Staining**                     | Any staining applied (e.g. “Coomassie Blue”, “Silver stain”).                                     |
| **Strain**                       | Animal strain (e.g. “BALB/c”, “FVB/N”).                                                           |
| **SyntheticPeptide**             | Indicates a synthetic peptide sample (e.g. “synthetic phosphopeptide”).                           |
| **TumorCellularity**             | Percentage of tumor cells in the sample (e.g. “80%”).                                             |
| **TumorGrade**                   | Histological grade (e.g. “Grade II”).                                                             |
| **TumorSize**                    | Physical size of the tumor (e.g. “3 cm diameter”).                                                |
| **TumorSite**                    | Anatomical site of tumor (e.g. “breast”, “pancreas”).                                             |
| **TumorStage**                   | Clinical staging (e.g. “Stage III”).                                                              |
| **Time**                         | Broad time parameter (e.g. “day 5”, “week 2”).                                                    |
| **Temperature**                  | Temperature during processing or incubation (e.g. “37 °C”).                                       |
| **Treatment**                    | Experimental treatment (e.g. “drug X 5 µM 24 h”).                                                 |
\* Not applicable on a per sample basis  

### Comment — data‐file and protocol parameters  
| Annotation Type             | Definition                                                                |
| --------------------------- | ------------------------------------------------------------------------- |
| **AcquisitionMethod**       | MS acquisition scheme (e.g. “DDA”, “DIA”, “PRM”).                         |
| **CollisionEnergy**         | Collision energy applied in MS/MS (e.g. “27 eV”).                         |
| **EnrichmentMethod**        | Peptide/enrichment protocol used (e.g. “TiO₂ phosphopeptide enrichment”). |
| **Experiment**              | Logical grouping or experiment identifier (e.g. “exp1”).                  |
| **FlowRateChromatogram**    | LC flow rate (e.g. “300 nL/min”).                                         |
| **FractionationMethod**     | Method used to fractionate peptides (e.g. “high-pH RP HPLC”).             |
| **FractionIdentifier**      | Numeric or text ID of each fraction (e.g. “F1”, “F2”).                    |
| **FragmentationMethod**     | Ion‐fragmentation technique (e.g. “HCD”, “CID”, “ETD”).                   |
| **FragmentMassTolerance**   | Mass tolerance for fragment matching (e.g. “0.02 Da”).                    |
| **GradientTime**            | Total LC gradient length (e.g. “120 min”).                                |
| **Instrument**              | Mass spec make/model (e.g. “Thermo Q-Exactive Plus”).                     |
| **IonizationType**          | Ionization source (e.g. “nanoESI”, “MALDI”).                              |
| **MS2MassAnalyzer**         | Analyzer used for MS2 (e.g. “orbitrap”, “ion trap”).                      |
| **NumberOfMissedCleavages** | Max missed cleavages allowed in database search (e.g. “2”).               |
| **NumberOfFractions**       | Total number of fractions generated from each sample.                     |
| **PrecursorMassTolerance**  | Mass tolerance for precursor matching (e.g. “10 ppm”).                    |
| **Separation**              | Chromatographic separation mode (e.g. “C18 reversed-phase”).              |

More details about specific annotation types can be found here: [SDRF_Proteomics_Specification_v1.0.0.pdf](../documents/SDRF_Proteomics_Specification_v1.0.0.pdf)     
Note: Most of the sample characteristics and comments above are detailed in this .pdf file but not all. These are an expanded set of annotated tags we use to ensure we get as much metadata from the manuscript as possible and some may not be applicable on a per sample basis (such as NumberBiologicalReplicates).  

## References and Resources
[1] Perez-Riverol Y; European Bioinformatics Community for Mass Spectrometry. Toward a Sample Metadata Standard in Public Proteomics Repositories. J Proteome Res. 2020 Oct 2;19(10):3906-3909. doi: 10.1021/acs.jproteome.0c00376. Epub 2020 Sep 22. PMID: 32786688; PMCID: PMC7116434. [link](https://pmc.ncbi.nlm.nih.gov/articles/PMC7116434/?utm_source=chatgpt.com)  
[2] [www.psidev.info/sdrf-sample-data-relationship-format](www.psidev.info/sdrf-sample-data-relationship-format)  
[3] [https://github.com/CompOmics](https://github.com/CompOmics)  
[4] [https://github.com/bigbio/proteomics-sample-metadata](https://github.com/bigbio/proteomics-sample-metadata)  
[5] Claeys, T., Van Den Bossche, T., Perez-Riverol, Y. et al. lesSDRF is more: maximizing the value of proteomics data through streamlined metadata annotation. Nat Commun 14, 6743 (2023). https://doi.org/10.1038/s41467-023-42543-5