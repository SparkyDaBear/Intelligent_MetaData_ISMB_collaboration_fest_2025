# Sample Data Relation File (SDRF) Generation
The SDRF-Proteomics format is a tab-delimited format that describes the sample characteristics and the relationships between samples and data files included in a dataset. The information in SDRF files is organized to follow the natural flow of a proteomics experiment.  

## Overall structure of an SDRF
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
Note: Most of the sample characteristics and comments above are detailed in this .pdf file but not all. These are an expanded set of annotated tags we use to ensure we get as much metadata from the manuscript as possible and some many not be applicable on a per sample basis (such as NumberBiologicalReplicates).  

## References and Resources
[1] Perez-Riverol Y; European Bioinformatics Community for Mass Spectrometry. Toward a Sample Metadata Standard in Public Proteomics Repositories. J Proteome Res. 2020 Oct 2;19(10):3906-3909. doi: 10.1021/acs.jproteome.0c00376. Epub 2020 Sep 22. PMID: 32786688; PMCID: PMC7116434. [link](https://pmc.ncbi.nlm.nih.gov/articles/PMC7116434/?utm_source=chatgpt.com)  
[2] [www.psidev.info/sdrf-sample-data-relationship-format](www.psidev.info/sdrf-sample-data-relationship-format)  
[3] [https://github.com/CompOmics](https://github.com/CompOmics)  
[4] [https://github.com/bigbio/proteomics-sample-metadata](https://github.com/bigbio/proteomics-sample-metadata)  
