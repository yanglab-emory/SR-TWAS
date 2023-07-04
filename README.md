# SR-TWAS

**SR-TWAS** stands for **Stacked Regression-Transcriptome-Wide Association Study**, which is developed using *Python* and *BASH* scripts for creating improved **Genetically Regulated gene eXpression (GReX) prediction models** by using an Ensemble Machine Learning technique of Stacked Regression to form optimal linear combinations of previously trained gene expression imputation models. SR-TWAS allows users to leverage multiple reference panels of the same tissue type in order to improve GReX prediction accuracy and TWAS power with increased effective training sample sizes.

---

- [Software Setup](#software-setup)
- [Input Files](#input-files)
	- [Validation Data](#validation-data)
		- [1. Genotype File](#1-genotype-file)
		- [2. SampleID File](#2-sampleid-file)
		- [3. Gene Expression File](#3-gene-expression-file)
	- [Weight (eQTL effect size) Files](#weight-eqtl-effect-size-files)
- [Example Usage](#example-usage)
	- [Arguments](#arguments)
	- [Example Command](#example-command)
	- [Output](#output)

<!-- - [VCF](#vcf-variant-call-format)
- [Dosage](#dosage-file) -->
---

## Software Setup

### 1. Install BGZIP, TABIX, Python 3.6, and the following Python libraries
- [BGZIP](http://www.htslib.org/doc/bgzip.html) 
- [TABIX](http://www.htslib.org/doc/tabix.html) 
- Python 3.6 modules/libraries:
	- [pandas](https://pandas.pydata.org/)
	- [numpy](https://numpy.org/)
	- [scipy](https://www.scipy.org/)
	- [sklearn](https://scikit-learn.org/stable/)
	- [statsmodels](https://www.statsmodels.org/stable/index.html)

### 2. Make files executable
- Make `*.sh` files in the **SR-TWAS** directory executable
```bash
SR_TWAS_dir="/home/jyang/GIT/SR-TWAS"
chmod 755 ${SR_TWAS_dir}/*.sh 
```

## Input Files
Example input files provided under `./ExampleData/` are generated artificially. All input files are *Tab Delimited Text Files*.

### Validation Data

#### 1. Genotype File

##### VCF ([Variant Call Format](http://www.internationalgenome.org/wiki/Analysis/Variant%20Call%20Format/vcf-variant-call-format-version-40/))

- `--genofile_type vcf`
	- `--format GT`
	- `--format DS`

| CHROM | POS |  ID | REF | ALT | QUAL | FILTER | INFO | FORMAT |  Sample1 | Sample...|
|:-----:|:---:|:---:|:---:|:---:|:----:|:------:|:----:|:------:|:--------:|:--------:|
|   1   | 100 | rs1 |  C  |  T  |   .  |  PASS  |   .  |  GT:DS | 0/0:0.01 |    ...   |

- Sorted by chromosome and base pair position, bgzipped by `bgzip`, and tabixed by `tabix`
- Example tabix command for a VCF file: `tabix -f -p vcf *.vcf.gz`.
- First 9 columns are Chromosome number, Base pair position, Variant ID, Reference allele, Alternative allele, Quality score, Filter status, Variant information, Genotype data format
- Genotype data starts from the 10th column. Each column denotes the genotype data per sample.
- Example: `./ExampleData/genotype.vcf.gz`


##### Dosage file

- `--genofile_type dosage`

| CHROM | POS |  ID | REF | ALT | Sample1 | Sample...|
|:-----:|:---:|:---:|:---:|:---:|:-------:|:--------:|
|   1   | 100 | rs** |  C  |  T  |   0.01  |    ...   |

- First 5 columns have the same format as the first 5 columns of a VCF file.
- Dosage genotype data start from the 6th column that are values ranging from 0 to 2, denoting the number of minor alleles. Each column denotes the genotype data per sample.


#### 2. SampleID File
- Headerless, single-column file containing sampleIDs to use.
- Examples:
	- `./ExampleData/train_sampleID.txt`
- Argument: `--train_sampleID`


#### 3. Gene Expression File

| CHROM | GeneStart | GeneEnd |   TargetID      | GeneName | Sample1 | Sample...|
|:-----:|:---------:|:-------:|:---------------:|:--------:|:-------:|:--------:|
|   1   |    100    |   200   |     ENSG0000    |     X    |   0.2   |     ...  |

- First 5 columns are *Chromosome number, Gene start position, Gene end position, Target gene ID, Gene name* (optional, could be the same as Target gene ID).
- Gene expression data start from the 6th column. Each column denotes the corresponding gene expression value per sample. 
- Example: `./ExampleData/gene_expression.txt`
- Argument: `--gene_exp`


### Weight (eQTL effect size) Files

| CHROM | POS | REF | ALT |     TargetID    |  ES  | MAF |
|:-----:|:---:|:---:|:---:|:---------------:|:----:|:---:|
|   1   | 100 |  C  |  T  |     ENSG0000    |  0.2 | 0.02 |

- Weight files contain cis-eQTL effect sizes estimated from reference data for TWAS.
- SR-TWAS requires that _at least_ two weight files be provided.
- Model training output `*_eQTLweights.txt` files obtained from the [TIGAR](https://github.com/yanglab-emory/SR-TWAS) tool may be used directly.
- Example bgzip/tabix command for a user-generated weight file: `bgzip -f my_eQTLweights.txt && tabix -f -b2 -e2 -S1 my_eQTLweights.txt.gz`.
- First 5 columns must be be: *Chromosome number, Base pair position, Reference allele, Alternative allele, and Target gene ID*. 
- Must contain a column named `ES` to denote variant weights (estimated eQTL effect sizes from a reference dataset) with for the gene (TargetID).
- The MAF column is only required if using a non-zero value of `--maf_diff` to exclude exclude eQTL weights obtained from data in which the absolute value between the MAF substantially different from that of the validation data.
- Each row denotes the variant weight (eQTL effect size) for testing the association of the gene (TargetID) with a phenotype
- Variants will be matched by their unique `CHROM:POS:REF:ALT` snpID
- Example: `./ExampleData/eQTLweights.txt`
- Argument: `--weights`


## Example Usage 

### Arguments
- `--chr`: Chromosome number need to be specified with respect to the genotype input data
- `--weights`: Space delimited list of paths to trained eQTL weight files (bgzipped and tabixed). 
- `--weights_names`: Space delimited list of names corresponding to base models specified by `--weights`; if user-submitted list is not the same length as number of base models or names are non-unique will use default (default: `[W0, W1, W2, ..., W{K}]`)
- `--gene_exp`: Path to Gene annotation and Expression file
- `--genofile`: Path to the training genotype file (bgzipped and tabixed)
- `--genofile_type`: Genotype file type: `vcf` or `dosage`
- `--format`: (Required if `genofile_type` is `vcf`) Genotype data format that should be used: `GT` or `DS`
	- `GT`: genotype data
	- `DS`: dosage data
- `--train_sampleID`: Path to a file with sampleIDs that will be used for training
- `--maf_diff`: Minor Allele Frequency threshold (ranges from 0 to 1) to exclude eQTL weights obtained from data in which the absolute value between the MAF substantially different from that of the validation data. (ie, if the absolute value of the difference between the validation genotype data and the weight file MAF exceeds `maf_diff`, then the effect size from that weight file is excluded). To use this option all weight files _MUST_ contain a `MAF` column. (default: `0` [no filtering])
- `--missing rate`: Missing rate threshold. If the rate of missing values for a SNP exceeds this value the SNP will be excluded. Otherwise, missing values for non-excluded SNPs will be imputed as the mean. (default: `0.2`)
- `--hwe`: Hardy Weinberg Equilibrium p-value threshold to exclude variants that violated HWE (default: `0.00001`)
- `weight_threshold`: Exclude eQTL weights with a magnitude less than this threshold (default: `0` [no filtering])
- `--window`: Window size (in base pairs) around gene region from which to include SNPs (default: `1000000` [`+- 1MB` region around gene region])
- `--cvR2`: Whether to perform 5-fold cross validation by average R2: `0` or `1` (default: `1`)
	- `0`: Skip 5-fold cross validation evaluation
	- `1`: Do 5-fold cross validation and only do final training if the average R2 of the 5-fold validations is at least >= cvR2_threshold
- `--cvR2_threshold`: Threshold for 5-fold CV. (default: `0.005`)
- `--thread`: Number of simultaneous *processes* to use for parallel computation (default: `1`)
- `--SR_TWAS_dir`: Specify the directory of **SR-TWAS** source code
- `--out_dir`: Output directory (will be created if it does not exist)
- `--sub_dir`: Whether to create a subdirectory for output files within the specified output directory. (default: `1`)
	- `0`: Output files will be placed in the output directory
	- `1`: Output files will be placed into subdirectory called `SR_CHR${chr}` within output directory
- `--out_prefix`: Filenames for the output weight, info, and log files begin with this string if filenames are not otherwise specified using `--out_weight_file`, `--out_info_file`, `--log_file`. (default: `CHR${chr}_SR_train`)
- `--out_weight_file`: Filename for output eQTL weights file. (default: `${out_prefix}_eQTLweights.txt`, `CHR${chr}_SR_train_eQTLweights.txt`)
- `--out_info_file`: Filename for output training info file. (default: `${out_prefix}_GeneInfo.txt`, `CHR${chr}_SR_train_GeneInfo.txt.txt`)
- `--log_file`: (default: `${out_prefix}_log.txt`, `CHR${chr}_SR_train_log.txt`)

### Example Command
```bash
gene_exp="${SR_TWAS_dir}/ExampleData/gene_expression.txt"
train_sampleID="${SR_TWAS_dir}/ExampleData/train_sampleID.txt"
genofile="${SR_TWAS_dir}/ExampleData/genotype.vcf.gz"
out_dir="${SR_TWAS_dir}/ExampleData/output"

weight0="${SR_TWAS_dir}/ExampleData/CHR1_DPR_cohort0_eQTLweights.txt.gz"
weight_name0=cohort0

weight1="${SR_TWAS_dir}/ExampleData/CHR1_DPR_cohort1_eQTLweights.txt.gz"
weight_name1=cohort1

weight2="${SR_TWAS_dir}/ExampleData/CHR1_DPR_cohort2_eQTLweights.txt.gz"
weight_name2=cohort2

${SR_TWAS_dir}/SR_TWAS.sh \
--gene_exp ${gene_exp} \
--train_sampleID ${train_sampleID} \
--chr 1 \
--genofile ${genofile} \
--genofile_type vcf \
--format GT \
--maf 0.01 \
--hwe 0.0001 \
--cvR2 1 \
--thread 2 \
--out_dir ${out_dir} \
--SR_TWAS_dir ${SR_TWAS_dir} \
--weights ${weight0} ${weight1} ${weight2} \
--weights_names ${weight_name0} ${weight_name1} ${weight_name2}
--
```

### Output
- `${out_dir}/SR_CHR${chr}/CHR${chr}_SR_train_eQTLweights.txt` is the file storing all eQTL effect sizes (`ES`) estimated from the gene expression imputation model per gene (`TargetID`)
- `${out_dir}/SR_CHR${chr}/CHR${chr}_SR_train_GeneInfo.txt` is the file storing information about the fitted gene expression imputation model per gene (per row), including gene annotation (`CHROM GeneStart GeneEnd TargetID GeneName`), sample size (`sample_size`), number of SNPs used in the model training (`N_SNP`), number of SNPs with non-zero eQTL effect sizes (`N_EFFECT_SNP`), imputation R2 by 5-fold cross validation (`CVR2`), imputation R2 using all given training samples (`R2`), p-value of the training R2 (`PVAL`), and the zeta weight used for each component weight model (`Z0`,`Z1`,...), and 4 columns for each component weight model describing performance on the validation data (`W0_N_SNP`,`W0_CVR2`,`W0_R2`,`W0_PVAL`,`W1_N_SNP`,`W1_CVR2`,`W1_R2`,`W1_PVAL`,...)
- `${out_dir}/logs/CHR${chr}_SR_train_log.txt` is the file storing all log messages for model training.

