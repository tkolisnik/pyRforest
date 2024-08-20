# pyRforest

Welcome to pyRforest, a comprehensive tool for genomic data analysis featuring scikit-learn Random Forests in R. Tailored for expression data, such as RNA-seq or Microarray, pyRforest is built for bioinformaticians and researchers looking to explore the relationship between biological features and matched binary or categorical outcome variables using Random Forest models. Please read on for instructions that will guide you through pyRforest's seamless integration of scikit-learn's Random Forest methodologies (imported to R via reticulate) for model development, evaluation, SHAPley additive explanations, and our custom feature reduction approach by way of rank-based permutation. You will also be directed you through our integration with clusterProfiler and g:Profiler for Gene Ontology and Enrichment Analysis.

Please see our [vignette](https://github.com/tkolisnik/pyRforest/blob/main/vignettes/pyRforest-vignette.pdf) for instructions that will guide you through pyRforest's seamless integration of [scikit-learn's](https://scikit-learn.org/stable/common_pitfalls.html) Random Forest methodologies (imported to R via [reticulate](https://rstudio.github.io/reticulate/)) for model development, evaluation, and our custom feature reduction approach by way of rank-based permutation. You will also be directed through our integration with [Enrichr Enrichment Analysis & Gene Ontology](https://maayanlab.cloud/Enrichr/), [SHAP](https://shap.readthedocs.io/en/latest/) and [gProfiler](https://biit.cs.ut.ee/gprofiler/gost).

## Features

- Integration of Python's scikit-learn Random Forest models in R.
- Custom rank-based feature reduction methodology.
- Compatibility with RNA-seq and Microarray data.
- Integration with Gene Ontology, Enrichment Analysis, SHAP, and gProfiler.

## Installation

```r
# Install devtools if not already available

if (!requireNamespace("devtools", quietly = TRUE))
    install.packages("devtools")

# Install pyRforest from GitHub

devtools::install_github("tkolisnik/pyRforest")

# Load pyRforest

library(pyRforest)

Note: pyRforest is developed on R version 4.3.1 for Apple Mac M1 arm64 architecture. It also works on Windows (intel x64) and Linux. 

Please see vignette for full installation and usage instructions. 

```

## Data Preparation

pyRforest is designed to work with structured genomic data with a focus on classification problems. Your data should be formatted as a list containing four tibbles (training, validation, testing, and target categories). See the [vignette](https://github.com/tkolisnik/pyRforest/blob/main/vignettes/pyRforest-vignette.pdf) for detailed structuring instructions.

[Demo data](https://github.com/tkolisnik/pyRforest/blob/main/data/demo_rnaseq_data.RData) is available using the command <code>data("demo_rnaseq_data", package = "pyRforest")</code>

## Usage

pyRforest includes functionalities for data preprocessing, model tuning, fitting, evaluation, and obtaining feature importances. Detailed usage examples and function documentation are available in the package [vignette](https://github.com/tkolisnik/pyRforest/blob/main/vignettes/pyRforest-vignette.pdf).

## Advanced Features

Explore advanced functionalities including SHAP value analysis and integration with Enrichment, Gene Ontology and gProfiler modules.

## Support and Contribution

For more information, detailed documentation, and how to contribute, visit the pyRforest GitHub repository.

## Acknowledgments

Created by Tyler Kolisnik with support from Dr. Olin Silander, Dr. Adam Smith & Faeze Keshavarz.

## Contact

For queries, contact Tyler Kolisnik at tkolisnik@gmail.com.
