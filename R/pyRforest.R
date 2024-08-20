#' pyRforest: A comprehensive approach to genomic analysis using scikit-learn's Random Forest models and rank-based feature reduction in R
#'
#' Welcome to pyRforest, a comprehensive tool for genomic data analysis featuring scikit-learn Random Forests in R. Tailored for expression data, such as RNA-seq or Microarray, pyRforest is built for bioinformaticians and researchers looking to explore the relationship between biological features and matched binary or categorical outcome variables using Random Forest models.
#' Please read on for instructions that will guide you through pyRforest's seamless integration of scikit-learn's Random Forest methodologies (imported to R via reticulate) for model development, evaluation, SHAPley additive explanations, and our custom feature reduction approach by way of rank-based permutation. You will also be directed you through our integration with clusterProfiler and g:Profiler for Gene Ontology and Enrichment Analysis.
#'
#' @details
#' The pyRforest package provides advanced tools for genomic data analysis, featuring seamless integration of scikit-learn's Random Forest methodologies in R. It's designed to be user-friendly for both beginners and advanced users, with a focus on providing accurate and interpretable results. Please see our comprehensive vignette for detailed instructions on using pyRforest.
#'
#' @section Installation:
#' Install pyRforest from GitHub using devtools:
#' \preformatted{devtools::install_github("tkolisnik/pyRforest")}
#'
#' @section Basic Usage:
#' \preformatted{
#' library(pyRforest)
#' library(reticulate)
#' Note: This package uses the reticuate R package to use Python's scikit-learn, see vignette for installation and setup instructions.
#' Python 3.9.18 is required and it is recommended to use Conda for Mac or Windows installations and reticulate venv for Linux.
#'
#' First time users are highly recommend to read the vignette and use the demo dataset data("demo_rnaseq_data").
#'
#' reticulate::use_condaenv("pyRforest-conda", conda = 'path to conda - see vignette', required = TRUE)
#'
#' data("demo_rnaseq_data")
#'
#' processed_training_data <- pyRforest::create_feature_matrix(demo_rnaseq_data$training_data,"training")
#'
#' tuning_results <- pyRforest::tune_and_train_rf_model_bayes(
#'     X = processed_training_data$X_training_mat,
#'     y = processed_training_data$y_training_vector,
#'     cv_folds = 5,
#'     scoring = 'roc_auc', # use 'roc_auc_ovr' for multiclass targets
#'      seed = 4,
#'     n_jobs = 1,
#'     n_cores = -2)
#' print(tuning_results$best_params)
#' print(tuning_results$best_score)
#'
#' ... Please see vignette for additional steps and functions, and any issues with setup.
#' }
#'
#' @section More Information:
#' Please see the comprehensive vignette for detailed instructions: www.github.com/tkolisnik/pyRforest
#'
#' @section Contact and Contribution:
#' We welcome contributions, bug reports, and questions. Please submit issues and pull requests on our GitHub repository or email to tkolisnik@gmail.com.
#'
"_PACKAGE"
