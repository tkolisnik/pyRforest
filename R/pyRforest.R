#' pyRforest: A comprehensive approach to genomic analysis using scikit-learn's Random Forest models and rank-based feature reduction in R
#'
#' Welcome to pyRforest, a comprehensive tool designed to revolutionize your approach to genomic data analysis with Random Forest Models in R.
#' Whether you're dealing with RNA-seq or Microarray data, pyRforest integrates scikit-learn's Random Forest methodologies (imported to R via reticulate) for model development, evaluation, and custom feature reduction through rank-based permutation. It also integrates with SHAP and gProfiler for advanced analysis.
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
#'
#' Note: This package uses the reticuate R package to use Python's scikit-learn, see vignette for installation and setup instructions. Python 3.9.18 and Conda are required,
#' Highly recommend to read the vignette and use the demo dataset data("demo_rnaseq_data") for first time users.
#'
#' reticulate::use_python("/usr/local/bin/python3")
#' reticulate::conda_create(envname = "pyRforest-conda")
#' reticulate::conda_install(envname = "pyRforest-conda", packages = c("scikit-learn", "numpy","shap"))
#' reticulate::use_condaenv("pyRforest-conda", required = TRUE)
#'
#' data("demo_rnaseq_data")
#'
#' processed_training_data <- pyRforest::create_feature_matrix(demo_rnaseq_data$training_data,"training")
#' processed_validation_data <- pyRforest::create_feature_matrix(demo_rnaseq_data$training_data,"validation")
#' processed_testing_data <- pyRforest::create_feature_matrix(demo_rnaseq_data$training_data,"testing")
#' tuned_params <- pyRforest::tune_and_train_rf_model(processed_training_data$X_training_mat, processed_training_data$y_training_vector)
#' fitting_results<- pyRforest::fit_and_evaluate_rf(tuning_results$best_params,processed_training_data$X_training_mat,processed_training_data$y_training_vector,processed_validation_data$X_validation_mat,processed_validation_data$y_validation_vector)
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
