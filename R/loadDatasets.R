#' Dataset Preparation Function for Scikit-Learn Random forest in pyRforest.
#'
#' This function prepares a dataset for scikit-learn random forest machine learning by creating a matrix from the dataset excluding certain columns, and extracting a target vector. It is flexible and can handle different types of data sets such as training, testing, or validation.
#' Data must be in the correct input format, see vignette or example dataset for details.
#'
#' @param dataset A data frame representing the dataset to be prepared. The dataset must contain the columns 'identifier', and 'target', which will be excluded from the main matrix.
#' @param set_type A character string specifying the type of dataset being prepared. Possible values are "training", "testing", or "validation". This parameter influences the naming of the output elements.
#'
#' @return A list containing three elements:
#' \itemize{
#'   \item{X_set_type}{ A data frame of the dataset after excluding specified columns.}
#'   \item{X_set_type_mat}{ A matrix conversion of 'X_set_type'.}
#'   \item{y_set_type_vector}{ A vector representing the 'target' column from the original dataset.}
#' }
#' The names of these elements are dynamically generated based on the 'set_type' parameter.
#'
#' @examples
#' library(pyRforest)
#'
#' Load sample data
#' data("demo_rnaseq_data")
#'
#' processed_training_data <- create_feature_matrix(demo_data_rnaseq_rf$training_data, "training")
#'
#' processed_validation_data <- create_feature_matrix(demo_data_rnaseq_rf$validation_data, "validation")
#'
#' processed_testing_data <- create_feature_matrix(demo_data_rnaseq_rf$testing_data, "testing")
#'
#' @export

create_feature_matrix <- function(dataset, set_type){
  # Ensure the input dataset is a data.frame
  if (!is.data.frame(dataset)) {
    stop("The input dataset must be a data frame or tibble.")
  }

  # Select and process the dataset
  X_set_type <- dataset %>% dplyr::select(-identifier, -target)
  X_set_type_mat <- as.matrix(X_set_type)
  rownames(X_set_type_mat) <- dataset$identifier
  y_set_type_vector <- as.vector(dataset$target)

  # Verify X_set_type_mat is a matrix
  if (!is.matrix(X_set_type_mat)) {
    stop("X_set_type_mat is not a matrix.")
  }

  # Verify if y_set_type_vector is a one-dimensional vector
  if (!(is.atomic(y_set_type_vector) && !is.list(y_set_type_vector) && is.null(dim(y_set_type_vector)))) {
    stop("y_set_type_vector is not a one-dimensional vector.")
  }

  # Dynamically create names for the list elements based on set_type
  names_X_set_type <- paste("X", set_type, sep = "_")
  names_X_set_type_mat <- paste("X", set_type, "mat", sep = "_")
  names_y_set_type_vector <- paste("y", set_type, "vector", sep = "_")

  # Use setNames to correctly name the list elements
  return(setNames(list(X_set_type, X_set_type_mat, y_set_type_vector),
                  c(names_X_set_type, names_X_set_type_mat, names_y_set_type_vector)))
}
