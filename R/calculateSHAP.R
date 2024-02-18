#' Calculate SHAP Values and Identify Significant Features
#'
#' This function calculates SHAP values for a given dataset using a provided model.
#' It then identifies significant features based on the SHAP values for the specified class.
#' Additionally, it prepares a long-format data frame of individual SHAP values suitable for visualization.
#'
#' @param model The trained model for which SHAP values are to be calculated.
#' @param data A matrix or data frame of input features for SHAP value calculation.
#' @param class_index Index of the class for which to calculate SHAP values.
#'   For binary classification (e.g., target 1 = "Right", target 0 = "Left"),
#'   this parameter determines the class perspective for SHAP analysis.
#'   In the demo_rnaseq_data dataset:
#'   - If class_index = 1, SHAP values represent feature contributions towards predicting "Right".
#'   - If class_index = 0, SHAP values represent feature contributions towards predicting "Left".
#'   In binary classification, SHAP values for one class are the negative of those for the other class,
#'   this reflects how each feature influences the model's output in opposite directions for each class.
#' @param shap_std_dev_factor Factor to determine the cutoff for significant SHAP values.
#'   Default is 0.5. For example 0.5 is considered conservative as it means you are selecting features whose mean absolute SHAP values are above the mean plus half of the standard deviation.
#' @return A list containing three elements:
#'   - shap_values: A data frame with SHAP values for each feature.
#'   - significant_features: A data frame with significant features based on the cutoff.
#'   - long_shap_data: A long-format data frame of individual SHAP values for each feature.
#' @examples
#' shapvals <- calculate_SHAP_values(fitting_results$model, processed_training_data$X_training_mat, class_index = 1, shap_std_dev_factor = 0.5)
#' @importFrom dplyr filter group_by summarize mutate arrange
#' @importFrom tidyr pivot_longer
#' @export
calculate_SHAP_values <- function(model, data, class_index = 1, shap_std_dev_factor = 0.5) {

  # Validate inputs
  if (is.null(model)) stop("Model input is NULL.")
  if (!is.matrix(data) && !is.data.frame(data)) stop("Data must be a matrix or data frame.")
  if (!is.numeric(class_index) || !(class_index %in% c(0, 1))) {
    stop("class_index must be either 0 or 1 for binary classification.")
  }
  if (!is.numeric(shap_std_dev_factor) || shap_std_dev_factor < 0) stop("shap_std_dev_factor must be a non-negative numeric value.")

  # Import necessary Python modules using reticulate
  python_pkgs<-setup_python_pkgs()

  # Create a SHAP explainer using the model
  explainer <- python_pkgs$shap$TreeExplainer(model)

  # Calculate SHAP values
  shap_values <- explainer$shap_values(data)

  # Select SHAP values for the specified class
  selected_shap_values <- shap_values[[class_index]]

  # Convert the NumPy array to an R matrix
  selected_shap_values_matrix <- reticulate::py_to_r(selected_shap_values)

  # Convert to data frame and assign column names
  feature_names <- colnames(data)
  shap_values_df <- as.data.frame(selected_shap_values_matrix)
  colnames(shap_values_df) <- feature_names

  # Identify non-zero columns
  non_zero_columns <- colnames(shap_values_df)[apply(shap_values_df, 2, function(x) any(x != 0))]

  # Filter data frame
  significant_shap_values_df <- shap_values_df %>% dplyr::select(all_of(non_zero_columns))
  #significant_shap_values_df <- shap_values_df[, non_zero_columns, drop = FALSE] # Deprecated method

  # Transform data to long format for visualization
  long_shap_data <- tidyr::pivot_longer(significant_shap_values_df, cols = non_zero_columns, names_to = "feature", values_to = "shap_value")

  # Calculate mean SHAP values
  mean_shap_values <- long_shap_data %>%
    group_by(feature) %>%
    summarize(mean_shap = mean(shap_value), .groups = 'drop') %>%
    mutate(abs_mean_shap = abs(mean_shap)) %>%
    arrange(desc(abs_mean_shap))

  # Determine cutoff for significance
  mean_abs_shap <- mean(mean_shap_values$abs_mean_shap)
  sd_abs_shap <- sd(mean_shap_values$abs_mean_shap)
  shap_cutoff <- mean_abs_shap + (shap_std_dev_factor * sd_abs_shap)

  # Filter significant SHAP values
  significant_features <- dplyr::filter(mean_shap_values, abs_mean_shap > shap_cutoff)

  return(list(shap_values = shap_values_df, significant_features = significant_features, long_shap_data = long_shap_data))
}
