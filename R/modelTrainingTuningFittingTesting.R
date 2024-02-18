#' Perform hyperparameter tuning and training for RandomForestClassifier
#'
#' This function uses scikit-learn's python based GridSearchCV to perform hyperparameter tuning and training of a RandomForestClassifier.
#' It allows for customizable parameter grids and includes preprocessing steps of one-hot encoding
#' and scaling. The function is designed to find the best hyperparameters based on accuracy. Please reference the scikit-learn GridSearchCV documentation for the full description of options, however our defaults are comprehensive.
#'
#' @param X The features for the model (data frame or matrix). Usually obtained from the create_feature_matrix function.
#' @param y The target variable for the model (vector). Usually obtained from the create_feature_matrix function.
#' @param cv_folds The number of splits in StratifiedKFold cross validation, (default: 5)
#' @param seed The random seed for reproducibility (default: 4).
#' @param param_grid An optional list of parameters for tuning the model.
#'                   If NULL, a default set of parameters is used. The list should follow
#'                   the format expected by GridSearchCV, with parameters requiring integers
#'                   suffixed with 'L' (e.g., 10L). This is to ensure compatibility when being passed from R to Python.
#'                   Default param_grid is as follows:
#'                      param_grid <- list(
#'                          bootstrap = list(TRUE),
#'                          class_weight = list(NULL),
#'                          max_depth = list(5L, 10L, 15L, 20L, NULL),
#'                          n_estimators = as.integer(seq(10, 100, 10)),
#'                          max_features = list("sqrt", "log2", 0.1, 0.2),
#'                          criterion = list("gini"),
#'                          warm_start = list(FALSE),
#'                          min_samples_leaf = list(1L, 2L, 5L, 10L, 20L, 50L),
#'                          min_samples_split = list(2L, 10L, 20L, 50L, 100L, 200L)
#'                          )
#' @param scoring_method The scoring method to be used. Options are 'accuracy', 'precision', 'recall', 'roc_auc', 'f1'... see scikit-learn GridSearchCV documentation for more info.
#' @param n_jobs An optional number of jobs to specify for parallel processing. Default is 1.
#' @param n_cores An optional number of cores to specify for parallel processing. Default is (-2), which is 2 less than the maximum available number of cores.
#' @return A list containing the best hyperparameters for the model, cross-validation scores on training set,
#'         and the fitted GridSearchCV object.
#' @examples
#' library(pyRforest)
#'
#' Load conda environment, which ensures the correct version of Python and the necessary python packages can be loaded. See vignette for more details.
#' use_condaenv("pyRforest-conda-arm64mac", required = TRUE)
#'
#' Load the demo data
#' data(demo_rnaseq_data)
#'
#' Prepare the sample data into a format ingestible by the ML algorithm
#' processed_training_data <- create_feature_matrix(demo_data_rnaseq_rf$training_data, "training")
#'
#' Model training (Warning: may take a long time if dataset is large and if param_grid has many options)
#' tuning_results <- tune_and_train_rf_model(processed_training_data$X_training_mat, processed_training_data$y_training_vector, cv_folds = 5, seed = 123, param_grid = list(max_depth = list(10L, 20L)))
#' print(tuning_results$best_params)
#' print(tuning_results$grid_search$best_score_)
#' @export

tune_and_train_rf_model <- function(X, y, cv_folds = 5, scoring_method = "roc_auc",  seed = 4, param_grid = NULL, n_jobs=1, n_cores=-2) {
  # Import necessary Python modules using reticulate
  python_pkgs<-setup_python_pkgs()

  # Import specific sklearn submodules
  GridSearchCV <- python_pkgs$sklearn$model_selection$GridSearchCV
  RandomizedSearchCV <- python_pkgs$sklearn$model_selection$RandomizedSearchCV
  OneHotEncoder <- python_pkgs$sklearn$preprocessing$OneHotEncoder
  StandardScaler <- python_pkgs$sklearn$preprocessing$StandardScaler
  StratifiedKFold <- python_pkgs$sklearn$model_selection$StratifiedKFold
  RandomForestClassifier <- python_pkgs$sklearn$ensemble$RandomForestClassifier

  # One-hot encode categorical variables and scale numeric variables
  ohe <- OneHotEncoder()
  scaler <- StandardScaler(with_mean = FALSE)

  X_encoded <- ohe$fit_transform(X)
  X_scaled <- scaler$fit_transform(X_encoded)

  # Create the Random Forest Classifier
  clf <- RandomForestClassifier(random_state = as.integer(seed), n_jobs = as.integer(n_jobs))

  # Default parameter grid if not provided
  if (is.null(param_grid)) {
    param_grid <- list(
      bootstrap = list(TRUE),
      class_weight = list(NULL),
      max_depth = list(5L, 10L, 15L, 20L, NULL),
      n_estimators = as.integer(seq(50, 200, 25)),
      max_features = list("sqrt", "log2", 0.1, 0.2),
      criterion = list("gini"),
      warm_start = list(FALSE),
      min_samples_leaf = list(2L, 3L, 4L, 5L, 10L),
      min_samples_split = list(2L, 3L, 4L, 5L, 10L)
    )
  }

  # Check if param_grid is in correct format
  if (!is.null(param_grid)) {
    if (!is.list(param_grid)) {
      stop("param_grid must be a list.")
    }
  }

  # Create a stratified k-fold cross validation object
  strat_k_fold <- StratifiedKFold(n_splits = as.integer(cv_folds), shuffle = TRUE, random_state = as.integer(seed))

  # Initialize the GridSearchCV object
  grid_search <- GridSearchCV(
    estimator = clf,
    param_grid = param_grid,
    cv = strat_k_fold,
    scoring = scoring_method,
    verbose = as.integer(2),
    n_jobs = as.integer(n_cores)
  )

  # Fit the GridSearchCV object to the data
  grid_search$fit(X_scaled, y)

  # Return the best parameters and the GridSearchCV object
  return(list(best_params = grid_search$best_params_, grid_search = grid_search))
}

#' Fit and Evaluate Random Forest Model
#'
#' This function fits a Random Forest model using the provided hyperparameters and
#' training data, then evaluates its performance on a validation set.
#'
#' @param best_params A list of best hyperparameters obtained from hyperparameter tuning.
#' @param X_train Training feature matrix.
#' @param y_train Training target vector.
#' @param X_val Validation feature matrix (optional).
#' @param y_val Validation target vector (optional).
#' @param save_path Optional path to save the trained model.
#' @param seed Optional random seed.
#' @return A list containing the trained model and, if testing or validation data is provided,
#'         the accuracy, f1, precision, recall and roc_auc scores on the testing or validation set.
#' @examples
#' fitting_results<-fit_and_evaluate_rf(tuning_results$best_params,processed_training_data$X_training_mat,processed_training_data$y_training_vector,processed_validation_data$X_validation_mat,processed_validation_data$y_validation_vector)
#'
#' Print the fitting results, provides accuracy, f1 score, precision, recall and roc_auc scores on the model as fitted to the validation set
#' print(fitting_results)
#' @export

fit_and_evaluate_rf <- function(best_params, X_train, y_train, X_val = NULL, y_val = NULL, save_path = NULL, seed=4) {
  # Import necessary Python modules using reticulate
  python_pkgs<-setup_python_pkgs()

  # Import RandomForestClassifier from sklearn.ensemble
  random_forest_classifier <- python_pkgs$sklearn$ensemble$RandomForestClassifier
  metrics <- python_pkgs$sklearn$metrics

  # Instantiate the model with the best parameters
  final_model <- random_forest_classifier(
    random_state = as.integer(seed),
    n_estimators = best_params$n_estimators,
    max_depth = best_params$max_depth,
    min_samples_split = best_params$min_samples_split,
    min_samples_leaf = best_params$min_samples_leaf,
    max_features = best_params$max_features,
    bootstrap = best_params$bootstrap,
    class_weight = best_params$class_weight,
    criterion = best_params$criterion,
    warm_start = best_params$warm_start,
    verbose = TRUE
  )

  # Fit the model on the training set
  final_model$fit(X_train, y_train)

  # Optionally save the model
  if (!is.null(save_path)) {
    saveRDS(final_model, file = save_path)
  }

  py$final_model <- final_model
  reticulate::py_run_string("params = final_model.get_params()")
  all_params <- py$params

  # Helper function for printing full list of model parameters (by default only a few print and it looks misleading)
  format_model_params <- function(params_list) {
    params_str <- "RandomForestClassifier("
    param_lines <- c()

    for (param_name in names(params_list)) {
      value <- params_list[[param_name]]

      # Handle different value types
      if (is.logical(value)) {
        value_str <- ifelse(value, "TRUE", "FALSE")
      } else if (is.null(value)) {
        value_str <- "NULL"
      } else {
        value_str <- toString(value)
      }

      param_lines <- c(param_lines, sprintf("%s=%s", param_name, value_str))
    }

    params_str <- paste0(params_str, paste(param_lines, collapse=", "), ")")

    return(params_str)
  }

  full_model_parameters <- format_model_params(all_params)

  # Initialize an empty list to store results
  results <- list(model = final_model, all_params = full_model_parameters)

  # If validation data is provided, compute the metrics on the validation set
  if (!is.null(X_val) && !is.null(y_val)) {
    y_pred <- final_model$predict(X_val)
    y_pred_proba <- final_model$predict_proba(X_val)[,1]  # Assuming binary classification

    # Compute accuracy
    accuracy <- metrics$accuracy_score(y_val, y_pred)

    # Compute F1 score
    f1 <- metrics$f1_score(y_val, y_pred, average = 'binary')  # Adjust 'average' as needed

    # Compute Precision and Recall
    precision <- metrics$precision_score(y_val, y_pred, average = 'binary')
    recall <- metrics$recall_score(y_val, y_pred, average = 'binary')

    # Compute ROC AUC Score
    roc_auc <- metrics$roc_auc_score(y_val, y_pred_proba)

    # Add scores to the results
    results$accuracy <- accuracy
    results$f1 <- f1
    results$precision <- precision
    results$recall <- recall
    results$roc_auc <- roc_auc
  }


  return(results)
}

#' Calculate True and Permuted Feature Importances
#'
#' This function fits a Random Forest model and calculates the true and permuted feature importances.
#' It performs permutations on the target variable to generate permuted importances for comparison.
#'
#' @param model The Random Forest model to be used.
#' @param X_train Training feature matrix.
#' @param y_train Training target vector.
#' @param n_permutations The number of permutations to perform (default: 1000).
#' @return A list containing three data frames: one for true feature importances, one for permuted importances, and one containing the top features (filtered non-zero true importances).
#' @examples
#' feat_importances <- pyRforest::calculate_feature_importances(fitting_results$model,processed_training_data$X_training_mat,processed_training_data$y_training_vector,n_permutations=1000)
#'
#' Print the fitting results, provides accuracy, f1 score, precision, recall and roc_auc scores on the model as fitted to the validation set
#' print(top_features)
#' @import dplyr
#' @import reticulate
#' @export

calculate_feature_importances <- function(model, X_train, y_train, n_permutations = 1000) {
  # Import necessary Python modules using reticulate
  python_pkgs<-setup_python_pkgs()

  # Fit the model on the training set
  model$fit(X_train, y_train)

  # Get true feature importances and convert to R vector
  true_feature_importances <- reticulate::py_to_r(model$feature_importances_)
  feature_names <- colnames(X_train)

  # Create a data frame with feature names and their importances
  true_feature_importances_df <- data.frame(
    feature = feature_names,
    importance = true_feature_importances
  ) %>% dplyr::arrange(desc(importance))

  # Function to convert true feature importances into the required format
  convert_true_importances <- function(df) {
    df <- df %>%
      dplyr::mutate(
        feature_rank = 1:nrow(df),
        permutation = 0,
        log_feature_importance = log(importance)
      ) %>%
      dplyr::select(feature_rank, feature_importance = importance, log_feature_importance, permutation, feature)
    return(df)
  }

  # Convert true feature importances
  true_feature_importances_data <- convert_true_importances(true_feature_importances_df)

  # Create a list to store permuted models' importances
  permuted_importances <- lapply(1:n_permutations, function(i) {
    permuted_y <- sample(y_train)
    model$fit(X_train, permuted_y)
    data.frame(
      feature_importance = reticulate::py_to_r(model$feature_importances_),
      permutation = i
    ) %>%
      dplyr::arrange(desc(feature_importance)) %>%
      dplyr::mutate(
        feature_rank = 1:nrow(.),
        log_feature_importance = log(feature_importance)
      )  %>%
      dplyr::select(feature_rank, feature_importance, log_feature_importance, permutation)
  })

  # Combine all permuted importances into a single tibble
  permuted_importances_data <- bind_rows(permuted_importances)

  # Calculate top true feature importances
  filter_non_zero_importance <- function(df) {
    df <- df[df$feature_importance != 0, ]  # Filter non-zero importance rows
    return(df)
  }

  filtered_true_feature_importances <- filter_non_zero_importance(true_feature_importances_data)

  # Return a list with true and permuted importances
  return(list(
    true_importances = true_feature_importances_data,
    permuted_importances = permuted_importances_data,
    top_features = filtered_true_feature_importances
  ))
}
