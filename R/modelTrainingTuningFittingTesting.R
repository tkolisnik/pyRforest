#' Perform hyperparameter tuning and training for RandomForestClassifier
#'
#' This function uses scikit-learn's python based GridSearchCV to perform hyperparameter tuning and training of a RandomForestClassifier.
#' It supports binary and categorical multiclass classification and it allows for customizable parameter grids and includes preprocessing steps of one-hot encoding
#' and scaling. The function is designed to find the best hyperparameters using a brute force search. Please reference the scikit-learn GridSearchCV documentation for the full description of options, however our defaults are comprehensive.
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
#' @examplesIf FALSE
#' library(pyRforest)
#' library(reticulate)
#'
#' # Load conda environment, which ensures the correct version of Python and the necessary python packages can be loaded. See vignette for more details.
#' use_condaenv("pyRforest-conda", conda = "conda path see vignette", required = TRUE)
#'
#' # Load the demo data
#' data(demo_rnaseq_data)
#'
#' # Prepare the sample data into a format ingestible by the ML algorithm
#' processed_training_data <- pyRforest::create_feature_matrix(dataset = demo_rnaseq_data$training_data,
#'     set_type = "training")
#'
#' # Model tuning and training (Warning: may take a long time if dataset is large and if param_grid has many options)
#' tuning_results <- pyRforest::tune_and_train_rf_model_grid(
#'     X = processed_training_data$X_training_mat,
#'     y = processed_training_data$y_training_vector,
#'     cv_folds = 5,
#'    scoring = 'roc_auc', #use 'roc_auc_ovr' for multiclass targets
#'     seed = 123,
#'     param_grid = custom_parameter_grid,
#'     n_jobs = 1,
#'     n_cores = -2)

#' print(tuning_results$best_params)
#' print(tuning_results$best_score_)
#' @export

tune_and_train_rf_model_grid <- function(X, y, cv_folds = 5, scoring_method = "roc_auc",  seed = 4, param_grid = NULL, n_jobs=1, n_cores=-2) {

  # Import necessary Python modules using reticulate
  python_pkgs<-setup_python_pkgs()

  # Import specific sklearn submodules
  GridSearchCV <- python_pkgs$sklearn$model_selection$GridSearchCV
  RandomizedSearchCV <- python_pkgs$sklearn$model_selection$RandomizedSearchCV
  OneHotEncoder <- python_pkgs$sklearn$preprocessing$OneHotEncoder
  StandardScaler <- python_pkgs$sklearn$preprocessing$StandardScaler
  StratifiedKFold <- python_pkgs$sklearn$model_selection$StratifiedKFold
  RandomForestClassifier <- python_pkgs$sklearn$ensemble$RandomForestClassifier
  memory_profiler <- python_pkgs$memprof$memory_usage

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
      stop("param_grid must be a list of parameter lists. Add capital L after all integers e.g. min_samples_leaf = list(2L, 3L, 4L, 5L, 10L) ")
    }
  }

  # Modify scoring method for multiclass roc_auc
  if (scoring_method == "roc_auc" && length(unique(y)) > 2) {
    scoring_method <- "roc_auc_ovr"  # Use the multiclass version
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
  train_model <- function() {
    grid_search$fit(X_scaled, y)
  }

  mem_usage <- memory_profiler(train_model)
  mem_usage <- py_to_r(mem_usage)
  #mem_usage_mb <- sapply(mem_usage, function(x) x * 1.048576)
  avg_mem_usage_mib <- mean(mem_usage)

  # Return the best parameters and the GridSearchCV object
  return(list(best_params = grid_search$best_params_, best_score=grid_search$best_score_, grid_search = grid_search, memory_usage = avg_mem_usage_mib))
}

#' Tune and train RandomForest model using Bayesian optimization
#'
#' This function uses scikit-learn's python-based `BayesSearchCV` from the `skopt` library
#' to perform hyperparameter tuning and training of a `RandomForestClassifier`.
#' It leverages Bayesian optimization for a more efficient hyperparameter search,
#' allowing more optimal parameter configurations to be discovered faster and with less compute time when compared to brute-force methods like GridSearchCV.
#' It supports binary and categorical multiclass classification, and the function includes preprocessing steps of one-hot encoding and scaling for compatibility with the random forest algorithm.
#' Please reference the scikit-learn `BayesSearchCV` documentation for more details on customization.
#'
#' @param X Training features (data frame or matrix). Typically obtained from the `create_feature_matrix` function.
#' @param y Target vector for the model. Usually obtained from the `create_feature_matrix` function.
#' @param cv_folds The number of cross-validation splits in `StratifiedKFold` (default: 5).
#' @param scoring_method The scoring method to be used during optimization (default: 'roc_auc'). Options include 'accuracy', 'precision', 'recall', 'f1', etc. See scikit-learn's `BayesSearchCV` documentation for a full list of scoring methods.
#' @param seed An optional random seed for reproducibility (default: 4).
#' @param param_grid An optional list of hyperparameters for Bayesian optimization.
#'                   If NULL, a default grid will be used. The list should follow the format expected by `BayesSearchCV`.
#'                   The default `param_grid` includes:
#'                   \itemize{
#'                     \item \code{bootstrap} (list): [TRUE]
#'                     \item \code{class_weight} (list): [NULL]
#'                     \item \code{max_depth} (list): [5L, 10L, 15L, 20L, NULL]
#'                     \item \code{n_estimators} (integer sequence): [50, 75, 100, 125, 150, 175, 200]
#'                     \item \code{max_features} (list): ["sqrt", "log2", 0.1, 0.2]
#'                     \item \code{criterion} (list): ["gini"]
#'                     \item \code{warm_start} (list): [FALSE]
#'                     \item \code{min_samples_leaf} (list): [2L, 3L, 4L, 5L, 10L]
#'                     \item \code{min_samples_split} (list): [2L, 3L, 4L, 5L, 10L]
#'                   }
#' @param n_jobs The number of jobs to run in parallel during training. Default is 1.
#' @param n_cores The number of cores to use for parallel processing. Default is -2, meaning it will use all but 2 cores.
#' @return A list containing the best hyperparameters for the model, the cross-validation score, the trained `BayesSearchCV` object, and memory usage statistics.
#' @examplesIf FALSE
#' library(pyRforest)
#' library(reticulate)
#'
#' # Load the conda environment for the package dependencies
#' use_condaenv("pyRforest-conda", conda = "conda path see vignette", required = TRUE)
#'
#' # Load sample data
#'  data(demo_rnaseq_data)
#'
#' # Prepare the sample data into a format ingestible by the ML algorithm
#' processed_training_data <- pyRforest::create_feature_matrix(dataset = demo_rnaseq_data$training_data,
#'     set_type = "training")
#'
#' # Model training and tuning
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
#' @export

tune_and_train_rf_model_bayes <- function(X, y, cv_folds = 5, scoring_method = "roc_auc",  seed = 4, param_grid = NULL, n_jobs=1, n_cores=-2)  {
  # Import necessary Python modules using reticulate
  python_pkgs<-setup_python_pkgs()

  BayesSearchCV <- python_pkgs$skopt$BayesSearchCV
  OneHotEncoder <- python_pkgs$sklearn$preprocessing$OneHotEncoder
  StandardScaler <- python_pkgs$sklearn$preprocessing$StandardScaler
  RandomForestClassifier <- python_pkgs$sklearn$ensemble$RandomForestClassifier
  StratifiedKFold <- python_pkgs$sklearn$model_selection$StratifiedKFold
  memory_profiler <- python_pkgs$memprof$memory_usage
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
      stop("param_grid must be a list of parameter lists. Add capital L after all integers e.g. min_samples_leaf = list(2L, 3L, 4L, 5L, 10L) ")
    }
  }

  # Adjust scoring method if multiclass ROC AUC is needed
  if (scoring_method == "roc_auc" && length(unique(y)) > 2) {
    scoring_method <- "roc_auc_ovr"
  }

  # Create a stratified k-fold cross validation object
  strat_k_fold <- StratifiedKFold(n_splits = as.integer(cv_folds), shuffle = TRUE, random_state = as.integer(seed))

  # Initialize the BayesSearchCV object
  bayes_search <- BayesSearchCV(
    estimator = clf,
    search_spaces = param_grid,
    cv = strat_k_fold,
    scoring = scoring_method,
    verbose = as.integer(2),
    n_jobs = as.integer(n_cores)
  )

  # Fit the BayesSearchCV object to the data
  train_model <- function(){
    bayes_search$fit(X_scaled, y)
  }

  mem_usage <- memory_profiler(train_model)

  # Return the best parameters and the BayesSearchCV object
  return(list(best_params = bayes_search$best_params_, best_score = bayes_search$best_score_, bayes_search = bayes_search, memory_usage = mem_usage))

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
#' @examplesIf FALSE
#' fitting_results<-fit_and_evaluate_rf(tuning_results$best_params,
#'       processed_training_data$X_training_mat,
#'       processed_training_data$y_training_vector,
#'       processed_validation_data$X_validation_mat,
#'       processed_validation_data$y_validation_vector)
#'
#' # Print the fitting results, provides accuracy, f1 score, precision, recall and roc_auc scores on the model as fitted to the validation set
#' print(fitting_results)
#' @export

fit_and_evaluate_rf <- function(best_params, X_train, y_train, X_val = NULL, y_val = NULL, save_path = NULL, seed=4) {
  # Import necessary Python modules using reticulate
  python_pkgs <- setup_python_pkgs()

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

  # Helper function for printing full list of model parameters
  format_model_params <- function(params_list) {
    param_lines <- sapply(names(params_list), function(param_name) {
      value <- params_list[[param_name]]
      value_str <- ifelse(is.logical(value), ifelse(value, "TRUE", "FALSE"), toString(value))
      sprintf("%s=%s", param_name, value_str)
    })
    paste0("RandomForestClassifier(", paste(param_lines, collapse = ", "), ")")
  }

  full_model_parameters <- format_model_params(all_params)

  # Initialize an empty list to store results
  results <- list(model = final_model, all_params = full_model_parameters)

  # If validation data is provided, compute the metrics on the validation set
  if (!is.null(X_val) && !is.null(y_val)) {
    y_pred <- final_model$predict(X_val)
    y_pred_proba <- final_model$predict_proba(X_val)

    # Compute accuracy
    accuracy <- metrics$accuracy_score(y_val, y_pred)

    # Compute F1 score with 'micro' averaging
    f1 <- metrics$f1_score(y_val, y_pred, average = 'micro')

    # Compute Precision and Recall with 'micro' averaging
    precision <- metrics$precision_score(y_val, y_pred, average = 'micro')
    recall <- metrics$recall_score(y_val, y_pred, average = 'micro')

    # Compute ROC AUC Score
    if (length(unique(y_val)) == 2) {
      # Binary classification: extract probabilities for the positive class
      roc_auc <- metrics$roc_auc_score(y_val, y_pred_proba[, 1])
    } else {
      # Multiclass classification ROC calculation.
      roc_auc <- metrics$roc_auc_score(y_val, y_pred_proba, multi_class = "ovr")
      print("Note: For multiclass classification, the ROC AUC score is calculated using the one-vs-rest (OVR) approach.")
    }

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
#' @examplesIf FALSE
#' feat_importances <- pyRforest::calculate_feature_importances(fitting_results$model,processed_training_data$X_training_mat,processed_training_data$y_training_vector,n_permutations=1000)
#'
#' # Print the fitting results, provides accuracy, f1 score, precision, recall and roc_auc scores on the model as fitted to the validation set
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

