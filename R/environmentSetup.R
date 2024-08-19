#' Centralized Setup for Python Environment
#'
#' This function sets up the Python environment for the package. It imports essential Python
#' libraries using the `reticulate` package. This setup is used across various functions in the
#' package that require Python interoperability. This package depends on Python 3.9.18 being installed and the conda environment being loaded.
#'
#' @details
#' The function imports the following Python libraries:
#' \itemize{
#'   \item \code{sklearn}: Comprehensive machine learning library in Python.
#'   \item \code{numpy}: Fundamental package for scientific computing in Python.
#'   \item \code{shap}: Library for SHAP (SHapley Additive exPlanations) values, which explain the output of machine learning models.
#'   \item \code{skopt}: Package for optimizing hyperparameter tuning includes BayesSearchCV.
#'   \item \code{memprof}: Package for monitoring python memory usage.
#' }
#' These libraries are essential for various machine learning tasks and analyses within the package.
#' This centralized setup ensures consistency and reduces redundancy.
#'
#' @return
#' A list containing imported Python modules: \code{sklearn}, \code{numpy}, \code{shap}, \code{skopt}, and \code{memprof}.
#'
#' @examples
#' \dontrun{
#' python_pkgs <- setup_python_pkgs()
#' }
#' @import reticulate
#' @export
setup_python_pkgs <- function() {
  require(reticulate)
  sklearn <- reticulate::import("sklearn", convert = FALSE)
  numpy <- reticulate::import("numpy", convert = FALSE)
  shap <- reticulate::import("shap", convert = FALSE)
  skopt <- reticulate::import("skopt", convert=FALSE) #scikit-optimize
  memprof <- reticulate::import("memory_profiler", convert=FALSE)
  return(list(sklearn = sklearn, numpy = numpy, shap = shap, skopt = skopt, memprof = memprof))
}
