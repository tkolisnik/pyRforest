% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/environmentSetup.R
\name{setup_python_pkgs}
\alias{setup_python_pkgs}
\title{Centralized Setup for Python Environment}
\usage{
setup_python_pkgs()
}
\value{
A list containing imported Python modules: \code{sklearn}, \code{numpy}, and \code{shap}.
}
\description{
This function sets up the Python environment for the package. It imports essential Python
libraries using the \code{reticulate} package. This setup is used across various functions in the
package that require Python interoperability. This package depends on Python 3.9.18 being installed and the conda environment being loaded.
}
\details{
The function imports the following Python libraries:
\itemize{
\item \code{sklearn}: Comprehensive machine learning library in Python.
\item \code{numpy}: Fundamental package for scientific computing in Python.
\item \code{shap}: Library for SHAP (SHapley Additive exPlanations) values, which explain the output of machine learning models.
}
These libraries are essential for various machine learning tasks and analyses within the package.
This centralized setup ensures consistency and reduces redundancy.
}
\examples{
python_pkgs <- setup_python_pkgs()
# You can now use python_pkgs$sklearn, python_pkgs$numpy, and python_pkgs$shap in other functions
}
