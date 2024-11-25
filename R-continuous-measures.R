# -*- coding: utf-8 -*-

#' Created on Fri Nov 17 17:56:48 2024
#' continuous-jaccard - Precision, Recall, F-score, and Jaccard Index for continuous, ratio-scale measurements
#' 
#' An approach to extend commonly used agreement measures estimated from a confusion matrix to non-negative ratio-scale attributes.
#' Useful for comparing agreement of gridded magnitude estimates with bounded, dimensionless measures.
#' 
#' related publication: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4865121
#' @author: Katarzyna Krasnodębska, Martino Pesaresi

# Define functions
RMSE <- function(pred, ref) {
  # Root Mean Square Error
  if (length(pred) != length(ref)) {
    stop(paste("Arrays must have the same shape. Found", length(pred), "and", length(ref)))
  }
  sqrt(mean((pred - ref)^2))
}

MAE <- function(pred, ref) {
  # Mean Absolute Error
  if (length(pred) != length(ref)) {
    stop(paste("Arrays must have the same shape. Found", length(pred), "and", length(ref)))
  }
  mean(abs(pred - ref))
}

ME <- function(pred, ref) {
  # Mean Error
  if (length(pred) != length(ref)) {
    stop(paste("Arrays must have the same shape. Found", length(pred), "and", length(ref)))
  }
  mean(pred) - mean(ref)
}

MAPE <- function(pred, ref) {
  # Mean Absolute Percentage Error
  if (length(pred) != length(ref)) {
    stop(paste("Arrays must have the same shape. Found", length(pred), "and", length(ref)))
  }
  pred <- round(pred, 4)
  ref <- round(ref, 4)
  pred_mask <- ifelse(ref > 0, pred, NA)
  ref_mask <- ifelse(ref > 0, ref, NA)
  mape <- mean(abs((pred_mask - ref_mask) / ref_mask), na.rm = TRUE) * 100
  return(mape)
}

rho <- function(pred, ref) {
  # Pearson's correlation coefficient
  if (length(pred) != length(ref)) {
    stop(paste("Arrays must have the same shape. Found", length(pred), "and", length(ref)))
  }
  cor(c(pred), c(ref), method = "pearson")
}

Slope <- function(pred, ref) {
  # Slope of the regression line
  if (length(pred) != length(ref)) {
    stop(paste("Arrays must have the same shape. Found", length(pred), "and", length(ref)))
  }
  fit <- lm(ref ~ pred)
  coef(fit)[2]
}

contJaccard <- function(pred, ref) {
  # Continuous Jaccard
  if (length(pred) != length(ref)) {
    stop(paste("Arrays must have the same shape. Found", length(pred), "and", length(ref)))
  }
  sum(pmin(pred, ref)) / sum(pmax(pred, ref))
}

contRecall <- function(pred, ref) {
  # Continuous recall
  if (length(pred) != length(ref)) {
    stop(paste("Arrays must have the same shape. Found", length(pred), "and", length(ref)))
  }
  sum(pmin(pred, ref)) / sum(ref)
}

contPrecision <- function(pred, ref) {
  # Continuous precision
  if (length(pred) != length(ref)) {
    stop(paste("Arrays must have the same shape. Found", length(pred), "and", length(ref)))
  }
  sum(pmin(pred, ref)) / sum(pred)
}

fscore <- function(precision, recall, beta) {
  if (precision == 0 && recall == 0) {
    return(NaN)
  } else {
    (1 + beta^2) * precision * recall / (beta^2 * precision + recall)
  }
}

# Generate reference and modeled arrays with continuous, ratio-scale, non-negative measurements
N <- 5
# set.seed(123)  # For reproducibility
example <- list(
  reference = matrix(round(runif(N * N), 2), nrow = N) * 10,
  modelled = matrix(round(runif(N * N), 2), nrow = N) * 10
)

# Plot reference and modelled arrays
par(mfrow = c(1, 2), mar = c(2,2,2,2))

for (data_name in names(example)) {
  data <- example[[data_name]]
  image(1:N, 1:N, data, col = rev(heat.colors(10)), axes = FALSE, 
        xlab = "", ylab="", main = paste(data_name, "data"))
  
  # Add text labels
  for (x in 1:N) {
    for (y in 1:N) {
      text(x, y, labels = round(data[y, x], 2), cex = 0.8, col = "black")
    }
  }
}

# Compare magnitude estimations  ----------------------------------------------------
# Print measures of error
cat(sprintf("ME: %.3f\n", ME(example$modelled, example$reference)))
cat(sprintf("MAE: %.3f\n", MAE(example$modelled, example$reference)))
# Print measures of association
cat(sprintf("r: %.3f\n", rho(example$modelled, example$reference)))
cat(sprintf("Slope: %.3f\n", Slope(example$modelled, example$reference)))
# Print measures of agreement
cont_Jaccard <- contJaccard(example$modelled, example$reference)
cont_Precision <- contPrecision(example$modelled, example$reference)
cont_Recall <- contRecall(example$modelled, example$reference)
cat(sprintf("cont. Jaccard: %.3f\n", cont_Jaccard))
cat(sprintf("cont. Precision: %.3f\n", cont_Precision))
cat(sprintf("cont. Recall: %.3f\n", cont_Recall))
cat(sprintf("cont. F1-score: %.3f\n", fscore(cont_Precision, cont_Recall, beta = 1)))
