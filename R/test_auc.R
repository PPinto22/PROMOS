data <- read.csv(file='data.csv', header=TRUE, sep=';')
targets <- data$target
data$target <- NULL

sigmoid_unsigned <- function(x, slope=1, shift=0){
  return (1.0 / (1.0 + exp( -slope*x - shift)))
}

predict <- function(input, weights, bias){
  activation = sum(input*weights)
  activation = activation + bias
  output = sigmoid_unsigned(activation)
  return(output)
}

# Pesos e bias (do nodo de output) encontrados pelo NEAT, que resultam numa AUC de ~0.84
weights = c(1.141573492437601,
            -0.8214713073102757,
            -0.2551822494715452,
            -0.3023685598745942,
            -0.1999088996089995,
            -1.7316514905542135,
            2.5767266862094402,
            0.8350600260309875,
            2.111528931185603,
            0.8204303244128823)
bias=-0.2663601029198617

predictions <- apply(data, 1, predict, weights=weights, bias=bias)
#png(filename = "predictions_hist.png")
hist(predictions)
#dev.off()

# install.packages('ROSE')
library(ROSE)
#png(filename = "roc_curve.png")
roc <- roc.curve(targets, predictions, n.thresholds=100)
legend("bottomright", sprintf("AUC = %f", roc$auc), lty=1, lwd=2)
#dev.off()
