df <- read.csv(file='../results/neat_2018-05-01_18:57:40_evaluations.csv', header=TRUE, sep=',')

gens_max <- aggregate(. ~ generation, data = df, max)

#install.packages("ggplot2")
library(ggplot2)

# Plot fitness (neurons/connections) over generations
ggplot(gens_max, aes(generation, fitness)) + 
  geom_line() +
  labs(x="Generation", y="Fitness (AUC)")

# Plot complexity (neurons/connections) over generations
ggplot(gens_max, aes(generation)) + 
  geom_line(aes(y = neurons, col = "neurons")) + 
  geom_line(aes(y = connections, col = "connections")) +
  labs(x="Generation", y="Network Complexity", colour="")
