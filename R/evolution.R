#install.packages("rstudioapi")
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#install.packages("data.table")
library(data.table)
dt <- data.table(read.csv(file='../results/hyperneat_BASELINE_evaluations.csv', header=TRUE, sep=','))

evolution <- dt[ , .(fitness.mean = mean(fitness), fitness.max = max(fitness),
                     neurons.mean = mean(neurons), neurons.max = max(neurons),
                     connections.mean = mean(connections), connections.max = max(connections),
                     time = mean(run_minutes)), by = generation]

#install.packages("ggplot2")
library(ggplot2)

# Plot fitness over time
ggplot(evolution, aes(time)) + 
  geom_line(aes(y=fitness.mean, col = 'mean')) +
  #  geom_smooth(aes(y=fitness.mean, col = 'mean')) + 
  geom_line(aes(y=fitness.max, col = 'best')) +
  labs(x="Run time (min)", y="Fitness (AUC)", colour="") +
  scale_y_continuous(breaks=seq(0.5,1,0.05))

# Plot complexity (neurons/connections) over time
ggplot(evolution, aes(time)) + 
  geom_line(aes(y = neurons.mean, col = "neurons")) + 
  geom_line(aes(y = connections.mean, col = "connections")) +
  labs(x="Run time (min)", y="Mean network complexity", colour="")

# Plot complexity (connections) over time
ggplot(evolution, aes(time)) + 
  geom_line(aes(y = connections.mean)) +
  labs(x="Run time (min)", y="Mean network connections", colour="")

