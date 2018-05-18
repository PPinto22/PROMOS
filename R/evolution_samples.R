#install.packages("rstudioapi")
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#install.packages("data.table")
library(data.table)
dt1 <- data.table(read.csv(file='../results/NEAT/samples/neat_ALL_evaluations.csv', header=TRUE, sep=','))
dt2 <- data.table(read.csv(file='../results/NEAT/samples/neat_10K_evaluations.csv', header=TRUE, sep=','))
dt3 <- data.table(read.csv(file='../results/NEAT/samples/neat_1K_evaluations.csv', header=TRUE, sep=','))
dt4 <- data.table(read.csv(file='../results/NEAT/samples/neat_100_evaluations.csv', header=TRUE, sep=','))

all <- dt1[ , .(fitness.mean = mean(fitness), fitness.max = max(fitness), 
                neurons.mean = mean(neurons), neurons.max = max(neurons),
                connections.mean = mean(connections), connections.max = max(connections),
                time = mean(run_minutes)), by = generation]
sample10K <- dt2[ , .(fitness.mean = mean(fitness), fitness.max = max(fitness),
                      neurons.mean = mean(neurons), neurons.max = max(neurons),
                      connections.mean = mean(connections), connections.max = max(connections),
                      time = mean(run_minutes)), by = generation]

sample1K <- dt3[ , .(fitness.mean = mean(fitness), fitness.max = max(fitness), 
                     neurons.mean = mean(neurons), neurons.max = max(neurons),
                     connections.mean = mean(connections), connections.max = max(connections),
                     time = mean(run_minutes)), by = generation]
sample100 <- dt4[ , .(fitness.mean = mean(fitness), fitness.max = max(fitness),
                      neurons.mean = mean(neurons), neurons.max = max(neurons),
                      connections.mean = mean(connections), connections.max = max(connections),
                      time = mean(run_minutes)), by = generation]

#install.packages("ggplot2")
library(ggplot2)
# install.packages("ggpubr")
library(ggpubr)

# Plot mean fitness over time
png(filename = "img/neat_samples_meanfitness.png")
gg.meanfit.all <- ggplot(data=all, aes(time, y=fitness.mean)) + geom_line() +
  labs(x="Run time (min)", y="Mean fitness (AUC)") +
  scale_y_continuous(breaks=seq(0.5,1,0.05))
gg.meanfit.10K <- ggplot(data=sample10K, aes(time, y=fitness.mean)) + geom_line() +
  labs(x="Run time (min)", y="Mean fitness (AUC)") +
  scale_y_continuous(breaks=seq(0.5,1,0.05))
gg.meanfit.1K <- ggplot(data=sample1K, aes(time, y=fitness.mean)) + geom_line() +
  labs(x="Run time (min)", y="Mean fitness (AUC)") +
  scale_y_continuous(breaks=seq(0.5,1,0.05))
gg.meanfit.100 <- ggplot(data=sample100, aes(time, y=fitness.mean)) + geom_line() +
  labs(x="Run time (min)", y="Mean fitness (AUC)") +
  scale_y_continuous(breaks=seq(0.5,1,0.05))
ggarrange(gg.meanfit.all, gg.meanfit.10K, gg.meanfit.1K, gg.meanfit.100, 
          labels=c('All (165K)', '10 000', '1 000', '100'), ncol=2, nrow=2)
dev.off()

# Plot max fitness over time
png(filename = "img/neat_samples_maxfitness.png")
ggplot(data=all, aes(time)) + 
  geom_smooth(data=all, aes(y=fitness.max, col = 'All (165K)'), method='loess') +
  geom_smooth(data=sample10K, aes(y=fitness.max, col = '10 000'), method='loess') +
  geom_smooth(data=sample1K, aes(y=fitness.max, col = '1 000'), method='loess') +
  geom_smooth(data=sample100, aes(y=fitness.max, col = '100'), method='loess') +
  labs(x="Run time (min)", y="Max fitness (AUC)", colour="Sample size") +
  scale_y_continuous(breaks=seq(0.5,1,0.05))
dev.off()

# Plot generations over time
png(filename = "img/neat_samples_generations.png")
ggplot(data=all, aes(time)) +
  geom_smooth(data=all, aes(y=generation, col = 'All (165K)'), method='loess') +
  geom_smooth(data=sample10K, aes(y=generation, col = '10 000'), method='loess') +
  geom_smooth(data=sample1K, aes(y=generation, col = '1 000'), method='loess') +
  geom_smooth(data=sample100, aes(y=generation, col = '100'), method='loess') +
  labs(x="Run time (min)", y="Generations", colour="Sample size")
dev.off()

# Plot complexity over time
png(filename = "img/neat_samples_meancomplexity.png")
ggplot(data=all, aes(time)) +
  geom_line(data=all, aes(y = connections.mean, col = "All (165K)")) +
  geom_line(data=sample10K, aes(y = connections.mean, col = "10 000")) +
  geom_line(data=sample1K, aes(y = connections.mean, col = "1 000")) +
  geom_line(data=sample100, aes(y = connections.mean, col = "100")) +
  labs(x="Run time (min)", y="Mean network connections", colour="")
dev.off()