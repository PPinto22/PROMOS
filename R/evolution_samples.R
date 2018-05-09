#install.packages("rstudioapi")
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#install.packages("data.table")
library(data.table)
dt1 <- data.table(read.csv(file='../results/neat_BASELINE_evaluations.csv', header=TRUE, sep=','))
dt2 <- data.table(read.csv(file='../results/neat_SAMPLE10K_evaluations.csv', header=TRUE, sep=','))
dt3 <- data.table(read.csv(file='../results/neat_SAMPLE1000_evaluations.csv', header=TRUE, sep=','))
dt4 <- data.table(read.csv(file='../results/neat_SAMPLE100_evaluations.csv', header=TRUE, sep=','))

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

# Plot mean fitness over time
png(filename = "img/neat_samples_meanfitness.png")
ggplot(data=all, aes(time)) + 
  geom_smooth(data=all, aes(y=fitness.mean, col = 'All (235K)'), method='loess') +
  geom_smooth(data=sample10K, aes(y=fitness.mean, col = '10 000'), method='loess') +
  geom_smooth(data=sample1K, aes(y=fitness.mean, col = '1 000'), method='loess') +
  geom_smooth(data=sample100, aes(y=fitness.mean, col = '100'), method='loess') +
  labs(x="Run time (log10(min))", y="Mean fitness (AUC)", colour="Sample size") +
  scale_y_continuous(breaks=seq(0.5,1,0.05)) + 
  scale_x_log10(breaks=c(0,1,10,100,1000))
dev.off()

png(filename = "img/neat_samples_maxfitness.png")
ggplot(data=all, aes(time)) + 
  geom_smooth(data=all, aes(y=fitness.max, col = 'All (235K)'), method='loess') +
  geom_smooth(data=sample10K, aes(y=fitness.max, col = '10 000'), method='loess') +
  geom_smooth(data=sample1K, aes(y=fitness.max, col = '1 000'), method='loess') +
  geom_smooth(data=sample100, aes(y=fitness.max, col = '100'), method='loess') +
  labs(x="Run time (log10(min))", y="Max fitness (AUC)", colour="Sample size") +
  scale_y_continuous(breaks=seq(0.5,1,0.05)) + 
  scale_x_log10(breaks=c(0,1,10,100,1000))
dev.off()
                