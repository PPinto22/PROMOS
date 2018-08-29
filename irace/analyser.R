# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

load('hyperneat_hyperparams/output/irace.Rdata')

results <- iraceResults$testing$experiments
# results <- results[, seq(1, 4)]
results <- apply(results, c(1,2), '*', -1) # Flip from negative to positive
colnames(results) <- paste("Config", seq(1, ncol(results)))

conf <- gl(ncol(results), nrow(results), labels=colnames(results))

sink('ttest.txt')
pairwise.t.test(as.vector(results), conf,  paired=TRUE)
sink()

png(filename = 'test_bp.png')
boxplot(results, ylab='Fitness')
dev.off()