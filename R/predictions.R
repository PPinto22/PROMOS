# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(data.table)
library(scales)
library(pROC)
source('util.R')

# CONFIG
PREDS_FILE <- '../results/predictions/2wks_pcp.csv'
OUT_DIR <- add_trailing_slash('out/2wks_pcp/predictions')

# SETUP
preds_dt <- data.table(read.csv(file=PREDS_FILE, header=TRUE, sep=','))
# Target frequency
target_freq <- data.table(prop.table(table(preds_dt$target)))
colnames(target_freq) <- c("target", "frequency")

# OUTPUTS
# Create OUT_DIR
dir.create(file.path(OUT_DIR), recursive=TRUE, showWarnings=FALSE)

# AUC
sink(file = paste(OUT_DIR, 'auc.txt', sep=''))
roc_obj <- roc(preds_dt$target, preds_dt$prediction)
print(auc(roc_obj))
sink()

# Target histogram
png(filename = paste(OUT_DIR, 'target_pie.png', sep=''))
gg_target_pie <- ggplot(target_freq, aes(x="", y=frequency, fill=target)) +
  geom_bar(stat = "identity", width = 1) +
  scale_fill_brewer(palette = 'Oranges') +
  coord_polar("y", start=0) +
  theme_void() +
  theme(axis.text.x=element_blank()) +
  labs(x=NULL, y=NULL, fill='Target') +
  geom_text(aes(label=percent(frequency)), position=position_fill(vjust = 0.5))
print(gg_target_pie)
dev.off()

# Prediction histogram
png(filename = paste(OUT_DIR, 'predictions_hist.png', sep=''))
gg_pred_hist <- ggplot(preds_dt, aes(x=prediction)) +
  geom_histogram() +
  labs(x='Prediction', y='Frequency') +
  theme_minimal()
print(gg_pred_hist)
dev.off()
