library(optparse)

option_list = list(
  make_option(c("-w","--workingdir"), type="character", default=".", help="working directory", metavar="DIR"),
  make_option(c("-s","--sales"), type="character", default="", help="path where to save the extracted sales", metavar="FILE"),
  make_option(c("-r","--redirects"), type="character", default="", help="path where to save the extracted redirects", metavar="FILE"),
  make_option(c("-t","--time"), type="double", default = 6, help="collection time duration in hours", metavar = "H")
);
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

dir.create(file.path(opt$workingdir), recursive=TRUE, showWarnings=FALSE)
setwd(opt$workingdir)

opt$sales