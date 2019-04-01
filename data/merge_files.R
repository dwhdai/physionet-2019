# Assuming script runs in home directory and files to merge are in ~/temp

library(dplyr)

outfile <- file("septrain.csv", "w")
flist <- list.files(path = "./temp", full.names = T)
allofit <- data_frame()
for (i in 1:length(flist)){
  # add patient nr to the end of each file and concatanate
  data <- read.csv(flist[i], sep = "|", na.strings = "NaN")
  data <- mutate(data, subject = substr(flist[i], nchar(flist[i])-9,nchar(flist[i])-4))
  allofit <- rbind(allofit, data)
}

write.table(allofit, outfile, row.names = F, quote = F, sep = "\t")

close(outfile)
rm(outfile)
