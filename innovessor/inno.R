install.packages("readxl")
library("readxl")
all <- read_excel("problem_dataset.xlsx",sheet = "Data")

#Loading R packages.
library(knitr)
library(ggplot2)
library(plyr)
library(dplyr)
library(corrplot)
library(caret)
library(gridExtra)
library(scales)
library(Rmisc)
library(ggrepel)
library(randomForest)
library(psych)
library(xgboost)

#Getting rid of the IDs but keeping the test IDs in a vector. 
data_labels <- cbind(all$aco_num,all$aco_name)
all$aco_num <-NULL
all$aco_name <- NULL
dim(all)
all=as.data.frame(all)

#Exploring some of the most important variables
#-----------------
#Missing data
miss_col <- which(colSums(is.na(all)) > 0)
sort(colSums(sapply(all[miss_col], is.na)), decreasing = TRUE)
cat('There are', length(miss_col), 'columns with missing values')
#since missing data for continous variable very less, better to put zero

summary(all$n_ben_race_native)
summary(all$n_ben_race_asian)
summary(all$n_ben_race_hisp)

ggplot(data=all[!is.na(all$n_ben_race_native),], aes(x=n_ben_race_native)) +
  geom_histogram(fill="blue", binwidth = 20) +
  scale_x_continuous(breaks= seq(-50, 1000, by=20), labels = comma)
#As you can see, the total no of assigned beneficiaries are right skewed.
summary(all)
ggplot(data=all[!is.na(all$n_ben_race_asian),], aes(x=n_ben_race_asian)) +
  geom_histogram(fill="blue", binwidth = 20) +
  scale_x_continuous(breaks= seq(-50, 10000, by=20), labels = comma)

ggplot(data=all[!is.na(all$n_ben_race_hisp),], aes(x=n_ben_race_hisp)) +
  geom_histogram(fill="blue", binwidth = 20) +
  scale_x_continuous(breaks= seq(-50, 3000, by=20), labels = comma)
#replacing missing value to 0
all$n_ben_race_native[which(is.na(all$n_ben_race_native))] <- 0
all$n_ben_race_asian[which(is.na(all$n_ben_race_asian))] <- 0
all$n_ben_race_hisp[which(is.na(all$n_ben_race_hisp))] <- 0
#------------------
summary(lapply(apply(all, 1, function(x)which(x == 'NA')), names))
indx <- grepl('NA', colnames(all))
all[indx]
min(all$aco19)

all[(all$per_capita_exp_total_py)== "NA"] <-NULL
all$per_capita_exp_total_py[is.na(all$per_capita_exp_total_py)] <- "None"
df[df == 0] <- NA
all[all$per_capita_exp_total_py == "NA"] <-0
#---------------------
ggplot(data=all[!is.na(all$per_capita_exp_total_py),], aes(x=per_capita_exp_total_py)) +
  geom_histogram(fill="blue", binwidth = 10000) +
  scale_x_continuous(breaks= seq(0, 800000, by=100000), labels = comma)
