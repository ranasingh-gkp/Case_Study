#House prices: Lasso, XGBoost, and a detailed EDA
#https://www.kaggle.com/erikbruin/house-prices-lasso-xgboost-and-a-detailed-eda
#----------------------------------------------
#install.packages("Rmisc")
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
train <- read.csv("train.csv", stringsAsFactors = F)
test <- read.csv("test.csv", stringsAsFactors = F)
#============Data size and structure
dim(train)
str(train[,c(1:10, 81)]) #display first 10 variables and the response variable
#Getting rid of the IDs but keeping the test IDs in a vector. These are needed to compose the submission file
test_labels <- test$Id
test$Id <- NULL
train$Id <- NULL

test$SalePrice <- NA
all <- rbind(train, test)
dim(all)
#Without the Id's, the dataframe consists of 79 predictors and our response variable SalePrice.

#===========Exploring some of the most important variables
#---The response variable; SalePrice
ggplot(data=all[!is.na(all$SalePrice),], aes(x=SalePrice)) +
  geom_histogram(fill="blue", binwidth = 10000) +
  scale_x_continuous(breaks= seq(0, 800000, by=100000), labels = comma)
#As you can see, the sale prices are right skewed. This was expected as few people can afford very expensive houses. I will keep this in mind, and take measures before modeling.
summary(all$SalePrice)
boxplot(all$SalePrice)
#-----The most important numeric predictors
#The character variables need some work before I can use them. To get a feel for the dataset, I decided to first see which numeric variables have a high correlation with the SalePrice.

#:::: Correlations with SalePrice
numericVars <- which(sapply(all, is.numeric)) #index vector numeric variables
numericVars
numericVarNames <- names(numericVars) #saving names vector for use later on
cat('There are', length(numericVars), 'numeric variables')

all_numVar <- all[, numericVars]
cor_numVar <- cor(all_numVar, use="pairwise.complete.obs") #correlations of all numeric variables
#sort on decreasing correlations with SalePrice
cor_sorted <- as.matrix(sort(cor_numVar[,'SalePrice'], decreasing = TRUE))
#select only high corelations
CorHigh <- names(which(apply(cor_sorted, 1, function(x) abs(x)>0.5)))
CorHigh
cor_numVar <- cor_numVar[CorHigh, CorHigh]
cor_numVar
#plot
corrplot.mixed(cor_numVar, tl.col="black", tl.pos = "lt")
#INFERENCE::
#Altogether, there are 10 numeric variables with a correlation of at least 0.5 with SalePrice. All those correlations are positive.
#It also becomes clear the multicollinearity is an issue. For example: the correlation between GarageCars and GarageArea is very high (0.89), and both have similar (high) correlations with SalePrice.

#::::Overall Quality
#Overall Quality has the highest correlation with SalePrice among the numeric variables (0.79). It rates the overall material and finish of the house on a scale from 1 (very poor) to 10 (very excellent).
ggplot(data=all[!is.na(all$SalePrice),], aes(x=factor(OverallQual), y=SalePrice))+
  geom_boxplot(col='blue') + labs(x='Overall Quality') +
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma)
#The positive correlation is certainly there indeed, and seems to be a slightly upward curve.
#Regarding outliers, I do not see any extreme values. If there is a candidate to take out as an outlier later on, it seems to be the expensive house with grade 4.

#::::Above Grade (Ground) Living Area (square feet)
#The numeric variable with the second highest correlation with SalesPrice is the Above Grade Living Area. This make a lot of sense; big houses are generally more expensive.
ggplot(data=all[!is.na(all$SalePrice),], aes(x=GrLivArea, y=SalePrice))+
  geom_point(col='blue') + geom_smooth(method = "lm", se=FALSE, color="black", aes(group=1)) +
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) +
  geom_text_repel(aes(label = ifelse(all$GrLivArea[!is.na(all$SalePrice)]>4500, rownames(all), '')))
#Especially the two houses with really big living areas and low SalePrices seem outliers (houses 524 and 1299, see labels in graph).
#For instance, a low score on the Overall Quality could explain a low price. However, as you can see below, these two houses actually also score maximum points on Overall Quality. 
#Therefore, I will keep houses 1299 and 524 in mind as prime candidates to take out as outliers.
all[c(524, 1299), c('SalePrice', 'GrLivArea', 'OverallQual')]
#///////////////============================================
#:::GarageCars
counts<- table(all$GarageCars)
barplot(counts, main="Car Distribution",xlab="Number of GarageCars")  #...names.arg=c("3 Gears", "4 Gears", "5 Gears"))

#:::GarageArea
ggplot(data=all[!is.na(all$SalePrice),], aes(x=GarageArea, y=SalePrice))+
  geom_point(col='blue') + geom_smooth(method = "lm", se=FALSE, color="black", aes(group=1)) +
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) +
  geom_text_repel(aes(label = ifelse(all$GarageArea[!is.na(all$SalePrice)]>1250, rownames(all), '')))+
  geom_text_repel(aes(label = ifelse(all$SalePrice[!is.na(all$SalePrice)]>600000, rownames(all), '')))
all[c(582, 1299,1191,1183,1170,692), c('SalePrice', 'GarageArea', 'OverallQual')]

#:::TotalBsmtSF
ggplot(data=all[!is.na(all$SalePrice),], aes(x=TotalBsmtSF, y=SalePrice))+
  geom_point(col='blue') + geom_smooth(method = "lm", se=FALSE, color="black", aes(group=1)) +
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) +
  geom_text_repel(aes(label = ifelse(all$TotalBsmtSF[!is.na(all$SalePrice)]>4000, rownames(all), '')))+
  geom_text_repel(aes(label = ifelse(all$SalePrice[!is.na(all$SalePrice)]>600000, rownames(all), '')))
all[c(1299,899,1183,1170,692), c('SalePrice', 'TotalBsmtSF', 'OverallQual')]

#:::X1stFlrSF-First Floor square feet
ggplot(data=all[!is.na(all$SalePrice),], aes(x=X1stFlrSF, y=SalePrice))+
  geom_point(col='blue') + geom_smooth(method = "lm", se=FALSE, color="black", aes(group=1)) +
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) +
  geom_text_repel(aes(label = ifelse(all$X1stFlrSF[!is.na(all$SalePrice)]>4000, rownames(all), '')))+
  geom_text_repel(aes(label = ifelse(all$SalePrice[!is.na(all$SalePrice)]>600000, rownames(all), '')))
all[c(1299,899,1183,1170,692), c('SalePrice', 'GarageArea', 'OverallQual')]

#:::FullBath-Full bathrooms above grade
ggplot(all[!is.na(all$SalePrice),], aes(x=FullBath, y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue') +
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..))

counts<- table(all$FullBath)
barplot(counts, main="Full bathrooms above grade",xlab="FullBath")  #...names.arg=c("3 Gears", "4 Gears", "5 Gears"))

#:::TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
ggplot(all[!is.na(all$SalePrice),], aes(x=TotRmsAbvGrd, y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue') +
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..))

counts<- table(all$TotRmsAbvGrd)
barplot(counts, main="Total rooms above grade (does not include bathrooms)",xlab="TotRmsAbvGrd")  #...names.arg=c("3 Gears", "4 Gears", "5 Gears"))

#:::YearBuilt: Original construction date
ggplot(all[!is.na(all$SalePrice),], aes(x=YearBuilt, y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue') +
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..))

counts<- table(all$YearBuilt)
barplot(counts, main=" Original construction date",xlab="YearBuilt")  #...names.arg=c("3 Gears", "4 Gears", "5 Gears"))

#:::YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)
counts<- table(all$YearRemodAdd)
barplot(counts, main=" Remodel date (same as construction date if no remodeling or additions)",xlab="YearRemodAdd")  #...names.arg=c("3 Gears", "4 Gears", "5 Gears"))

#/////////////////==============================================


#=====Missing data, label encoding, and factorizing variables
#---Completeness of the data
#First of all, I would like to see which variables contain missing values.
NAcol <- which(colSums(is.na(all)) > 0)
sort(colSums(sapply(all[NAcol], is.na)), decreasing = TRUE)
cat('There are', length(NAcol), 'columns with missing values')
#Of course, the 1459 NAs in SalePrice match the size of the test set perfectly. This means that I have to fix NAs in 34 predictor variables.

#---Imputing missing data
#converted character variables into ordinal integers if there is clear ordinality, or into factors if levels are categories without ordinality.
#convert these factors into numeric later on by using one-hot encoding (using the model.matrix function).

#>>Pool Quality and the PoolArea variable
#So, it is obvious that I need to just assign 'No Pool' to the NAs. Also, the high number of NAs makes sense as normally only a small proportion of houses have a pool.
all$PoolQC[is.na(all$PoolQC)] <- 'None'

counts<- table(all$PoolQC)
barplot(counts, main=" PoolQC",xlab="PoolQC")

all[!is.na(all$SalePrice),] %>% group_by(PoolQC) %>% summarise(median = median(SalePrice), counts=n())
#My conclusion is that the values  seem ordinal (Ex is best).

#It is also clear that I can label encode this variable as the values are ordinal. As there a multiple variables that use the same quality levels, I am going to create a vector that I can reuse later on.
Qualities <- c('None' = 0, 'Po' = 1, 'Fa' = 2, 'TA' = 3, 'Gd' = 4, 'Ex' = 5)
all$PoolQC<-as.integer(revalue(all$PoolQC, Qualities))
table(all$PoolQC)
#First, I checked if there was a clear relation between the PoolArea and the PoolQC. As I did not see a clear relation (bigger of smaller pools with better PoolQC),
# I am going to impute PoolQC values based on the Overall Quality of the houses (which is not very high for those 3 houses).
all[all$PoolArea>0 & all$PoolQC==0, c('PoolArea', 'PoolQC', 'OverallQual')]
all$PoolQC[2421] <- 2
all$PoolQC[2504] <- 3
all$PoolQC[2600] <- 2
#check type of pool area
counts<- table(all$PoolArea)
barplot(counts, main=" PoolArea",xlab="PoolArea")
    
#>>Miscellaneous feature
#segregate of MiscFeature or #table(all$MiscFeature)
counts<- table(all$MiscFeature)
barplot(counts, main=" MiscFeature",xlab="MiscFeature")
table(all$MiscFeature)

#As the values are not ordinal, I will convert MiscFeature into a factor. Values:
all$MiscFeature[is.na(all$MiscFeature)] <- 'None'
all$MiscFeature <- as.factor(all$MiscFeature)
ggplot(all[!is.na(all$SalePrice),], aes(x=MiscFeature, y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue') +
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..))
table(all$MiscFeature)
# while it makes a lot of sense that a house with a Tennis court is expensive, there is only one house with a tennis court in the training set.
# the variable seems irrelevant to me. Having a shed probably means 'no Garage', which would explain the lower sales price for Shed.

#>>alley
#segregate of MiscFeature or #table(all$MiscFeature)
counts<- table(all$Alley)
barplot(counts, main=" Alley",xlab="Alley")
table(all$Alley)
#As the values are not ordinal, I will convert Alley into a factor. Values:
all$Alley[is.na(all$Alley)] <- 'None'
all$Alley <- as.factor(all$Alley)

ggplot(all[!is.na(all$SalePrice),], aes(x=Alley, y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue')+
  scale_y_continuous(breaks= seq(0, 200000, by=50000), labels = comma)
table(all$Alley)

#>>Fence quality(confusion in ordinal)
#segregate of MiscFeature or #table(all$MiscFeature)
counts<- table(all$Fence)
barplot(counts, main=" Fence",xlab="Fence")
table(all$Fence)

#The values seem to be ordinal. Values:(categorical value in series)
all$Fence[is.na(all$Fence)] <- 'None'
table(all$Fence)
all[!is.na(all$SalePrice),] %>% group_by(Fence) %>% summarise(median = median(SalePrice), counts=n())
#My conclusion is that the values do not seem ordinal (no fence is best). Therefore, I will convert Fence into a factor
all$Fence <- as.factor(all$Fence)

#>>Fireplace quality
#segregate of MiscFeature or #table(all$MiscFeature)
counts<- table(all$FireplaceQu)
barplot(counts, main=" FireplaceQu",xlab="FireplaceQu")
table(all$FireplaceQu)
sum(table(all$FireplaceQu))

#The number of NAs in FireplaceQu matches the number of houses with 0 fireplaces. This means that I can safely replace the NAs in FireplaceQu with 'no fireplace'. 
#The values are ordinal, and I can use the Qualities vector that I have already created for the Pool Quality. Values:
all$FireplaceQu[is.na(all$FireplaceQu)] <- 'None'
all$FireplaceQu<-as.integer(revalue(all$FireplaceQu, Qualities))
table(all$FireplaceQu)
table(all$Fireplaces)
sum(table(all$Fireplaces))
#>>LotFrontage(Linear feet of street connected to property)(median)
#The most reasonable imputation seems to take the median per neigborhood.
ggplot(all[!is.na(all$LotFrontage),], aes(x=as.factor(Neighborhood), y=LotFrontage)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue') +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
for (i in 1:nrow(all)){
  if(is.na(all$LotFrontage[i])){
    all$LotFrontage[i] <- as.integer(median(all$LotFrontage[all$Neighborhood==all$Neighborhood[i]], na.rm=TRUE)) 
  }
}

#>>LotShape (General shape of property)
#segregate of MiscFeature or #table(all$MiscFeature)
counts<- table(all$LotShape)
barplot(counts, main=" LotShape",xlab="LotShape")
table(all$LotShape)
sum(table(all$LotShape))
all[!is.na(all$SalePrice),] %>% group_by(LotShape) %>% summarise(median = median(SalePrice), counts=n())
# Values seem ordinal (Regular=best)
all$LotShape<-as.integer(revalue(all$LotShape, c('IR3'=0, 'IR2'=1, 'IR1'=2, 'Reg'=3)))
table(all$LotShape)
sum(table(all$LotShape))

#>>>LotConfig (Lot configuration)(ordinal doubt)
#segregate of MiscFeature or #table(all$MiscFeature)
counts<- table(all$LotConfig)
barplot(counts, main=" LotConfig",xlab="LotConfig")
table(all$LotConfig)
sum(table(all$LotConfig))
all[!is.na(all$SalePrice),] %>% group_by(LotConfig) %>% summarise(median = median(SalePrice), counts=n())

#The values seemed possibly ordinal to me, but the visualization does not show this. Therefore, I will convert the variable into a factor.
ggplot(all[!is.na(all$SalePrice),], aes(x=as.factor(LotConfig), y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue')+
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..))

all$LotConfig <- as.factor(all$LotConfig)
table(all$LotConfig)
sum(table(all$LotConfig))

#>>Garage variable
#Two of those have one NA (GarageCars and GarageArea), one has 157 NAs (GarageType), 4 variables have 159 NAs.
#First of all, I am going to replace all 159 missing GarageYrBlt: Year garage was built values with the values in YearBuilt (this is similar to YearRemodAdd, which also defaults to YearBuilt if no remodeling or additions).
all$GarageYrBlt[is.na(all$GarageYrBlt)] <- all$YearBuilt[is.na(all$GarageYrBlt)]

#check if all 157 NAs are the same observations among the variables with 157/159 NAs
length(which(is.na(all$GarageType) & is.na(all$GarageFinish) & is.na(all$GarageCond) & is.na(all$GarageQual)))
#Find the 2 additional NAs
kable(all[!is.na(all$GarageType) & is.na(all$GarageFinish), c('GarageCars', 'GarageArea', 'GarageType', 'GarageCond', 'GarageQual', 'GarageFinish')])
# As you can see, house 2127 actually does seem to have a Garage and house 2577 does not. 
#Therefore, there should be 158 houses without a Garage. To fix house 2127, I will imputate the most common values (modes) for GarageCond, GarageQual, and GarageFinish.
#Imputing modes.
all$GarageCond[2127] <- names(sort(-table(all$GarageCond)))[1]
all$GarageQual[2127] <- names(sort(-table(all$GarageQual)))[1]
all$GarageFinish[2127] <- names(sort(-table(all$GarageFinish)))[1]
#display "fixed" house
kable(all[2127, c('GarageYrBlt', 'GarageCars', 'GarageArea', 'GarageType', 'GarageCond', 'GarageQual', 'GarageFinish')])
#::GarageCars and GarageArea: Size of garage in car capacity and Size of garage in square
#Both have 1 NA. As you can see above, it is house 2577 for both variables. The problem probably occured as the GarageType for this house is "detached", while all other Garage-variables seem to indicate that this house has no Garage.
#fixing 3 values for house 2577
all$GarageCars[2577] <- 0
all$GarageArea[2577] <- 0
all$GarageType[2577] <- NA
#check if NAs of the character variables are now all 158
length(which(is.na(all$GarageType) & is.na(all$GarageFinish) & is.na(all$GarageCond) & is.na(all$GarageQual)))
#Now, the 4 character variables related to garage all have the same set of 158 NAs, which correspond to 'No Garage'. I will fix all of them in the remainder of this section

#>>GarageType: Garage location
#segregate of GarageType or #table(all$GarageType)
counts<- table(all$GarageType)
barplot(counts, main=" GarageType",xlab="GarageType")
table(all$GarageType)
sum(table(all$GarageType))
all[!is.na(all$SalePrice),] %>% group_by(GarageType) %>% summarise(median = median(SalePrice), counts=n())
#The values do not seem ordinal, so I will convert into a factor.
all$GarageType[is.na(all$GarageType)] <- 'No Garage'
all$GarageType <- as.factor(all$GarageType)
table(all$GarageType)

#>>GarageFinish: Interior finish of the garage
#segregate of GarageFinish or #table(all$GarageFinish)
counts<- table(all$GarageFinish)
barplot(counts, main=" GarageFinish",xlab="GarageFinish")
table(all$GarageFinish)
sum(table(all$GarageFinish))
all[!is.na(all$SalePrice),] %>% group_by(GarageFinish) %>% summarise(median = median(SalePrice), counts=n())

#The values are ordinal.
all$GarageFinish[is.na(all$GarageFinish)] <- 'None'
Finish <- c('None'=0, 'Unf'=1, 'RFn'=2, 'Fin'=3)

all$GarageFinish<-as.integer(revalue(all$GarageFinish, Finish))
table(all$GarageFinish)

#>>GarageQual: Garage quality
#segregate of GarageQual or #table(all$GarageQual)
counts<- table(all$GarageQual)
barplot(counts, main=" GarageQual",xlab="GarageQual")
table(all$GarageQual)
sum(table(all$GarageQual))
all[!is.na(all$SalePrice),] %>% group_by(GarageQual) %>% summarise(median = median(SalePrice), counts=n())

#Another variable than can be made "ordinal" with the Qualities vector.
all$GarageQual[is.na(all$GarageQual)] <- 'None'
all$GarageQual<-as.integer(revalue(all$GarageQual, Qualities))
table(all$GarageQual)

#>>GarageCond: Garage condition
#segregate of GarageCond or #table(all$GarageCond)
counts<- table(all$GarageCond)
barplot(counts, main=" GarageCond",xlab="GarageCond")
table(all$GarageCond)
sum(table(all$GarageCond))
all[!is.na(all$SalePrice),] %>% group_by(GarageCond) %>% summarise(median = median(SalePrice), counts=n())
#The values seemed possibly ordinal to me, but the visualization does not show this. Therefore, I will convert the variable into a factor.
ggplot(all[!is.na(all$SalePrice),], aes(x=as.factor(GarageCond), y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue')+
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..))

#Another variable than can be made ordinal with the Qualities vector.
all$GarageCond[is.na(all$GarageCond)] <- 'None'
all$GarageCond<-as.integer(revalue(all$GarageCond, Qualities))
table(all$GarageCond)

#>>basement variable
#check if all 79 NAs are the same observations among the variables with 80+ NAs
length(which(is.na(all$BsmtQual) & is.na(all$BsmtCond) & is.na(all$BsmtExposure) & is.na(all$BsmtFinType1) & is.na(all$BsmtFinType2)))
#Find the additional NAs; BsmtFinType1 is the one with 79 NAs
all[!is.na(all$BsmtFinType1) & (is.na(all$BsmtCond)|is.na(all$BsmtQual)|is.na(all$BsmtExposure)|is.na(all$BsmtFinType2)), c('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2')]
#So altogether, it seems as if there are 79 houses without a basement, because the basement variables of the other houses with missing values are all 80% complete (missing 1 out of 5 values). I am going to impute the modes to fix those 9 houses
#Imputing modes.
all$BsmtFinType2[333] <- names(sort(-table(all$BsmtFinType2)))[1]
all$BsmtExposure[c(949, 1488, 2349)] <- names(sort(-table(all$BsmtExposure)))[1]
all$BsmtCond[c(2041, 2186, 2525)] <- names(sort(-table(all$BsmtCond)))[1]
all$BsmtQual[c(2218, 2219)] <- names(sort(-table(all$BsmtQual)))[1]
#Now that the 5 variables considered agree upon 79 houses with 'no basement', I am going to factorize/hot encode them below.

#>>BsmtQual: Evaluates the height of the basement
#segregate of BsmtQual or #table(all$BsmtQual)
counts<- table(all$BsmtQual)
barplot(counts, main=" BsmtQual",xlab="BsmtQual")
table(all$BsmtQual)
sum(table(all$BsmtQual))
all[!is.na(all$SalePrice),] %>% group_by(BsmtQual) %>% summarise(median = median(SalePrice), counts=n())
#The values seemed possibly ordinal to me, but the visualization does not show this. Therefore, I will convert the variable into a factor.
ggplot(all[!is.na(all$SalePrice),], aes(x=as.factor(BsmtQual), y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue')+
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..))

#A variable than can be made ordinal with the Qualities vector.
all$BsmtQual[is.na(all$BsmtQual)] <- 'None'
all$BsmtQual<-as.integer(revalue(all$BsmtQual, Qualities))
table(all$BsmtQual)

#>>BsmtCond: Evaluates the general condition of the basement
#segregate of BsmtCond or #table(all$BsmtCond)
counts<- table(all$BsmtCond)
barplot(counts, main=" BsmtCond",xlab="BsmtCond")
table(all$BsmtCond)
sum(table(all$BsmtCond))
all[!is.na(all$SalePrice),] %>% group_by(BsmtCond) %>% summarise(median = median(SalePrice), counts=n())
#The values seemed possibly ordinal to me, but the visualization does not show this. Therefore, I will convert the variable into a factor.
ggplot(all[!is.na(all$SalePrice),], aes(x=as.factor(BsmtCond), y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue')+
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..))
#A variable than can be made ordinal with the Qualities vector.
all$BsmtCond[is.na(all$BsmtCond)] <- 'None'
all$BsmtCond<-as.integer(revalue(all$BsmtCond, Qualities))
table(all$BsmtCond)

#>>BsmtExposure: Refers to walkout or garden level walls
#segregate of BsmtExposure or #table(all$BsmtExposure)
counts<- table(all$BsmtExposure)
barplot(counts, main=" BsmtExposure",xlab="BsmtExposure")
table(all$BsmtExposure)
sum(table(all$BsmtExposure))
all[!is.na(all$SalePrice),] %>% group_by(BsmtExposure) %>% summarise(median = median(SalePrice), counts=n())
#The values seemed possibly ordinal to me, but the visualization does not show this. Therefore, I will convert the variable into a factor.
ggplot(all[!is.na(all$SalePrice),], aes(x=as.factor(BsmtExposure), y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue')+
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..))

#A variable than can be made ordinal.
all$BsmtExposure[is.na(all$BsmtExposure)] <- 'None'
Exposure <- c('None'=0, 'No'=1, 'Mn'=2, 'Av'=3, 'Gd'=4)

all$BsmtExposure<-as.integer(revalue(all$BsmtExposure, Exposure))
table(all$BsmtExposure)

#>>BsmtFinType1: Rating of basement finished area
#segregate of BsmtFinType1 or #table(all$BsmtFinType1)
counts<- table(all$BsmtFinType1)
barplot(counts, main=" BsmtFinType1",xlab="BsmtFinType1")
table(all$BsmtFinType1)
sum(table(all$BsmtFinType1))
#The values seemed possibly ordinal to me, but the visualization does not show this. Therefore, I will convert the variable into a factor.
ggplot(all[!is.na(all$SalePrice),], aes(x=as.factor(BsmtFinType1), y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue')+
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..))

#A variable than can be made ordinal.
all$BsmtFinType1[is.na(all$BsmtFinType1)] <- 'None'
FinType <- c('None'=0, 'Unf'=1, 'LwQ'=2, 'Rec'=3, 'BLQ'=4, 'ALQ'=5, 'GLQ'=6)

all$BsmtFinType1<-as.integer(revalue(all$BsmtFinType1, FinType))
table(all$BsmtFinType1)

#>>BsmtFinType2: Rating of basement finished area (if multiple types)
#segregate of BsmtFinType2 or #table(all$BsmtFinType2)
counts<- table(all$BsmtFinType2)
barplot(counts, main=" BsmtFinType2",xlab="BsmtFinType2")
table(all$BsmtFinType2)
sum(table(all$BsmtFinType2))
all[!is.na(all$SalePrice),] %>% group_by(BsmtFinType2) %>% summarise(median = median(SalePrice), counts=n())
#The values seemed possibly ordinal to me, but the visualization does not show this. Therefore, I will convert the variable into a factor.
ggplot(all[!is.na(all$SalePrice),], aes(x=as.factor(BsmtFinType2), y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue')+
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..))

all$BsmtFinType2[is.na(all$BsmtFinType2)] <- 'None'
FinType <- c('None'=0, 'Unf'=1, 'LwQ'=2, 'Rec'=3, 'BLQ'=4, 'ALQ'=5, 'GLQ'=6)

all$BsmtFinType2<-as.integer(revalue(all$BsmtFinType2, FinType))
table(all$BsmtFinType2)

#>>Remaining Basement variabes with just a few NAs
#I now still have to deal with those 6 variables that have 1 or 2 NAs.
#display remaining NAs. Using BsmtQual as a reference for the 79 houses without basement agreed upon earlier
all[(is.na(all$BsmtFullBath)|is.na(all$BsmtHalfBath)|is.na(all$BsmtFinSF1)|is.na(all$BsmtFinSF2)|is.na(all$BsmtUnfSF)|is.na(all$TotalBsmtSF)), c('BsmtQual', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF')]
all$BsmtFullBath[is.na(all$BsmtFullBath)] <-0#An integer variable.
table(all$BsmtFullBath)
all$BsmtHalfBath[is.na(all$BsmtHalfBath)] <-0#An integer variable.
table(all$BsmtHalfBath)
all$BsmtFinSF1[is.na(all$BsmtFinSF1)] <-0#An integer variable.
all$BsmtFinSF2[is.na(all$BsmtFinSF2)] <-0#An integer variable.
all$BsmtUnfSF[is.na(all$BsmtUnfSF)] <-0#An integer variable.
all$TotalBsmtSF[is.na(all$TotalBsmtSF)] <-0#An integer variable.

#>==Masonry veneer type, and masonry veneer area
#Masonry veneer type has 24 NAs. Masonry veneer area has 23 NAs. If a house has a veneer area, it should also have a masonry veneer type. Let's fix this one first.
#check if the 23 houses with veneer area NA are also NA in the veneer type
length(which(is.na(all$MasVnrType) & is.na(all$MasVnrArea)))
#find the one that should have a MasVnrType
all[is.na(all$MasVnrType) & !is.na(all$MasVnrArea), c('MasVnrType', 'MasVnrArea')]
#fix this veneer type by imputing the mode
all$MasVnrType[2611] <- names(sort(-table(all$MasVnrType)))[2] #taking the 2nd value as the 1st is 'none'
all[2611, c('MasVnrType', 'MasVnrArea')]

#>>Masonry veneer type
all$MasVnrType[is.na(all$MasVnrType)] <- 'None'
#segregate of MasVnrType or #table(all$MasVnrType)
counts<- table(all$MasVnrType)
barplot(counts, main=" MasVnrType",xlab="MasVnrType")
table(all$MasVnrType)
sum(table(all$MasVnrType))
all[!is.na(all$SalePrice),] %>% group_by(MasVnrType) %>% summarise(median = median(SalePrice), counts=n()) %>% arrange(median)
# show ordinality 
#There seems to be a significant difference between "common brick/none" and the other types. I assume that simple stones and for instance wooden houses are just cheaper. I will make the ordinality accordingly.
Masonry <- c('None'=0, 'BrkCmn'=0, 'BrkFace'=1, 'Stone'=2)
all$MasVnrType<-as.integer(revalue(all$MasVnrType, Masonry))
table(all$MasVnrType)
all$MasVnrArea[is.na(all$MasVnrArea)] <-0#An integer variable.

#==MSZoning: Identifies the general zoning classification of the sale
#Values are categorical
#segregate of MSZoning or #table(all$MSZoning)
counts<- table(all$MSZoning)
barplot(counts, main=" MSZoning",xlab="MSZoning")
table(all$MSZoning)
sum(table(all$MSZoning))
all[!is.na(all$SalePrice),] %>% group_by(MSZoning) %>% summarise(median = median(SalePrice), counts=n())
#The values seemed possibly ordinal to me, but the visualization does not show this. Therefore, I will convert the variable into a factor.
ggplot(all[!is.na(all$SalePrice),], aes(x=as.factor(MSZoning), y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue')+
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..))
#imputing the mode
all$MSZoning[is.na(all$MSZoning)] <- names(sort(-table(all$MSZoning)))[1]
all$MSZoning <- as.factor(all$MSZoning)
table(all$MSZoning)
sum(table(all$MSZoning))

#==Kitchen quality and numer of Kitchens above grade
#>>Kitchen quality
#segregate of KitchenQual or #table(all$KitchenQual)
counts<- table(all$KitchenQual)
barplot(counts, main=" KitchenQual",xlab="KitchenQual")
table(all$KitchenQual)
sum(table(all$KitchenQual))
#Can be made ordinal with the qualities vector.
all$KitchenQual[is.na(all$KitchenQual)] <- 'TA' #replace with most common value
all$KitchenQual<-as.integer(revalue(all$KitchenQual, Qualities))
table(all$KitchenQual)
sum(table(all$KitchenQual))

#==Utilities: Type of utilities available
#segregate of Utilities or #table(all$Utilities)
counts<- table(all$Utilities)
barplot(counts, main=" Utilities",xlab="Utilities")
table(all$Utilities)
sum(table(all$Utilities))
#Ordinal as additional utilities is better.
#However, the table below shows that only one house does not have all public utilities. This house is in the train set. Therefore, imputing 'AllPub' for the NAs means that all houses in the test set will have 'AllPub'.
#This makes the variable useless for prediction. Consequently, I will get rid of it.
table(all$Utilities)
kable(all[is.na(all$Utilities) | all$Utilities=='NoSeWa', 1:9])
all$Utilities <- NULL

#==Functional: Home functionality
#segregate of Functional or #table(all$Functional)
counts<- table(all$Functional)
barplot(counts, main=" Functional",xlab="Functional")
table(all$Functional)
sum(table(all$Functional))
#can be made ordinal (salvage only is worst, typical is best).
#impute mode for the 1 NA
all$Functional[is.na(all$Functional)] <- names(sort(-table(all$Functional)))[1]

all$Functional <- as.integer(revalue(all$Functional, c('Sal'=0, 'Sev'=1, 'Maj2'=2, 'Maj1'=3, 'Mod'=4, 'Min2'=5, 'Min1'=6, 'Typ'=7)))
table(all$Functional)
sum(table(all$Functional))

#==Exterior1st: Exterior covering on house
#segregate of Exterior1st or #table(all$Exterior1st)
counts<- table(all$Exterior1st)
barplot(counts, main=" Exterior1st",xlab="Exterior1st")
table(all$Exterior1st)
sum(table(all$Exterior1st))
#Values are categorical.
#imputing mode
all$Exterior1st[is.na(all$Exterior1st)] <- names(sort(-table(all$Exterior1st)))[1]

all$Exterior1st <- as.factor(all$Exterior1st)
table(all$Exterior1st)
sum(table(all$Exterior1st))

#==Exterior2nd: Exterior covering on house (if more than one material)
#Values are categorical.
#imputing mode
all$Exterior2nd[is.na(all$Exterior2nd)] <- names(sort(-table(all$Exterior2nd)))[1]

all$Exterior2nd <- as.factor(all$Exterior2nd)
table(all$Exterior2nd)
sum(table(all$Exterior2nd))

#==ExterQual: Evaluates the quality of the material on the exterior
#segregate of ExterQual or #table(all$ExterQual)
counts<- table(all$ExterQual)
barplot(counts, main=" ExterQual",xlab="ExterQual")
table(all$ExterQual)
sum(table(all$ExterQual))
#Can be made ordinal using the Qualities vector.
all$ExterQual<-as.integer(revalue(all$ExterQual, Qualities))
table(all$ExterQual)
sum(table(all$ExterQual))

#==ExterCond: Evaluates the present condition of the material on the exterior
#segregate of ExterCond or #table(all$ExterCond)
counts<- table(all$ExterCond)
barplot(counts, main=" ExterCond",xlab="ExterCond")
table(all$ExterCond)
sum(table(all$ExterCond))
# Can be made ordinal using the Qualities vector.
all$ExterCond<-as.integer(revalue(all$ExterCond, Qualities))
table(all$ExterCond)
sum(table(all$ExterCond))

#==Electrical: Electrical system
#segregate of Electrical or #table(all$Electrical)
counts<- table(all$Electrical)
barplot(counts, main=" Electrical",xlab="Electrical")
table(all$Electrical)
sum(table(all$Electrical))
#1 NA. Values are categorical.
#imputing mode
all$Electrical[is.na(all$Electrical)] <- names(sort(-table(all$Electrical)))[1]

all$Electrical <- as.factor(all$Electrical)
table(all$Electrical)
sum(table(all$Electrical))

#==SaleType: Type of sale
#segregate of SaleType or #table(all$SaleType)
counts<- table(all$SaleType)
barplot(counts, main=" SaleType",xlab="SaleType")
table(all$SaleType)
sum(table(all$SaleType))
#1 NA. Values are categorical
#imputing mode
all$SaleType[is.na(all$SaleType)] <- names(sort(-table(all$SaleType)))[1]

all$SaleType <- as.factor(all$SaleType)
table(all$SaleType)
sum(table(all$SaleType))

#==SaleCondition: Condition of sale
#segregate of SaleCondition or #table(all$SaleCondition)
counts<- table(all$SaleCondition)
barplot(counts, main=" SaleCondition",xlab="SaleCondition")
table(all$SaleCondition)
sum(table(all$SaleCondition))
#No NAs. Values are categorical.
all$SaleCondition <- as.factor(all$SaleCondition)
table(all$SaleCondition)
sum(table(all$SaleCondition))


#=====Label encoding/factorizing the remaining character variables=====
Charcol <- names(all[,sapply(all, is.character)])
Charcol
cat('There are', length(Charcol), 'remaining columns with character values')
#>>Foundation: Type of foundation
#segregate of Foundation or #table(all$Foundation)
counts<- table(all$Foundation)
barplot(counts, main=" Foundation",xlab="Foundation")
table(all$Foundation)
sum(table(all$Foundation))
#No ordinality, so converting into factors
all$Foundation <- as.factor(all$Foundation)
table(all$Foundation)
sum(table(all$Foundation))

#>>Heating: Type of heating
#segregate of Heating or #table(all$Heating)
counts<- table(all$Heating)
barplot(counts, main=" Heating",xlab="Heating")
table(all$Heating)
sum(table(all$Heating))
#No ordinality, so converting into factors
all$Heating <- as.factor(all$Heating)
table(all$Heating)
sum(table(all$Heating))

#>>HeatingQC: Heating quality and condition
#segregate of HeatingQC or #table(all$HeatingQC)
counts<- table(all$Heating)
barplot(counts, main=" HeatingQC",xlab="HeatingQC")
table(all$HeatingQC)
sum(table(all$HeatingQC))
#making the variable ordinal using the Qualities vector
all$HeatingQC<-as.integer(revalue(all$HeatingQC, Qualities))
table(all$HeatingQC)
sum(table(all$HeatingQC))

#>>CentralAir: Central air conditioning
#segregate of CentralAir or #table(all$CentralAir)
counts<- table(all$CentralAir)
barplot(counts, main=" CentralAir",xlab="HeatingQC")

all$CentralAir<-as.integer(revalue(all$CentralAir, c('N'=0, 'Y'=1)))
table(all$CentralAir)
sum(table(all$CentralAir))

#>>RoofStyle: Type of roof
#segregate of RoofStyle or #table(all$RoofStyle)
counts<- table(all$RoofStyle)
barplot(counts, main=" RoofStyle",xlab="RoofStyle")
#No ordinality, so converting into factors
all$RoofStyle <- as.factor(all$RoofStyle)
table(all$RoofStyle)
sum(table(all$RoofStyle))

#>>RoofMatl: Roof material
#segregate of RoofMatl or #table(all$RoofMatl)
counts<- table(all$RoofMatl)
barplot(counts, main=" RoofMatl",xlab="RoofMatl")
#No ordinality, so converting into factors
all$RoofMatl <- as.factor(all$RoofMatl)
table(all$RoofMatl)
sum(table(all$RoofMatl))

#>>LandContour: Flatness of the property
#segregate of LandContour or #table(all$LandContour)
counts<- table(all$LandContour)
barplot(counts, main=" LandContour",xlab="LandContour")
#No ordinality, so converting into factors
all$LandContour <- as.factor(all$LandContour)
table(all$LandContour)
sum(table(all$LandContour))

#>>LandSlope: Slope of property
#segregate of LandSlope or #table(all$LandSlope)
counts<- table(all$LandSlope)
barplot(counts, main=" LandSlope",xlab="LandSlope")
#Ordinal, so label encoding
all$LandSlope<-as.integer(revalue(all$LandSlope, c('Sev'=0, 'Mod'=1, 'Gtl'=2)))
table(all$LandSlope)
sum(table(all$LandSlope))

#>>BldgType: Type of dwelling
#segregate of BldgType or #table(all$BldgType)
counts<- table(all$BldgType)
barplot(counts, main=" BldgType",xlab="BldgType")
#This seems ordinal to me (single family detached=best). Let's check it with visualization.
ggplot(all[!is.na(all$SalePrice),], aes(x=as.factor(BldgType), y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue')+
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..))
#However, the visualization does not show ordinality.
#No ordinality, so converting into factors
all$BldgType <- as.factor(all$BldgType)
table(all$BldgType)
sum(table(all$BldgType))

#>>HouseStyle: Style of dwelling
ggplot(all[!is.na(all$SalePrice),], aes(x=as.factor(HouseStyle), y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue')+
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..))
#No ordinality, so converting into factors
all$HouseStyle <- as.factor(all$HouseStyle)
table(all$HouseStyle)
sum(table(all$HouseStyle))

#>>Neighborhood: Physical locations within Ames city limits
#Note: as the number of levels is really high, I will look into binning later on.
ggplot(all[!is.na(all$SalePrice),], aes(x=as.factor(Neighborhood), y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue')+
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..))
#No ordinality, so converting into factors
all$Neighborhood <- as.factor(all$Neighborhood)
table(all$Neighborhood)
sum(table(all$Neighborhood))

#>>Condition1: Proximity to various conditions
ggplot(all[!is.na(all$SalePrice),], aes(x=as.factor(Condition1), y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue')+
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..))
#No ordinality, so converting into factors
all$Condition1 <- as.factor(all$Condition1)
table(all$Condition1)
sum(table(all$Condition1))

#>>Condition2: Proximity to various conditions (if more than one is present)
ggplot(all[!is.na(all$SalePrice),], aes(x=as.factor(Condition2), y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue')+
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..))
#No ordinality, so converting into factors
all$Condition2 <- as.factor(all$Condition2)
table(all$Condition2)
sum(table(all$Condition2))

#>>Street: Type of road access to property
ggplot(all[!is.na(all$SalePrice),], aes(x=as.factor(Street), y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue')+
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..))
#Ordinal, so label encoding
all$Street<-as.integer(revalue(all$Street, c('Grvl'=0, 'Pave'=1)))
table(all$Street)
sum(table(all$Street))

#>>PavedDrive: Paved driveway
ggplot(all[!is.na(all$SalePrice),], aes(x=as.factor(PavedDrive), y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue')+
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..))
#Ordinal, so label encoding
all$PavedDrive<-as.integer(revalue(all$PavedDrive, c('N'=0, 'P'=1, 'Y'=2)))
table(all$PavedDrive)
sum(table(all$PavedDrive))

#========Changing some numeric variables into factors
#there are 3 variables that are recorded numeric but should actually be categorical.
#>>Year and Month Sold
#I wil convert YrSold into a factor before modeling, but as I need the numeric version of YrSold to create an Age variable, I am not doing that yet.
#Month Sold is also an Integer variable. However, December is not "better" than January. Therefore, I will convert MoSold values back into factors.
str(all$YrSold)
str(all$MoSold)
all$MoSold <- as.factor(all$MoSold)
# the effects of the Banking crises that took place at the end of 2007 can be seen indeed. After the highest median prices in 2007, the prices gradually decreased.
#However, seasonality seems to play a bigger role, as you can see below.
ys <- ggplot(all[!is.na(all$SalePrice),], aes(x=as.factor(YrSold), y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue')+
  scale_y_continuous(breaks= seq(0, 800000, by=25000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..)) +
  coord_cartesian(ylim = c(0, 200000)) +
  geom_hline(yintercept=163000, linetype="dashed", color = "red") #dashed line is median SalePrice
ms <- ggplot(all[!is.na(all$SalePrice),], aes(x=MoSold, y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue')+
  scale_y_continuous(breaks= seq(0, 800000, by=25000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..)) +
  coord_cartesian(ylim = c(0, 200000)) +
  geom_hline(yintercept=163000, linetype="dashed", color = "red") #dashed line is median SalePrice
grid.arrange(ys, ms, widths=c(1,2))

#>>MSSubClass
ggplot(all[!is.na(all$SalePrice),], aes(x=as.factor(MSSubClass), y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue')+
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..))
#MSSubClass: Identifies the type of dwelling involved in the sale.
#These classes are coded as numbers, but really are categories.
str(all$MSSubClass)
all$MSSubClass <- as.factor(all$MSSubClass)
#revalue for better readability
all$MSSubClass<-revalue(all$MSSubClass, c('20'='1 story 1946+', '30'='1 story 1945-', '40'='1 story unf attic', '45'='1,5 story unf', '50'='1,5 story fin', '60'='2 story 1946+', '70'='2 story 1945-', '75'='2,5 story all ages', '80'='split/multi level', '85'='split foyer', '90'='duplex all style/age', '120'='1 story PUD 1946+', '150'='1,5 story PUD all', '160'='2 story PUD 1946+', '180'='PUD multilevel', '190'='2 family conversion'))
str(all$MSSubClass)

#===========Visualization of important variables
numericVars <- which(sapply(all, is.numeric)) #index vector numeric variables
factorVars <- which(sapply(all, is.factor)) #index vector factor variables
cat('There are', length(numericVars), 'numeric variables, and', length(factorVars), 'categoric variables')

#>>Correlations again(numeric variable)
#Although the correlations are giving a good overview of the most important numeric variables and multicolinerity among those variables, 
#Below I am checking the correlations again. As you can see, the number of variables with a correlation of at least 0.5 with the SalePrice has increased from 10 to 16.
all_numVar <- all[, numericVars]
cor_numVar <- cor(all_numVar, use="pairwise.complete.obs") #correlations of all numeric variables

#sort on decreasing correlations with SalePrice
cor_sorted <- as.matrix(sort(cor_numVar[,'SalePrice'], decreasing = TRUE))
#select only high corelations
CorHigh <- names(which(apply(cor_sorted, 1, function(x) abs(x)>0.5)))
cor_numVar <- cor_numVar[CorHigh, CorHigh]

corrplot.mixed(cor_numVar, tl.col="black", tl.pos = "lt", tl.cex = 0.7,cl.cex = .7, number.cex=.7)

#>>Finding variable importance with a quick Random Forest(categorical variable)
# I wanted to get an overview of the most important variables including the categorical variables before moving on to visualization
set.seed(2018)
quick_RF <- randomForest(x=all[1:1460,-79], y=all$SalePrice[1:1460], ntree=100,importance=TRUE)
imp_RF <- importance(quick_RF)
imp_DF <- data.frame(Variables = row.names(imp_RF), MSE = imp_RF[,1])
imp_DF <- imp_DF[order(imp_DF$MSE, decreasing = TRUE),]

ggplot(imp_DF[1:20,], aes(x=reorder(Variables, MSE), y=MSE, fill=MSE)) + geom_bar(stat = 'identity') + labs(x = 'Variables', y= '% increase MSE if variable is randomly permuted') + coord_flip() + theme(legend.position="none")
#Only 3 of those most important variables are categorical according to RF; Neighborhood, MSSubClass, and GarageType.

#>>Above Ground Living Area, and other surface related variables (in square feet)
s1 <- ggplot(data= all, aes(x=GrLivArea)) +
  geom_density() + labs(x='Square feet living area')
s2 <- ggplot(data=all, aes(x=as.factor(TotRmsAbvGrd))) +
  geom_histogram(stat='count') + labs(x='Rooms above Ground')
s3 <- ggplot(data= all, aes(x=X1stFlrSF)) +
  geom_density() + labs(x='Square feet first floor')
s4 <- ggplot(data= all, aes(x=X2ndFlrSF)) +
  geom_density() + labs(x='Square feet second floor')
s5 <- ggplot(data= all, aes(x=TotalBsmtSF)) +
  geom_density() + labs(x='Square feet basement')
s6 <- ggplot(data= all[all$LotArea<100000,], aes(x=LotArea)) +
  geom_density() + labs(x='Square feet lot')
s7 <- ggplot(data= all, aes(x=LotFrontage)) +
  geom_density() + labs(x='Linear feet lot frontage')
s8 <- ggplot(data= all, aes(x=LowQualFinSF)) +
  geom_histogram() + labs(x='Low quality square feet 1st & 2nd')
layout <- matrix(c(1,2,5,3,4,8,6,7),4,2,byrow=TRUE)
multiplot(s1, s2, s3, s4, s5, s6, s7, s8, layout=layout)
# I am taking the opportunity to bundle them in this section. Note: GarageArea is taken care of in the Garage variables section.
#I am also adding 'Total Rooms Above Ground' (TotRmsAbvGrd) as this variable is highly correlated with the Above Ground Living Area(0.81).
#The correlation between those 3 variables and GrLivArea is exactely 1.
cor(all$GrLivArea, (all$X1stFlrSF + all$X2ndFlrSF + all$LowQualFinSF))
head(all[all$LowQualFinSF>0, c('GrLivArea', 'X1stFlrSF', 'X2ndFlrSF', 'LowQualFinSF')])

#>>The most important categorical variable; Neighborhood
#Th first graph shows the median SalePrice by Neighorhood. The frequency (number of houses) of each Neighborhood in the train set is shown in the labels.
#The second graph below shows the frequencies across all data.
n1 <- ggplot(all[!is.na(all$SalePrice),], aes(x=Neighborhood, y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue') +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_y_continuous(breaks= seq(0, 800000, by=50000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..), size=3) +
  geom_hline(yintercept=163000, linetype="dashed", color = "red") #dashed line is median SalePrice
n2 <- ggplot(data=all, aes(x=Neighborhood)) +
  geom_histogram(stat='count')+
  geom_label(stat = "count", aes(label = ..count.., y = ..count..), size=3)+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
grid.arrange(n1, n2)
#>>Overall Quality, and other Quality variables
#I have already visualized the relation between Overall Quality and SalePrice in my initial explorations, but I want to visualize the frequency distribution as well. As there are more quality measurements, I am taking the opportunity to bundle them in this section.
q1 <- ggplot(data=all, aes(x=as.factor(OverallQual))) +
  geom_histogram(stat='count')
q2 <- ggplot(data=all, aes(x=as.factor(ExterQual))) +
  geom_histogram(stat='count')
q3 <- ggplot(data=all, aes(x=as.factor(BsmtQual))) +
  geom_histogram(stat='count')
q4 <- ggplot(data=all, aes(x=as.factor(KitchenQual))) +
  geom_histogram(stat='count')
q5 <- ggplot(data=all, aes(x=as.factor(GarageQual))) +
  geom_histogram(stat='count')
q6 <- ggplot(data=all, aes(x=as.factor(FireplaceQu))) +
  geom_histogram(stat='count')
q7 <- ggplot(data=all, aes(x=as.factor(PoolQC))) +
  geom_histogram(stat='count')

layout <- matrix(c(1,2,8,3,4,8,5,6,7),3,3,byrow=TRUE)
multiplot(q1, q2, q3, q4, q5, q6, q7, layout=layout)
#Overall Quality is very important, and also more granular than the other variables. External Quality is also improtant, but has a high correlation with Overall Quality (0.73).
#Kitchen Quality also seems one to keep, as all houses have a kitchen and there is a variance with some substance.
#Garage Quality does not seem to distinguish much, as the majority of garages have Q3. Fireplace Quality is in the list of high correlations, and in the important variables list.
#The PoolQC is just very sparse (the 13 pools cannot even be seen on this scale). I will look at creating a 'has pool' variable later on.

#>>The second most important categorical variable; MSSubClass
#The first visualization shows the median SalePrice by MSSubClass. The frequency (number of houses) of each MSSubClass in the train set is shown in the labels.
#The histrogram shows the frequencies across all data. Most houses are relatively new, and have one or two stories.
ms1 <- ggplot(all[!is.na(all$SalePrice),], aes(x=MSSubClass, y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue') +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_y_continuous(breaks= seq(0, 800000, by=50000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..), size=3) +
  geom_hline(yintercept=163000, linetype="dashed", color = "red") #dashed line is median SalePrice
ms2 <- ggplot(data=all, aes(x=MSSubClass)) +
  geom_histogram(stat='count')+
  geom_label(stat = "count", aes(label = ..count.., y = ..count..), size=3) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
grid.arrange(ms1, ms2)

#>>Garage variables
#Several Garage variables have a high correlation with SalePrice, and are also in the top-20 list of the quick random forest. However, there is multicolinearity among them and I think that 7 garage variables is too many anyway. 
# I feel that something like 3 variables should be sufficient (possibly GarageCars, GarageType, and a Quality measurement), but before I do any selection I am visualizing all of them in this section.
#correct error
all$GarageYrBlt[2593] <- 2007 #this must have been a typo. GarageYrBlt=2207, YearBuilt=2006, YearRemodAdd=2007.
g1 <- ggplot(data=all[all$GarageCars !=0,], aes(x=GarageYrBlt)) +
  geom_histogram()
g2 <- ggplot(data=all, aes(x=as.factor(GarageCars))) +
  geom_histogram(stat='count')
g3 <- ggplot(data= all, aes(x=GarageArea)) +
  geom_density()
g4 <- ggplot(data=all, aes(x=as.factor(GarageCond))) +
  geom_histogram(stat='count')
g5 <- ggplot(data=all, aes(x=GarageType)) +
  geom_histogram(stat='count')
g6 <- ggplot(data=all, aes(x=as.factor(GarageQual))) +
  geom_histogram(stat='count')
g7 <- ggplot(data=all, aes(x=as.factor(GarageFinish))) +
  geom_histogram(stat='count')

layout <- matrix(c(1,5,5,2,3,8,6,4,7),3,3,byrow=TRUE)
multiplot(g1, g2, g3, g4, g5, g6, g7, layout=layout)
#As already mentioned in section 4.2, GarageCars and GarageArea are highly correlated. Here, GarageQual and GarageCond also seem highly correlated, and both are dominated by level =3.

#>>Basement variables
#Similar the garage variables, multiple basement variables are important in the correlations matrix and the Top 20 RF predictors list. However, 11 basement variables seems an overkill.
# Before I decide what I am going to do with them, I am visualizing 8 of them below. 
b1 <- ggplot(data=all, aes(x=BsmtFinSF1)) +
  geom_histogram() + labs(x='Type 1 finished square feet')
b2 <- ggplot(data=all, aes(x=BsmtFinSF2)) +
  geom_histogram()+ labs(x='Type 2 finished square feet')
b3 <- ggplot(data=all, aes(x=BsmtUnfSF)) +
  geom_histogram()+ labs(x='Unfinished square feet')
b4 <- ggplot(data=all, aes(x=as.factor(BsmtFinType1))) +
  geom_histogram(stat='count')+ labs(x='Rating of Type 1 finished area')
b5 <- ggplot(data=all, aes(x=as.factor(BsmtFinType2))) +
  geom_histogram(stat='count')+ labs(x='Rating of Type 2 finished area')
b6 <- ggplot(data=all, aes(x=as.factor(BsmtQual))) +
  geom_histogram(stat='count')+ labs(x='Height of the basement')
b7 <- ggplot(data=all, aes(x=as.factor(BsmtCond))) +
  geom_histogram(stat='count')+ labs(x='Rating of general condition')
b8 <- ggplot(data=all, aes(x=as.factor(BsmtExposure))) +
  geom_histogram(stat='count')+ labs(x='Walkout or garden level walls')

layout <- matrix(c(1,2,3,4,5,9,6,7,8),3,3,byrow=TRUE)
multiplot(b1, b2, b3, b4, b5, b6, b7, b8, layout=layout)
#So it seemed as if the Total Basement Surface in square feet (TotalBsmtSF) is further broken down into finished areas (2 if more than one type of finish), and unfinished area.
#I did a check between the correlation of total of those 3 variables, and TotalBsmtSF. The correlation is exactely 1, so that's a good thing (no errors or small discrepancies)!

#========Feature engineering
#>>Total number of Bathrooms
#There are 4 bathroom variables. Individually, these variables are not very important. However, I assume that I if I add them up into one predictor, this predictor is likely to become a strong one.
#"A half-bath, also known as a powder room or guest bath, has only two of the four main bathroom components-typically a toilet and sink." Consequently, I will also count the half bathrooms as half.
all$TotBathrooms <- all$FullBath + (all$HalfBath*0.5) + all$BsmtFullBath + (all$BsmtHalfBath*0.5)
#As you can see in the first graph, there now seems to be a clear correlation (it's 0.63). The frequency distribution of Bathrooms in all data is shown in the second graph
tb1 <- ggplot(data=all[!is.na(all$SalePrice),], aes(x=as.factor(TotBathrooms), y=SalePrice))+
  geom_point(col='blue') + geom_smooth(method = "lm", se=FALSE, color="black", aes(group=1)) +
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma)
tb2 <- ggplot(data=all, aes(x=as.factor(TotBathrooms))) +
  geom_histogram(stat='count')
grid.arrange(tb1, tb2)

#>>Adding 'House Age', 'Remodeled (Yes/No)', and IsNew variables
#I will use YearRemodeled and YearSold to determine the Age. 
# I will also introduce a Remodeled Yes/No variable. This should be seen as some sort of penalty parameter that indicates that if the Age is based on a remodeling date, it is probably worth less than houses that were built from scratch in that same year.
all$Remod <- ifelse(all$YearBuilt==all$YearRemodAdd, 0, 1) #0=No Remodeling, 1=Remodeling
all$Age <- as.numeric(all$YrSold)-all$YearRemodAdd
ggplot(data=all[!is.na(all$SalePrice),], aes(x=Age, y=SalePrice))+
  geom_point(col='blue') + geom_smooth(method = "lm", se=FALSE, color="black", aes(group=1)) +
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma)
#As expected, the graph shows a negative correlation with Age (old house are worth less).
cor(all$SalePrice[!is.na(all$SalePrice)], all$Age[!is.na(all$SalePrice)])
#As you can see below, houses that are remodeled are worth less indeed, as expected.
ggplot(all[!is.na(all$SalePrice),], aes(x=as.factor(Remod), y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue') +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..), size=6) +
  scale_y_continuous(breaks= seq(0, 800000, by=50000), labels = comma) +
  theme_grey(base_size = 18) +
  geom_hline(yintercept=163000, linetype="dashed") #dashed line is median SalePrice
#Finally, I am creating the IsNew variable below. Altogether, there are 116 new houses in the dataset.
all$IsNew <- ifelse(all$YrSold==all$YearBuilt, 1, 0)
table(all$IsNew)
#These 116 new houses are fairly evenly distributed among train and test set, and as you can see new houses are worth considerably more on average.
ggplot(all[!is.na(all$SalePrice),], aes(x=as.factor(IsNew), y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue') +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..), size=6) +
  scale_y_continuous(breaks= seq(0, 800000, by=50000), labels = comma) +
  theme_grey(base_size = 18) +
  geom_hline(yintercept=163000, linetype="dashed") #dashed line is median SalePrice
all$YrSold <- as.factor(all$YrSold) #the numeric version is now not needed anymore

#>>Binning Neighborhood
nb1 <- ggplot(all[!is.na(all$SalePrice),], aes(x=reorder(Neighborhood, SalePrice, FUN=median), y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue') + labs(x='Neighborhood', y='Median SalePrice') +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_y_continuous(breaks= seq(0, 800000, by=50000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..), size=3) +
  geom_hline(yintercept=163000, linetype="dashed", color = "red") #dashed line is median SalePrice
nb2 <- ggplot(all[!is.na(all$SalePrice),], aes(x=reorder(Neighborhood, SalePrice, FUN=mean), y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "mean", fill='blue') + labs(x='Neighborhood', y="Mean SalePrice") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_y_continuous(breaks= seq(0, 800000, by=50000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..), size=3) +
  geom_hline(yintercept=163000, linetype="dashed", color = "red") #dashed line is median SalePrice
grid.arrange(nb1, nb2)

#Both the median and mean Saleprices agree on 3 neighborhoods with substantially higher saleprices. The separation of the 3 relatively poor neighborhoods is less clear, but at least both graphs agree on the same 3 poor neighborhoods.
# I am only creating categories for those 'extremes'.
all$NeighRich[all$Neighborhood %in% c('StoneBr', 'NridgHt', 'NoRidge')] <- 2
all$NeighRich[!all$Neighborhood %in% c('MeadowV', 'IDOTRR', 'BrDale', 'StoneBr', 'NridgHt', 'NoRidge')] <- 1
all$NeighRich[all$Neighborhood %in% c('MeadowV', 'IDOTRR', 'BrDale')] <- 0
table(all$NeighRich)

#>>Total Square Feet
#As the total living space generally is very important when people buy houses, I am adding a predictors that adds up the living space above and below ground
all$TotalSqFeet <- all$GrLivArea + all$TotalBsmtSF
ggplot(data=all[!is.na(all$SalePrice),], aes(x=TotalSqFeet, y=SalePrice))+
  geom_point(col='blue') + geom_smooth(method = "lm", se=FALSE, color="black", aes(group=1)) +
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) +
  geom_text_repel(aes(label = ifelse(all$GrLivArea[!is.na(all$SalePrice)]>4500, rownames(all), '')))
# the correlation with SalePrice is very strong indeed (0.78).
cor(all$SalePrice, all$TotalSqFeet, use= "pairwise.complete.obs")
#The two potential outliers seem to 'outlie' even more than before. By taking out these two outliers, the correlation increases by 5%.
cor(all$SalePrice[-c(524, 1299)], all$TotalSqFeet[-c(524, 1299)], use= "pairwise.complete.obs")

#>> Consolidating Porch variables
#As far as I know, porches are sheltered areas outside of the house, and a wooden deck is unsheltered. Therefore, I am leaving WoodDeckSF alone, and are only consolidating the 4 porch variables.
all$TotalPorchSF <- all$OpenPorchSF + all$EnclosedPorch + all$X3SsnPorch + all$ScreenPorch
# the correlation with SalePrice is not very strong.
cor(all$SalePrice, all$TotalPorchSF, use= "pairwise.complete.obs")
ggplot(data=all[!is.na(all$SalePrice),], aes(x=TotalPorchSF, y=SalePrice))+
  geom_point(col='blue') + geom_smooth(method = "lm", se=FALSE, color="black", aes(group=1)) +
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma)

#==========Preparing data for modeling
#Although the correlations are giving a good overview of the most important numeric variables and multicolinerity among those variables, 
#Below I am checking the correlations again. As you can see, the number of variables with a correlation of at least 0.5 with the SalePrice has increased from 10 to 16.
all_numVar <- all[, numericVars]
cor_numVar <- cor(all_numVar, use="pairwise.complete.obs") #correlations of all numeric variables

#sort on decreasing correlations with SalePrice
cor_sorted <- as.matrix(sort(cor_numVar[,'SalePrice'], decreasing = TRUE))
#select only high corelations
CorHigh <- names(which(apply(cor_sorted, 1, function(x) abs(x)>0.5)))
cor_numVar <- cor_numVar[CorHigh, CorHigh]

corrplot.mixed(cor_numVar, tl.col="black", tl.pos = "lt", tl.cex = 0.7,cl.cex = .7, number.cex=.7)


#>>Dropping highly correlated INDEPENDENT variables
#GarageCars and GarageArea have a correlation of 0.89. Of those two, I am dropping the variable with the lowest correlation with SalePrice (which is GarageArea with a SalePrice correlation of 0.62. GarageCars has a SalePrice correlation of 0.64).
dropVars <- c('YearRemodAdd', 'GarageYrBlt', 'GarageArea', 'GarageCond', 'TotalBsmtSF', 'TotalRmsAbvGrd', 'BsmtFinSF1')

all <- all[,!(names(all) %in% dropVars)]

#>>Removing outliers
#For the time being, I am keeping it simple and just remove the two really big houses with low SalePrice manually. However, I intend to investigate this more thorough in a later stage (possibly using the 'outliers' package).
all <- all[-c(524, 1299),]

#>>====== PreProcessing predictor variables
#Before modeling I need to center and scale the 'true numeric' predictors (so not variables that have been label encoded), and create dummy variables for the categorical predictors.
#Below, I am splitting the dataframe into one with all (true) numeric variables, and another dataframe holding the (ordinal) factors.
numericVarNames <- numericVars[!(numericVars %in% c('MSSubClass', 'MoSold', 'YrSold', 'SalePrice', 'OverallQual', 'OverallCond'))] #numericVarNames was created before having done anything
numericVarNames <- append(numericVarNames, c('Age', 'TotalPorchSF', 'TotBathrooms', 'TotalSqFeet'))
numericVarNames
numericVars
length(numericVarNames)
DFnumeric <- all[,names(all) %in% numericVarNames]

DFfactors <- all[, !(names(all) %in% numericVarNames)]
DFfactors <- DFfactors[, names(DFfactors) != 'SalePrice']

cat('There are', length(numericVarNames), 'numeric variables, and', length(DFfactors), 'factor variables')

#>>Skewness and normalizing of the numeric predictors
#Skewness essentially measures the relative size of the two tails. As a rule of thumb, skewness should be between -1 and 1. In this range, data are considered fairly symmetrical.
#In order to fix the skewness, I am taking the log for all numeric predictors with an absolute skew greater than 0.8 (actually: log+1, to avoid division by zero issues).
for(i in 1:ncol(DFnumeric)){
  if (abs(skew(DFnumeric[,i]))>0.8){
    DFnumeric[,i] <- log(DFnumeric[,i] +1)
  }
}
#>>=====Normalizing the data
PreNum <- preProcess(DFnumeric, method=c("center", "scale"))
print(PreNum)
DFnorm <- predict(PreNum, DFnumeric)
dim(DFnorm)

#>>=====One hot encoding the categorical variables(model.matrix())
#The last step needed to ensure that all predictors are converted into numeric columns (which is required by most Machine Learning algorithms) is to 'one-hot encode' the categorical variables.
#This basically means that all (not ordinal) factor values are getting a seperate colums with 1s and 0s (1 basically means Yes/Present). To do this one-hot encoding, I am using the model.matrix() function.
DFdummies <- as.data.frame(model.matrix(~.-1, DFfactors))
dim(DFdummies)

#>>=====Removing levels with few or no observations in train or test
#---check if some values are absent in the test set
ZerocolTest <- which(colSums(DFdummies[(nrow(all[!is.na(all$SalePrice),])+1):nrow(all),])==0)
colnames(DFdummies[ZerocolTest])

DFdummies <- DFdummies[,-ZerocolTest] #removing predictors

#---check if some values are absent in the train set
ZerocolTrain <- which(colSums(DFdummies[1:nrow(all[!is.na(all$SalePrice),]),])==0)
colnames(DFdummies[ZerocolTrain])
DFdummies <- DFdummies[,-ZerocolTrain] #removing predictor

#Also taking out variables with less than 10 'ones' in the train set.
fewOnes <- which(colSums(DFdummies[1:nrow(all[!is.na(all$SalePrice),]),])<10)
colnames(DFdummies[fewOnes])
DFdummies <- DFdummies[,-fewOnes] #removing predictors
dim(DFdummies)
#Altogether, I have removed 49 one-hot encoded predictors with little or no variance.
#Altough this may seem a significant number, it is actually much less than the number of predictors that were taken out by using caret'snear zero variance function (using its default thresholds).
combined <- cbind(DFnorm, DFdummies) #combining all (now numeric) predictors into one dataframe

#=======Dealing with skewness of response variable
skew(all$SalePrice)
qqnorm(all$SalePrice)
qqline(all$SalePrice)
#The skew of 1.87 indicates a right skew that is too high, and the Q-Q plot shows that sale prices are also not normally distributed. To fix this I am taking the log of SalePrice.
#all$SalePrice <- log(all$SalePrice) #default is the natural logarithm, "+1" is not necessary as there are no 0's
all$SalePrice <- log(all$SalePrice) #default is the natural logarithm, "+1" is not necessary as there are no 0's
skew(all$SalePrice)
#As you can see,the skew is now quite low and the Q-Q plot is also looking much better.
qqnorm(all$SalePrice)
qqline(all$SalePrice)

#>>Composing train and test sets
train1 <- combined[!is.na(all$SalePrice),]
test1 <- combined[is.na(all$SalePrice),]

#=======Modeling
#---Lasso regression model
#tried Ridge and Elastic Net models, but since lasso gives the best results of those 3 models I am only keeping the lasso model in the document.
#The elastic-net penalty is controlled by alpha, and bridges the gap between lasso (alpha=1) and ridge (alpha=0).
#The tuning parameter lambda controls the overall strength of the penalty. It is known that the ridge penalty shrinks the coefficients of correlated predictors towards each other while the lasso tends to pick one of them and discard the others.
#I am using caret cross validation to find the best value for lambda, which is the only hyperparameter that needs to be tuned for the lasso model.
set.seed(27042018)
my_control <-trainControl(method="cv", number=5)
lassoGrid <- expand.grid(alpha = 1, lambda = seq(0.001,0.1,by = 0.0005))

lasso_mod <- train(x=train1, y=all$SalePrice[!is.na(all$SalePrice)], method='glmnet', trControl= my_control, tuneGrid=lassoGrid) 
lasso_mod$bestTune
min(lasso_mod$results$RMSE)
#The documentation of the caret `varImp' function says: for glmboost and glmnet the absolute value of the coefficients corresponding to the tuned model are used.
#Although this means that a real ranking of the most important variables is not stored, it gives me the opportunity to find out how many of the variables are not used in the model (and hence have coefficient 0).
lassoVarImp <- varImp(lasso_mod,scale=F)
lassoImportance <- lassoVarImp$importance

varsSelected <- length(which(lassoImportance$Overall!=0))
varsNotSelected <- length(which(lassoImportance$Overall==0))

cat('Lasso uses', varsSelected, 'variables in its model, and did not select', varsNotSelected, 'variables.')
#So lasso did what it is supposed to do: it seems to have dealt with multicolinearity well by not using about 45% of the available variables in the model.
LassoPred <- predict(lasso_mod, test1)
predictions_lasso <- exp(LassoPred) #need to reverse the log to the real values
head(predictions_lasso)

#=======XGBoost model
#. The main reason for this was that the package uses its own efficient datastructure (xgb.DMatrix). 
#The package also provides a cross validation function. However, this CV function only determines the optimal number of rounds, and does not support a full grid search of hyperparameters.
#Although caret does not seem to use the (fast) datastructure of the xgb package, I eventually decided to do hyperparameter tuning with it anyway, as it at least supports a full grid search.
#As far as I understand it, the main parameters to tune to avoid overfitting are max_depth, and min_child_weight (see XGBoost documentation). Below I am setting up a grid that tunes both these parameters, and also the eta (learning rate).

xgb_grid = expand.grid(
  nrounds = 1000,
  eta = c(0.1, 0.05, 0.01),
  max_depth = c(2, 3, 4, 5, 6),
  gamma = 0,
  colsample_bytree=1,
  min_child_weight=c(1, 2, 3, 4 ,5),
  subsample=1
)
#The next step is to let caret find the best hyperparameter values (using 5 fold cross validation).
xgb_caret <- train(x=train1, y=all$SalePrice[!is.na(all$SalePrice)], method='xgbTree', trControl= my_control, tuneGrid=xgb_grid) 
xgb_caret$bestTune
#In the remainder of this section, I will continue to work with the xgboost package directly. Below, I am starting with the preparation of the data in the recommended format.
label_train <- all$SalePrice[!is.na(all$SalePrice)]
# put our testing & training data into two seperates Dmatrixs objects
dtrain <- xgb.DMatrix(data = as.matrix(train1), label= label_train)
dtest <- xgb.DMatrix(data = as.matrix(test1))

#In addition, I am taking over the best tuned values from the caret cross validation.
default_param<-list(
  objective = "reg:linear",
  booster = "gbtree",
  eta=0.05, #default = 0.3
  gamma=0,
  max_depth=3, #default=6
  min_child_weight=4, #default=1
  subsample=1,
  colsample_bytree=1
)
#The next step is to do cross validation to determine the best number of rounds (for the given set of parameters).
xgbcv <- xgb.cv( params = default_param, data = dtrain, nrounds = 500, nfold = 5, showsd = T, stratified = T, print_every_n = 40, early_stopping_rounds = 10, maximize = F)
#Although it was a bit of work, the hyperparameter tuning definitly paid of, as the cross validated RMSE inproved considerably (from 0.1225 without the caret tuning, to 0.1162 in this version)!
#train the model using the best iteration found by cross validation
xgb_mod <- xgb.train(data = dtrain, params=default_param, nrounds = 375)
XGBpred <- predict(xgb_mod, dtest)
predictions_XGB <- exp(XGBpred) #need to reverse the log to the real values
head(predictions_XGB)

#view variable importance plot
#install.packages("Ckmeans.1d.dp")
library(Ckmeans.1d.dp) #required for ggplot clustering
mat <- xgb.importance (feature_names = colnames(train1),model = xgb_mod)
xgb.ggplot.importance(importance_matrix = mat[1:20], rel_to_first = TRUE)

#>>Averaging predictions
#Since the lasso and XGBoost algorithms are very different, averaging predictions likely improves the scores. As the lasso model does better regarding the cross validated RMSE score (0.1121 versus 0.1162), I am weigting the lasso model double.
sub_avg <- data.frame(Id = test_labels, SalePrice = (predictions_XGB+2*predictions_lasso)/3)
head(sub_avg)
write.csv(sub_avg, file = 'average.csv', row.names = F)
