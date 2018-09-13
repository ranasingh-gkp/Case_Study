#Exploratory Analysis Zillow
#https://www.kaggle.com/philippsp/exploratory-analysis-zillow
#-------------------------------
library(data.table)
library(dplyr)
library(ggplot2)
library(stringr)
library(DT)
library(tidyr)
library(corrplot)
library(leaflet)
library(lubridate)
install.packages("leaflet")
properties <- fread('properties_2016.csv')
transactions <- fread('train_2016.csv')
sample_submission <- fread('sample_submission.csv')

properties <- properties %>% rename(
  id_parcel = parcelid,
  build_year = yearbuilt,
  area_basement = basementsqft,
  area_patio = yardbuildingsqft17,
  area_shed = yardbuildingsqft26, 
  area_pool = poolsizesum,  
  area_lot = lotsizesquarefeet, 
  area_garage = garagetotalsqft,
  area_firstfloor_finished = finishedfloor1squarefeet,
  area_total_calc = calculatedfinishedsquarefeet,
  area_base = finishedsquarefeet6,
  area_live_finished = finishedsquarefeet12,
  area_liveperi_finished = finishedsquarefeet13,
  area_total_finished = finishedsquarefeet15,  
  area_unknown = finishedsquarefeet50,
  num_unit = unitcnt, 
  num_story = numberofstories,  
  num_room = roomcnt,
  num_bathroom = bathroomcnt,
  num_bedroom = bedroomcnt,
  num_bathroom_calc = calculatedbathnbr,
  num_bath = fullbathcnt,  
  num_75_bath = threequarterbathnbr, 
  num_fireplace = fireplacecnt,
  num_pool = poolcnt,  
  num_garage = garagecarcnt,  
  region_county = regionidcounty,
  region_city = regionidcity,
  region_zip = regionidzip,
  region_neighbor = regionidneighborhood,  
  tax_total = taxvaluedollarcnt,
  tax_building = structuretaxvaluedollarcnt,
  tax_land = landtaxvaluedollarcnt,
  tax_property = taxamount,
  tax_year = assessmentyear,
  tax_delinquency = taxdelinquencyflag,
  tax_delinquency_year = taxdelinquencyyear,
  zoning_property = propertyzoningdesc,
  zoning_landuse = propertylandusetypeid,
  zoning_landuse_county = propertycountylandusecode,
  flag_fireplace = fireplaceflag, 
  flag_tub = hashottuborspa,
  quality = buildingqualitytypeid,
  framing = buildingclasstypeid,
  material = typeconstructiontypeid,
  deck = decktypeid,
  story = storytypeid,
  heating = heatingorsystemtypeid,
  aircon = airconditioningtypeid,
  architectural_style= architecturalstyletypeid
)

transactions <- transactions %>% rename(
  id_parcel = parcelid,
  date = transactiondate)

properties <- properties %>% 
  mutate(tax_delinquency = ifelse(tax_delinquency=="Y",1,0),
         flag_fireplace = ifelse(flag_fireplace=="Y",1,0),
         flag_tub = ifelse(flag_tub=="Y",1,0))


tmp <- transactions %>% mutate(year_month = make_date(year=year(date),month=month(date)))
tmp %>% 
  group_by(year_month) %>% count() %>% 
  ggplot(aes(x=year_month,y=n)) +
  geom_bar(stat="identity", fill="red")+
  geom_vline(aes(xintercept=as.numeric(as.Date("2016-10-01"))),size=2)

#Outcome
#To get a feel for the data let's first have a look at the distribution of our outcome (logerror), i.e. the difference in log(Zestimate)-log(Saleprice)
transactions %>% 
  ggplot(aes(x=logerror)) + 
  geom_histogram(bins=400, fill="red")+
  theme_bw()+theme(axis.title = element_text(size=16),axis.text = element_text(size=14))+
  ylab("Count")+coord_cartesian(x=c(-0.5,0.5))

#Absolute logerror
transactions <- transactions %>% mutate(abs_logerror = abs(logerror))
transactions %>% 
  ggplot(aes(x=abs_logerror)) + 
  geom_histogram(bins=400, fill="red")+
  theme_bw()+theme(axis.title = element_text(size=16),axis.text = element_text(size=14))+
  ylab("Count")+coord_cartesian(x=c(0,0.5))
#How does absolute log error change with time
transactions %>% 
  mutate(year_month = make_date(year=year(date),month=month(date)) ) %>% 
  group_by(year_month) %>% summarize(mean_abs_logerror = mean(abs_logerror)) %>% 
  ggplot(aes(x=year_month,y=mean_abs_logerror)) + 
  geom_line(size=1.5, color="red")+
  geom_point(size=5, color="red")+theme_bw()
#How does log error change with time
transactions %>% 
  mutate(year_month = make_date(year=year(date),month=month(date)) ) %>% 
  group_by(year_month) %>% summarize(mean_logerror = mean(logerror)) %>% 
  ggplot(aes(x=year_month,y=mean_logerror)) + 
  geom_line(size=1.5, color="red")+geom_point(size=5, color="red")+theme_bw()

#Missing values
#We have seen many missing values in the data peeking. How many missing values are there for each feature? In fact, some features are missing nearly completely. So, we probably have to work more with the others.
missing_values <- properties %>% summarize_each(funs(sum(is.na(.))/n()))

missing_values <- gather(missing_values, key="feature", value="missing_pct")
missing_values %>% 
  ggplot(aes(x=reorder(feature,-missing_pct),y=missing_pct)) +
  geom_bar(stat="identity",fill="red")+
  coord_flip()+theme_bw()
good_features <- filter(missing_values, missing_pct<0.75)

#Correlation with absolute logerror
#num_ features:
vars <- good_features$feature[str_detect(good_features$feature,'num_')]

cor_tmp <- transactions %>% left_join(properties, by="id_parcel") 
tmp <- cor_tmp %>% select(one_of(c(vars,"abs_logerror")))

corrplot(cor(tmp, use="complete.obs"),type="lower")

#Correlation with absolute logerror
#area_ features

vars <- good_features$feature[str_detect(good_features$feature,'area_')]

tmp <- cor_tmp %>% select(one_of(c(vars,"abs_logerror")))

corrplot(cor(tmp, use="complete.obs"), type="lower")

#Correlation with absolute logerror
#tax_ features

vars <- setdiff(good_features$feature[str_detect(good_features$feature,'tax_')],c("tax_delinquency","tax_year"))

tmp <- cor_tmp %>% select(one_of(c(vars,"abs_logerror")))

corrplot(cor(tmp, use="complete.obs"), type="lower")
#Correlation with logerror
#num_ features: There seems to be small negative correlations between the num features and logerror.

vars <- good_features$feature[str_detect(good_features$feature,'num_')]

cor_tmp <- transactions %>% left_join(properties, by="id_parcel") 
tmp <- cor_tmp %>% select(one_of(c(vars,"logerror")))

corrplot(cor(tmp, use="complete.obs"),type="lower")

#Overall, correlations with log error are quite small. This is a hint that predictions are quite good already. However, we might still find some features or at least ranges of feature values where predictions can still be improved.

#When were the houses built?
#Let's plot the distribution of build year for the houses. Most houses were built around 1950. There are not many older houses, neither many new houses >2000.

cor_tmp %>% 
  ggplot(aes(x=build_year))+geom_line(stat="density", color="red", size=1.2)+theme_bw()

#How does the absolute logerror change with build_year?
#Predictions are better for newer houses. As we saw in the figure above we have way fewer older houses. However, what is interesting is that there are many houses with build year around 1950, but predictions for those houses are not too good.
cor_tmp %>% 
  group_by(build_year) %>% 
  summarize(mean_abs_logerror = mean(abs(logerror)),n()) %>% 
  ggplot(aes(x=build_year,y=mean_abs_logerror))+
  geom_smooth(color="grey40")+
  geom_point(color="red")+coord_cartesian(ylim=c(0,0.25))+theme_bw()
#How does the logerror change with build_year?
cor_tmp %>% 
  group_by(build_year) %>% 
  summarize(mean_logerror = mean(logerror)) %>% 
  ggplot(aes(x=build_year,y=mean_logerror))+
  geom_smooth(color="grey40")+
  geom_point(color="red")+coord_cartesian(ylim=c(0,0.075))+theme_bw()
## `geom_smooth()` using method = 'loess' and formula 'y ~ x'

#Where does Zestimate predict well?
#To get a quick feel where zestimate predicts well, we can group our absolute logerror into different percentiles, e.g. the percentile with best predictions (top 10%), worst predictions (worst 10%) and typical predictions (50% around the median).
transactions <- transactions %>% mutate(percentile = cut(abs_logerror,quantile(abs_logerror, probs=c(0, 0.1, 0.25, 0.75, 0.9, 1),names = FALSE),include.lowest = TRUE,labels=FALSE))

tmp1 <- transactions %>% 
  filter(percentile == 1) %>% 
  sample_n(5000) %>% 
  left_join(properties, by="id_parcel")
tmp2 <- transactions %>% 
  filter(percentile == 5) %>% 
  sample_n(5000) %>% 
  left_join(properties, by="id_parcel")
tmp3 <- transactions %>% 
  filter(percentile == 3) %>% 
  sample_n(5000) %>% 
  left_join(properties, by="id_parcel")

tmp1 <- tmp1 %>% mutate(type="best_fit")
tmp2 <- tmp2 %>% mutate(type="worst_fit")
tmp3 <- tmp3 %>% mutate(type="typical_fit")
tmp <- bind_rows(tmp1,tmp2,tmp3)
tmp <- tmp %>% mutate(type = factor(type,levels = c("worst_fit", "typical_fit", "best_fit")))
#If the distributions of features are largely overlapping for these three groups of transactions the feature most likely does not have a large effect on the goodness of estimation. Let's see one example.
col_pal <- "Set1"

tmp %>% ggplot(aes(x=latitude, fill=type, color=type)) + geom_line(stat="density", size=1.2) + theme_bw()+scale_fill_brewer(palette=col_pal)+scale_color_brewer(palette=col_pal)
#We can see that rows resulting in the worst predictions have a lower density for lower latitude values, but a higher density for intermediate latitudes (around 34000000).
#We can examine this effect more closely and plot the absolute logerror as a function of latitude.

tmptrans <- transactions %>% 
  left_join(properties, by="id_parcel")

tmptrans %>% 
  ggplot(aes(x=latitude,y=abs_logerror))+geom_smooth(color="red")+theme_bw()
#Having seen the example, we can look at other features quickly, to see which are associated with absolute logerror.
