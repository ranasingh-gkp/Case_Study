#Exploratory Data Analysis (EDA)
install.packages("arules")
dataset=BlackFriday
library(tidyverse)
library(scales)
library(arules)
library(gridExtra)
summary(dataset)

head(dataset)

#Gender
dataset_gender = dataset %>%
  select(User_ID, Gender) %>%
  group_by(User_ID) %>%
  distinct()  

head(dataset_gender)

summary(dataset_gender$Gender)
options(scipen=10000)   # To remove scientific numbering

genderDist  = ggplot(data = dataset_gender) +
  geom_bar(mapping = aes(x = Gender, y = ..count.., fill = Gender)) +
  labs(title = 'Gender of Customers') + 
  scale_fill_brewer(palette = 'PuBuGn')
print(genderDist)

total_purchase_user = dataset %>%
  select(User_ID, Gender, Purchase) %>%
  group_by(User_ID) %>%
  arrange(User_ID) %>%
  summarise(Total_Purchase = sum(Purchase))

user_gender = dataset %>%
  select(User_ID, Gender) %>%
  group_by(User_ID) %>%
  arrange(User_ID) %>%
  distinct()

head(user_gender)
head(total_purchase_user)

user_purchase_gender = full_join(total_purchase_user, user_gender, by = "User_ID")
head(user_purchase_gender)

average_spending_gender = user_purchase_gender %>%
  group_by(Gender) %>%
  summarize(Purchase = sum(as.numeric(Total_Purchase)), 
            Count = n(), 
            Average = Purchase/Count)
head(average_spending_gender)
#We can see that that the average transaction for Females was 699054.00 and the average transaction for Males was 911963.20. Let visualize our results.
  genderAverage  = ggplot(data = average_spending_gender) +
  geom_bar(mapping = aes(x = Gender, y = Average, fill = Gender), stat = 'identity') +
  labs(title = 'Average Spending by Gender') +
  scale_fill_brewer(palette = 'PuBuGn')
print(genderAverage)


#-----Top Sellers
top_sellers = dataset %>%
  count(Product_ID, sort = TRUE)

top_5 = head(top_sellers, 5)

top_5
best_seller = dataset[dataset$Product_ID == 'P00265242', ]

head(best_seller)
#We can see that this product fits into Product_Category_1 = 5 and Product_Category_2 = 8. As mentioned in the introduction, it would be useful to have a key to reference the item name in order to determine what it is
#Another interesting finding is that even though people are purchasing the same product, they are paying different prices. This could be due to various Black Friday promotions, discounts, or coupon codes. Otherwise, investigation would need to be done regarding the reason for different purchase prices of the same product between customers.
#Lets continue to analyze our best seller to see if any relationship to Gender exits.

  genderDist_bs  = ggplot(data = best_seller) +
  geom_bar(mapping = aes(x = Gender, y = ..count.., fill = Gender)) +
  labs(title = 'Gender of Customers (Best Seller)') +
  scale_fill_brewer(palette = 'PuBuGn')
print(genderDist_bs)
#We see a similar distribution between genders to our overall dataset gender split - lets confirm.
genderDist_bs_prop = ggplot(data = best_seller) + 
  geom_bar(fill = 'lightblue', mapping = aes(x = Gender, y = ..prop.., group = 1, fill = Gender)) +
  labs(title = 'Gender of Customers (Best Seller - Proportion)') +
  theme(plot.title = element_text(size=9.5))

genderDist_prop = ggplot(data = dataset_gender) + 
  geom_bar(fill = "lightblue4", mapping = aes(x = Gender, y = ..prop.., group = 1)) +
  labs(title = 'Gender of Customers (Total Proportion)') +
  theme(plot.title = element_text(size=9.5)) 

grid.arrange(genderDist_prop, genderDist_bs_prop, ncol=2)
#We can see that between the overall observation set, both purchasers of the best seller and purchasers of all products are roughly ~25% female and ~75% male. A slight difference does exist but it seems like we can generally conclude that our best seller does not cater to a specific gender.

#----Age
customers_age = dataset %>%
  select(User_ID, Age) %>%
  distinct() %>%
  count(Age)
customers_age

customers_age_vis = ggplot(data = customers_age) + 
  geom_bar(color = 'black', stat = 'identity', mapping = aes(x = Age, y = n, fill = Age)) +
  labs(title = 'Age of Customers') +
  theme(axis.text.x = element_text(size = 10)) +
  scale_fill_brewer(palette = 'Blues') +
  theme(legend.position="none")
print(customers_age_vis)
#We can also plot a similar chart depicting the distribution of age within our "best seller" category. This will show us if there is a specific age category that purchased the best selling product more than other shoppers.
ageDist_bs  = ggplot(data = best_seller) +
  geom_bar(color = 'black', mapping = aes(x = Age, y = ..count.., fill = Age)) +
  labs(title = 'Age of Customers (Best Seller)') +
  theme(axis.text.x = element_text(size = 10)) +
  scale_fill_brewer(palette = 'GnBu') + 
  theme(legend.position="none")
print(ageDist_bs)
#It seems as though younger people (18-25 & 26-35) account for the highest number of purchases of the best selling product. Lets compare this observation to the overall dataset.
grid.arrange(customers_age_vis, ageDist_bs, ncol=2)

#------City
#Let's create a table of each User_ID and their corresponding City_Category.

customers_location =  dataset %>%
  select(User_ID, City_Category) %>%
  distinct()
head(customers_location)


customers_location_vis = ggplot(data = customers_location) +
    geom_bar(color = 'white', mapping = aes(x = City_Category, y = ..count.., fill = City_Category)) +
    labs(title = 'Location of Customers') + 
    scale_fill_brewer(palette = "Dark2") + 
    theme(legend.position="none")
print(customers_location_vis)
#We can see that most of our customers live in City C. Now, we can compute the total purchase amount by City to see the which city's customers spent the most at our store.

  purchases_city = dataset %>%
    group_by(City_Category) %>%
    summarise(Purchases = sum(Purchase))
  
  purchases_city_1000s = purchases_city %>%
    mutate(purchasesThousands = purchases_city$Purchases / 1000)
  
  purchases_city_1000s

  
  purchaseCity_vis = ggplot(data = purchases_city_1000s, aes(x = City_Category, y = purchasesThousands, fill = City_Category)) +
    geom_bar(color = 'white', stat = 'identity') +
    labs(title = 'Total Customer Purchase Amount (by City)', y = '($000s)', x = 'City Category') +
    scale_fill_brewer(palette = "Dark2") + 
    theme(legend.position="none", plot.title = element_text(size = 9))
  print(purchaseCity_vis)  

  grid.arrange(customers_location_vis, purchaseCity_vis, ncol=2)
#  Here we can see that customers from City C were the most frequent shoppers at our store on Black Friday but Customers from City B had the highest amount of total purchases.
#  Lets find how many purchases were made by customers from each city. First, we will get the total number of purchases for each corresponding User_ID.
  customers = dataset %>%
    group_by(User_ID) %>%
    count(User_ID)
  head(customers)
#  This tells us how many times a certain user made a purchase. To dive deeper lets compute the total purchase amount for each user, then join it with the other table
  customers_City =  dataset %>%
    select(User_ID, City_Category) %>%
    group_by(User_ID) %>%
    distinct() %>%
    ungroup() %>%
    left_join(customers, customers_City, by = 'User_ID') 
  head(customers_City)
  
  city_purchases_count = customers_City %>%
    select(City_Category, n) %>%
    group_by(City_Category) %>%
    summarise(CountOfPurchases = sum(n))
  city_purchases_count

  city_count_purchases_vis = ggplot(data = city_purchases_count, aes(x = City_Category, y = CountOfPurchases, fill = City_Category)) +
    geom_bar(color = 'white', stat = 'identity') +
    labs(title = 'Total Purchase Count (by City)', y = 'Count', x = 'City Category') +
    scale_fill_brewer(palette = "Dark2") +
    theme(legend.position="none", plot.title = element_text(size = 9))
  print(city_count_purchases_vis)  
  grid.arrange(purchaseCity_vis, city_count_purchases_vis, ncol = 2)               
#One inference we can make from these charts is that customers from City B are simply making more purchases than residence of City A + City C, and not necessarily buying more expensive products.
#  Now, since we have identified that the purchase counts across City_Category follow a similar distribution to total purchase amount, lets examine the distribution of our best selling product (P00265242) within each City_Category.
  head(best_seller)
  
best_seller_city = best_seller %>%
    select(User_ID, City_Category) %>%
    distinct() %>%
    count(City_Category)
best_seller_city

best_seller_city_vis = ggplot(data = best_seller_city, aes(x = City_Category, y = n, fill = City_Category)) +
  geom_bar(color = 'white', stat = 'identity') +
  labs(title = 'Best Seller Purchase Count (by City)', y = 'Count', x = 'City Category') +
  scale_fill_brewer(palette = "Blues") +
  theme(legend.position="none", plot.title = element_text(size = 9))
grid.arrange(city_count_purchases_vis,best_seller_city_vis, ncol = 2)
#An interesting revelation has been made! Although customers residing in City C purchase more of our "best seller" than City A + B, residents of City C fall behind City B in overall number of purchases.

#-----Stay in Current City
#Lets now examine the distribution of customers who have lived in their city the longest.
customers_stay = dataset %>%
  select(User_ID, City_Category, Stay_In_Current_City_Years) %>%
  group_by(User_ID) %>%
  distinct()
head(customers_stay)
#Lets see where most of our customers are living.
residence = customers_stay %>%
  group_by(City_Category) %>%
  tally()
head(residence)
#Looks like most of our customers are living in City C.,Now, lets investigate further.
customers_stay_vis = ggplot(data = customers_stay, aes(x = Stay_In_Current_City_Years, y = ..count.., fill = Stay_In_Current_City_Years)) +
  geom_bar(stat = 'count') +
  scale_fill_brewer(palette = 15) +
  labs(title = 'Customers Stay in Current City', y = 'Count', x = 'Stay in Current City', fill = 'Number of Years in Current City')
print(customers_stay_vis)
#It looks like most of our customers have only been living in their respective cities for 1 year. In order to see a better distribution, lets make a stacked bar chart according to each City_Category

stay_cities = customers_stay %>%
  group_by(City_Category, Stay_In_Current_City_Years) %>%
  tally() %>%
  mutate(Percentage = (n/sum(n))*100)
head(stay_cities)


ggplot(data = stay_cities, aes(x = City_Category, y = n, fill = Stay_In_Current_City_Years)) + 
  geom_bar(stat = "identity", color = 'white') + 
  scale_fill_brewer(palette = 2) + 
  labs(title = "City Category + Stay in Current City", 
       y = "Total Count (Years)", 
       x = "City", 
       fill = "Stay Years")
