#!/usr/bin/env python
# coding: utf-8

# # IMPORTING THE LIBRARIES

# In[1]:


#for data manipulation and numerical tasks 
import pandas as pd 
import numpy as np 
#for data visualization and multidimensional plots
import matplotlib.pyplot as plt 
import seaborn as sns 


# # IMPORTING THE DATA

# In[2]:


df = pd.read_csv(r"C:\Users\pooji\Downloads\hotel_bookings.csv") #import the data in the project
df.head()


# # DATA CLEANING 

# In[3]:


df.shape


# In[4]:


df.isna().sum() #finding the empty/null values


# In[5]:


#filling the spaces of non available data
def data_clean(df):
    df.fillna(0,inplace = True) #zero imputation
    print(df.isnull().sum())


# In[6]:


data_clean(df)


# In[7]:


#finding the unique no of values for children, adults and babies
list_cols = ["children", "adults", "babies"]

for i in list_cols:
    print(f"{i} has unique values as {df[i].unique()}")


# In[8]:


#cleaning and filtering the data 
filtered_data = (df['children'] == 0) & (df['adults'] == 0) & (df['babies'] == 0)
final_data = df[~filtered_data] #filter it out


# In[9]:


final_data.shape


# # WHERE DO THE GUESTS COME FROM (SPATIAL ANALYSIS)

# In[10]:


final_data[final_data['is_canceled'] == 0]['country']    # '0' = booking was NOT cancelled ; '1' = booking WAS cancelled      


# In[11]:


final_data[final_data['is_canceled'] == 0]['country'].value_counts().reset_index()


# In[12]:


country_wise_data =  final_data[final_data['is_canceled'] == 0]['country'].value_counts().reset_index()
country_wise_data.columns = ["Country", "No.of guests"]
print(country_wise_data)


# In[13]:


import plotly.express as px


# In[14]:


#plotting the map (to find out from which country the guests come from)
map_guests = px.choropleth(country_wise_data, locations = country_wise_data['Country'],
                           color = country_wise_data["No.of guests"],
                           hover_name = country_wise_data['Country'],
                           title = "Home country of guests"
                          )

map_guests.show()


# # HOW MUCH DO THE GUESTS PAY PER ROOM PER NIGHT

# In[15]:


final_data.head()


# In[16]:


final_data["hotel"].unique()


# In[17]:


df["adr"]


# In[18]:


data = final_data[final_data["is_canceled"] == 0]


# In[19]:


#boxplot
plt.figure(figsize = (12,8)) #default

sns.boxplot(x = "reserved_room_type",
           y = "adr", 
           hue = "hotel", data = data)

plt.title("Price of room types per night per person", fontsize = 16)
plt.xlabel("Room type")
plt.ylabel("Price [EUR]")
plt.legend(loc = "upper right")
plt.ylim(0,600)
plt.show()


# We can now analyze the median (black line inside box) prices of each graphed type, and see which were the most economic or most expenssive classes ("which room type to book given your specifications").
# 
# We can give recommendations to our customers/guests, so they can book the hotel on any given season.

# # HOW DOES THE PRICE PER NIGHT (ADR) VARY OVER THE YEARS?

# In[20]:


final_data.head()


# In[21]:


final_data["hotel"].unique()


# In[22]:


data_resort = final_data[(final_data["hotel"] == "Resort Hotel") & (final_data["is_canceled"] == 0)]
data_city = final_data[(final_data["hotel"] == "City Hotel") & (final_data["is_canceled"] == 0)]


# In[23]:


#how many people came to the resort hotel per month
resort_hotel = data_resort.groupby(['arrival_date_month'])['adr'].mean().reset_index()


# In[24]:


#how many people came to the city hotel per month
city_hotel = data_city.groupby(['arrival_date_month'])['adr'].mean().reset_index()


# In[25]:


#combining both the data variables into 1
final = resort_hotel.merge(city_hotel, on = "arrival_date_month")

final.columns = ["month", "price_for_resort_hotel", "price_for_city_hotel"]


# In[26]:


print(final)


# # SORTING THE MONTH

# In[27]:


# from calendar import month_name
# for i , name in enumerate(month_name) : 
   # print(i , name)                               # for list of months with their index #


# In[28]:


from calendar import month_name

def sort_month(df, colname):
    month_dict = { j:i for i, j in enumerate(month_name) } #dictionary comprehension
    df["month_num"] = df[colname].apply(lambda x: month_dict[x])
    return df.sort_values(by = "month_num").reset_index().drop(['index', 'month_num'], axis = 1)


# In[29]:


sort_month(final, "month")


# In[30]:


final.plot(kind = "line", x = "month", y = ['price_for_resort_hotel','price_for_city_hotel'])


# Our graph is now plotted, and we'll get the following conclusions from it:
# 
# Plot shows price rises in certain months;(ugust-Resort and May-City)
# Resort rooms have higher peaks (up and down), so price ranges significantly in specific months;
# Etc.

# # MOST BUSIEST MONTHS

# In[31]:


data_resort.head()


# In[32]:


#creating a  variable to know the rush months/seasons
rush_resort = data_resort["arrival_date_month"].value_counts().reset_index()
rush_resort.columns = ["month", "no of guests"]

print(rush_resort)


# In[33]:


rush_city = data_city["arrival_date_month"].value_counts().reset_index()
rush_city.columns = ["month", "no of guests"]

print(rush_city)


# In[34]:


#merge two data frame rush_resort , rush_city

final_rush = rush_resort.merge(rush_city, on = "month")

final_rush.columns = ["month", "no of guests in resort hotel", "no of guests in city hotel"]

final_rush


# In[35]:


final_rush = sort_month(final_rush, "month")
print(final_rush)


# In[36]:


#plot showing month vs no of guests
final_rush.plot(kind = "line", x = "month", 
                y = ["no of guests in resort hotel","no of guests in city hotel" ])


# Our graph is now plotted, and we'll get the following conclusions from it:
# 
# Plot shows rushs in certain months;(August for both city and resort-hotels)
# City hotels have higher rush during the summer holidays.

# # HOW LONG DO PEOPLE STAY AT THE HOTELS?

# In[37]:


filter_condition = final_data['is_canceled'] == 0

clean_data = final_data[filter_condition]


# In[38]:


clean_data.head()


# In[39]:


clean_data["total_nights"] = clean_data["stays_in_weekend_nights"] + clean_data["stays_in_week_nights"]


# In[40]:


clean_data.head()


# In[41]:


stay = clean_data.groupby(["total_nights", "hotel"]).agg('count').reset_index()

stay = stay.iloc[:, 0:3]
print(stay)


# In[42]:


stay = stay.rename(columns = {'is_canceled': 'Number of stays'})


# In[43]:


print(stay)


# In[44]:


#plotting the barplot
sns.barplot(x = "total_nights", y = "Number of stays", hue = "hotel",
           hue_order = ["City Hotel", "Resort Hotel"], data = stay)


# # SELECTIOM OF IMPORTANT NUMERICAL FEATURES USING CORRELATION

# In[45]:


import warnings
from warnings import filterwarnings
filterwarnings("ignore")


# In[46]:


correlation = final_data.corr()


# In[47]:


#correlation["is_canceled"]  
correlation = correlation["is_canceled"][1 : ] #filter parameter


# In[51]:


correlation.abs().sort_values(ascending = False)


# In[48]:


list_not = ["days_in_waiting_list", "arrival_date_year"]


# In[49]:


num_features = [col for col in final_data.columns if final_data[col].dtype != "O" and col not in list_not]  #numerical one


# In[50]:


print(num_features) #we have ALL the numerical features, except the 'list_not' attributes we previously filtered.


# # SELECT IMPORTANT CATEGORICAL FEATURES

# In[52]:


final_data["reservation_status"].value_counts() #checking the total canceled/check-Out, No show


# In[53]:


final_data.columns


# In[54]:


cat_not = ["country", "reservation_status", "booking_changes", "assigned_room_type", "days_in_waiting_list" ]


# In[55]:


#creating a new variable for categorical features
cat_features = [col for col in final_data.columns 
                if final_data[col].dtype == "O" and col not in cat_not]


# In[56]:


print(cat_features)


# In[57]:


print(num_features)


# All of these 'features' listed above, will be used for our ML Prediction Model.

# In[58]:


data_cat = final_data[cat_features]


# In[59]:


print(data_cat.head())


# In[60]:


import warnings
from warnings import filterwarnings
filterwarnings("ignore")

data_cat["reservation_status_date"] = pd.to_datetime(data_cat["reservation_status_date"])


# In[61]:


data_cat["year"] = data_cat["reservation_status_date"].dt.year

data_cat["month"] = data_cat["reservation_status_date"].dt.month

data_cat["day"] = data_cat["reservation_status_date"].dt.day


# In[62]:


data_cat.drop("reservation_status_date" , axis = 1 , inplace = True)   
data_cat.head()


# # FEATURE ENCODING
# MEAN ENCODING TECHNIQUE

# In[63]:


data_cat.columns


# In[64]:


data_cat["cancellation"] = final_data["is_canceled"]


# In[65]:


#Define a function for our Mean Encoding (while also creating a dictionary)
def mean_encode(df , col , mean_col) :     # 'mean_col' = by which column do you want to create the 'mean encoding technique'? #
    
    df_dict = df.groupby([col])[mean_col].mean().to_dict()
    
    df[col] = df[col].map(df_dict)
    
    return df



# ° Let's call in our desired columns to encode, 
for col in data_cat.columns[0 : 8] :     
    
    data_cat = mean_encode(data_cat , col , "cancellation")


# In[66]:


print(data_cat)


# In[67]:


data_cat.drop(["cancellation"] , axis = 1 , inplace = True)   


# # PREPARING OUR DATA

# In[68]:


num_data = final_data[num_features] #numerical data

cat_data = data_cat #categorical data

dataframe = pd.concat([num_data, cat_data], axis = 1) #final dataframe


# In[69]:


print(dataframe.head())


# # HANDLING THE OUTLIERS

# In[70]:


dataframe.describe()["adr"]


# Here, we'll see the max values are way-off or way-too-high if we compare them to the mean values of the different Quartiles (25%, 50%, 75%).

# In[71]:


#using seaborn library for visualizing the range
sns.distplot(dataframe["lead_time"])


# We can conclude that our values are not evenly distributed, therefore: our Algorithm will be biased (favouring the higher value in our 'training' process, while decreasing the accuracy of our prediction).

# In[72]:


#function to handle the outlier
def handle_outlier(col):
    dataframe[col] = np.log1p(dataframe[col])


# In[73]:


handle_outlier("lead_time")


# In[74]:


sns.distplot(dataframe["lead_time"].dropna())


# It's no longer positively-schewed, but it's also not in SNL either (this graph is called density plot)

# In[75]:


sns.distplot(dataframe["adr"])


# In[76]:


handle_outlier("adr")
sns.distplot(dataframe["adr"])


# In[77]:


dataframe.isnull().sum()# impuataion of non-null values


# In[78]:


dataframe.dropna(inplace = True)
dataframe.isnull().sum()


# We can do 'log' transformations for the remaining attributes we want to prepare, [ log(1 + "max_num") = log(ex."99") = will be added, in place of "98" ]

# # FEATURE IMPORTANCE

# In[79]:


##separate dependent and independent variables

y = dataframe["is_canceled"] #dependent variable

X = dataframe.drop("is_canceled", axis = 1) #independent


# In[80]:


X.columns


# In[81]:


#To predict which attributes will be important, [ we'll use 'scikit-learn' lib for our LASSO model and our feature selection ]
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel


# In[82]:


feature_sel_model = SelectFromModel(Lasso(alpha = 0.005, random_state = 0))


# In[83]:


feature_sel_model.fit(X, y)


# In[84]:


feature_sel_model.get_support()


# In[85]:


cols = X.columns

print(cols)


# In[86]:


selected_feature = cols[(feature_sel_model.get_support())]

print(selected_feature)


# In[87]:


print(f"Total features {X.shape[1]}")      


# In[88]:


print(f"Selected features {len(selected_feature)}")


# In[89]:


X = X[selected_feature]

X.columns


# In[90]:


X.columns


# In[91]:


X.head


# # SPLITTING THE DATA AND MODEL BUILDING

# In[94]:


#spliting our data into 'training' and 'testing', [ using scikit-learn ]
from sklearn.model_selection import train_test_split


# In[95]:


x_train , x_test , y_train , y_test = train_test_split(X , y , train_size = 0.75 , random_state = 45)   # '45' = random, no special meaning the amount you put in #


# We have a (binary) classification problem, since our Y Prediction 'is_canceled' has 2 classes of values.
# 
# To solve this, we can use logistic regression. (we can't use other models like 'linear regression' in the case of a classification problem).

# # IMPLIMENT LOGISTIC REGRESSION

# In[96]:


from sklearn.linear_model import LogisticRegression


# In[97]:


logistic_model = LogisticRegression()
#we fit our LR Model to our 'training' set we created above, [ this step is the actual training of our Machine Learning Model
logistic_model.fit(x_train, y_train) #training of the model


# In[98]:


#using our fresh model to predict our 'testing' data (and create a variable for it), [ this step is the actual prediction of our Model ]
y_pred = logistic_model.predict(x_test) #prediction by model


# In[99]:


from sklearn.metrics import confusion_matrix


# In[100]:


confusion_matrix(y_test, y_pred)


# In[101]:


from sklearn.metrics import accuracy_score


# In[102]:


accuracy_score(y_test, y_pred)


# # implementing different classification algorithms
# logistic regression
# Naive Bayes
# Random Forest
# Decision Tree
# KNN

# Algorithms for Classification we'll implement:
# a) Logistic Regression
# b) Naive Bayes
# c) Random Forest
# d) KNN
# e) Decision Tree

# In[103]:


from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# In[104]:


models = []

models.append(("Naive Bayes", GaussianNB()))
models.append(("Random Forest", RandomForestClassifier()))
models.append(("Decision Tree", DecisionTreeClassifier()))
models.append(("KNN", KNeighborsClassifier(n_neighbors = 5)))


# In[105]:


for name, model in models:
    print(name)
    model.fit(x_train, y_train)
    
    #make a predictions
    predictions = model.predict(x_test)
    
    #evaluate a model
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(predictions, y_test))
    
    from sklearn.metrics import accuracy_score
    print(accuracy_score(predictions, y_test))
    
    print("\n")
    
    
    


# #### So we can analyze from our results that there are some results lower than our first prediction, but almost all the others are way higher that the one we decided to use first.
