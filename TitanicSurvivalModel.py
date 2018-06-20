
# coding: utf-8

# In[1]:


#Titanic Survival Model with Kaggle Dataset


# In[ ]:


#Data Processing


# In[2]:


#Import Required Library


# In[2]:


import pandas as pd #pandas is python library for data processing
import numpy as np #numpy is python library for scientific calculation


# In[3]:


#import CSV Dataset


# In[4]:


train_df = pd.read_csv('titanic_data.csv') #We read csv with pandas and assigned it to train_df


# In[5]:


#Now lets look at it


# In[6]:


train_df.head() #Head only displays some top rows and tail displays last rows


# In[7]:


#Now we look at data Info and analyze it


# In[8]:


train_df.info() #info gives information about overall data structure


# In[9]:


#now time to process the data
#Total Values are 891 
#cabin, embarked, age has some null values
#Cabin, Embarked, Ticket, Sex, Name are object.


# In[10]:


#what we need to do
#Drop cabin because it has huge null values and has no relation to survival
#drop name and ticket because it has no relation to survival
#fill missing age values with average
#convert sex to numerical values
#Fill Embarked with highest possible values


# In[11]:


train_df.drop('Cabin', axis =1, inplace=True) # drop the cabin


# In[12]:


train_df.drop(['Name','Ticket'], axis=1, inplace=True) #drop name and ticket. this is how we pass multiple var with list


# In[13]:


#now lets look at data. There will be no cabin, name and ticket


# In[14]:


train_df.head()


# In[15]:


#Lets Analyze The age


# In[16]:


age_null = train_df['Age'].isnull() #This store train_df age null status on boolen on age_null


# In[17]:


train_df[age_null] #this display all null ages datas


# In[18]:


#Now lets fill them with mean


# In[19]:


age_mean =train_df['Age'].mean() # we calculate mean of age and stored it on age_mean


# In[20]:


age_mean # this is calculated value


# In[21]:


#Filling the age


# In[22]:


age_null = train_df['Age'].fillna(age_mean, inplace =True) #we fill all the null ages with mean


# In[23]:


train_df['Age'].head()


# In[24]:


train_df.info() #Now there is no null age


# In[25]:


#Time for embarked


# In[26]:


null_embarked = train_df["Embarked"].isnull() #we assigned null embark value on null_embarked variable


# In[27]:


train_df[null_embarked] #retrive null contains embark


# In[28]:


#Now we look at statistics of embarked


# In[29]:


train_df['Embarked'].describe()


# In[30]:


#Because S has highest frequency we gonna replace null with S


# In[31]:


train_df['Embarked'].fillna('S', inplace=True)


# In[32]:


new_null_embarked = train_df['Embarked'].isnull() #we gonna check if there is any null value


# In[33]:


train_df[new_null_embarked] #There is no null value on embarked


# In[34]:


train_df.info() #Now there is no null embarked


# In[35]:


#Now we need to convert object to numerical, as there are two object


# In[36]:


#Lets deal with gender


# In[37]:


#we gonna convert male, frmale to integer with simple dictionary 
gender_maps ={
    'male':1,
    'female':2
}


# In[38]:


train_df['Sex'] = train_df['Sex'].map(gender_maps) #we replace Sex with gender_maps dictionary


# In[39]:


#Now lets look at the info


# In[40]:


train_df.info() #sex must be integer by now


# In[41]:


#Lets Deal with Embarked


# In[42]:


#now we gonna convert embarked value to integer with dictionary 


# In[43]:


embarked_map ={
    'S':1,
    'Q':2,
    'C':3
}


# In[44]:


train_df['Embarked'] = train_df['Embarked'].map(embarked_map) #we replace embarked with embarked_maps dictionary


# In[45]:


#Lets look at data status


# In[46]:


train_df.info()


# In[47]:


#The Data is perfect for accurate analysis now


# In[48]:


#Now we export processed data to CSV
train_df.to_csv("titanic.csv", index=False) #import to csv


# In[49]:


#The data is cleaned and managed now we can make model via machine learning


# In[50]:


#Machine Learning


# In[55]:


train_df.info()


# In[66]:


#Lets Import sklearn dicission tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# In[67]:


X_train, X_test, Y_train, Y_test = train_test_split(train_df, test_df,test_size=.20) #split training and test data


# In[70]:


X_test.info() #testing data information


# In[72]:


X_train.info() #training data info


# In[76]:


#Desession tree
decision_tree = DecisionTreeClassifier(random_state=4) #We Define classifier moudle with desission tree algorithm


# In[77]:


decision_tree.fit(X_train,Y_train) #We fit Training data and testing data on desision tree algorithm and trained the model


# In[79]:


decision_tree.predict(X_test[0:1]) #time for prediction


# In[80]:


decision_tree.predict(X_test[0:5]) #predicting for multiple values 0-5


# In[84]:


#Time to check accuracy for model
decision_tree.score(X_test, Y_test) #wonder how this is possibele

