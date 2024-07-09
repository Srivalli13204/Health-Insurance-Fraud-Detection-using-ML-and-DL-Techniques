#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing_libraries


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[3]:


#loading_dataset


# In[4]:


df = pd.read_csv("C:/Users/MADHUSUDAN/Downloads/insurance_data.csv")
df


# In[5]:


#replace?


# In[6]:


df.replace('?', np.nan, inplace = True)
df


# In[7]:


df.head()


# In[8]:


df.describe()


# In[9]:


df.info()


# In[10]:


df['fraud_reported'].hist()
plt.xlabel('Fraud found')
plt.ylabel('Frequency')
plt.title('Class distribution')


# In[11]:


#null_values


# In[12]:


df['fraud_reported'].shape


# In[13]:


df.isna().sum()


# In[14]:


#missing_values


# In[15]:


df.isnull().sum()


# In[16]:


#percentage_graph


# In[17]:


sns.heatmap(df.isnull(), cbar=False)


# In[18]:


missing = df.isnull().sum() / len(df)
missing = missing[missing>0]
missing.sort_values(inplace=True)
missing = missing.to_frame()
missing.columns = ['Null_Count']
missing.index.names  = ['Col_Name']
missing = missing.reset_index()
sns.set(style='whitegrid', color_codes=True)
sns.barplot(x='Col_Name', y='Null_Count', data=missing)
plt.xticks(rotation = 45)
plt.show()


# In[19]:


#filling_values


# In[20]:


df['collision_type'] = df['collision_type'].fillna(df['collision_type'].mode()[0])
df['property_damage'] = df['property_damage'].fillna(df['property_damage'].mode()[0])
df['police_report_available'] = df['police_report_available'].fillna(df['police_report_available'].mode()[0])


# In[21]:


#correlation_multicolonality-problems


# In[22]:


plt.figure(figsize = (18,15))
corr = df.corr()
sns.heatmap(data = corr, annot = True, fmt = '.1g',linewidth = 2)
plt.show()


# In[23]:


unique = df.nunique().to_frame()
unique.columns = ['Count']
unique.index.names = ['ColName']
unique = unique.reset_index()
sns.set(style='whitegrid', color_codes = True)
sns.barplot(x='ColName', y='Count', data = unique)
plt.xticks(rotation = 90)
plt.show()


# In[24]:


#sorting_values


# In[25]:


unique.sort_values(by='Count', ascending = True)


# In[26]:


#droping_columns


# In[27]:


to_drop = ['policy_number','policy_bind_date','policy_state','insured_zip','incident_location','incident_date',
           'incident_state','incident_city','insured_hobbies','auto_make','auto_model','auto_year','_c39']
df.drop(to_drop, inplace = True, axis = 1)


# In[28]:


df.head()


# In[29]:


#multi_colonality


# In[30]:


plt.figure(figsize = (18,15))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype = bool))
sns.heatmap(data = corr, mask = mask, annot = True, fmt = '.2g',linewidth = 1)
plt.show()


# In[31]:


#dropping_columns_age,total_claim_amount


# In[32]:


df.drop(columns = ['age','total_claim_amount'],inplace = True, axis = 1)


# In[33]:


df.head()


# In[34]:


#figure


# In[35]:


plt.figure(figsize = (18,15))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype = bool))
sns.heatmap(data = corr, mask = mask, annot = True, fmt = '.2g',linewidth = 1)
plt.show()


# In[36]:


#dependent_independent_variables


# In[37]:


X = df.drop('fraud_reported', axis = 1)
y = df['fraud_reported']


# In[38]:


X.shape


# In[39]:


y.shape


# In[41]:


#converting_label_columns_to_numerical_columns


# In[42]:


categorical_cols = X.select_dtypes(include = ['object'])
categorical_cols = pd.get_dummies(categorical_cols, drop_first = True)
categorical_cols


# In[43]:


categorical_cols.head()


# In[44]:


numerical_col = X.select_dtypes(include = ['int64'])
X = pd.concat([numerical_col,categorical_cols],axis = 1)
X


# In[45]:


X.head()


# In[46]:


#outliers


# In[47]:


plt.figure(figsize =(20,15))
plotnumber = 1
for col in X.columns:
    if plotnumber <= 24:
        ax = plt.subplot(5, 5, plotnumber)
        sns.boxplot(X[col])
        plt.xlabel(col, fontsize = 15)
    plotnumber += 1
plt.tight_layout()
plt.show()


# In[48]:


#to_remove_outliers


# In[49]:


#train_test_split


# In[50]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
X_train


# In[51]:


X_train.head()


# In[52]:


y_train


# In[53]:


X_test


# In[54]:


y_test


# In[55]:


numerical_data = X_train[['months_as_customer','policy_deductable','umbrella_limit','capital-gains','capital-loss',
                         'incident_hour_of_the_day','number_of_vehicles_involved','bodily_injuries','witnesses',
                         'injury_claim','property_claim','vehicle_claim']]
numerical_data


# In[56]:


#standardization


# In[57]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)


# In[58]:


scaled_num_df = pd.DataFrame(data = scaled_data, columns = numerical_data.columns, index = X_train.index)
scaled_num_df.head()


# In[59]:


X_train.drop(columns = scaled_num_df.columns, inplace = True)


# In[60]:


X_train = pd.concat([scaled_num_df, X_train],axis = 1)


# In[61]:


X_train.head()


# In[62]:


#modelling_through_SVM


# In[63]:


from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X_train, y_train)
y_pred = svc_model.predict(X_test)


# In[64]:


#calculations


# In[65]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
svc_model_train_acc = accuracy_score(y_train, svc_model.predict(X_train))
svc_model_test_acc = accuracy_score(y_test, y_pred)
print("Training Accuracy : ",svc_model_train_acc)
print("Testing Accuracy : ",svc_model_test_acc)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[66]:


#k_nearest_neighbours


# In[67]:


from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors = 30)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)


# In[68]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
knn_model_train_acc = accuracy_score(y_train, knn_model.predict(X_train))
knn_model_test_acc = accuracy_score(y_test, y_pred)
print("Training Accuracy : ",knn_model_train_acc)
print("Testing Accuracy : ",knn_model_test_acc)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[69]:


#decision_tree_classifier


# In[70]:


from sklearn.tree import DecisionTreeClassifier
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train, y_train)
y_pred = decision_tree_model.predict(X_test)


# In[71]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
decision_tree_model_train_acc = accuracy_score(y_train, decision_tree_model.predict(X_train))
decision_tree_model_test_acc = accuracy_score(y_test, y_pred)
print("Training Accuracy : ",decision_tree_model_train_acc)
print("Testing Accuracy : ",decision_tree_model_test_acc)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[72]:


#hyper_parameter_tuning


# In[73]:


from sklearn.model_selection import GridSearchCV
grid_params = {
    'criterion' : ['gini','entropy'],
    'max_depth' : [3, 5, 7, 10],
    'min_samples_split' : range(2, 10, 1),
    'min_samples_leaf' : range(2, 10, 1)
}
grid_search = GridSearchCV(decision_tree_model, grid_params, cv=5, n_jobs = -1, verbose = 1)
grid_search.fit(X_train, y_train)


# In[74]:


print(grid_search.best_params_)
print(grid_search.best_score_)


# In[75]:


decision_tree_model = grid_search.best_estimator_
y_pred = decision_tree_model.predict(X_test)


# In[76]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
decision_tree_model_train_acc = accuracy_score(y_train, decision_tree_model.predict(X_train))
decision_tree_model_test_acc = accuracy_score(y_test, y_pred)
print("Training Accuracy : ",decision_tree_model_train_acc)
print("Testing Accuracy : ",decision_tree_model_test_acc)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[77]:


#Ada_boost_classifier


# In[80]:


from sklearn.ensemble import AdaBoostClassifier
ada_model = AdaBoostClassifier(estimator = decision_tree_model)
parameters = {
    'n_estimators' : [50, 70, 90, 120, 180, 200],
    'learning_rate' : [0.0001, 0.01, 0.1, 1, 10],
    'algorithm' : ['SAMME', 'SAMME.R']
}
grid_search = GridSearchCV(ada_model, parameters, n_jobs = -1, cv=5, verbose = 1)
grid_search.fit(X_train, y_train)


# In[81]:


print(grid_search.best_params_)
print(grid_search.best_score_)


# In[82]:


ada_model = grid_search.best_estimator_
y_pred = ada_model.predict(X_test)


# In[83]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
ada_model_train_acc = accuracy_score(y_train, ada_model.predict(X_train))
ada_model_test_acc = accuracy_score(y_test, y_pred)
print("Training Accuracy : ",ada_model_train_acc)
print("Testing Accuracy : ",ada_model_test_acc)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[84]:


#voting_classifier


# In[85]:


from sklearn.ensemble import VotingClassifier
classifiers = [('Support Vector Classifier', svc_model), ('KNN - Model', knn_model), 
               ('Decision Tree', decision_tree_model),('Ada Boost', ada_model)]
vc = VotingClassifier(estimators = classifiers)
vc.fit(X_train, y_train)
y_pred = vc.predict(X_test)


# In[86]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
vc_model_train_acc = accuracy_score(y_train, ada_model.predict(X_train))
vc_model_test_acc = accuracy_score(y_test, y_pred)
print("Training Accuracy : ",vc_model_train_acc)
print("Testing Accuracy : ",vc_model_test_acc)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[87]:


#models_comparison


# In[88]:


models = pd.DataFrame({
    'Model' : ['SVC-Model','KNN-Model','Decision-Tree','Ada-Boost','Voting-Model'],
    'Score' : [svc_model_test_acc, knn_model_test_acc, decision_tree_model_test_acc, ada_model_test_acc, vc_model_test_acc]
})
models.sort_values(by = 'Score', ascending = False)


# In[89]:


plt.bar(models['Model'],models['Score'], width = 0.4, color = 'maroon')
plt.xlabel("Models")
plt.xticks(rotation = 10)
plt.ylabel("Scores")
plt.title("Models For Health Insurance Fraud Detection")
plt.show()


# In[ ]:




