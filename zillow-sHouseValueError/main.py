import pandas as pd
import os
from sklearn.preprocessing import Imputer
from sklearn import decomposition
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from exploreData import exploreData
from plot import plotScatter
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn import ensemble
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

#-Data Exploration
explorer = exploreData( 'data', 'property.csv', 'train_2016.csv', 'train_2017.csv')
validfeatures = explorer.getFeatureInfo()
explorer.getLabelsInfo()
print "valid features which inclued 250,000 valid values"

#'propertycountylandusecode' is ID number of code. Training doesn't need 'propertycountylandusecode'
validfeatures.remove('propertycountylandusecode')
print validfeatures

#load data
featuresF = os.path.abspath(os.path.join('data', 'property.csv'))
features = pd.read_csv(featuresF, low_memory=False)
print 'raw features shape', features.shape
features = features.loc[ : , validfeatures]


print 'select valid features shape', features.shape

#load 2016  transections
labels2016F = os.path.abspath(os.path.join('data', 'train_2016.csv'))
labels2016 = pd.read_csv(labels2016F, low_memory=False)
print 'raw labels2016 shape', labels2016.shape

#remove outlier
labels2016=labels2016[ labels2016.logerror > -0.33]
labels2016=labels2016[ labels2016.logerror < 0.33]
labels2016 = labels2016.reset_index(drop=True)
print 'labels2016 shape after remove outlier',labels2016.shape

#load 2017  transections
labels2017F = os.path.abspath(os.path.join('data', 'train_2017.csv'))
labels2017 = pd.read_csv(labels2017F, low_memory=False)
print  'labels2017.shape',labels2017.shape

#remove outlier
labels2017=labels2017[ labels2017.logerror > -0.33]
labels2017=labels2017[ labels2017.logerror < 0.33]
labels2017 = labels2017.reset_index(drop=True)
print 'labels2017 shape after remove outlier',labels2017.shape

# plot logerror scatter
#2016
plt2 = plotScatter(labels2016.index, labels2016['logerror'])
plt2.get()
#2017
plt3 = plotScatter(labels2017.index, labels2017['logerror'])
plt3.get()

#create feature 'month' for 2016
transactiondate2016 = labels2016.loc[ : , 'transactiondate']
print transactiondate2016.shape
month2016 = list()
for i in range(0,transactiondate2016.shape[0]):
    a = transactiondate2016[i].split('/')[0]
    month2016.append(a)
month2016 ={'month': month2016}
df_month2016 = pd.DataFrame(month2016)
labels2016['month'] = df_month2016

#create feature 'month' for 2017
transactiondate2017 = labels2017.loc[ : , 'transactiondate']
print transactiondate2017.shape
month2017 = list()
for i in range(0,transactiondate2017.shape[0]):
    b = transactiondate2017[i].split('/')[0]
    c = (int(b)+ 12)
    month2017.append(c)
month2017 ={'month': month2017}
df_month2017 = pd.DataFrame(month2017)
labels2017['month'] = df_month2017

#merge transactions
frames_lables = [labels2016, labels2017]
labels  = pd.concat(frames_lables).reset_index(drop=True)
month = labels.loc[:,('parcelid','month')]

X = month.merge(features,how = 'left', on = 'parcelid')
y = labels.loc[:,'logerror']

print 'X shape, y shape'
print(X.shape, y.shape)

#'parcelid' is ID of properties, training doesn't need it.
X.set_index('parcelid', inplace = True)
train_columns = list(X)
print 'features in training'
print train_columns

#Imputing and scalling data
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
scaler = MinMaxScaler()
X = imp.fit_transform(X)
X = scaler.fit_transform(X)

#spliting data to x_train, x_test, y_train, y_test
x_train, x_test, y_train, y_test =train_test_split(X, y, test_size= 0.4, random_state= 42)

#GradientBoostingRegressor
print 'GradientBoostingRegressor'
pca = decomposition.PCA()
reg_GB= ensemble.GradientBoostingRegressor()
pipe = Pipeline(steps=[('pca', pca),('GradientBoostingRegressor', reg_GB)])
n_components = [23] # 21
n_estimators = [300] #150
learning_rate = [0.05] #0.01
max_depth = [4] #6
min_samples_leaf =[200] #100



estimator = GridSearchCV( pipe, param_grid = dict(pca__n_components=n_components,
                                        GradientBoostingRegressor__n_estimators = n_estimators,
                                        GradientBoostingRegressor__learning_rate = learning_rate,
                                        GradientBoostingRegressor__max_depth = max_depth,
                                        GradientBoostingRegressor__min_samples_leaf = min_samples_leaf
                                        ))
estimator.fit(x_train, y_train)

y_pred_train = estimator.predict(x_train)
print 'training set MAE value:',mean_absolute_error(y_train, y_pred_train)

y_pred = estimator.predict(x_test)
print 'testing set MAE value:',mean_absolute_error(y_test, y_pred)
# print 'best parameters'
# print estimator.best_params_


#DecisionTreeRegressor
print 'DecisionTreeRegressor'
reg_GB_2 = DecisionTreeRegressor()
pca_2 = decomposition.PCA()
pipe_2 = Pipeline(steps=[('pca_2', pca_2),('DecisionTreeRegressor_2', reg_GB_2)])
estimator_2 = GridSearchCV( pipe_2, param_grid = dict(pca_2__n_components=n_components,
                                                      DecisionTreeRegressor_2__max_depth = max_depth,
                                                      DecisionTreeRegressor_2__min_samples_leaf = min_samples_leaf
                                        ))
estimator_2.fit(x_train, y_train)
y_pred_2 = estimator_2.predict(x_test)
print 'testing set MAE value',mean_absolute_error(y_test, y_pred_2)



result = pd.DataFrame()
#predict October logerror using GradientBoostingRegressor
featuresF = os.path.abspath(os.path.join('data', 'property.csv'))
features = pd.read_csv(featuresF, low_memory=False)
features = features.loc[ : , validfeatures]
features['month'] = 22
result['parcelid']= features.loc[ : , 'parcelid']
features.set_index('parcelid', inplace = True)
Oct_X = features.loc[ : , train_columns]
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
scaler = MinMaxScaler()
Oct_X = imp.fit_transform(Oct_X)
Oct_X = scaler.fit_transform(Oct_X)
Oct_y = estimator.predict(Oct_X)
result['Oct'] = Oct_y

#predict November logerror using GradientBoostingRegressor
featuresF = os.path.abspath(os.path.join('data', 'property.csv'))
features = pd.read_csv(featuresF, low_memory=False)
features = features.loc[ : , validfeatures]
features['month'] = 23
features.set_index('parcelid', inplace = True)
Nov_X = features.loc[ : , train_columns]
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
scaler = MinMaxScaler()
Nov_X = imp.fit_transform(Nov_X)
Nov_X = scaler.fit_transform(Nov_X)
Nov_y = estimator.predict(Nov_X)
result['Nov'] = Nov_y


#predict December logerror using GradientBoostingRegressor
featuresF = os.path.abspath(os.path.join('data', 'property.csv'))
features = pd.read_csv(featuresF, low_memory=False)
features = features.loc[ : , validfeatures]
features['month'] = 24
features.set_index('parcelid', inplace = True)
Dec_X = features.loc[ : , train_columns]
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
scaler = MinMaxScaler()
Dec_X = imp.fit_transform(Dec_X)
Dec_X = scaler.fit_transform(Dec_X)
Dec_y = estimator.predict(Dec_X)
result['Dec'] = Dec_y

#Save prediction to csv file
result.to_csv('prediction.csv', sep=',',encoding='utf-8')








