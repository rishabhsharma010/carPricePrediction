import pandas as pd
import numpy as np
import seaborn as sns

df=pd.read_csv("car data.csv")

final_dataset=df[['Year','Selling_Price','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]
#removing car name as there is many number of car names so  car name is not going to play any important role.

final_dataset['Current Year']=2020

final_dataset['no_year']=final_dataset['Current Year']- final_dataset['Year']

final_dataset.drop(['Year'],axis=1,inplace=True)

final_dataset.drop(['Current Year'],axis=1,inplace=True)

final_dataset=pd.get_dummies(final_dataset,drop_first=True)


#independent and dependent featrures
X=final_dataset.iloc[:,1:]
y=final_dataset.iloc[:,0]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

rf = RandomForestRegressor(n_estimators=700,
                             criterion='mse',
                             max_depth=20,
                             min_samples_split=15,
                             min_samples_leaf=1,
                             max_features='auto',
                             n_jobs=1,
                             random_state=42,
                             verbose=2)

rf.fit(X_train,y_train)

#making prediction using 
predictions=rf.predict(X_test)

#plot of residual
sns.distplot(y_test-predictions)


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


import pickle
# open a file, where you want to store the data
file = open('model.pkl', 'wb')

# dump information to that file
pickle.dump(rf, file)
