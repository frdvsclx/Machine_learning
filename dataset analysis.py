import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

pd.set_option('display.max_columns',None)
pd.set_option('display.width', 500)


df_full = pd.read_csv("datasets/train.csv",index_col='Id')
df_test = pd.read_csv("datasets/train.csv", index_col= 'Id')

y = df_full.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X= df_full[features].copy()
X_test = df_test[features].copy()

X_train, X_valid, y_train, y_valid = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=0)

X_train.head()

model1= RandomForestRegressor(n_estimators=50,random_state=0)
model2= RandomForestRegressor(n_estimators=100,random_state=0)
model3= RandomForestRegressor(n_estimators=100,criterion='absolute_error',random_state=0)
model4= RandomForestRegressor(n_estimators=50,min_samples_split=20 ,random_state=0)
model5= RandomForestRegressor(n_estimators=50,max_depth=7,random_state=0)

models= [model1,model2,model3,model4,model5]

def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

for i in range(0,len(models)):
    mae= score_model(models[i])
    print("Model %d MAE: %d" % (i + 1, mae))

'''
Model 1 MAE: 24015
Model 2 MAE: 23740
Model 3 MAE: 23528 #best one
Model 4 MAE: 24051
Model 5 MAE: 23669
'''

my_model= model3

my_model.fit(X, y)

preds_test = my_model.predict(X_test)

'''
array([209042.6 , 173834.87, 222898.  , ..., 243630.03, 132717.93, 152829.  ])
'''

output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})

###################################### missing values #####################################
#some solutions: drop/imputation/ imputation and add one column

melb_data = pd.read_csv("datasets/melb_data.csv")
melb_pre = melb_data.drop(['Price'], axis=1) #drop objects

y2= melb_data.Price
X2= melb_pre.select_dtypes(exclude= ['object'])

X2_train, X2_valid, y2_train, y2_valid = train_test_split(X2,y2,train_size=0.8,test_size=0.2,random_state=0)


def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


#1)drop:

missing_cols = [col for col in X2_train.columns if X2_train[col].isnull().any()]
#Out[19]: ['Car', 'BuildingArea', 'YearBuilt']

reduced_X2_train = X2_train.drop(missing_cols, axis=1)
reduced_X2_valid = X2_valid.drop(missing_cols, axis=1)

print("MAE from Approach 1 (Drop columns with missing values):")
print(score_dataset(reduced_X2_train, reduced_X2_valid, y2_train, y2_valid))
#183550.22137772635


#2)imputation:

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

#replace missing values with the mean value along each column.
imputed_X2_train = pd.DataFrame(my_imputer.fit_transform(X2_train))
imputed_X2_valid = pd.DataFrame(my_imputer.transform(X2_valid))

# Imputation removed column names; put them back
imputed_X2_train.columns = X2_train.columns
imputed_X2_valid.columns = X2_valid.columns

print("MAE from Approach 2 (Imputation):")
print(score_dataset(imputed_X2_train, imputed_X2_valid, y2_train, y2_valid))
#178166.46269899711

#3)imputation and add one column:

X2_train_ext = X2_train.copy()
X2_valid_ext = X2_valid.copy()

for col in missing_cols:
    X2_train_ext[col + 'missing'] = X2_train_ext[col].isnull()
    X2_valid_ext[col + 'missing'] = X2_valid_ext[col].isnull()

imputed_X2_train_ext= pd.DataFrame(my_imputer.fit_transform(X2_train_ext))
imputed_X2_valid_ext = pd.DataFrame(my_imputer.fit_transform(X2_valid_ext))

imputed_X2_train_ext.columns = X2_train_ext.columns
imputed_X2_valid_ext.columns = X2_valid_ext.columns

print("MAE from Approach 3 (An Extension to Imputation):")
print(score_dataset(imputed_X2_train_ext, imputed_X2_valid_ext, y2_train, y2_valid))
#179986.2708570026

