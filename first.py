import  pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns',None)
pd.set_option('display.width', 500)

df_data = pd.read_csv("datasets/melb_data.csv")
df_data.head(10)

newest_home = 2025 - df_data["YearBuilt"].max()
print(newest_home)
print(df_data.YearBuilt)

df_data.columns
df_data= df_data.dropna(axis=0)


# Define model. Specify a number for random_state to ensure same results each run
a= ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = df_data[a]
y = df_data.Price
df_model = DecisionTreeRegressor(random_state=1)
df_model.fit(X,y)

print("Making predictions for the following 5 houses:")
print(X.head())
'''
Making predictions for the following 5 houses:
   Rooms  Bathroom  Landsize  Lattitude  Longtitude
1      2       1.0     156.0   -37.8079    144.9934
2      3       2.0     134.0   -37.8093    144.9944
4      4       1.0     120.0   -37.8072    144.9941
6      3       2.0     245.0   -37.8024    144.9993
7      2       1.0     256.0   -37.8060    144.9954
'''

print("The predictions are")
print(df_model.predict(X))
'''
The predictions are
[1035000. 1465000. 1600000. 1876000. 1636000.]
'''


#Mean Absolute Error
df_data2 = pd.read_csv("datasets/melb_data.csv")

filtered_df_data2 = df_data2.dropna(axis=0)
b= ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea','YearBuilt', 'Lattitude', 'Longtitude']
y2 = filtered_df_data2.Price
X2= filtered_df_data2[b]

df_model2=  DecisionTreeRegressor()
df_model2.fit(X2, y2)

predicted_home_prices= df_model2.predict(X2)
mean_absolute_error(y2,predicted_home_prices)
#Out[26]: 434.71594577146544 --> in-sample score

#validation value:

train_X, val_X, train_y, val_y = train_test_split(X2, y2, random_state = 0)

df_model3= DecisionTreeRegressor()
df_model3.fit(train_X,train_y)

val_prediction = df_model3.predict(val_X)
print(mean_absolute_error(val_y,val_prediction))
#out: 260042.310522918

#UNDERFITTING AND OVERFITTING



