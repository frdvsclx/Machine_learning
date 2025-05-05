import  pandas as pd
from sklearn.tree import DecisionTreeRegressor

pd.set_option('display.max_columns',None)
pd.set_option('display.width', 500)

df_data = pd.read_csv("datasets/melb_data.csv")
df_data.head(10)

newest_home = 2025 - df_data["YearBuilt"].max()
print(newest_home)
print(df_data.YearBuilt)

df_data.columns()
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






