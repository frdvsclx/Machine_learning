import  pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

pd.set_option('display.max_columns',None)
pd.set_option('display.width', 500)

df_data3 = pd.read_csv("datasets/melb_data.csv")

c= ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

y= df_data3.Price
X= df_data3[c]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

normal_model= DecisionTreeRegressor(random_state=1)
normal_model.fit(train_X,train_y)
val_prediction = normal_model.predict(val_X)
val_mae = mean_absolute_error(val_prediction,val_y)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

#Validation MAE when not specifying max_leaf_nodes: 241,632

normal_model2 = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
normal_model2.fit(train_X, train_y)
val_predictions = normal_model2.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))

#Validation MAE for best value of max_leaf_nodes: 243,732

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
forest_preds = forest_model.predict(val_X)
forest_model_mae = mean_absolute_error(val_y,forest_preds)
print("Validation MAE for Random Forest Model: {}".format(forest_model_mae))

#Validation MAE for Random Forest Model: 180544.06532524488


