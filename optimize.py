import  pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns',None)
pd.set_option('display.width', 500)

# Mean Absolute Error
df_data2 = pd.read_csv("datasets/melb_data.csv")

filtered_df_data2 = df_data2.dropna(axis=0)
b = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
y = filtered_df_data2.Price
X = filtered_df_data2[b]

df_model2 = DecisionTreeRegressor()
df_model2.fit(X, y)

predicted_home_prices = df_model2.predict(X)
mean_absolute_error(y, predicted_home_prices)
# Out[26]: 434.71594577146544 --> in-sample score

# validation value(just for test,model doesn't see):

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
# random_state= 0,1,42 --> it divides same for evey time
# 42: origin num (The Hitchhiker's Guide to the Galaxy ref.)
# if none: it divides different for evey time

df_model3 = DecisionTreeRegressor()
df_model3.fit(train_X, train_y)

val_prediction = df_model3.predict(val_X)
print(mean_absolute_error(val_y, val_prediction))
# out: 260042.310522918

# UNDERFITTING AND OVERFITTING

'''
Decision trees leave you with a difficult decision. A deep tree with lots of leaves will overfit 
because each prediction is coming from historical data from only the few houses at its leaf. But 
a shallow tree with few leaves will perform poorly because it fails to capture as many distinctions 
in the raw data.
'''

df_data2 = pd.read_csv("datasets/melb_data.csv")
a = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

y2 = df_data2.Price  # target variable
X2 = df_data2[a]  # features matris

train_X2, val_X2, train_y2, val_y2 = train_test_split(X2, y2, random_state=1)
# train_test_split(X2, y2, test_size=0.2, random_state=1)
# 1,3,4. rows... %80 train, 2,5,6. rows... %20 test
# random_state=

df_model4 = DecisionTreeRegressor(random_state=1)
df_model4.fit(train_X2, train_y2)

val2_prediction = df_model4.predict(val_X2)
val2_mae = mean_absolute_error(val2_prediction, val_y2)
print("Validation MAE: {:,.0f}".format(val2_mae))


# Validation MAE: 241,632

####################optimize: #######################

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return (mae)


candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

results = {leaf_size: get_mae(leaf_size, train_X2, val_X2, train_y2, val_y2) for leaf_size in candidate_max_leaf_nodes}
best_tree_size = min(results, key=results.get)
'''
Out[19]: 
{5: 356157.8826013371,
 25: 281687.6016869558,
 50: 264538.5759958539,
 100: 243715.1555402675,
 250: 229271.75328742765,
 500: 222979.25446807843}

best_tree_size
Out[20]: 500
'''

final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)
final_model.fit(X2, y2)
