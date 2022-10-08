import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import klib
from automl import auto_ml

# Load the data
df = pd.read_csv('Train.csv')
df.head()

# data preprocessing
df.Item_Fat_Content.replace({'low fat':'Low Fat', 'LF':'Low Fat', 'reg':'Regular'}, inplace = True)

for i, val in enumerate(df['Outlet_Size']):

    if (df['Outlet_Type'][i] == 'Grocery Store' and df['Outlet_Size'].isnull()[i] == True):
        df['Outlet_Size'][i] = 'Small'

    elif (df['Outlet_Type'][i] == 'Supermarket Type1' and df['Outlet_Size'].isnull()[i] == True):
        df['Outlet_Size'][i] = 'Small'

    elif (df['Outlet_Type'][i] == 'Supermarket Type2' and df['Outlet_Size'].isnull()[i] == True):
        df['Outlet_Size'][i] = 'Medium'

    elif (df['Outlet_Type'][i] == 'Supermarket Type3' and df['Outlet_Size'].isnull()[i] == True):
        df['Outlet_Size'][i] = 'Medium'

df["Item_Weight"].fillna(df["Item_Weight"].mean(), inplace = True)

df['Item_Visibility'].replace(0, df['Item_Visibility'].mean(), inplace = True)

df['Item_Type'].replace(['Fruits and Vegetables','Snack Foods','Household','Frozen Foods','Dairy','Canned','Baking Goods','Health and Hygiene','Soft Drinks','Meat','Breads','Hard Drinks','Starchy Foods','Breakfast','Seafood','Others'],['edible','edible','non-edible','edible','edible','edible','edible','non-edible','edible','edible','edible','edible','edible','edible','edible','non-edible'],inplace = True)
df['Item_Type'].value_counts()

# Encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['Item_Fat_Content'] = le.fit_transform(df['Item_Fat_Content'])
df['Item_Type'] = le.fit_transform(df['Item_Type'])
df['Outlet_Size'] = le.fit_transform(df['Outlet_Size'])
df['Outlet_Location_Type'] = le.fit_transform(df['Outlet_Location_Type'])
df['Outlet_Type'] = le.fit_transform(df['Outlet_Type'])

# Split the data
X = df.drop(columns=['Outlet_Establishment_Year', 'Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'])
y = df['Item_Outlet_Sales']
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state = 42, test_size = 0.3)

# Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)
X_train_std = pd.DataFrame(X_train,columns = X_val.columns)

X_val_std = sc.transform(X_val)
X_val_std = pd.DataFrame(X_val,columns = X_train.columns)

# Models

# Linear Regression:
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

lr = linear_model.LinearRegression(normalize = True)
lr.fit(X_train_std, y_train)
lr_pred = lr.predict(X_val_std)

lr_score = r2_score((y_val), (lr_pred))
lr_mae = mean_absolute_error((y_val), (lr_pred))
lr_mse = mean_squared_error((y_val), (lr_pred))
lr_rmse = np.sqrt(lr_mse)
print(lr_score, lr_mae, lr_mse, lr_rmse)

# XGB Regressor:
from xgboost import XGBRegressor

xgb_model = XGBRegressor(learning_rate=0.1, n_estimators=34, random_state=0)

xgb_model.fit(X_train_std, y_train)
xgb_pred = xgb_model.predict(X_val_std)

xgb_score = r2_score((y_val), (xgb_pred))
xgb_mae = mean_absolute_error((y_val), (xgb_pred))
xgb_mse = mean_squared_error((y_val), (xgb_pred))
xgb_rmse = np.sqrt(xgb_mse)
print(xgb_score, xgb_mae, xgb_mse, xgb_rmse)

# Gradient Boosting regression:
from sklearn.ensemble import GradientBoostingRegressor

gb_reg = GradientBoostingRegressor(learning_rate = 0.1, n_estimators = 37, random_state = 0)

gb_model = gb_reg.fit(X_train_std, y_train)
gb_pred = gb_reg.predict(X_val_std)

gb_score = r2_score(y_val, gb_pred)
gb_mae = mean_absolute_error(y_val, gb_pred)
gb_mse = mean_squared_error(y_val, gb_pred)
gb_rmse = np.sqrt(gb_mse)

print(gb_score, gb_mae, gb_mse, gb_rmse)

# Save the model
import pickle
pickle.dump(gb_reg, open('models/gb.pkl','wb'))
pickle.dump(sc, open('models/scaler.pkl','wb'))

# load the model
model = pickle.load(open('models/gb.pkl','rb'))

# Custom Predictions
pred = np.array([[31, 1, 0.032, 1, 30, 0, 0, 0]])
print(model.predict(pred))


