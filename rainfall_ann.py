#!/usr/bin/env python
# coding: utf-8

# In[8]:


# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

sns.set(style='whitegrid')

# Load Dataset
df = pd.read_csv('data/rainfall in india 1901-2015.csv')

# Fill missing monthly values
for month in ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']:
    df[month] = df.groupby('SUBDIVISION')[month].transform(lambda x: x.fillna(x.mean()))

# Reshape wide → long for month-wise prediction
df_long = df.melt(id_vars=['YEAR','SUBDIVISION'],
                  value_vars=['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'],
                  var_name='Month', value_name='Rainfall')

month_map = {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,
             'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}
df_long['Month_Num'] = df_long['Month'].map(month_map)

# Create season
def get_season(month):
    if month in [12,1,2]:
        return 'Winter'
    elif month in [3,4,5]:
        return 'Pre-monsoon'
    elif month in [6,7,8,9]:
        return 'Monsoon'
    else:
        return 'Post-monsoon'

df_long['Season'] = df_long['Month_Num'].apply(get_season)

# Create lag feature: previous year same month rainfall
df_long = df_long.sort_values(['SUBDIVISION','YEAR','Month_Num'])
df_long['Rainfall_Lag1'] = df_long.groupby('SUBDIVISION')['Rainfall'].shift(12)
df_long = df_long.dropna()

# Encode categorical features
df_encoded = pd.get_dummies(df_long, columns=['SUBDIVISION','Season'], drop_first=True)

X = df_encoded.drop(['Rainfall','Month'], axis=1)
y = df_encoded['Rainfall']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build ANN
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# Train ANN
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=100,
                    batch_size=32,
                    verbose=1)

# Evaluate
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# Visualizations
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Rainfall")
plt.ylabel("Predicted Rainfall")
plt.title("Actual vs Predicted Rainfall")
plt.savefig('actual_vs_predicted_rainfall.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10,6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig('training_validation_loss.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(x='Season', y='Rainfall', data=df_long)
plt.title("Seasonal Rainfall Distribution")
plt.xticks(rotation=45)
plt.savefig('seasonal_rainfall_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Save model
model.save('models/ann_rainfall_model.h5')


# In[ ]:




