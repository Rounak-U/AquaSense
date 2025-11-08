#!/usr/bin/env python3
"""
Fix Model Compatibility Script
This script retrains and saves the model with proper compatibility
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import warnings
warnings.filterwarnings('ignore')

def fix_model():
    print("🔧 Fixing model compatibility...")
    
    try:
        # Load data
        df = pd.read_csv('data/rainfall in india 1901-2015.csv')
        print("✅ Data loaded successfully")
        
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
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print("✅ Data preprocessing completed")
        
        # Build ANN with compatible configuration
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        
        # Compile with compatible optimizer and loss
        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )
        
        print("✅ Model architecture created")
        
        # Train model with fewer epochs for quick fix
        print("🚀 Training model...")
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=20,  # Reduced for quick fix
            batch_size=32,
            verbose=1
        )
        
        # Save model with compatible format
        model.save('models/ann_rainfall_model_fixed.h5')
        print("✅ Model saved as 'models/ann_rainfall_model_fixed.h5'")
        
        # Also save scaler
        import pickle
        with open('models/scaler_fixed.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        print("✅ Scaler saved as 'models/scaler_fixed.pkl'")
        
        # Evaluate model
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        y_pred = model.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"📊 Model Performance:")
        print(f"   MAE: {mae:.2f}")
        print(f"   MSE: {mse:.2f}")
        print(f"   R²: {r2:.4f}")
        
        print("🎉 Model fixed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing model: {str(e)}")
        return False

if __name__ == "__main__":
    fix_model()
