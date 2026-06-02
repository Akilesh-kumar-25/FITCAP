import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
df = pd.read_csv('Activity_Tracker.csv')
df = df.drop(['User_ID'], axis=1)

df['Mass_Duration'] = df['Weight'] * df['Duration']

Q1_bt = df['Body_Temp'].quantile(0.25)
Q3_bt = df['Body_Temp'].quantile(0.75)
IQR_bt = Q3_bt - Q1_bt
lower_bt = Q1_bt - 1.5 * IQR_bt
median_bt = df['Body_Temp'].median()
df.loc[df['Body_Temp'] < lower_bt, 'Body_Temp'] = median_bt

Q1_md = df['Mass_Duration'].quantile(0.25)
Q3_md = df['Mass_Duration'].quantile(0.75)
IQR_md = Q3_md - Q1_md
upper_md = Q3_md + 1.5 * IQR_md
median_md = df['Mass_Duration'].median()
df.loc[df['Mass_Duration'] > upper_md, 'Mass_Duration'] = median_md

thresholds = {
    'body_temp_lower': lower_bt,
    'body_temp_median': median_bt,
    'mass_duration_upper': upper_md,
    'mass_duration_median': median_md
}

df_dummies = pd.get_dummies(df[['Gender', 'Activity_Type']], drop_first=True, dtype=int)
df = pd.concat([df, df_dummies], axis=1)
df = df.drop(['Gender', 'Activity_Type'], axis=1)

X = df.drop(['Calories'], axis=1)
y = df['Calories']
feature_columns = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

imputer = KNNImputer(n_neighbors=7)
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Model Performance:")
print(f"R² Score: {r2:.4f}")
print(f"MAE: {mae:.2f} calories")
print(f"RMSE: {rmse:.2f} calories")

os.makedirs('model_artifacts', exist_ok=True)
joblib.dump(model, 'model_artifacts/linear_model.pkl')
joblib.dump(imputer, 'model_artifacts/knn_imputer.pkl')
joblib.dump(scaler, 'model_artifacts/scaler.pkl')
joblib.dump(feature_columns, 'model_artifacts/feature_columns.pkl')
joblib.dump(thresholds, 'model_artifacts/thresholds.pkl')

print("All artifacts saved to 'model_artifacts/'")


