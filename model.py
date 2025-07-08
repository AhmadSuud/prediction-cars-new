import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load dataset
df = pd.read_csv('depreciated-clear.csv')

X = df[['Brand', 'Year', 'Fuel_simple', 'Transmission_simple',
        'Price', 'Simulation_Year', 'Vehicle_Age', 'Simulation_Kilometer']]
y = df['Estimated_Value']

# Pipeline
categorical_cols = ['Brand', 'Fuel_simple', 'Transmission_simple']
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
], remainder='passthrough')

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', HistGradientBoostingRegressor(random_state=42))
])

# Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Evaluasi
y_pred = pipeline.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Simpan model ke file
joblib.dump(pipeline, 'model_histgradientboosting.pkl')
print("✅ Model berhasil disimpan di lokal.")
