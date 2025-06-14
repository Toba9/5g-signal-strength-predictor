import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

np.random.seed(42)
num_samples = 500

distance = np.random.uniform(0.1, 5.0, num_samples)  # in km
frequency = np.random.choice([700, 1800, 2600], size=num_samples)  # MHz
terrain = np.random.choice([0, 1, 2], size=num_samples)  # 0 = Urban, 1 = Suburban, 2 = Rural
noise = np.random.normal(0, 2, num_samples)

signal_strength = -30 - 20 * np.log10(distance) - 0.02 * frequency - terrain * 3 + noise

df = pd.DataFrame({
    'Distance_km': distance,
    'Frequency_MHz': frequency,
    'Terrain_Type': terrain,
    'Signal_Strength_dBm': signal_strength
})

X = df[['Distance_km', 'Frequency_MHz', 'Terrain_Type']]
y = df['Signal_Strength_dBm']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=43)

model = RandomForestRegressor(n_estimators=150, random_state=43)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

joblib.dump(model, "signal_strength_model.pkl")

plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=1, color='skyblue')
plt.xlabel('Actual Signal Strength (dBm)')
plt.ylabel('Predicted Signal Strength (dBm)')
plt.title('Actual vs Predicted Signal Strength')
plt.grid(True)
plt.tight_layout()
plt.savefig("signal_strength_plot.png")
plt.show()
