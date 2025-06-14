# 📡 5G Signal Strength Predictor

This is a machine learning project that predicts 5G signal strength (in dBm) based on distance from the tower, frequency band, and terrain type. A Random Forest Regressor model is used with synthetically generated training data.

## 👩‍💻 Author: Umma Jarin Toba

## 🚀 Features
- 📊 Synthetic dataset generation (distance, frequency, terrain)
- 🌲 Random Forest Regressor model
- 💾 Model saving with `joblib`
- 📉 Graphical visualization of predictions

## 🛠️ Technologies Used
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Joblib

## 📦 Installation

```bash
git clone https://github.com/Toba9/5g-signal-strength-predictor.git
cd 5g-signal-strength-predictor
pip install -r requirements.txt
```

## ▶️ Run

```bash
python main.py
```

## 📁 File Description

| File | Description |
|------|-------------|
| `main.py` | Core ML training and prediction code |
| `signal_strength_model.pkl` | Saved Random Forest model |
| `signal_strength_plot.png` | Visualization of predictions |
| `README.md` | Project documentation |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Ignore files for Git |

## 🙏 Acknowledgements

This project was assisted with OpenAI's ChatGPT for guidance and explanation. Modified and developed by **Umma Jarin Toba**.
