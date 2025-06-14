📈 Stock Price Prediction using CNN + BiLSTM
This project leverages a hybrid deep learning model combining Convolutional Neural Networks (CNN) and Bidirectional Long Short-Term Memory (BiLSTM) networks to predict stock prices based on historical data.

🧠 Model Architecture
CNN Layers: Capture short-term temporal features and local patterns in the time series data.

BiLSTM Layers: Model long-term dependencies in both forward and backward directions.

Dense Layers: Output predicted stock prices.

🗂️ Project Structure
bash
Copy
Edit
├── CNN+BiLSTM.ipynb     # Main notebook with preprocessing, model training and evaluation
├── README.md            # Project documentation
└── requirements.txt     # Python dependencies (optional, create if needed)

📊 Features Used
Normalized stock price data using MinMaxScaler.

Technical indicators via ta (optional).

Multistep prediction for future stock values.

🚀 How to Run
Clone this repository:

git clone https://github.com/Prakashreddy-Alla/Stock-Price-Pridiction-Using-Machine-Learning.git
cd stock-prediction-cnn-bilstm

Install dependencies:
pip install -r requirements.txt

Launch the notebook:
jupyter notebook CNN+BiLSTM.ipynb

✅ Results
The model achieves reasonable accuracy based on RMSE and MAE scores. Visualizations of actual vs predicted prices are plotted in the notebook.

🛠 Technologies Used
Python

TensorFlow / Keras

Scikit-learn

Matplotlib

Pandas, NumPy

TA-lib (optional, for technical analysis features)

📌 TODO
Add sentiment features

Tune hyperparameters

Experiment with other hybrid models

📄 License
This project is licensed under the MIT License.
