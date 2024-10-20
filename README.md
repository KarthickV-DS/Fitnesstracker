This project is a fitness tracker designed to monitor and analyze exercise performance using data from accelerometer and gyroscope sensors. The system applies signal processing techniques and machine learning algorithms, including Random Forest, Neural Networks, K-Nearest Neighbors (KNN), and Decision Trees, to detect exercise patterns and count repetitions accurately.

Features
Sensor Data Processing: Utilizes accelerometer and gyroscope data from fitness devices to compute the total acceleration (acc_r) and gyroscope magnitude (gyr_r).

Low-Pass Filtering: Smooths the sensor data using a low-pass filter to remove noise and make pattern recognition easier.

Repetition Counting: Automatically detects and counts repetitions for different exercises like squats, bench press, deadlifts, rows, and overhead presses.

Machine Learning Models: Implements several machine learning models to predict exercise metrics and validate accuracy:
  Random Forest Classifier: A powerful ensemble method used to predict the number of reps based on sensor features.
  Neural Networks: A deep learning model that learns complex patterns from sensor data to improve accuracy in rep prediction.
  K-Nearest Neighbors (KNN): A distance-based model used for classification of exercise data.
  Decision Trees: A simple, interpretable model that helps classify exercise types and predict repetitions.
  Model Evaluation: Performance evaluation is done using metrics like mean absolute error (MAE) and accuracy to compare the models’ effectiveness in rep prediction.

Exercises Supported
  Bench Press
  Rows
  Squats
  Deadlifts
  Overhead Press (OHP)
Technologies Used
  Python: Core programming language used for data processing, machine learning, and visualization.
  Pandas & NumPy: For data manipulation and handling sensor data.
  Matplotlib: To visualize sensor data and detected repetitions.
  SciPy: For signal processing, including low-pass filtering and peak detection.
  Scikit-learn: For machine learning models such as Random Forest, KNN, and Decision Trees, and evaluation metrics like MAE and accuracy.
  TensorFlow/Keras: For building and training Neural Network models.
  Machine Learning Models
  Random Forest Classifier: A powerful ensemble model that uses multiple decision trees to classify and predict exercise repetitions.
  Neural Networks: A multi-layer perceptron (MLP) used for deeper learning of the sensor data patterns.
  K-Nearest Neighbors (KNN): A simple model that classifies data points based on the nearest neighbors in feature space.
  Decision Tree Classifier: A decision-making model that splits data into branches to classify exercise types and predict repetitions.
How It Works
  Data Loading: Load pre-processed sensor data containing multiple sets of workout data.
  Data Transformation: Compute the magnitude of acceleration and gyroscope readings for analysis.
  Low-Pass Filtering: Apply a low-pass filter to smooth the data and detect key patterns corresponding to repetitions.
  Repetition Counting: Use machine learning models to classify exercises and count repetitions based on sensor data.
  Model Evaluation: Compare actual reps with predicted reps using different machine learning models and evaluate performance using metrics like mean absolute error and accuracy.
  Future Improvements
  Real-Time Tracking: Extend the system to handle real-time data streams from fitness devices for immediate feedback.
  Additional Exercises: Add support for more exercises and tailor machine learning models for better predictions.
  Hyperparameter Tuning: Improve model performance by optimizing hyperparameters for each machine learning model.

Usage
  Run the Fitness(F).py script to process sensor data, apply machine learning models, and visualize results.
