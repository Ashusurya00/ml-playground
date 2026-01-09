🚀 Machine Learning Algorithm Playground

An interactive, end-to-end Machine Learning playground built with Python, Scikit-Learn, and Streamlit that allows users to experiment with multiple ML algorithms, tune hyperparameters in real time, and visualize model behavior on real-world datasets.

This project is designed to bridge the gap between ML theory and real-world implementation.

📌 Features
🔹 Algorithms Supported

Logistic Regression

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

Decision Tree

Random Forest

Gradient Boosting

Naive Bayes

🔹 Data Handling (Production-Grade)

✔ Upload any CSV dataset
✔ Automatic detection of:

Numerical features

Categorical features
✔ Handles missing values (NaN) using imputation
✔ Encodes categorical features using One-Hot Encoding
✔ Scales numerical features using StandardScaler
✔ Prevents data leakage using Scikit-Learn Pipelines

🔹 Hyperparameter Tuning

Each algorithm exposes important hyperparameters through an interactive UI:

Regularization strength, solvers, kernels

Tree depth, number of estimators, subsampling

Distance metrics, neighbor counts

Smoothing parameters

Changes are reflected instantly in model performance and visualizations.

🔹 Advanced Visualizations

📊 Confusion Matrix

📈 Accuracy Metrics

📉 PCA 2D Projection (works for high-dimensional datasets)

🌲 Feature Importance (tree-based models)

📊 Prediction Probability Distribution

🔗 Correlation Heatmap (numerical features)

Note: Instead of forcing misleading decision boundaries on high-dimensional data, PCA is used for honest and interpretable visualization.

🧠 Why This Project Matters

Most ML demos stop at training a model in a notebook.
This project demonstrates how ML is actually built in real applications:

Robust preprocessing

Clean model pipelines

Safe handling of real datasets

Visualization-driven understanding

Scalable and extensible architecture

This makes the project resume-ready, interview-ready, and production-inspired.

🏗️ Tech Stack

Python

Streamlit

Scikit-Learn

Pandas

NumPy

Matplotlib

Seaborn

📁 Project Structure
ml-algorithm-playground/
│
├── app.py
├── requirements.txt
├── README.md

▶️ How to Run Locally
1️⃣ Clone the Repository
git clone https://github.com/your-username/ml-algorithm-playground.git
cd ml-algorithm-playground

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the App
streamlit run app.py


The app will open in your browser at:

http://localhost:8501

📊 Example Use Cases

Learning ML algorithms interactively

Understanding hyperparameter effects

Testing models on real datasets

Demonstrating ML skills in interviews

Academic or final-year project

🚀 Future Enhancements

Regression mode (continuous targets)

Model comparison dashboard

Cross-validation visualization

AutoML integration

Model export (pickle/joblib)

Deployment on Streamlit Cloud
