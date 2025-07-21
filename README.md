🏨 Hotel Booking Cancellation Prediction
📌 Project Overview
This machine learning project aims to predict hotel booking cancellations using structured data related to guest details, booking attributes, and stay characteristics. By analyzing patterns in both numerical and categorical variables, we build predictive models that help hotels optimize bookings, manage inventory, and reduce revenue loss.

📂 Dataset Information
The dataset used in this project is publicly available from the Hotel Booking Demand Dataset published by Antonio, Almeida, and Nunes (2019) and hosted on Kaggle.

🔍 Dataset Source:
"hotel_bookings.csv" from Kaggle: Hotel Booking Demand

📊 Columns Overview:
The dataset contains over 30 features including:

Hotel Type: City Hotel or Resort Hotel

Lead Time: Number of days between booking and arrival

Arrival Dates: Year, Month, Weekday

Stay Duration: Nights during week/weekend

Guest Details: Number of adults, children, and babies

Meal Type, Country, Booking Channel, Deposit Type

Reservation Status and Date

ADR (Average Daily Rate): Price per room per night

🔧 Preprocessing Steps:
Handled missing values using zero imputation

Removed records with no guests (adults, children, and babies all = 0)

Created new features (e.g., total_nights, month number)

Mean encoding applied to categorical features

Log transformation on skewed variables (like lead_time, adr)

Feature selection using Lasso Regularization

Final dataset prepared with top predictive features for modeling

📈 Exploratory Data Analysis
A thorough EDA was conducted to understand:

Country-wise Guest Origin (Choropleth Map)

Room Pricing Trends by hotel and room type (Boxplots)

Seasonal Pricing & Rush Months (Line charts)

Stay Duration Patterns (Barplots)

Key insights included:

August is the peak month for both city and resort hotels

Room type and hotel type significantly impact price

Lead time and ADR have positive skewness, addressed using log transformation

🤖 Machine Learning Models Used
The target variable is is_canceled (binary: 0 = Not Canceled, 1 = Canceled)

Models Implemented:
✅ Logistic Regression

✅ Naive Bayes

✅ Random Forest

✅ Decision Tree

✅ K-Nearest Neighbors (KNN)

🔍 Evaluation Metric:
Confusion Matrix

Accuracy Score

Final accuracy scores were used to compare the model performance.

🧠 Results & Observations
Random Forest and Decision Tree classifiers performed well in terms of accuracy.

Cancellations were often correlated with high lead times and specific market segments.

Feature importance analysis helped eliminate noise and overfitting, improving model generalization.

🚀 Future Improvements
Hyperparameter tuning using GridSearchCV

Use of advanced models like XGBoost or LightGBM

Deploy as a web app using Flask or Streamlit for real-time predictio
