# Machine Learning Projects - Supervised Learning 
This project involves predictive maintenance using machine learning to classify 
types of equipment failure. The dataset, predictive_maintenance.csv, was 
processed to remove irrelevant columns ('UDI', 'Product ID', 'Target', 'Type') 
and encoded the target variable 'Failure Type' using LabelEncoder.
Data visualization included a heatmap to show feature correlations and several 
seaborn relational plots to explore relationships between different features 
colored by 'Failure Type'. For example, 'Air temperature [K]' vs 'Process 
temperature [K]' and 'Torque [Nm]' vs 'Rotational speed [rpm]'.

The data was split into training and test sets using train_test_split with an 80-20 
ratio. The preprocessing pipeline handled categorical and numerical features 
separately: categorical features were one-hot encoded, while numerical features 
underwent power transformation and standard scaling.
A RandomForestClassifier wrapped in an OutputCodeClassifier was trained on 
the data. The model's performance was evaluated using a confusion matrix and 
classification report, which provided precision, recall, F1-scores for each class, 
and overall accuracy
