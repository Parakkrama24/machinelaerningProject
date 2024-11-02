import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_data(data):
    """
    Prepare the cricket dataset for classification using Linear SVM
    """
    # Create DataFrame
    df = pd.DataFrame([data])
    
    # Select relevant features for prediction
    features = ['season', 'city', 'match_type', 'venue', 'team1', 'team2', 
               'toss_winner', 'toss_decision']
    
    # Target variable
    target = 'winner'
    
    # Initialize label encoders for categorical variables
    label_encoders = {}
    
    # Convert categorical variables to numerical
    for column in features + [target]:
        if column in df.columns:
            label_encoders[column] = LabelEncoder()
            df[column] = label_encoders[column].fit_transform(df[column].astype(str))
    
    # Create feature matrix X and target vector y
    X = df[features]
    y = df[target]
    
    # Scale the features (important for SVM)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, label_encoders, scaler, features

def train_svm_model(X, y):
    """
    Train a Linear SVM classifier for cricket match prediction
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define parameter grid for GridSearchCV
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'max_iter': [1000],
        'dual': [True, False]
    }
    
    # Initialize Linear SVM
    svm = LinearSVC(random_state=42)
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.3f}")
    
    return best_model, X_test, y_test, y_pred

def evaluate_model(model, X_test, y_test, y_pred, feature_names):
    """
    Evaluate the SVM model and visualize results
    """
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Feature importance for Linear SVM (using absolute values of coefficients)
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(model.coef_[0])
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance (Absolute Values of SVM Coefficients)')
    plt.show()

def predict_match(model, new_data, label_encoders, scaler):
    """
    Make predictions on new match data using the trained SVM model
    """
    # Transform new data using label encoders
    encoded_data = {}
    for column, value in new_data.items():
        if column in label_encoders:
            encoded_data[column] = label_encoders[column].transform([str(value)])[0]
    
    # Create feature vector
    X_new = pd.DataFrame([encoded_data])
    
    # Scale the features
    X_new_scaled = scaler.transform(X_new)
    
    # Make prediction
    prediction = model.predict(X_new_scaled)
    
    # Calculate decision function values (distance from hyperplane)
    decision_values = model.decision_function(X_new_scaled)
    
    # Convert decision values to probabilities using sigmoid function
    probability = 1 / (1 + np.exp(-decision_values))
    
    # Decode prediction
    predicted_winner = label_encoders['winner'].inverse_transform(prediction)[0]
    
    return predicted_winner, probability

# Example usage
if __name__ == "__main__":
    # Sample data preparation
    data = {
        'season': '2007/08',
        'city': 'Bangalore',
        'match_type': 'League',
        'venue': 'M Chinnaswamy Stadium',
        'team1': 'Royal Challengers Bangalore',
        'team2': 'Kolkata Knight Riders',
        'toss_winner': 'Royal Challengers Bangalore',
        'toss_decision': 'field',
        'winner': 'Kolkata Knight Riders'
    }
    
    # Prepare data
    X, y, label_encoders, scaler, features = prepare_data(data)
    
    # Train model
    model, X_test, y_test, y_pred = train_svm_model(X, y)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test, y_pred, features)
    
    # Make prediction for a new match
    new_match = {
        'season': '2007/08',
        'city': 'Mumbai',
        'match_type': 'League',
        'venue': 'Wankhede Stadium',
        'team1': 'Mumbai Indians',
        'team2': 'Chennai Super Kings',
        'toss_winner': 'Mumbai Indians',
        'toss_decision': 'bat'
    }
    
    predicted_winner, confidence = predict_match(model, new_match, label_encoders, scaler)
    print(f"\nPredicted Winner: {predicted_winner}")
    print(f"Confidence Score: {confidence[0]*100:.2f}%")
