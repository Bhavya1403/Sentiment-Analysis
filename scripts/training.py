import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


# Function for training the logistic regression model
def train_logistic_regression_model(X_train, X_test, y_train):
    
    pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression(solver='liblinear')),
    ])
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    
    return y_pred

# # Function for training the complex model
# def train_complex_model(X, y):
#     # Your training code for complex model here
#     pass


# Function for evaluating the logistic regression model
def evaluate_logistic_regression_model(y_test, y_pred):
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\nLogistic Regression Model Evaluation:")
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix as a heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    # Extract values from classification report
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    precision = report_dict['weighted avg']['precision']
    recall = report_dict['weighted avg']['recall']
    f1 = report_dict['weighted avg']['f1-score']
    
    # Plot metrics as a bar chart
    metrics = [accuracy_score(y_test, y_pred), precision, recall, f1]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    plt.figure(figsize=(10, 7))
    sns.barplot(x=metric_names, y=metrics)
    plt.title('Model Metrics')
    plt.show()

    return accuracy, precision, recall, f1

##Function for evaluating the complex model
# def evaluate_complex_model(y_true, y_pred):
#     # Your evaluation metrics calculation for complex model here
#     accuracy = accuracy_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred)
#     recall = recall_score(y_true, y_pred)

#     return accuracy, precision, recall


# Main function to execute evaluation
if __name__ == "__main__":
    
    filename = "../data/tweets_main_sentiment.csv"
    df = pd.read_csv(filename)
    
    df['content'] = df['content'].apply(lambda x: x.lower())  # convert to lowercase
    le = LabelEncoder()
    df['sentiment'] = le.fit_transform(df['sentiment'])
    
    X_train, X_test, y_train, y_test = train_test_split(df['content'], df['sentiment'], test_size=0.2, random_state=42)

    # Call training functions for each model
    logistic_regression_y_pred = train_logistic_regression_model(X_train, X_test, y_train)
    # trained_complex_model = train_complex_model(complex_model_X, complex_model_y)
    
    # Call evaluation functions for each model
    logistic_regression_accuracy, logistic_regression_precision, logistic_regression_recall, logistic_regression_f1 = evaluate_logistic_regression_model(y_test, logistic_regression_y_pred)
    # complex_accuracy, complex_precision, complex_recall = evaluate_complex_model(complex_model_y_true, complex_model_y_pred)
    