# EDA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# ML algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# AI Chatbot
from openai import OpenAI
import openai


def visualize_data(df, save=False):
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()
    plt.figure(figsize=(20,10), dpi =500)
    sns.heatmap(corr, annot=False, linewidth=.5, cmap='coolwarm')
    plt.show()
    
    if save:
        plt.savefig('assets/correlation_matrix.png')


def create_model(df):
    # Select specified features and the target variable
    features = df.drop(columns=['Y', 'Data'])
    target = df['Y']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a logistic regression model
    log_reg = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000, C=2)
    log_reg.fit(X_train_scaled, y_train)
    
    # Predict probabilities and classes for the training set
    y_train_pred = log_reg.predict(X_train_scaled)
    y_train_pred_prob = log_reg.predict_proba(X_train_scaled)[:, 1]
    
    # Predict on the test set
    y_pred = log_reg.predict(X_test_scaled)
    y_pred_prob = log_reg.predict_proba(X_test_scaled)[:, 1]
    
    # Add predictions for the training and test data
    df.loc[X_train.index, 'Predicted_Prob'] = y_train_pred_prob
    df.loc[X_train.index, 'Predicted_Class'] = y_train_pred
    df.loc[X_test.index, 'Predicted_Prob'] = y_pred_prob
    df.loc[X_test.index, 'Predicted_Class'] = y_pred
    
    # Define actions based on predictions
    df['Action'] = df['Predicted_Prob'].apply(lambda x: 'Reduce Risk' if x > 0.5 else 'Invest More')
    
    df.to_csv('docs/output.csv', index=False)
    
    return df


def validate_api_key(api_key):
    client = OpenAI(api_key=api_key)
    try:
        client.models.list()  # Lightweight call to check API key
        print("API key is valid.")
        return False
    except openai.AuthenticationError:
        print("Invalid API key. Please check.")
        return True
    except openai.RateLimitError:
        print("Rate limit exceeded. Check your plan.")
        return True


def search_csv(file_path, column_name, date):
    df = pd.read_csv(file_path)
    result = df[df[column_name] == date]

    while result.empty:
        print(f"Could not find data for the date {date}.")
        date = input('Please enter another date that you\'d like an investment explanation on as it appears in the docs/output.csv file: ')
        result = df[df[column_name] == date]
    
    return result


# TO-DO: fix RateLimitError, or at least find the cause of it
def get_response(prompt, api_key):
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    print(response.choices[0].message.content)
    
    
def main():
    df = pd.read_csv("docs/data.csv")
    visualize_data(df)
    
    # ML training
    df = create_model(df)
    
    # AI Chatbot
    print('To use this bot, you\'ll need an OpenAI API key.')
    print('Get your API key at https://platform.openai.com/api-keys')
    
    validation = True
    while validation:
        api_key = input('Please enter your OpenAI API key: ')
        print(f'You entered {api_key}')
        validation = validate_api_key(api_key)
    
    date = input('Please enter a date that you\'d like an investment explanation on as it appears in the docs/output.csv file: ')
    csv_data = search_csv('docs/output.csv', 'Data', date).to_string()
    
    bot_input = f'''The model predicts anomalies if probability > 0.5; otherwise, invest.
                    Given {date} and the following data:\n{csv_data}, explain the strategy.'''
    # ERROR
    get_response(bot_input, api_key)

if __name__ == "__main__":
    main()