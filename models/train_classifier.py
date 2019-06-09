import sys
import nltk
import pandas as pd
from collections import defaultdict
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, fbeta_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine

import seaborn as sns
import matplotlib.pyplot as plt


def load_data(database_filepath):
    '''
    Function to load table from database into a dataframe
    Args: database_filepath = string containing filepath to sqlite database
    Returns: X = dataframe containaing all messages
             Y = dataframe containing all target labels
             category_names = list of strings
    '''
    
    # Load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    # Get table names
    table = engine.table_names()

    # Read in the sqlite table
    df = pd.read_sql_table(table[0], con=engine)
    X = df['message']
    Y = df[df.columns[4:]]
    category_names = Y.columns
    
    return X, Y, category_names

def tokenize(text):
    '''
    Function splitting messages into words, converting to lower case and removing punctuation
    Args: text = message in form of string
    Return: clean_tokens = list of cleaned tokens
    '''
    
    # Tokenize message into words and initialize lemmatizer
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    # Lemmatize words, convert them to lower case and strip whitespace
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
        
    return clean_tokens


def build_model():
    '''
    Function to create machine learning model
    Args: None
    Returns: model = model pipeline
    '''
    
    # Text processing and model pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('adaboost', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    # Define parameters for GridSearchCV
    parameters = {
        'vect__max_df': [0.7, 0.9, 1],
        'tfidf__use_idf': [True, False],
        'adaboost__estimator__n_estimators': [20, 50]
    }

    # Create gridsearch object and return as final model pipeline
    model = GridSearchCV(pipeline, param_grid=parameters, cv=3)

    return model

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function to train the model and output predicted labels
    Args: model = machine learning model
          X_test = dataframe with test data of X
          Y_test = dataframe with test data of Y
          category_names = list of names for the target labels
    Returns: None
    '''

    # Return the predictions for the optimal parameter combination
    y_pred = model.predict(X_test)
    # Convert y_pred from array to dataframe
    y_pred = pd.DataFrame(y_pred, columns=category_names)
    
    # Print the results of the predictive algorithm
    for col in y_pred.columns:
        print(col, classification_report(Y_test[col], y_pred[col]))

def save_model(model, model_filepath):
    '''
    Function to save the model as a pickled file
    Args: model = machine learning model
          model_filepath = string containing the name to save the model as
    Returns: None
    '''
    
    # Filename of the pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()