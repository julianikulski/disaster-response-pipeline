import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Function to load message and category files into a dataframe
    Args: messages_filepath = csv file
          categories_filepath = csv file
    Returns: df = dataframe
    '''
    # read in csv files as dataframes
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge both dataframes
    df = messages.merge(categories, on='id')
    
    return df


def clean_data(df):
    '''
    Function to clean dataframe
    Args: df = dataframe
    Return: df = dataframe
    '''
    
    # Create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Select the first row of the categories dataframe
    row = categories.loc[0]
    # Use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2])
    # Rename the columns of `categories`
    categories.columns = category_colnames

    # Drop all rows with value 'related-2' in the 'related' column
    categories = categories[categories['related'] != 'related-2']
    
    # Convert category values to just 0 and 1
    for column in categories:
        # Set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # Convert column from string to numeric
        categories[column] = categories[column].apply(lambda x: int(x))

    # Drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # Drop duplicates
    df.drop_duplicates(subset='message', keep=False, inplace=True)
    
    # Remove missing values
    df = df.dropna(axis=0, subset=df.columns[4:], how='any')
    
    return df


def save_data(df, database_filename):
    '''
    Function to save dataframe to sql database
    Args: df = dataframe
          database_filename = string declaring name of database
    Returns: None
    '''
    
    # Create sqlite database
    engine = create_engine('sqlite:///'+database_filename)
    # Save dataframe to table in database
    df.to_sql('DataTable', engine, if_exists='replace', index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()