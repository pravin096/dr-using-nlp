import sys

import nltk
import string
import numpy as np
import pandas as pd
import re

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

def get_unique_list_of_values(input_df,col,delimiter=";"):
    '''
    INPUT
        input_df: Input data frame containing the column which contains multiple categories separated by ";" delimiter
        colname : Column name which is to be used for computation of standard deviation value
    OUTPUT 
        input_df: Updated data frame containing engineered features
    '''
    col_series = input_df[col]
    col_series_new = col_series.dropna()
    i=0
    for s in col_series_new:
        col_series_new_list = s.split(delimiter)
        col_series_new_list = [val [:-2] for val in col_series_new_list]
        if i==0:
            consol_list = col_series_new_list
        else:
            for j in col_series_new_list:
                consol_list.insert(len(consol_list), j)
        i += 1
    list_of_unique_values = list(set(consol_list))
    print ("List of unique values for ",col," :\n" , list_of_unique_values)
    print ("# of unique values for ",col," :" , len(list_of_unique_values))
    
    #Create category labels for each of those items
    label_prefix = "Cat_"
    cat_labels = [label_prefix + val for val in list_of_unique_values]
    
    return list_of_unique_values, cat_labels


def create_cat_cols_for_multvalcols(source_col,source_value_list,cat_col_list,input_df):
    '''
    INPUT
        source col - source column from which category columns to be created
        source_value_list  - list containing values to be searched in the source column
        cat_col_list       - list containing category column names to be created
        colname_prefix     - Prefix for column name to be created for rows with missing values
        others_categ_list  - List of categories that need to be cmbined in "Others" category
        input_df                 - source df
    OUTPUT
        output_df                 - Updated source df containing category columns created as per columns listed in cat_col_list
    '''
    output_df = input_df
    i=0
    for source_value in source_value_list:
        cat_col = cat_col_list[i]
        pattern = source_value + '-1'
        print("cat col = ",cat_col,"source col = ",source_col)
        output_df [cat_col] = np.where(output_df[source_col].str.contains(pattern),1,0)
        i += 1
    
    #output_df = output_df.drop(['id','categories'],axis=1)
    return output_df


def load_data(messages_filepath, categories_filepath):
    #Load data into respective dataframes
    df = pd.read_csv(messages_filepath)
    df2 = pd.read_csv(categories_filepath)
  
    #Update the categories dataframe to include new columns - one column for each category i.e. one hot encoding form
    target_label_lst,cat_labels = get_unique_list_of_values(df2,'categories')
    df2 = create_cat_cols_for_multvalcols('categories',target_label_lst,cat_labels,df2)
    print("df shape ", df.shape,"df2 shape",df2.shape)
    
    #Create a new dataframe to include the message dataframe and the labels dataframe
    df_new = pd.concat([df,df2],axis=1)
    
    #Create a new column that sums up the category labels (which are in binary format) for each observation. 
    #This would be used for EDA
    df_new ['LabelSum'] = df_new.iloc[:,5:].sum(axis=1)
    return df_new


def clean_text(text):
    
    ## Remove puncuation
    text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower().split()
    
    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]
    
    text = " ".join(text)    ## Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)    ## Stemming
    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)
    
    return text

def clean_data(df):
    '''
    INPUT
        df                 - source df
        
    OUTPUT
        df                 - Updated source dataframe 
    '''
    #Get index list of messages where message is null and delete the same from the dataframe
    null_message_index_list = df[df['message'].isnull()].index.tolist()
    print("No of observations with null messages", len(null_message_index_list))
    print("Df shape before deleting observations with null messages",df.shape)
    df = df.drop(null_message_index_list)
    print("Df shape after deleting observations with null messages",df.shape)
    
    # apply the clean_text function to df['text']
    df['cleaned_message'] = df['message'].map(lambda x: clean_text(x))
    null_message_index_list = df[df['cleaned_message'].isnull()].index.tolist()
    print("No of observations with null values in cleaned messages", len(null_message_index_list))
    print("Df shape before deleting observations with null messages",df.shape)
    df = df.drop(null_message_index_list)
    print("Df shape after deleting observations with null messages",df.shape)
    
    #Drop the id columns from the dataframe that are present as a result of concatenation from message and category dataframes
    df = df.drop(['id'],axis=1)
    print("Df shape after dropping id columns",df.shape)
    
    return df


def save_data(df, database_filename):
    from sqlalchemy import create_engine
    import sqlite3
    
    df.to_csv('df_consol.csv')
    
    # connect to the database
    # the database filename will include path and db file name including .db extension
    # note that sqlite3 will create this database file if it does not exist already
    engine_string = 'sqlite:///' + database_filename
    engine = create_engine(engine_string,echo=True)
    sqlite_connection = engine.connect()
    
    #table_name = database_filename[:-3] #Drops the .db extension to create a table with the rest of the string
    #Extract the table name from filepath
    import re

    search_for_dbname = re.search('/(.+?).db', database_filename)

    if search_for_dbname:
        tablename_incl_path = search_for_dbname.group(1)

    word_list = tablename_incl_path.split("/")
    table_name = word_list[len(word_list) -1]
    
    sqlite_table = table_name
    
    conn = sqlite3.connect(database_filename)

    # get a cursor
    cur = conn.cursor()

    # drop the test table in case it already exists
    #df_merged.to_sql('merged', con = conn, if_exists='replace', index=False)
    #drop_table_sql = "DROP TABLE IF EXISTS " + table_name
    #cur.execute(drop_table_sql)
    
    df.to_sql (sqlite_table,sqlite_connection,if_exists='replace',index=False)
    
    sqlite_connection.close()
    conn.close()
    pass  


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