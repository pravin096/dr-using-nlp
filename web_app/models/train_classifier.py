import sys
import sqlite3
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, \
    confusion_matrix, multilabel_confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold


def load_data(database_filepath):
    '''
        INPUT
            database_filepath : Database_filepath containing the databse (i.e. .db) filename
        
        OUTPUT
            X                 : Pandas series containing cleaned messages (retrieved from the SQLite database)
            y                 : Dataframe containing labels for each of the 36 categories (retrieved from the SQLite database)
            categories        : List containing category names 
    '''
    # Read sqlite query results into a pandas DataFrame
    con = sqlite3.connect(database_filepath)
    
    #Extract the table name from filepath
    import re

    #filepath = "../data/DisasterResponse.db"

    search_for_dbname = re.search('/(.+?).db', database_filepath)

    if search_for_dbname:
        tablename_incl_path = search_for_dbname.group(1)

    word_list = tablename_incl_path.split("/")
    table_name = word_list[len(word_list) -1]
    
    #Construct query string
    query_string = "SELECT * from " + table_name
    df = pd.read_sql_query(query_string, con)

    # Verify that result of SQL query is stored in the dataframe
    #df = df.drop(['index'],axis=1)

    con.close()
    
    category_names = [col for col in df.columns if 'Cat_' in col]
    print("Df before cleaning ",df.shape)
    df = df[df['LabelSum'] > 0]
    print("Df after deletion of rows without labels ",df.shape)
    #drop child alone column - single class
    #df  = df.drop(['Cat_child_alone','Cat_related'],axis=1)
    #print("Df after dropping child_alone & related columns ",df.shape)
        
    cat_cols = [col for col in df.columns if 'Cat_' in col]
    
    X = df.cleaned_message
    y = df.loc[:, cat_cols]

    return X,y, category_names

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    ''' Extracts information (through part of speech tagging) from a series containing input string 
         METHODS
            starting_verb - Returns a boolean if first tag is either a verb (verb or verb present (VBP))
                            or first word is a retweet (RT)
            fit           - No specific operation applied here
            transform     - Applies the starting_verb method to each individual observation containing 
                            input text string 
    '''  
    def starting_verb(self, text):
        '''
            INPUTS
                text - observation containing input text string
            OUTPUTS
                boolean - Returns True if first part of speech tag is VB or VBP or if first word is RT,
                          False otherwise
        '''
        import nltk
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        '''
          INPUTS
                 x - Series containing text string
                 y - labels; if not specified, defaulted to None
          OUTPUTS
                 Object of class StartingVerbExtractor  
        '''
        return self

    def transform(self, X):
        '''
            INPUTS
                X - Series containing text string
            OUTPUTS
                X_tagged - Series containing additional feature containing boolean variable that 
                           indicates if first POS tag is verb or first word is a retweet  
        '''
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def tokenize(text):
    '''
        INPUT
            text            : list containing cleaned messages
        
        OUTPUT
            clean_tokens    : list containing clean tokens (after lemmatization)
    '''
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    '''
        INPUT
            None
            
        OUTPUT
            cv     : model object
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'features__text_pipeline__vect__ngram_range': [(1, 2)],
        'features__text_pipeline__vect__max_df': [1.0],
        'features__text_pipeline__vect__max_features': [5000],
        'features__text_pipeline__tfidf__use_idf': [False],
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [3, 4],
        'features__transformer_weights': [
            {'text_pipeline': 0.5, 'starting_verb': 1},
            #{'text_pipeline': 0.8, 'starting_verb': 1},
        ]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters,verbose = 2,cv=KFold(2, random_state=2))
    
    return cv


def evaluate_model(model, X_test, Y_test):
    '''
    INPUT
        model          : model object (created by the build_model() function )
        X_test         : test dataset created by the train_test_split, containing cleaned text messages
        Y_test         : labels corresponding to X_test
    
    OUTPUT
        None
    '''
    labels = Y_test.columns
    y_pred = model.predict(X_test)

    conf_mat_dict={}

    for label_col in range(len(labels)):
        y_true_label = Y_test.values[:, label_col]
        y_pred_label = y_pred[:, label_col]
        conf_mat_dict[labels[label_col]] = confusion_matrix(y_pred=y_pred_label, y_true=y_true_label)


    print("Labels:", labels)
    for label, matrix in conf_mat_dict.items():
        print("Confusion matrix for label {}:".format(label))
        print(matrix)
    
    print('\n Classification report \n',classification_report(Y_test, y_pred, digits=3))
    accuracy = (y_pred == Y_test).mean()
    print("Accuracy:", accuracy)
    
    print("\nBest Parameters:", model.best_params_)
    pass


def save_model(model, model_filepath):
    '''
    INPUT
        model          : model object resulting from the fit function and GridSearch
        model_filepath : filepath including the filename to store the model as a serialized .pkl object
    
    OUTPUT
        None
    '''
    #Store the model
    import pickle

    # It is important to use binary access
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)
    
    save_model = 0
    pass


def main():
    '''
        INPUT
            database_filepath   : filepath including the filename of the database file that contains cleaned messages and labels dataset
            model_filepath      : filepath including the filename to store the model as a serialized .pkl object
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')



if __name__ == '__main__':
    main()
