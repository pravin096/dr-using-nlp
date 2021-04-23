import json
import plotly
import pandas as pd

import nltk
import string
import re

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.offline import plot, iplot
from plotly.graph_objs import Bar
import plotly.graph_objs as go
import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin


app = Flask(__name__)

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
        INPUTS
            text - Observation containing text string
        OUTPUTS
            clean_tokens - List containing tokens from cleaned text
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse',engine)

# load model
model = joblib.load("../models/classifier.pkl")

def clean_text(text):
    '''
        INPUTS
            text - Observation containing text string
        OUTPUTS
            clean_tokens - List containing tokens from cleaned text
    '''
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

def create_plot():
    '''
        INPUTS
            None
        OUTPUTS
            graphJSON - JSON object containing the plot depicting counts by genre
    '''
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    data = [
        Bar(
            x=genre_names, # assign x as the dataframe column 'x'
            y=genre_counts
        )
    ]
    
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON


def create_plot_percentage_of_messages_by_class():
    '''
        INPUTS
            None
        OUTPUTS
            graphJSON - JSON object containing the plot depicting message count (as % 
                        of total number of messages) for each category/ label
    '''
    import plotly.graph_objects as go
    # extract data needed for visuals    
    df2 = df.iloc[:,4:-2]
    class_counts = pd.DataFrame(data=round(df2.sum()/df2.shape[0],4)*100,columns=['PercentageOfTotalMessages'])
    class_counts['Category'] = class_counts.index
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=class_counts['PercentageOfTotalMessages']
                       , y=class_counts['Category']
                       , text = '%'
                       , textposition = 'auto'
                       , orientation='h'
                       , marker=dict(color = class_counts['PercentageOfTotalMessages']
                       , colorbar=dict(
                           title="<b>%ofTotalMessages<b>"
                           )
                       , colorscale='Viridis')
                       ))
    
    fig.update_layout(
                         title='<b>Messages per category as (%) of Total Messages<b>'
                       , xaxis_title="<b>Percentage of Total Messages<b>"
                       , yaxis_title="<b>Category<b>"
                       , autosize=False
                       , width=600
                       , height=600
                       , margin=dict(l=20,r=20,b=50,t=50,pad=2)
                       , yaxis=dict(showgrid=False,zeroline=False,)
                     )      
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

def create_plot_message_count_by_classcount():
    '''
        INPUTS
            None
        OUTPUTS
            graphJSON - JSON object containing the plot depicting message count by class/label count
    '''
    import plotly.graph_objects as go
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    
    no_of_labels_per_message = df.LabelSum.value_counts().reset_index()

    fig = go.Figure(go.Bar(y=no_of_labels_per_message['LabelSum'], x=no_of_labels_per_message['index']))#, orientation='h'))
#            , title = '<b>Messages per category as (%) of Total Messages<b>'))
    fig.update_layout(
     title='<b>Count of Messages with multiple labels<b>',
     xaxis_title="<b>Count of labels<b>",
     yaxis_title="<b>Counts of Messages with multiple labels<b>",
     width=500,
     height=500,
    )
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    '''
        INPUTS
            None
        OUTPUTS
            render_template - object containing master.html page and bar plot to be displayed on the page
    '''
    bar = create_plot()
    return render_template('master.html',plot=bar)

#web page that displays EDA visualizations
@app.route('/eda_orig')
def eda_orig():
    '''
        INPUTS
            None
        OUTPUTS
            render_template - object containing eda.html page and plots to be displayed on the page
    '''
    # create visuals

    bar0 = create_plot_percentage_of_messages_by_class()
    bar1 = create_plot_message_count_by_classcount()
    
    return render_template('eda_orig.html',plot=bar0,plot1=bar1)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    '''
        INPUTS
            query  - input text string entered by the user in the flask application
        OUTPUTS
            render_template - object containing go.html page, that displays classification results from
                              execution of the model, and input text string received from the user for
                              the classification
    '''
    # save user input in query
    query = request.args.get('query', '') 
    cleaned_text = clean_text(query)

    # use model to predict classification for query
    classification_labels = model.predict([cleaned_text])[0]
    classification_columns = df.iloc[:,4:-2].columns #include all category columns -> Cat_
    classification_results = dict(zip(classification_columns, classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


@app.route('/classify_sample1',methods = ['GET','POST'])
def classify_sample1(x = None, y = None):
    '''
        INPUTS
            sample_query  - input text string containing the sample message selected by the user for 
                            classification (this function is executed when user chooses to classify 
                            sample message (the one displayed on bottom left on the main screen)
        OUTPUTS
            render_template - object containing go.html page, that displays classification results from
                              execution of the model, and input text string received from the user for
                              the classification
    '''
    # save user input in query
    sample_query = df.loc[25790,['message']].values[0] 
    cleaned_text = clean_text(sample_query)

    # use model to predict classification for query
    classification_labels = model.predict([cleaned_text])[0]
    classification_columns = df.iloc[:,4:-2].columns #include all category columns -> Cat_
    classification_results = dict(zip(classification_columns, classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=sample_query,
        classification_result=classification_results
    )

@app.route('/classify_sample2',methods = ['GET','POST'])
def classify_sample2(x = None, y = None):
    '''
        INPUTS
            sample_query  - input text string containing the sample message selected by the user for 
                            classification (this function is executed when user chooses to classify 
                            sample message (the one displayed on bottom centre on the main screen)
        OUTPUTS
            render_template - object containing go.html page, that displays classification results from
                              execution of the model, and input text string received from the user for
                              the classification
    '''
    # save user input in query
    sample_query = df.loc[13201,['message']].values[0] 
    cleaned_text = clean_text(sample_query) 
    print('in sample2')

    # use model to predict classification for query
    classification_labels = model.predict([cleaned_text])[0]
    classification_columns = df.iloc[:,4:-2].columns #include all category columns -> Cat_
    classification_results = dict(zip(classification_columns, classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=sample_query,
        classification_result=classification_results
    )
@app.route('/classify_sample3',methods = ['GET','POST'])
def classify_sample3(x = None, y = None):
    '''
        INPUTS
            sample_query  - input text string containing the sample message selected by the user for 
                            classification (this function is executed when user chooses to classify 
                            sample message (the one displayed on bottom right on the main screen)
        OUTPUTS
            render_template - object containing go.html page, that displays classification results from
                              execution of the model, and input text string received from the user for
                              the classification
    '''
    # save user input in query
    sample_query = df.loc[23110,['message']].values[0] 
    cleaned_text = clean_text(sample_query) 

    # use model to predict classification for query
    classification_labels = model.predict([cleaned_text])[0]
    classification_columns = df.iloc[:,4:-2].columns #include all category columns -> Cat_
    classification_results = dict(zip(classification_columns, classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=sample_query,
        classification_result=classification_results
    )
def main():
    '''
        Main function that runs the app hosted on the local machine, port 3001
        INPUTS
            None
        OUTPUTs
            None
    '''
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
