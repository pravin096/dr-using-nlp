import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

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
     
    def starting_verb(self, text):
        import nltk
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def tokenize(text):
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
    df2 = df.iloc[:,5:-2]
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
            render_template - object containing master.html page and plot(s) to be displayed on the page
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
            render_template - object containing eda.html page and plot(s) to be displayed on the page
    '''
    # create visuals

    bar0 = create_plot_percentage_of_messages_by_class()
    bar1 = create_plot_message_count_by_classcount()
    
    return render_template('eda_orig.html',plot=bar0,plot1=bar1)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_columns = df.iloc[:,5:-2].columns #include all category columns -> Cat_
    classification_results = dict(zip(classification_columns, classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


@app.route('/classify_sample1',methods = ['GET','POST'])
def classify_sample1(x = None, y = None):
    # save user input in query
    text = 'Storm in London' 
    print('in sample1')

    # use model to predict classification for query
    classification_labels = model.predict([text])[0]
    classification_columns = df.iloc[:,5:-2].columns #include all category columns -> Cat_
    classification_results = dict(zip(classification_columns, classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=text,
        classification_result=classification_results
    )

@app.route('/classify_sample2',methods = ['GET','POST'])
def classify_sample2(x = None, y = None):
    # save user input in query
    text = 'Storm in Berlin' 
    print('in sample2')

    # use model to predict classification for query
    classification_labels = model.predict([text])[0]
    classification_columns = df.iloc[:,5:-2].columns #include all category columns -> Cat_
    classification_results = dict(zip(classification_columns, classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=text,
        classification_result=classification_results
    )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
