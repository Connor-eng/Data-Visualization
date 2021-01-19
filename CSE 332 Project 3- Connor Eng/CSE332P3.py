import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import flask
import pandas as pd
import plotly.express as px
import plotly
from datetime import datetime as dt
import calendar
from time import strptime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio
from plotly.graph_objs import *
import plotly.figure_factory as ff
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.manifold import MDS
import math

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
df = pd.read_csv('NBA Games 2019 dataset.csv')

def mapper(month):
    return month.strftime('%b')

df['GAME_DATE_EST'] = pd.to_datetime(df['GAME_DATE_EST'])
#df['month'] = pd.DatetimeIndex(df['GAME_DATE_EST']).month
df['GAME_DATE_EST'] = pd.to_datetime(df['GAME_DATE_EST']).dt.date
df['month']= df['GAME_DATE_EST'].apply(mapper)

data = df[["PTS_home", "FG_PCT_home", "FT_PCT_home", "AST_home", "REB_home", "PTS_away", "FG_PCT_away",	"FT_PCT_away", "AST_away", "REB_away"]]

#Correlation Matrix
corrMatrix = data.corr()
corrM = np.corrcoef(data)
#cmap = sns.diverging_palette(240, 10, n=9)
#sns.heatmap(corrMatrix, cmap = cmap, annot=False, square=True, linewidths=.5)
#figCM = plt.figure()
#Z values are found from data.corr()
x= ["PTS_home", "FG_PCT_home", "FT_PCT_home", "AST_home", "REB_home", "PTS_away", "FG_PCT_away", "FT_PCT_away", "AST_away", "REB_away"] 
y= ["PTS_home", "FG_PCT_home", "FT_PCT_home", "AST_home", "REB_home", "PTS_away", "FG_PCT_away", "FT_PCT_away", "AST_away", "REB_away"] 
z= [
    [1.000000 ,   0.724101  ,   0.195431 , 0.519739 , 0.092183,  0.329098, 0.067424 ,    0.056871,  0.096593 ,-0.354363], 
    [0.724101 ,    1.000000,     0.000161,  0.522784, -0.163973 , 0.071604, 0.014171,     0.033902,  0.018643, -0.609333], 
    [0.195431 ,    0.000161,     1.000000, -0.004835, -0.054997 , 0.069270,0.039494  ,   0.014316 , 0.051379, -0.103968], 
    [0.519739 ,    0.522784,    -0.004835,  1.000000 , 0.002670,  0.083270,0.011967 ,    0.056420,  0.102086, -0.257082],
    [0.092183  ,  -0.163973   , -0.054997,  0.002670 , 1.000000, -0.337288, -0.572648 ,   -0.134625 ,-0.248170 , 0.004443], 
    [0.329098 ,    0.071604  ,   0.069270,  0.083270, -0.337288,  1.000000, 0.711770 ,    0.184199,  0.504399,  0.112462], 
    [0.067424  ,   0.014171 ,    0.039494 , 0.011967, -0.572648,  0.711770,1.000000 ,    0.010463,  0.511445, -0.143784],
    [0.056871  ,   0.033902  ,   0.014316,  0.056420, -0.134625,  0.184199, 0.010463  ,   1.000000 , -0.036478, -0.032468],
    [0.096593 ,    0.018643 ,    0.051379,  0.102086, -0.248170 , 0.504399,0.511445 ,   -0.036478,  1.000000, -0.002387],
    [-0.354363 ,   -0.609333 ,   -0.103968 ,-0.257082 , 0.004443 , 0.112462, -0.143784 ,   -0.032468, -0.002387 , 1.000000]
    ]
font_colors = ['black', 'white']
figCM = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='RdBu', font_colors= font_colors, reversescale= True, showscale=True)
figCM.update_layout(
    hoverlabel=dict(
        bgcolor="white",
        font_size=14,
        font_family="Rockwell"
    )
)

#Scatter Plot Matrix
data2 = df[["PTS_home", "FG_PCT_home", "PTS_away", "FG_PCT_away", "REB_away", "HOME_TEAM_WINS"]]
figSM = px.scatter_matrix(data2, dimensions= ["PTS_home", "FG_PCT_home", "PTS_away", "FG_PCT_away", "REB_away"], color = "HOME_TEAM_WINS", title = 'Scatter Plot Matrix', height=800,  color_continuous_scale='Bluered_r')
figSM.update_layout(
    hoverlabel=dict(
        bgcolor="white",
        font_size=14,
        font_family="Rockwell"
    )
)

#Parallel Coordinates Plot
dataPC = df[["PTS_home", "FG_PCT_home", "FT_PCT_home", "AST_home", "REB_home", "PTS_away", "FG_PCT_away",	"FT_PCT_away", "AST_away", "REB_away", "HOME_TEAM_WINS"]]
figPC = px.parallel_coordinates(dataPC, dimensions=["PTS_home", "FG_PCT_home", "REB_away", "AST_home", "AST_away", "FG_PCT_away", "PTS_away", "REB_home", "FT_PCT_away", "FT_PCT_home"], title = 'Parallel Coordinates Plot', color='HOME_TEAM_WINS', height=1000)

#PCA
dataPCA = df[["PTS_home", "PTS_away"]]
pca = PCA(n_components=2)
compPCA = pca.fit_transform(dataPCA)
figPCA = px.scatter(compPCA, x=0, y=1, color=df['HOME_TEAM_WINS'], title='PCA Plot',  color_continuous_scale='Bluered_r')
figPCA.update_xaxes(title_text="PC1")
figPCA.update_yaxes(title_text="PC2")
figPCA.update_layout(coloraxis_colorbar=dict(title = 'Home Team Wins'))

PCAx = StandardScaler().fit_transform(data)
pcamodel = PCA(n_components=10)
pca = pcamodel.fit_transform(PCAx)
PCAy = pcamodel.explained_variance_
figScree = go.Figure([go.Bar(x=[1,2,3,4,5,6,7,8,9,10], y=PCAy)])
figScree.update_xaxes(title_text="Components")
figScree.update_yaxes(title_text="Explained Variance")

#Biplot
features = ["PTS_home", "FG_PCT_home", "FT_PCT_home", "AST_home", "REB_home", "PTS_away", "FG_PCT_away",	"FT_PCT_away", "AST_away", "REB_away"]
dataBP = df[features]

pca = PCA(n_components=2)
components = pca.fit_transform(dataBP)

loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

figBP = px.scatter(components, x=0, y=1, color=df['HOME_TEAM_WINS'], title='Biplot', color_continuous_scale=[(0,'yellow'),(1,'green')])

for i, feature in enumerate(features):
    if feature == 'PTS_home' or feature == 'PTS_away':
        figBP.add_shape(
            type='line',
            x0=0, y0=0,
            x1=loadings[i, 0],
            y1=loadings[i, 1]
        )
        figBP.add_annotation(
            x=loadings[i, 0],
            y=loadings[i, 1],
            ax=0, ay=0,
            xanchor="center",
            yanchor="bottom",
            text=feature,
        )
figBP.update_xaxes(title_text="PC1")
figBP.update_yaxes(title_text="PC2")
figBP.update_layout(coloraxis_colorbar=dict(title = 'Home Team Wins'))

#MDS Data
mds = MDS(n_components=2)
mdsC = mds.fit_transform(data)
figMDS = px.scatter(mdsC, x=0, y=1, color=df['HOME_TEAM_WINS'], title='MDS of Data', color_continuous_scale='Bluered_r')
figMDS.update_xaxes(title_text="Dimension 1")
figMDS.update_yaxes(title_text="Dimension 2")
figMDS.update_layout(coloraxis_colorbar=dict(title = 'Home Team Wins'))

#MDS Attributes
mdsA = mds.fit_transform(1-abs(corrMatrix))
figMDSa = px.scatter(mdsA, x=0, y=1, title='MDS of Attributes')
figMDSa.update_xaxes(title_text="Dimension 1")
figMDSa.update_yaxes(title_text="Dimension 2")

app = dash.Dash(__name__)

url_bar_and_content_div = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

layout_index = html.Div([
    html.Br(),
    html.H1(
        'Project 3 Part 1: Correlation Matrix',
        style={'text-align': 'center'}),
    html.H3(
        'By: Connor Eng',
        style={'text-align': 'center'}
    ),
    html.Br(),
    dcc.Link('Part 2: Scatter Plot Matrix', href='/page-1'),
    html.Br(),
    dcc.Link('Part 3: Parallel Coordinates Plot', href='/page-2'),
    html.Br(),
    dcc.Link('Part 4: PCA Plot', href='/page-3'),
    html.Br(),
    dcc.Link('Part 5: Biplot', href='/page-4'),
    html.Br(),
    dcc.Link('Part 6: MDS Data', href='/page-5'),
    html.Br(),
    dcc.Link('Part 7: MDS Attributes', href='/page-6'),
    html.Br(),
    html.Br(),
    html.H4(
        'Correlation Matrix'
    ),
    dcc.Graph(
        id='corrMatrix',
        figure = figCM
    ),
])

layout_page_1 = html.Div([
    html.Br(),
    html.H1(
        'Project 3 Part 2: Scatter Plot Matrix',
        style={'text-align': 'center'}),
    html.H3(
        'By: Connor Eng',
        style={'text-align': 'center'}
    ),
    html.Br(),
    dcc.Link('Part 1: Correlation Matrix', href='/'),
    html.Br(),
    dcc.Link('Part 3: Parallel Coordinates Plot', href='/page-2'),
    html.Br(),
    dcc.Link('Part 4: PCA Plot', href='/page-3'),
    html.Br(),
    dcc.Link('Part 5: Biplot', href='/page-4'),
    html.Br(),
    dcc.Link('Part 6: MDS Data', href='/page-5'),
    html.Br(),
    dcc.Link('Part 7: MDS Attributes', href='/page-6'),
    html.Br(),
    html.Br(),
    dcc.Graph(
        id='scatterMatrix',
        figure = figSM
    ),
])

layout_page_2 = html.Div([
    html.Br(),
    html.H1(
        'Project 3 Part 3: Parallel Coordinates Plot',
        style={'text-align': 'center'}),
    html.H3(
        'By: Connor Eng',
        style={'text-align': 'center'}
    ),
    html.Br(),
    dcc.Link('Part 1: Correlation Matrix', href='/'),
    html.Br(),
    dcc.Link('Part 2: Scatter Plot Matrix', href='/page-1'),
    html.Br(),
    dcc.Link('Part 4: PCA Plot', href='/page-3'),
    html.Br(),
    dcc.Link('Part 5: Biplot', href='/page-4'),
    html.Br(),
    dcc.Link('Part 6: MDS Data', href='/page-5'),
    html.Br(),
    dcc.Link('Part 7: MDS Attributes', href='/page-6'),
    html.Br(),
    html.Br(),
    dcc.Graph(
        id='parallelCoordinate',
        figure = figPC
    ),
])

layout_page_3 = html.Div([
    html.Br(),
    html.H1(
        'Project 3 Part 4: PCA Plot',
        style={'text-align': 'center'}),
    html.H3(
        'By: Connor Eng',
        style={'text-align': 'center'}
    ),
    html.Br(),
    dcc.Link('Part 1: Correlation Matrix', href='/'),
    html.Br(),
    dcc.Link('Part 2: Scatter Plot Matrix', href='/page-1'),
    html.Br(),
    dcc.Link('Part 3: Parallel Coordinates Plot', href='/page-2'),
    html.Br(),
    dcc.Link('Part 5: Biplot', href='/page-4'),
    html.Br(),
    dcc.Link('Part 6: MDS Data', href='/page-5'),
    html.Br(),
    dcc.Link('Part 7: MDS Attributes', href='/page-6'),
    html.Br(),
    html.Br(),
    dcc.Graph(
        id='PCA',
        figure = figPCA
    ),
    html.Br(),
    html.H4(
        "Scree Plot"
    ),
    dcc.Graph(
        id='Scree',
        figure = figScree
    ),
])

layout_page_4 = html.Div([
    html.Br(),
    html.H1(
        'Project 3 Part 5: Biplot',
        style={'text-align': 'center'}),
    html.H3(
        'By: Connor Eng',
        style={'text-align': 'center'}
    ),
    html.Br(),
    dcc.Link('Part 1: Correlation Matrix', href='/'),
    html.Br(),
    dcc.Link('Part 2: Scatter Plot Matrix', href='/page-1'),
    html.Br(),
    dcc.Link('Part 3: Parallel Coordinates Plot', href='/page-2'),
    html.Br(),
    dcc.Link('Part 4: PCA Plot', href='/page-3'),
    html.Br(),
    dcc.Link('Part 6: MDS Data', href='/page-5'),
    html.Br(),
    dcc.Link('Part 7: MDS Attributes', href='/page-6'),
    html.Br(),
    html.Br(),
    dcc.Graph(
        id='biplot',
        figure = figBP
    ),
])

layout_page_5 = html.Div([
    html.Br(),
    html.H1(
        'Project 3 Part 6: MDS Data',
        style={'text-align': 'center'}),
    html.H3(
        'By: Connor Eng',
        style={'text-align': 'center'}
    ),
    html.Br(),
    dcc.Link('Part 1: Correlation Matrix', href='/'),
    html.Br(),
    dcc.Link('Part 2: Scatter Plot Matrix', href='/page-1'),
    html.Br(),
    dcc.Link('Part 3: Parallel Coordinates Plot', href='/page-2'),
    html.Br(),
    dcc.Link('Part 4: PCA Plot', href='/page-3'),
    html.Br(),
    dcc.Link('Part 5: Biplot', href='/page-4'),
    html.Br(),
    dcc.Link('Part 7: MDS Attributes', href='/page-6'),
    html.Br(),
    html.Br(),
    dcc.Graph(
        id='MDSD',
        figure = figMDS
    ),
])

layout_page_6 = html.Div([
    html.Br(),
    html.H1(
        'Project 3 Part 7: MDS Attribute',
        style={'text-align': 'center'}),
    html.H3(
        'By: Connor Eng',
        style={'text-align': 'center'}
    ),
    html.Br(),
    dcc.Link('Part 1: Correlation Matrix', href='/'),
    html.Br(),
    dcc.Link('Part 2: Scatter Plot Matrix', href='/page-1'),
    html.Br(),
    dcc.Link('Part 3: Parallel Coordinates Plot', href='/page-2'),
    html.Br(),
    dcc.Link('Part 4: PCA Plot', href='/page-3'),
    html.Br(),
    dcc.Link('Part 5: Biplot', href='/page-4'),
    html.Br(),
    dcc.Link('Part 6: MDS Data', href='/page-5'),
    html.Br(),
    html.Br(),
    dcc.Graph(
        id='MDSa',
        figure = figMDSa
    ),
])

# index layout
app.layout = url_bar_and_content_div

# "complete" layout
app.validation_layout = html.Div([
    url_bar_and_content_div,
    layout_index,
    layout_page_1,
    layout_page_2,
    layout_page_3,
    layout_page_4,
    layout_page_5,
    layout_page_6,
])


# Index callbacks
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == "/page-1":
        return layout_page_1
    elif pathname == "/page-2":
        return layout_page_2
    elif pathname == "/page-3":
        return layout_page_3
    elif pathname == "/page-4":
        return layout_page_4
    elif pathname == "/page-5":
        return layout_page_5
    elif pathname == "/page-6":
        return layout_page_6
    else:
        return layout_index


if __name__ == '__main__':
    app.run_server(debug=True)