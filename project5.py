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



data = df[["PTS_home", "FG_PCT_home", "FT_PCT_home", "AST_home", "REB_home", "PTS_away", "FG_PCT_away",	"FT_PCT_away", "AST_away", "REB_away"]]

palette = sns.color_palette(None, 30)

figB = px.bar(df, x= 'NBA Teams.NICKNAME(home)', y= "HOME_TEAM_WINS", hover_name='NBA Teams.NICKNAME(home)', title='Home Team Wins', color='NBA Teams.NICKNAME(home)')

app.layout = html.Div(children=[
    html.Br(),
    html.H1(
        'Lab 5: NBA 2019-2020 Game Data Dashboard',
        style={'text-align': 'center'}),
    html.H3(
        'By: Connor Eng',
        style={'text-align': 'center'}
    ),

    html.Br(),
    html.Br(),
    html.Div([
        dcc.Graph(
            id='bar',
            figure=figB
        ),
        dcc.Graph(
            id='scatter',
        ),
    ], style={'width': '49%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(id='PC'),
        dcc.Graph(id='BP'),
    ], style={'display': 'inline-block', 'width': '49%'}),
])

@app.callback(
    Output('scatter', 'figure'),
    Input('bar', 'clickData')
)
def update_figureS(hoverData):
    if not hoverData:
        team = ''   
    else:
        team = hoverData['points'][0]['hovertext']
    
    filterdf = df[df['NBA Teams.NICKNAME(home)'] == team]
    figS = px.scatter(filterdf, x="FG_PCT_home", y='REB_away', hover_name='GAME_ID', color='HOME_TEAM_WINS', color_continuous_scale='Bluered_r', title='Home Points vs Away Rebounds: {}'.format(team))
    figS.update_layout(transition_duration=500)
    figS.update_layout(
        hoverlabel=dict(
            font_size=14,
            font_family="Rockwell"
        )
    )
    return figS

@app.callback(
    Output('PC', 'figure'),
    Input('bar', 'clickData')
)
def update_figurePC(hoverData):
    if not hoverData:
        team = ''   
    else:
        team = hoverData['points'][0]['hovertext']
    
    filterdf = df[df['NBA Teams.NICKNAME(home)'] == team]
    figPC = px.parallel_coordinates(filterdf, dimensions=["PTS_home", "FG_PCT_home", "REB_away", "AST_home", "AST_away", "FG_PCT_away", "PTS_away", "REB_home", "FT_PCT_away", "FT_PCT_home"], title = 'Parallel Coordinates Plot: {}'.format(team))
    figPC.update_layout(transition_duration=500)
    figPC.update_layout(
        hoverlabel=dict(
            font_size=14,
            font_family="Rockwell"
        )
    )
    return figPC

@app.callback(
    Output('BP', 'figure'),
    Input('scatter', 'selectedData'),
)
def update_figureBP(selectData):
    if not selectData:
        date = ''   
        features = ["PTS_home", "FG_PCT_home", "FT_PCT_home", "AST_home", "REB_home", "PTS_away", "FG_PCT_away", "FT_PCT_away", "AST_away", "REB_away"]
        dataBP = df[features]

        pca = PCA(n_components=2)
        components = pca.fit_transform(dataBP)

        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    

        figBP = px.scatter(components, x=0, y=1, color=df['HOME_TEAM_WINS'], title='Biplot', color_continuous_scale=[(0,'yellow'),(1,'green')])

        for i, feature in enumerate(features):
            if feature == "FG_PCT_home" or feature == 'REB_away':
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
        figBP.update_layout(transition_duration=500)
    else:
        features = ["PTS_home", "FG_PCT_home", "FT_PCT_home", "AST_home", "REB_home", "PTS_away", "FG_PCT_away", "FT_PCT_away", "AST_away", "REB_away"]
        #date = selectData['points'][0]['hovertext']
        count = []
        for x in selectData['points']:
            count.append(x['hovertext'])
        ddf = df[df['GAME_ID'].isin(count)]
        dataBP = ddf[features]

        pca = PCA(n_components=2)
        components = pca.fit_transform(dataBP)

        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

        figBP = px.scatter(components, x=0, y=1, title='Biplot')

        for i, feature in enumerate(features):
            if feature == "FG_PCT_home" or feature == 'REB_away':
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
        figBP.update_layout(transition_duration=500)

    return figBP

if __name__ == '__main__':
    app.run_server(debug=True)