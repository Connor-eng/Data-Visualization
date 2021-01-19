import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import plotly
from dash.dependencies import Input, Output
from datetime import datetime as dt
import calendar
from time import strptime

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
df = pd.read_csv('NBA Games 2019 dataset.csv')


def mapper(month):
    return month.strftime('%b')

df['GAME_DATE_EST'] = pd.to_datetime(df['GAME_DATE_EST'])
#df['month'] = pd.DatetimeIndex(df['GAME_DATE_EST']).month
df['GAME_DATE_EST'] = pd.to_datetime(df['GAME_DATE_EST']).dt.date
df['month']= df['GAME_DATE_EST'].apply(mapper)

app.layout = html.Div(children=[
    html.Br(),
    html.H1(
        'NBA 2019-2020 Season Game Data',
        style={'text-align': 'center'}),
    html.H3(
        'By: Connor Eng',
        style={'text-align': 'center'}
    ),

    html.Br(),
    html.Br(),
    dcc.Graph(
        id='bar',
    ),
    html.P(
        "Select a y-axis attribute for the bar chart:"
    ),
    dcc.Dropdown(
        id='bDropY',
        options=[
            {'label': 'Home team- points', 'value': 'PTS_home'},
            {'label': 'Home team- Field goal %', 'value': 'FG_PCT_home'},
            {'label': 'Home team- Free throw %', 'value': 'FT_PCT_home'},
            {'label': 'Home team- 3 point %', 'value': 'FG3_PCT_home'},
            {'label': 'Home team- assists', 'value':'AST_home'},
            {'label': 'Home team- rebounds', 'value': 'REB_home'},
            {'label': 'Away team- points', 'value': 'PTS_away'},
            {'label': 'Away team- Field goal %', 'value': 'FG_PCT_away'},
            {'label': 'Away team- Free throw %', 'value': 'FT_PCT_away'},
            {'label': 'Away team- 3 point %', 'value': 'FG3_PCT_away'},
            {'label': 'Away team- assists', 'value':'AST_away'},
            {'label': 'Away team- rebounds', 'value': 'REB_away'},
            {'label': 'Home team wins', 'value': 'HOME_TEAM_WINS'},
        ],
        value = 'HOME_TEAM_WINS'
    ),

    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    dcc.Graph(
        id='pie',
    ),
    html.P(
        "Select a home team to analyze the team's performance in each month:"
    ),
    dcc.Dropdown(
        id='pDropT',
        options=[
            {'label': '76ers', 'value': '76ers'},
            {'label': 'Bucks', 'value': 'Bucks'},
            {'label': 'Bulls', 'value': 'Bulls'},
            {'label': 'Cavaliers', 'value': 'Cavaliers'},
            {'label': 'Celtics', 'value': 'Celtics'},
            {'label': 'Clippers', 'value': 'Clippers'},
            {'label': 'Grizzlies', 'value': 'Grizzlies'},
            {'label': 'Hawks', 'value': 'Hawks'},
            {'label': 'Heat', 'value': 'Heat'},
            {'label': 'Hornets', 'value': 'Hornets'},
            {'label': 'Jazz', 'value': 'Jazz'},
            {'label': 'Kings', 'value': 'Kings'},
            {'label': 'Knicks', 'value': 'Knicks'},
            {'label': 'Lakers', 'value': 'Lakers'},
            {'label': 'Magic', 'value': 'Magic'},
            {'label': 'Mavericks', 'value': 'Mavericks'},
            {'label': 'Nets', 'value': 'Nets'},
            {'label': 'Nuggets', 'value': 'Nuggets'},
            {'label': 'Pacers', 'value': 'Pacers'},
            {'label': 'Pelicans', 'value': 'Pelicans'},
            {'label': 'Pistons', 'value': 'Pistons'},
            {'label': 'Raptors', 'value': 'Raptors'},
            {'label': 'Rockets', 'value': 'Rockets'},
            {'label': 'Spurs', 'value': 'Spurs'},
            {'label': 'Suns', 'value': 'Suns'},
            {'label': 'Thunder', 'value': 'Thunder'},
            {'label': 'Timberwolves', 'value': 'Timberwolves'},
            {'label': 'Trail Blazers', 'value': 'Trail Blazers'},
            {'label': 'Warriors', 'value': 'Warriors'},
            {'label': 'Wizards', 'value': 'Wizards'},
        ],
        value = 'Knicks'
    ),
    html.P(
        "Select an attribute for the pie chart:"
    ),
    dcc.Dropdown(
        id='pDrop',
        options=[
            {'label': 'Home team- points', 'value': 'PTS_home'},
            {'label': 'Home team- Field goal %', 'value': 'FG_PCT_home'},
            {'label': 'Home team- Free throw %', 'value': 'FT_PCT_home'},
            {'label': 'Home team- 3 point %', 'value': 'FG3_PCT_home'},
            {'label': 'Home team- assists', 'value':'AST_home'},
            {'label': 'Home team- rebounds', 'value': 'REB_home'},
            {'label': 'Away team- points', 'value': 'PTS_away'},
            {'label': 'Away team- Field goal %', 'value': 'FG_PCT_away'},
            {'label': 'Away team- Free throw %', 'value': 'FT_PCT_away'},
            {'label': 'Away team- 3 point %', 'value': 'FG3_PCT_away'},
            {'label': 'Away team- assists', 'value':'AST_away'},
            {'label': 'Away team- rebounds', 'value': 'REB_away'},
            {'label': 'Home team wins', 'value': 'HOME_TEAM_WINS'},
        ],
        value = 'PTS_home'
    ),

    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    dcc.Graph(
        id='scatter',
    ),
    html.P(
        "Select an x-axis attribute for the scatter plot:"
    ),
    dcc.Dropdown(
        id='sDropX',
        options=[
            {'label': 'Home team- points', 'value': 'PTS_home'},
            {'label': 'Home team- Field goal %', 'value': 'FG_PCT_home'},
            {'label': 'Home team- Free throw %', 'value': 'FT_PCT_home'},
            {'label': 'Home team- 3 point %', 'value': 'FG3_PCT_home'},
            {'label': 'Home team- assists', 'value':'AST_home'},
            {'label': 'Home team- rebounds', 'value': 'REB_home'},
            {'label': 'Away team- points', 'value': 'PTS_away'},
            {'label': 'Away team- Field goal %', 'value': 'FG_PCT_away'},
            {'label': 'Away team- Free throw %', 'value': 'FT_PCT_away'},
            {'label': 'Away team- 3 point %', 'value': 'FG3_PCT_away'},
            {'label': 'Away team- assists', 'value':'AST_away'},
            {'label': 'Away team- rebounds', 'value': 'REB_away'},
        ],
        value = 'PTS_home'
    ),
    html.P(
        "Select a y-axis attribute for the scatter plot:"
    ),
    dcc.Dropdown(
        id='sDropY',
        options=[
            {'label': 'Home team- points', 'value': 'PTS_home'},
            {'label': 'Home team- Field goal %', 'value': 'FG_PCT_home'},
            {'label': 'Home team- Free throw %', 'value': 'FT_PCT_home'},
            {'label': 'Home team- 3 point %', 'value': 'FG3_PCT_home'},
            {'label': 'Home team- assists', 'value':'AST_home'},
            {'label': 'Home team- rebounds', 'value': 'REB_home'},
            {'label': 'Away team- points', 'value': 'PTS_away'},
            {'label': 'Away team- Field goal %', 'value': 'FG_PCT_away'},
            {'label': 'Away team- Free throw %', 'value': 'FT_PCT_away'},
            {'label': 'Away team- 3 point %', 'value': 'FG3_PCT_away'},
            {'label': 'Away team- assists', 'value':'AST_away'},
            {'label': 'Away team- rebounds', 'value': 'REB_away'},
        ],
        value = 'PTS_away'
    )
])

@app.callback(
    Output('bar', 'figure'),
    Input('bDropY', 'value')
)
def update_figureB(selected_y):
    figB = px.bar(df, x= 'NBA Teams.NICKNAME(home)', y= selected_y, hover_name='GAME_DATE_EST', title='Home Team Statistics', labels={'NBA Teams.NICKNAME(home)': 'NBA Home Team'})
    figB.update_layout(transition_duration=500)
    figB.update_yaxes(showspikes=True)
    figB.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Rockwell"
        )
    )
    return figB

@app.callback(
    Output('pie', 'figure'),
    Input('pDropT', 'value'), Input('pDrop', 'value'), 
)
def update_figureP(team, value):
    filterDf = df[df['NBA Teams.NICKNAME(home)'] == team]
    figP = px.pie(filterDf, values= value, names='month', color='month', title='NBA Team Monthly Statistics', labels={'month': "Month Played"})
    figP.update_layout(transition_duration=500)
    figP.update_layout(
        hoverlabel=dict(
            font_size=14,
            font_family="Rockwell"
        )
    )
    return figP

@app.callback(
    Output('scatter', 'figure'),
    Input('sDropX', 'value'), Input('sDropY', 'value')
)
def update_figureS(selected_x, selected_y):
    figS = px.scatter(df, x=selected_x, y=selected_y,color="NBA Teams.NICKNAME(home)", hover_name='GAME_DATE_EST', title='NBA Season Statistics', labels={"NBA Teams.NICKNAME(home)": "NBA Home Team"})
    figS.update_layout(transition_duration=500)
    figS.update_layout(
        hoverlabel=dict(
            font_size=14,
            font_family="Rockwell"
        )
    )
    return figS

if __name__ == '__main__':
    app.run_server(debug=True)
