import dash
from dash import dash_table
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

from src.Environment.Env import load_p, sgen_p, gen_mw, env
from src.Environment.Env import lineload_max, num_congested, line_max


def create_figure():
    fig = go.Figure()
    #fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=load_p.index,
        y=load_p.sum(axis=1),
        mode='lines',
        hoverinfo='text',
        hovertext=[f'Time: {t}, Power: {round(p, 2)}' for t, p in zip(load_p.index, load_p.sum(axis=1))],
        name='Demand'
    ))

    fig.add_trace(go.Scatter(
        x=sgen_p.index,
        y=sgen_p.sum(axis=1),
        mode='lines',
        hoverinfo='text',
        hovertext=[f'Time: {t}, Power: {round(p, 2)}' for t, p in zip(sgen_p.index, sgen_p.sum(axis=1))],
        name='RES Production'
    ))

    fig.add_trace(go.Scatter(
        x=gen_mw.index,
        y=gen_mw.sum(axis=1),
        mode='lines',
        hoverinfo='text',
        hovertext=[f'Time: {t}, Power: {round(p, 2)}' for t, p in zip(gen_mw.index, gen_mw.sum(axis=1))],
        name='Conventional Generation'
    ))

    fig.update_layout(
        title='Total Power',
        xaxis_title='Time',
        yaxis_title='Power (MW)'
    )
    #fig.update_xaxes(range=[0, time_interval])

    return fig

def create_figure_reward_congestion():
    x = np.linspace(0, 5, 50)  # x values
    decay_rate = 1.0  # decay rate
    initial_value = 1.0  # initial value

    # Calculate the y values for the exponential decay curve
    y = initial_value * np.exp(-decay_rate * x)

    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))

    # Customize the figure
    fig.update_layout(title='Exponential Decay Curve',
                  xaxis_title='Time',
                  yaxis_title='Value')
    return fig

def create_figure_reward_voltage():
    vmax = 100
    vmin = 50
    v = np.linspace(0, 150, 200)  # x values
    y = np.zeros(len(v))

    # Calculate the y values for the exponential decay curve
    for i in range(0,len(v)):
        y[i] = -((max((v[i]-vmax), 0))**2 + (max(vmin-v[i], 0))**2)
    Nvol = max(abs(y))
    y = y/Nvol

    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=v, y=y, mode='lines'))

    # Customize the figure
    fig.update_layout(title='Voltage Deviation Penalty',
                      xaxis_title='Voltage',
                      yaxis_title='Reward')
    return fig

def create_figure_reward_losses():
    x = np.linspace(0, 5, 100)  # x values
    Nloss = max(x)
    # Calculate the y values for the exponential decay curve
    y = -x/Nloss

    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))

    # Customize the figure
    fig.update_layout(title='Losses Penalty',
                  xaxis_title='Total Losses',
                  yaxis_title='Reward')
    return fig

def update_step():
    ## place holder for step function

    num_cong = num_congested
    l_max = line_max
    ll_max = "{:.2g}".format(lineload_max)
    return num_cong,  l_max, ll_max


def create_figure_metric_cumulative_reward():
    x = np.linspace(0, 50000, 500)
    y = np.piecewise(x, [x < 10000, x >= 10000], [lambda x: x / 100, lambda x: 80 + np.sin(x / 5000) * 10])

    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))

    # Customize the figure
    fig.update_layout(title='Cumulative Reward',
                  xaxis_title='Episode',
                  yaxis_title='Cumulative Reward')

    return fig

def create_figure_metric_episode_length():
    x = np.linspace(0, 50000, 500)
    y = np.piecewise(x, [x < 10000, x >= 10000], [lambda x: x / 10, lambda x: 900 + np.sin(x / 5000) * 100])

    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))

    # Customize the figure
    fig.update_layout(title='Episode Length',
                  xaxis_title='Episode',
                  yaxis_title='Episode Length')

    return fig

def create_figure_metric_entropy():
    x = np.linspace(0, 50000, 500)
    y = 1.42 - np.log(x + 1) / 200

    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))

    # Customize the figure
    fig.update_layout(title='Entropy',
                  xaxis_title='Episode',
                  yaxis_title='Entropy')

    return fig

def create_figure_metric_entropy():
    x = np.linspace(0, 50000, 500)
    y = 1.42 - np.log(x + 1) / 200

    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))

    # Customize the figure
    fig.update_layout(title='Entropy',
                  xaxis_title='Episode',
                  yaxis_title='Entropy')

    return fig

def create_figure_metric_learning_rate():
    x = np.linspace(0, 50000, 500)
    y = 3e-4 * (1 - x / 50000)

    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))

    # Customize the figure
    fig.update_layout(title='Learning Rate',
                  xaxis_title='Episode',
                  yaxis_title='Learning Rate')

    return fig


def create_figure_metric_value_estimate():
    x = np.linspace(0, 50000, 500)
    y = 10 * (1 - np.exp(-x / 15000))

    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))

    # Customize the figure
    fig.update_layout(title='Value Estimate',
                  xaxis_title='Episode',
                  yaxis_title='Value Estimate')

    return fig

def create_figure_metric_policy_loss():
    x = np.linspace(0, 50000, 500)
    y = np.piecewise(x,
                     [x < 5000, (x >= 5000) & (x < 15000), x >= 15000],
                     [lambda x: 0.6 + 0.001 * x, lambda x: 2.2 - 0.0001 * (x - 5000),
                      lambda x: 1.8 + 0.00002 * (x - 15000)])
    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))

    # Customize the figure
    fig.update_layout(title='Policy Loss',
                  xaxis_title='Episode',
                  yaxis_title='Policy Loss')

    return fig

def create_figure_metric_value_loss():
    x = np.linspace(0, 50000, 500)
    y = 2.2 - np.exp(-x / 8000) + 0.5 * np.sin(x / 2000)  # Simulates the rise and fall pattern

    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))

    # Customize the figure
    fig.update_layout(title='Value Loss',
                  xaxis_title='Episode',
                  yaxis_title='Value Loss')

    return fig


# Constants
TIME_INTERVALS = [
    {'label': 'year', 'value': 96 * 365},
    {'label': 'month, 31 days', 'value': 96 * 31},
    {'label': 'week', 'value': 96 * 7},
    {'label': 'day', 'value': 96}
]

click = 0

fig_profiles = create_figure()

fig_metric_cumulative_reward = create_figure_metric_cumulative_reward()
fig_metric_episode_length = create_figure_metric_episode_length()
fig_metric_entropy = create_figure_metric_entropy()
fig_metric_learnign_rate = create_figure_metric_learning_rate()
fig_metric_value_estimate = create_figure_metric_value_estimate()
fig_metric_policy_loss = create_figure_metric_policy_loss()
fig_metric_value_loss = create_figure_metric_value_loss()



# Initialize app
app = dash.Dash(__name__)

# Define layout
app.layout = html.Div([
    html.H1('Topology Optimizer Dashboard'),
    dcc.Tabs(id="tabs-example-graph", value='tab-1', children=[
        dcc.Tab(label='Overview', value='tab-1', children=[
            dcc.Tabs(id="subtabs", value="subtab1", children=[
                dcc.Tab(label='Profiles', value='tab-1.1', children=[
                    html.Div([
                        html.H2('Profiles'),
                        dcc.Graph(id='graph-profiles', figure=fig_profiles),
                        html.H3('Time Interval'),
                        # dcc.DatePickerRange(id='date-picker-range'),
                        dcc.Dropdown(
                            id='time-interval-dropdown',
                            options=TIME_INTERVALS,
                            value=TIME_INTERVALS[0]['value']
                        )
                    ])
                ]),
                dcc.Tab(label='Grid View & Congestion', value='tab-1.2', children=[
                    html.Div([
                        html.H2('Transmission Grid:'),
                        html.H2('Congestion:'),
                        html.Span(children=['Number of Congested Lines: ']),
                        html.Span(id='num-congested'),
                        html.Br(),
                        html.Span(children=['Line with the Highest Line Loading: ']),
                        html.Span(id='line-max'),
                        html.Br(),
                        html.Span(children=['Highest Line Loading(%): ']),
                        html.Span(id='lineload-max'),
                        html.Br(),
                        html.Button('Step', id='button_step', n_clicks=0),
                        html.Span(id='tabs-content')
                    ])
                ])
            ])
        ]),
        dcc.Tab(label='Environment', value='tab-2'),
        dcc.Tab(label='Agent', value='tab-3', children=[
            html.Div([
                dcc.Graph(id='graph-metric-cumu-reward', figure=fig_metric_cumulative_reward),
                dcc.Graph(id='graph-metric-epi-length', figure=fig_metric_episode_length),
                dcc.Graph(id='graph-metric-entropy', figure=fig_metric_entropy),
                dcc.Graph(id='graph-metric-learning-rate', figure=fig_metric_learnign_rate),
                dcc.Graph(id='graph-metric-value-estimate', figure=fig_metric_value_estimate),
                dcc.Graph(id='graph-metric-policy-loss', figure=fig_metric_policy_loss),
                dcc.Graph(id='graph-metric-value-loss', figure=fig_metric_value_loss)
            ])
        ])
    ]),
    html.Div(id='tabs-content-example-graph')
])


@app.callback(Output('tabs-content-example-graph', 'children'),
              Input('tabs-example-graph', 'value'),
              Input('subtabs', 'value'))
def render_content(tab, subtabs):
    if tab == 'tab-1':
        if subtabs == 'tab-1.1':
            return ()
        elif subtabs == 'tab-1.2':
            return ()
    elif tab == 'tab-2':
        return html.Div([
            html.H3('Reward Function (congestion)'),
            dcc.Graph(
                id='graph-reward-congestion',
                figure = create_figure_reward_congestion()
            ),
            html.H3('Reward Function (Voltage)'),
            dcc.Graph(
                id='graph-reward-voltage',
                figure=create_figure_reward_voltage()
            ),
            html.H3('Reward Function (Losses)'),
            dcc.Graph(
                id='graph-reward-losses',
                figure=create_figure_reward_losses()
            )])
    elif tab == 'tab-3':
        return html.Div([
            html.H3('Tab content 3')
        ])
# Define callback function
@app.callback(
    Output('graph-profiles', 'figure'),
    #Output('text-area', 'value')
    #Output('date-picker-range', 'value'),
    Input('time-interval-dropdown', 'value'),
    State('graph-profiles','figure')
)
def update_power_graph(time_interval, fig_dict):
    fig = go.Figure(fig_dict) # Convert dictionary to Plotly figure
    if fig.layout.xaxis.range is not None:
        xstrat = fig.layout.xaxis.range[0]
    else:
        xstrat = 0

    xstrat = 0 #test
    print(xstrat)
    fig.update_xaxes(dict(range=[xstrat, xstrat + time_interval]))

    return fig


@app.callback(
    Output('lineload-max', 'children'),
    Output('num-congested', 'children'),
    Output('line-max', 'children'),
    [Input('button_step', 'n_clicks')]
)
def update_output(n_clicks):
    print('Callback triggered')
    if n_clicks != 0:
        num_congested, line_max, lineload_max= update_step()
        #lineload_max = 80  # Example data
        #num_congested = 2  # Example data
        #line_max = "Line 4"  # Example data
    else:
        print('No clicks yet')
        lineload_max = '-'  # Example data
        num_congested = '-'  # Example data
        line_max = '-'  # Example data


    print('Button has been clicked {} times'.format(n_clicks))

    return lineload_max, num_congested, line_max





#def create_figure_metric_episode_length():

 #   return fig

# Run app
if __name__ == '__main__':
    app.run_server()