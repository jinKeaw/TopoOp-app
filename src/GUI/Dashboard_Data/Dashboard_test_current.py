# Standard library imports
from datetime import date
from logging import warning
import time

# Third-party library imports
import pandapower as pp
import dash
from dash import Dash, callback, dcc, html, dash_table, ctx
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import gymnasium as gym
from gymnasium import spaces

import dash_bootstrap_components as dbc  # Import for collapsible components


# Local application imports
import src.Environment.Env_test_ as Env
from src.Environment.Env_test_ import (
    load_p, sgen_p, gen_mw, env, num_lines, num_switches, network, profiles,
    profiles_time_step, limit, network_code, avg_ll_congested, lineload_max,
    num_congested, line_max_indices)

#import src.Environment.Env as Env
#from src.Environment.Env import (
#    load_p, sgen_p, gen_mw, env, num_lines, num_switches, network, profiles,
#    profiles_time_step, limit, network_code, avg_ll_congested, lineload_max,
#    num_congested, line_max)

from src.GUI.Dashboard_Data.Agent_metrics import (
    steps_ep_len_mean, ep_len_mean,
    steps_rew_mean, ep_rew_mean,
    steps_fps, fps,
    steps_time_elapsed, time_elapsed,
    steps_total_timesteps, total_timesteps,
    steps_approx_kl, approx_kl,
    steps_clip_fraction, clip_fraction,
    steps_clip_range, clip_range,
    steps_entropy_loss, entropy_loss,
    steps_explained_variance, explained_variance,
    steps_learning_rate, learning_rate,
    steps_loss, loss,
    steps_policy_gradient_loss, policy_gradient_loss,
    steps_value_loss, value_loss
)

class GlobalState:
    action = None
    num_congested_op = None
    line_max_op = None
    lineload_max_op = None
    avg_ll_congested_op =None

def create_figure():
    fig = go.Figure()
    date_time = date_time_2016['Date Time'].astype(str)
    fig.add_trace(go.Scatter(
        #x=load_p.index,
        x=date_time,
        y=load_p.sum(axis=1),
        mode='lines',
        hoverinfo='text',
        hovertext=[f'Time: {t}, Power: {round(p, 2)}' for t, p in zip(date_time, load_p.sum(axis=1))],
        name='Demand'
    ))

    fig.add_trace(go.Scatter(
        #x=sgen_p.index,
        x=date_time,
        y=sgen_p.sum(axis=1),
        mode='lines',
        hoverinfo='text',
        hovertext=[f'Time: {t}, Power: {round(p, 2)}' for t, p in zip(date_time, sgen_p.sum(axis=1))],
        name='RES Production'
    ))

    fig.add_trace(go.Scatter(
        #x=gen_mw.index,
        x=date_time,
        y=gen_mw.sum(axis=1),
        mode='lines',
        hoverinfo='text',
        hovertext=[f'Time: {t}, Power: {round(p, 2)}' for t, p in zip(date_time, gen_mw.sum(axis=1))],
        name='Conventional Generation'
    ))

    fig.update_layout(
        title='Total Power',
        xaxis_title='Time',
        yaxis_title='Power (MW)'
    )
    if fig.layout.xaxis.range is None:
        fig.layout.xaxis.range =["2016-01-01 00:00","2016-12-31 23:45"]

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
    vmax = 0.75
    vmin = 0.25
    v = np.linspace(0, 1, 50)  # x values
    y = np.zeros(len(v))

    # Calculate the y values for the exponential decay curve
    for i in range(0,len(v)):
        y[i] = ((max((v[i]-vmax), 0))**2 + (max(vmin-v[i], 0))**2)
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
    y = x/Nloss

    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))

    # Customize the figure
    fig.update_layout(title='Losses Penalty',
                  xaxis_title='Total Losses',
                  yaxis_title='Reward')
    return fig


"""
def create_figure_metric_cumulative_reward():
    x = np.linspace(0, 50000, 500)
    y = np.piecewise(x, [x < 10000, x >= 10000], [lambda x: x / 100, lambda x: 80 + np.sin(x / 5000) * 10])

    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))

    # Customize the figure
    fig.update_layout(title='Cumulative Reward',
                  xaxis_title='Time Step',
                  yaxis_title='Cumulative Reward')

    return fig
"""
def create_figure_metric_episode_length_mean():
    x = steps_ep_len_mean
    y = ep_len_mean

    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))

    # Customize the figure
    fig.update_layout(#title='Mean Episode Length per Time Step',
                  xaxis_title='Time Step',
                  yaxis_title='Mean Episode Length')

    return fig

def create_figure_metric_ep_rew_mean():

    x = steps_rew_mean
    y = ep_rew_mean

    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))

    # Customize the figure
    fig.update_layout(#title='Mean Reward per Episode',
                  xaxis_title='Time Step',
                  yaxis_title='Mean Reward')

    return fig

def create_figure_metric_fps():

    x = steps_fps
    y = fps

    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))

    # Customize the figure
    fig.update_layout(#title='Frames Per Second (fps), per Episode',
                  xaxis_title='Time Step',
                  yaxis_title='Frames Per Second (fps)')

    return fig

def create_figure_metric_time_elapsed():

    x = steps_time_elapsed
    y = time_elapsed

    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))

    # Customize the figure
    fig.update_layout(#title=' Elapsed Time per Episode',
                  xaxis_title='Time Step',
                  yaxis_title=' Elapsed Time')

    return fig
"""
def create_figure_metric_total_timesteps():
    x = steps_total_timesteps
    y = total_timesteps

    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))

    # Customize the figure
    fig.update_layout(title='Total Timesteps',
                  xaxis_title='Time Step',
                  yaxis_title='Total Timesteps')

    return fig
"""
def create_figure_metric_approx_kl():
    x = steps_approx_kl
    y = approx_kl

    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))

    # Customize the figure
    fig.update_layout(#title='Approximated KL Divergence',
                  xaxis_title='Time Step',
                  yaxis_title='Approximated KL Divergence')

    return fig

def create_figure_metric_clip_fraction():
    x = steps_clip_fraction
    y = clip_fraction

    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))

    # Customize the figure
    fig.update_layout(#title='Clipping Fraction',
                  xaxis_title='Time Step',
                  yaxis_title='Clipping Fraction')

    return fig

def create_figure_metric_clip_range():
    x = steps_clip_range
    y = clip_range

    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))

    # Customize the figure
    fig.update_layout(#title='Clipping Range',
                  xaxis_title='Time Step',
                  yaxis_title='Clipping Range')

    return fig

def create_figure_metric_entropy_loss():
    x = steps_entropy_loss
    y = entropy_loss

    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))

    # Customize the figure
    fig.update_layout(#title='Entropy Loss per Episode',
                  xaxis_title='Time Step',
                  yaxis_title='Entropy Loss')

    return fig

def create_figure_metric_explained_variance():
    x = steps_explained_variance
    y = explained_variance

    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))

    # Customize the figure
    fig.update_layout(#title='Explained Variance',
                  xaxis_title='Time Step',
                  yaxis_title='Explained Variance')

    return fig

def create_figure_metric_learning_rate():
    x = steps_learning_rate
    y = learning_rate

    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))

    # Customize the figure
    fig.update_layout(#title='Learning Rate',
                  xaxis_title='Time Step',
                  yaxis_title='Learning Rate')

    return fig

def create_figure_metric_loss():
    x = steps_loss
    y = loss
    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))

    # Customize the figure
    fig.update_layout(#title='Loss',
                  xaxis_title='Time Step',
                  yaxis_title='Loss')

    return fig

"""
def create_figure_metric_value_estimate():
    x = np.linspace(0, 50000, 500)
    y = 10 * (1 - np.exp(-x / 15000))

    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))

    # Customize the figure
    fig.update_layout(title='Value Estimate',
                  xaxis_title='Time Step',
                  yaxis_title='Value Estimate')

    return fig
"""

def create_figure_metric_policy_gradient_loss():
    x = steps_policy_gradient_loss
    y = policy_gradient_loss
    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))

    # Customize the figure
    fig.update_layout(#title='Policy Gradient Loss',
                  xaxis_title='Time Step',
                  yaxis_title='Policy Gradient Loss')

    return fig

def create_figure_metric_value_loss():

    x = steps_value_loss
    y = value_loss

    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))

    # Customize the figure
    fig.update_layout(#title='Value Loss',
                  xaxis_title='Time Step',
                  yaxis_title='Value Loss')

    return fig

def update_profiles_time_step(chosen_date, hours, minutes):
    if not isinstance(chosen_date, date):
        raise ValueError("chosen_date must be a datetime.date object")

    picked_date = chosen_date
    hours = int(hours)
    minutes = int(minutes)
    #print('hours:', type(hours))
    #print('minutes:', type(minutes))
    time_steps = ((
            (days_passed_in_year(picked_date) - 1) * 24 * 4
            + hours * 4
            + (minutes / 15) + 1)-1
    )

    # Update both Env.profiles_time_step and profiles_time_step in Env.py
    Env.profiles_time_step = time_steps
    Env.update_profiles_time_step(time_steps)
    #print('Env.profiles_time_step:',Env.profiles_time_step)

    return time_steps

def days_passed_in_year(given_date):
    # Ensure the given date is a datetime object
    if not isinstance(given_date, date):
        raise ValueError("The given date must be a datetime object")

    # Calculate the start of the year for the given date
    start_of_year = date(2016, 1, 1)

    # Calculate the difference in days
    days_passed = (given_date - start_of_year).days + 1

    return days_passed

def update_kpi_step(action):

    #env.step(np.ones(num_switches, dtype=int))
    #lineload = Env.lineload
    if action is not None:
        if action is True:
            for i in range(Env.num_switches):
                        Env.network.switch.at[i, 'closed'] = True
        else:
            for i in range(Env.num_switches):
                        Env.network.switch.at[i, 'closed'] = bool(action[i])
    num_cong, l_max, ll_max, avg_ll_cong = step()

    return num_cong,  l_max, ll_max, avg_ll_cong

def step():

    pp.runpp(Env.network)
    lineload = 0.01 * Env.network.res_line.loading_percent
    lineload_max = max(lineload)
    line_max = lineload[lineload == lineload_max].index[0]
    num_congested = np.sum(lineload > 1)
    avg_ll_congested = np.mean(lineload[lineload > 1])

    ll_max = "{:.4g}".format(lineload_max*100)
    l_max = line_max
    num_cong = num_congested
    avg_ll_cong = "{:.4g}".format(avg_ll_congested*100)

    limit = 0.5


    return num_cong,  l_max, ll_max, avg_ll_cong

def reward_calc(line_loadings_percent, limit):
    # line_loadings_percent = net.res_line.loading_percent
    num_line = len(line_loadings_percent)
    ll = line_loadings_percent
    ll_max = max(ll)
    # print("max line loading = ", ll_max)
    if ll_max >= 1:
        # If the maximum line loading is greater than or equal to 1, calculate the margin
        # Calculate the margin by subtracting the limit from the line loadings
        margin = ll - limit * np.ones(num_line)
        # Set any margins smaller than 1-limit to 0
        margin[margin < 1 - limit] = 0
        # Calculate the congestion metric 'u' as the sum of the margins (ignore nan values)
        u = np.nansum(margin)

    else:
        # Calculate the congestion metric 'u' when the maximum line loading is less than 1
        # This is done by subtracting the limit from the maximum line loading,
        # and ensuring the result is not negative
        u = max(ll_max - limit, 0)

    reward = np.exp(-u)
    return reward


# Define the start and end date
start_date = "2016-01-01 00:00"
end_date = "2016-12-31 23:45"

# Create a date range with 15-minute intervals
date_range = pd.date_range(start=start_date, end=end_date, freq='15T')

# Create a DataFrame with the date range
date_time_2016 = pd.DataFrame({
    'Time Step': range(len(date_range)),
    'Date Time': date_range
})


#print(date_time_2016 )

# Constants
TIME_INTERVALS = [
    {'label': 'year', 'value': 96 * 365},
    {'label': 'month, 31 days', 'value': 96 * 31},
    {'label': 'week', 'value': 96 * 7},
    {'label': 'day', 'value': 96}
]

AGENT = [
    {'label': 'PPO (Proximal policy optimization)','value':1},
    {'label': 'TRPO (Trust Region Policy Optimization)', 'value':2, 'disabled':True},
    {'label': 'DQN (Deep Q-Network)', 'value':3, 'disabled':True}
   #{'label': '...', 'value':4, 'disabled':True}
]

num_cong = '-'
lin_max = '-'
ll_max ='-'

num_sw = num_switches
num_l = num_lines
profiles_time_st = profiles_time_step
grid_code = network_code

click = 0

hours = [f"{i:0{2}}" for i in range(24)]
minutes = [f"{i:0{2}}" for i in range(0, 60, 15)]
#seconds = [f"{i:0{2}}" for i in range(60)]

picker_style = {
    "display": "inline-block",
    "width": "40px",
    "cursor": "pointer",
    "border": "none",
}

separator = html.Span(":")

# to be implemented
### don't have the exact log file, doesn't mean you can't plot something related to it

###fig_metric_cumulative_reward = create_figure_metric_cumulative_reward()
##rollout/ep_len_mean
fig_metric_episode_length_mean = create_figure_metric_episode_length_mean()
##rollout/ep_rew_mean
fig_metric_ep_rew_mean = create_figure_metric_ep_rew_mean()
#time/fps
fig_metric_fps = create_figure_metric_fps()
#time_elapsed
fig_metric_time_elapsed = create_figure_metric_time_elapsed()
#total_timesteps
#fig_metric_total_timesteps = create_figure_metric_total_timesteps()
#train/approx_kl
fig_metric_approx_kl = create_figure_metric_approx_kl()
#train/clip_fraction
fig_metric_clip_fraction = create_figure_metric_clip_fraction()
#train/clip_range
fig_metric_clip_range = create_figure_metric_clip_range()
#train/entropy_loss
fig_metric_entropy_loss = create_figure_metric_entropy_loss()
#train/explained_variance
fig_metric_explained_variance = create_figure_metric_explained_variance()
##train/learning_rate
fig_metric_learnign_rate = create_figure_metric_learning_rate()
#train/loss
fig_metric_loss = create_figure_metric_loss()
###fig_metric_value_estimate = create_figure_metric_value_estimate()
##train/policy_gradient_loss
fig_metric_policy_gradient_loss = create_figure_metric_policy_gradient_loss()
##train/value_loss
fig_metric_value_loss = create_figure_metric_value_loss()


#action_space = spaces.MultiDiscrete([2]*(num_switches))
action_test = np.array([1]*(num_switches//3) + [0]*(num_switches - num_switches//3))

table_data = [
    {"Substation": "","Subnet": "", "Bus-1": "", "Switch-ID": "","Type of Switch": "", "Switching Action": ""}]

# Initialize app
#app = dash.Dash(__name__)
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define layout
app.layout = html.Div([
    html.H1('Topology Optimizer Dashboard'),

    dcc.Tabs(id="tabs-example-graph", value='tab-1', children=[

        # Overview Tab
        dcc.Tab(label='Overview', value='tab-1', children=[

            # Transmission Grid Section
            html.Div(
                style={
                    'padding': '20px',
                    'border': '1px solid #ccc',
                    'border-radius': '10px',
                    'background-color': '#ffffff',
                    'box-shadow': '0px 4px 6px rgba(0, 0, 0, 0.1)',
                    #'max-width': '600px',
                    'width': '100%',
                    'margin': '20px 0',  # Align to the left,
                    'font-family': 'Arial, sans-serif',
                    'align-items': 'left'
                },
                children=[
                    html.H3('Transmission Grid Information', style={'textAlign': 'left', 'fontSize': '18px', 'fontWeight': 'bold'}),
                    html.Div(
                        style={
                            'display': 'flex',
                            'justify-content': 'space-between',
                            'align-items': 'left'
                        },
                        children=[
                            html.Div(
                                style={
                                    'flex': '1',
                                    'text-align': 'left',
                                    'background-color': '#f9f9f9',
                                    'padding': '15px',
                                    'border-radius': '8px',
                                    'margin': '0 10px',
                                    'box-shadow': '0px 2px 4px rgba(0, 0, 0, 0.1)'
                                },
                                children=[
                                    html.Span('Grid Code', style={'color': '#888', 'font-size': '14px'}),
                                    html.Br(),
                                    html.Span(id='grid-code', children=grid_code, style={
                                        'font-size': '20px',
                                        'font-weight': 'bold',
                                        'color': '#333'
                                    })
                                ]
                            ),
                            html.Div(
                                style={
                                    'flex': '1',
                                    'text-align': 'left',
                                    'background-color': '#f9f9f9',
                                    'padding': '15px',
                                    'border-radius': '8px',
                                    'margin': '0 10px',
                                    'box-shadow': '0px 2px 4px rgba(0, 0, 0, 0.1)'
                                },
                                children=[
                                    html.Span('Number of Switches', style={'color': '#888', 'font-size': '14px'}),
                                    html.Br(),
                                    html.Span(id='num-switch', children=num_sw, style={
                                        'font-size': '20px',
                                        'font-weight': 'bold',
                                        'color': '#333'
                                    })
                                ]
                            ),
                            html.Div(
                                style={
                                    'flex': '1',
                                    'text-align': 'left',
                                    'background-color': '#f9f9f9',
                                    'padding': '15px',
                                    'border-radius': '8px',
                                    'margin': '0 10px',
                                    'box-shadow': '0px 2px 4px rgba(0, 0, 0, 0.1)'
                                },
                                children=[
                                    html.Span('Number of Lines', style={'color': '#888', 'font-size': '14px'}),
                                    html.Br(),
                                    html.Span(id='num-line', children=num_l, style={
                                        'font-size': '20px',
                                        'font-weight': 'bold',
                                        'color': '#333'
                                    })
                                ]
                            )
                        ]
                    )
                ]
            ),

    html.Div(
                style={
                    'padding': '20px',
                    'border': '1px solid #ccc',
                    'border-radius': '10px',
                    'background-color': '#ffffff',
                    'box-shadow': '0px 4px 6px rgba(0, 0, 0, 0.1)',
                    #'max-width': '600px',
                    'width': '100%',
                    'margin': '20px 0',  # Align to the left,
                    'font-family': 'Arial, sans-serif',
                    'align-items': 'left'
                },
                children=[
            # Profiles Section
                    html.Div(
                        style={
                            'flex': '1',
                            'text-align': 'left',
                            'background-color': '#f9f9f9',
                            'padding': '15px',
                            'border-radius': '8px',
                            'margin': '0 10px',
                            'box-shadow': '0px 2px 4px rgba(0, 0, 0, 0.1)'
                        },
                        children=[
                html.H2('Scenario',style={'textAlign': 'left', 'fontSize': '18px', 'fontWeight': 'bold'}),
                dcc.Graph(id='graph-profiles', figure=create_figure()),

                # Time Interval Dropdown
                html.H2('Time Interval',style={'textAlign': 'left', 'fontSize': '18px', 'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='time-interval-dropdown',
                    options=TIME_INTERVALS,
                    value=TIME_INTERVALS[0]['value'],
                    style={'margin-bottom': '10px'}
                ),

                # Date and Time Selection
                        html.H2('Chosen Profiles Time Step for Load Flow Calculation',style={'textAlign': 'left', 'fontSize': '18px', 'fontWeight': 'bold'}),

                        # Date Picker Section
                        html.Div([
                            html.Label("Date (YYYY/MM/DD)", style={'display': 'block'}),
                            dcc.DatePickerSingle(
                                id='my-date-picker-single',
                                min_date_allowed=date(2016, 1, 1),
                                max_date_allowed=date(2016, 12, 31),
                                placeholder="YYYY-MM-DD",
                                initial_visible_month=date(2016, 1, 1),
                                date=date(2016, 1, 1),
                                style={       # Set the height of the date picker
                                        #'marginBottom': '10px'
                                        }  # Add margin below the date picker
                            )
                        ], style={'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '10px'}),

                        # Time Picker Section
                        html.Div([
                            html.Label("Select Time (HH:MM)", style={'display': 'block'}),
                            html.Div([
                                dcc.Dropdown(id='hours', options=hours, placeholder="HH", style=picker_style),
                                html.Span(":", style={'padding': '0 5px'}),
                                dcc.Dropdown(id='minutes', options=minutes, placeholder="MM", style=picker_style),
                            ], style={'width': '150px',      # Set the width of the date picker
                                        'height': '40px',"border": "1px solid lightgray",'display': 'flex', 'alignItems': 'center'})
                        ], style={'display': 'inline-block', 'verticalAlign': 'top'}),

                        # Apply Button and Profile Time Step
                        html.Div([
                            dbc.Button('Apply Date and Time', id='button_apply_date_time', color="primary",n_clicks=click,
                                        style={'marginTop': '10px'}
                                        ),
                            html.Br(),
                            html.Span('Scenario Time Step: '),
                            html.Span(id='profiles-time-step-in-profiles', children=profiles_time_st),
                            #html.Div(id='d-t-alert'),
                            dbc.Alert(
                                        "Changes detected. Please press 'Apply Date and Time' to update.",
                                        id="d-t-alert",
                                        dismissable=True,
                                        is_open=True,
                                        color = 'warning'
                                    )

                        ], style={'display': 'block', 'marginTop': '10px'})  # Make button appear below the selectors
                    ]
                )
    ]),

            html.Div(
                style={
                    'padding': '20px',
                    'border': '1px solid #ccc',
                    'border-radius': '10px',
                    'background-color': '#ffffff',
                    'box-shadow': '0px 4px 6px rgba(0, 0, 0, 0.1)',
                    # 'max-width': '600px',
                    'width': '100%',
                    'margin': '20px 0',  # Align to the left,
                    'font-family': 'Arial, sans-serif',
                    'align-items': 'left'
                },
                children=[
                    # KPI Table
                    html.Div(
                        style={
                            'flex': '1',
                            'text-align': 'left',
                            'background-color': '#f9f9f9',
                            'padding': '15px',
                            'border-radius': '8px',
                            'margin': '0 10px',
                            'margin-bottom': '10px',
                            'box-shadow': '0px 2px 4px rgba(0, 0, 0, 0.1)'
                        },
                        children=[html.H2('Congestion Key Performance Indicators',
                                           style={'textAlign': 'left', 'fontSize': '18px', 'fontWeight': 'bold'}),
                                   dash_table.DataTable(
                                       id='computed-table',
                                       columns=[
                                           {'name': 'Congestion Key Performance Indicators', 'id': 'KPIs'},
                                           {'name': 'Before Optimization', 'id': 'bf-op'},
                                           {'name': 'After Optimization', 'id': 'af-op'}
                                       ],
                                       data=[
                                           {'KPIs': 'Number of Congested Lines', 'bf-op': '', 'af-op': ''},
                                           {'KPIs': 'Line with the Highest Line Loading', 'bf-op': '', 'af-op': ''},
                                           {'KPIs': 'Highest Line Loading(%)', 'bf-op': '', 'af-op': ''},
                                           {'KPIs': 'Average Line Loading of Congested Lines(%)', 'bf-op': '',
                                            'af-op': ''}
                                       ],
                                       editable=False,
                                       style_table={'width': '100%', 'overflowX': 'auto','marginBottom': '10px'},
                                       style_cell={
                                           'textAlign': 'center',
                                           'padding': '10px',
                                           'minWidth': '150px', 'width': '150px', 'maxWidth': '150px',
                                           'whiteSpace': 'normal'
                                       },
                                       style_header={
                                           'backgroundColor': 'lightgrey',
                                           'fontWeight': 'bold'
                                       },
                                       style_data={
                                           'backgroundColor': 'white',
                                           'color': 'black'
                                       }
                                   ),
                                   dbc.Button('Update KPIs Before Optimization', id='button_update_kpi_bf',color="primary", n_clicks=click), dbc.Button('Update KPIs After Optimization', id='button_update_kpi_af',color="primary", n_clicks=click),

                                  dbc.Alert(
                                      "Changes detected. Please press 'Update KPIs Before Optimization'.",
                                      id="bf-kpi-alert",
                                      dismissable=True,
                                      is_open=False,
                                      color='warning'
                                  ),
                                  dbc.Alert(
                                      "Changes detected. Please press 'Update KPIs After Optimization'.",
                                      id="af-kpi-alert",
                                      dismissable=True,
                                      is_open=False,
                                      color='warning'
                                  )
                                   ]),


                # Switching Section with Table
                    html.Div(
                        style={
                            'flex': '1',
                            'text-align': 'left',
                            'background-color': '#f9f9f9',
                            'padding': '15px',
                            'border-radius': '8px',
                            'margin': '0 10px',
                            'box-shadow': '0px 2px 4px rgba(0, 0, 0, 0.1)'
                        },
                        children=[
                             html.H2('Switching Action Recommendation',
                                     style={'textAlign': 'left', 'fontSize': '18px', 'fontWeight': 'bold'}),

                             # Switching Table Layout
                            dbc.Button("Show/Hide Switching Action Table", id="toggle-table", color="primary", n_clicks=0),dbc.Button('Run Optimization', id='button_run_op',color="success", n_clicks=click),
                             dbc.Spinner(html.Div(id="loading-output")),
                             html.Div(id="output"),  # Output to display confirmation
                             dbc.Collapse(

                             dash_table.DataTable(
                                 id='switching-table',
                                 columns=[

                                     {"name": ["Connection Points", "Substation"], "id": "Substation"},
                                     {"name": ["Connection Points", "Sub-network"], "id": "Subnet"},
                                     {"name": ["Connection Points", "Bus"], "id": "Bus-1"},

                                     #{"name": ["Connection Points", "Line"], "id": "Line"},
                                     {"name": ["Switch", "ID of the Switch with Recommended Action"], "id": "Switch-ID"},

                                     {"name": ["Switch", "Type of Switch"], "id": "Type of Switch"},
                                    {"name": ["Switch", "Suggested Switching Action"], "id": "Switching Action"}

                                 ],
                                 data=table_data,  # use empty data for now, can be dynamically populated
                                 merge_duplicate_headers=True,  # Merges headers across columns
                                 style_table={'width': '100%', 'marginTop': '20px', 'marginBottom': '10px', 'border': '1px solid #ccc'},
                                 style_header={
                                     'backgroundColor': 'lightgrey',
                                     'fontWeight': 'bold',
                                     'textAlign': 'center'
                                 },
                                 style_cell={
                                     'textAlign': 'center',
                                     'padding': '8px',
                                     'border': '1px solid #ccc',
                                     'whiteSpace': 'normal',
                                     'height': 'auto'
                                 },
                                 style_data={
                                     'backgroundColor': 'white',
                                     'color': 'black'
                                 },
                                 column_selectable="multi",
                                 style_cell_conditional=[

                                     {'if': {'column_id': 'Substation'}, 'width': '20%'},

                                     {'if': {'column_id': 'Subnet'}, 'width': '20%'},
                                     {'if': {'column_id': 'Busbar-1'}, 'width': '20%'},
                                     #{'if': {'column_id': 'Line'}, 'width': '20%'},
                                     #{'if': {'column_id': 'Measure Take'}, 'width': '20%'},
                                     {'if': {'column_id': 'Switch-ID'}, 'width': '20%'},
                                     {'if': {'column_id': 'Type-Switch'}, 'width': '15%'},
                                     {'if': {'column_id': 'Recommended Action'}, 'width': '15%'}
                                 ]
                             )
                        ,
                        id="collapse-table",
                        is_open=False
                    )

                         ])
        ])

        ])  # Close dcc.Tab for Overview
        ,  # End of Overview Tab

        # Environment Tab
        dcc.Tab(
            label='Reward Function',
            value='tab-2',
            children=[
html.Div(
                style={
                    'padding': '20px',
                    'border': '1px solid #ccc',
                    'border-radius': '10px',
                    'background-color': '#ffffff',
                    'box-shadow': '0px 4px 6px rgba(0, 0, 0, 0.1)',
                    #'max-width': '600px',
                    'width': '100%',
                    'margin': '20px 0',  # Align to the left,
                    'font-family': 'Arial, sans-serif',
                    'align-items': 'left'
                },
                children=[

html.Div(
                        style={
                            'flex': '1',
                            'text-align': 'left',
                            'background-color': '#f9f9f9',
                            'padding': '15px',
                            'border-radius': '8px',
                            'margin': '0 10px',
                            'box-shadow': '0px 2px 4px rgba(0, 0, 0, 0.1)',
                            'margin-bottom': '10px',
                            'margin-top': '10px'
                        },
                        children=[
                        html.H3('Reward Function (Congestion)'),
                        dbc.Button("?", id="open", n_clicks=0),
                            dbc.Modal(
                                [
                                    dbc.ModalHeader(dbc.ModalTitle("Reward Function (Congestion)")),
                                    dbc.ModalBody(
                                        html.Div(
                                            [
                    html.Div(
                        [
                            html.Img(
                                src="assets/rw_congest_2.png",
                                alt="Equation 2: Definition of R_congestion,t",
                                style={"width": "30%", "margin-bottom": "10px"}
                            ),
                            html.P(
                                "The reward function R_congestion,t is an exponential function of the coefficient u."
                            )
                        ]
                    ),
                    html.Div(
                        [
                            html.Img(
                                src="assets/rw_congest_3.png",
                                alt="Equation 3: Definition of u",
                                style={"width": "80%", "margin-bottom": "10px"}
                            ),
                            html.P(
                                "The coefficient u is defined based on the maximum normalized load "
                                "and the cumulative excess of load flow beyond 50%."
                            )
                        ]
                    ),html.Div(
                                            [html.Img(
                                src="assets/rw_congest_1.png",
                                alt="Equation 1: Definition of ρ_l",
                                style={"width": "10%", "margin-bottom": "10px"}
                            ),
                            html.P(
                                "The normalized load flow in line l, "
                                "where ρ_l is the ratio of the current flow (i_l) to the maximum capacity (i_100%)."
                            )
                        ]
                    ),
                                                html.P(
                                                    "To further explain with example edge cases, consider the following scenarios:"
                                                ),
                                                html.P(
                                                    [
                                                        html.B("1. No Congestion and No Excessive Line Loading: "),
                                                        "When there is no congestion (ρ_max < 1) and no line loading exceeds 50%, the margin (u) will have a value of 0. This results in the reward reaching its highest value of 1."
                                                    ]
                                                ),
                                                html.P(
                                                    [
                                                        html.B("2. No Congestion but Excessive Line Loading: "),
                                                        "If there is no congestion (ρ_max < 1), but some lines have loadings exceeding 50%, the margin (u) will correspond to how much the highest line loading exceeds 50%. In this case, u will lie between 0 and 0.5, causing the reward R_congestion,t to fall between 1 and e^(-0.5)."
                                                    ]
                                                ),
                                                html.P(
                                                    [
                                                        html.B("3. Congestion Occurs (ρ_max ≥ 1): "),
                                                        "When congestion occurs (ρ_max ≥ 1), the margin (u) is calculated as the sum of the excess loadings above 50% for all congested lines. For example, if only one line has a loading of 100% while all others remain uncongested, u = 0.5, and the reward R_congestion,t equals e^(-0.5). This shows that when ρ_max ≥ 1, the reward value will always satisfy R_congestion,t ≤ e^(-0.5)."
                                                    ]
                                                ),
                                                html.P(
                                                    [
                                                        "In a more severe scenario where five lines each have a loading of 100% and the rest remain uncongested, the margin (u) will be 2.5, resulting in R_congestion,t = e^(-2.5). As illustrated in Fig. 1, the reward value in this case decreases significantly compared to when only one line has a 100% loading. This demonstrates the desirable penalization mechanism, where actions leading to both more lines experiencing congestion and more severe congestion are penalized more heavily."
                                                    ]
                                                ),
                                            ]
                                        )
                                    ),
                                    dbc.ModalFooter(
                                        dbc.Button(
                                            "Close", id="close", className="ms-auto", n_clicks=0
                                        )
                                    ),
                                ],
                                id="modal",
                                is_open=False,
                            )
                            ,
                        dcc.Graph(id='graph-reward-congestion', figure=create_figure_reward_congestion()),])])
                    ,
                    html.Span("Further possible rewards (not implemented here)  ", style = {"textDecoration": "bold", "fontSize": "20px"}),
                    dbc.Button("Show/Hide", id="toggle-gap", color="primary", n_clicks=0),

                    dbc.Collapse(

                        html.Div(
                            style={
                                'padding': '20px',
                                'border': '1px solid #ccc',
                                'border-radius': '10px',
                                'background-color': '#ffffff',
                                'box-shadow': '0px 4px 6px rgba(0, 0, 0, 0.1)',
                                # 'max-width': '600px',
                                'width': '100%',
                                'margin': '20px 0',  # Align to the left,
                                'font-family': 'Arial, sans-serif',
                                'align-items': 'left'
                            },
                            children=[

                                html.Div(
                                    style={
                                        'flex': '1',
                                        'text-align': 'left',
                                        'background-color': '#f9f9f9',
                                        'padding': '15px',
                                        'border-radius': '8px',
                                        'margin': '0 10px',
                                        'margin-bottom': '10px',
                                        'margin-top': '10px',
                                        'box-shadow': '0px 2px 4px rgba(0, 0, 0, 0.1)'
                                    },
                                    children=[
                                html.H3('Reward Function (Voltage)'),
                                        dbc.Button("?", id="open_rw_vol", n_clicks=0),
                                        dbc.Modal(
                                            [
                                                dbc.ModalHeader(dbc.ModalTitle("Reward Function (Voltage)")),
                                                dbc.ModalBody(
                                                    html.Div(
                                                        [html.P([
                                                                    html.B("U_k,t is within the acceptable voltage range, [U_min, U_max]: "), "The penalty is zero, and the function achieves its minimum value, effectively imposing no cost."
                                                                ]),
                                                                html.P([
                                                                    html.B("U_k,t > U_max:"), "The penalty increases quadratically. When U_k,t > U_max, the penalty grows progressively as the voltage exceeds the upper limit."
                                                                ]),
                                                                html.P([
                                                                    html.B("U_k,t < U_min:"), "Similarly, when U_k,t < U_min, the penalty increases as the voltage drops further below the lower threshold."
                                                                ]),
                                                                html.P(
                                                                    "The quadratic growth ensures smooth transitions and progressively higher penalties for larger deviations."
                                                                ),
                                                        ]
                                                    )
                                                ),
                                                dbc.ModalFooter(
                                                    dbc.Button(
                                                        "Close", id="close_rw_vol", className="ms-auto", n_clicks=0
                                                    )
                                                ),
                                            ],
                                            id="modal_rw_vol",
                                            is_open=False,
                                        ),

                                        dcc.Graph(id='graph-reward-voltage', figure=create_figure_reward_voltage())
                                    ]),
                                html.Div(
                                    style={
                                        'flex': '1',
                                        'text-align': 'left',
                                        'background-color': '#f9f9f9',
                                        'padding': '15px',
                                        'border-radius': '8px',
                                        'margin': '0 10px',
                                        'margin-bottom': '10px',
                                        'box-shadow': '0px 2px 4px rgba(0, 0, 0, 0.1)'
                                    },
                                    children=[
                                html.H3('Reward Function (Losses)'),
                                        dbc.Button("?", id="open_rw_loss", n_clicks=0),
                                        dbc.Modal(
                                            [
                                                dbc.ModalHeader(dbc.ModalTitle("Reward Function (Losses)")),
                                                dbc.ModalBody(
                                                    html.Div(
                                                        [
                                                            html.P(
                                                                "This graph illustrates the relationship between total losses and the corresponding cost.",
                                                                "The graph demonstrates a linear relationship, where the reward increases proportionally with higher losses. "
                                                                "This suggests that the system is designed to provide higher rewards as total losses increase, indicating an incentive structure that encourages minimizing losses."
                                                            ),
                                                        ]
                                                    )
                                                ),
                                                dbc.ModalFooter(
                                                    dbc.Button(
                                                        "Close", id="close_rw_loss", className="ms-auto", n_clicks=0
                                                    )
                                                ),
                                            ],
                                            id="modal_rw_loss",
                                            is_open=False,
                                        ),

                                        dcc.Graph(id='graph-reward-losses', figure=create_figure_reward_losses())
                                    ]),
                            ])
                        ,
                        id="collapse-gap",
                        is_open=False
                    )
                #])
            ]
        )
        ,  # End of Reward Tab

        # Agent Tab
        dcc.Tab(label='Trained Agent', value='tab-3', children=[

html.Div(
    style={
        'padding': '20px',
        'border': '1px solid #ccc',
        'border-radius': '10px',
        'background-color': '#ffffff',
        'box-shadow': '0px 4px 6px rgba(0, 0, 0, 0.1)',
        # 'max-width': '600px',
        'width': '100%',
        'margin': '20px 0',  # Align to the left,
        'font-family': 'Arial, sans-serif',
        'align-items': 'left'
    },
children=[
        html.H2(
            "Agent Algorithm:",
            id="tooltip-target",
            style={'textAlign': 'left', 'fontSize': '18px', 'fontWeight': 'bold'},
        ),
        dbc.Tooltip(
            [html.Div([html.Span(
            "Agent Classification:",
            style={"textDecoration": "bold","fontSize": "20px"}),
            html.Img(src=app.get_asset_url("RLMineMap.jpg"))])],
            target="tooltip-target", placement='auto'
        ),
        dcc.Dropdown(
                    id='agent-dropdown',
                    options=AGENT,
                    value=AGENT[0]['value']
                )
,

                    html.H2('Hyperparameters',
                            style={'textAlign': 'left', 'fontSize': '18px', 'fontWeight': 'bold','margin-top': '10px'}),
        html.Div(
            style={
                'display': 'flex',
                'justify-content': 'space-between',
                'align-items': 'left'
            },
            children=[
                html.Div(
                    style={
                        'flex': '1',
                        'text-align': 'left',
                        'background-color': '#f9f9f9',
                        'padding': '15px',
                        'border-radius': '8px',
                        'margin': '0 10px',
                        'box-shadow': '0px 2px 4px rgba(0, 0, 0, 0.1)'
                    },
                    children=[

                        html.Span('Learning Rate', id= "learnign-rate",style={'color': '#888', 'font-size': '14px'}),
                        html.Br(),
                        html.Span(children=3e-04, style={
                            'font-size': '20px',
                            'font-weight': 'bold',
                            'color': '#333'
                        })
                        ,
                    dbc.Tooltip(
                        "The rate at which the model updates its parameters. This metric helps monitor if the learning rate is too high (which might cause instability) or too low (leading to slower progress). Some algorithms dynamically adjust the learning rate during training.",
                        target="learnign-rate",
                        placement="auto"
                    ),
                #dcc.Graph(id='graph-metric-learning-rate', figure=fig_metric_learnign_rate)
                    ]),
                html.Div(
                    style={
                        'flex': '1',
                        'text-align': 'left',
                        'background-color': '#f9f9f9',
                        'padding': '15px',
                        'border-radius': '8px',
                        'margin': '0 10px',
                        'box-shadow': '0px 2px 4px rgba(0, 0, 0, 0.1)'
                    },
                    children=[

                        html.Span('Clipping Fraction',
                                  id="clip-fraction", style={'color': '#888', 'font-size': '14px'}),
                        html.Br(),
                        html.Span(children=0, style={
                            'font-size': '20px',
                            'font-weight': 'bold',
                            'color': '#333'
                        }),
                    dbc.Tooltip(
                        "Shows the fraction of actions where gradients have been clipped to prevent excessive updates. In algorithms like PPO, clipping prevents large policy updates that could destabilize training.",
                        target="clip-fraction",
                        placement="auto"
                    )
                    # dcc.Graph(id='graph-metric-clip-fraction', figure=fig_metric_clip_fraction)
                ]),
                html.Div(
                    style={
                        'flex': '1',
                        'text-align': 'left',
                        'background-color': '#f9f9f9',
                        'padding': '15px',
                        'border-radius': '8px',
                        'margin': '0 10px',
                        'box-shadow': '0px 2px 4px rgba(0, 0, 0, 0.1)'
                    },
                    children=[

                                    html.Span('Clipping Range',
                        id="clip-range", style={'color': '#888', 'font-size': '14px'}),
                                    html.Br(),
                                    html.Span(children=0.2, style={
                                        'font-size': '20px',
                                        'font-weight': 'bold',
                                        'color': '#333'
                                    }),
                    dbc.Tooltip(
                        "Refers to the range within which gradients or policy updates are clipped. This metric helps you monitor the extent of clipping applied. It directly impacts the stability of training, as excessive clipping can lead to overly conservative updates, while too little clipping can result in instability.",
                        target="clip-range",
                        placement="auto"
                    ),
                    # dcc.Graph(id='graph-metric-clip-range', figure=fig_metric_clip_range)
                ]),html.Br()
            ])
        ,

            html.Div([
                html.Div(
                    style={
                        'flex': '1',
                        'text-align': 'left',
                        'background-color': '#f9f9f9',
                        'padding': '15px',
                        'border-radius': '8px',
                        'margin': '0 0px',
                        'margin-bottom': '10 px',
                        'box-shadow': '0px 2px 4px rgba(0, 0, 0, 0.1)'
                    },
                    children=[

                        html.Span(
                        "Mean Episode Length:",
                        id="Mean-Episode-Length",
                        style={"cursor": "pointer", "fontSize": "20px"}
                    ),
                    dbc.Tooltip(
                        "Measures the average number of steps taken in each episode over time.\n\n An increasing trend could indicate that the agent is learning to survive longer.",
                        target="Mean-Episode-Length",
                        placement="auto"
                    ),
                    dcc.Graph(id='graph-metric-epi-length', figure=fig_metric_episode_length_mean)
                ]),
                html.Div(
                    style={
                        'flex': '1',
                        'text-align': 'left',
                        'background-color': '#f9f9f9',
                        'padding': '15px',
                        'border-radius': '8px',
                        'margin': '0 0px',
                        'margin-bottom': '10 px',
                        'box-shadow': '0px 2px 4px rgba(0, 0, 0, 0.1)'
                    },
                    children=[

                        html.Span(
                        "Mean Episode Reward:",
                        id="Mean-Episode-Reward",
                        style={"cursor": "pointer", "fontSize": "20px"}
                    ),
                    dbc.Tooltip(
                        "Shows the average reward achieved by the agent per episode.  it reflects how well the agent is performing. A steady increase suggests that the agent is improving its behavior over time, as seen from it achieving higher rewards.",
                        target="Mean-Episode-Reward",
                        placement="auto"
                    ),
                dcc.Graph(id='graph-metric-epi-rew-mean', figure=fig_metric_ep_rew_mean)
                        ])
                ,
                html.Div(
                    style={
                        'flex': '1',
                        'text-align': 'left',
                        'background-color': '#f9f9f9',
                        'padding': '15px',
                        'border-radius': '8px',
                        'margin': '0 0px',
                        'margin-bottom': '10 px',
                        'box-shadow': '0px 2px 4px rgba(0, 0, 0, 0.1)'
                    },
                    children=[

                        html.Span(
                        "Time Elapsed:",
                        id="Time-Elapsed",
                        style={"cursor": "pointer", "fontSize": "20px"}
                    ),
                    dbc.Tooltip(
                        "The amount of time in seconds indicates overall training duration.",
                        target="Time-Elapsed",
                        placement="auto"
                    ),
                dcc.Graph(id='graph-metric-time-elapsed', figure=fig_metric_time_elapsed)
                    ]),
                html.Div(
                    style={
                        'flex': '1',
                        'text-align': 'left',
                        'background-color': '#f9f9f9',
                        'padding': '15px',
                        'border-radius': '8px',
                        'margin': '0 0px',
                        'margin-bottom': '10 px',
                        'box-shadow': '0px 2px 4px rgba(0, 0, 0, 0.1)'
                    },
                    children=[

                        html.Span(
                        "Approximated KL Divergence:",
                        id="approx-kl",
                        style={"cursor": "pointer", "fontSize": "20px"}
                    ),
                    dbc.Tooltip(
                        "Measures the difference between the agent's current policy and its previous policy.  KL Divergence is often used to prevent drastic policy changes. High values can indicate instability in learning, while low values generally indicate more stable and incremental learning.",
                        target="approx-kl",
                        placement="auto"
                    ),
                dcc.Graph(id='graph-metric-approx-kl', figure=fig_metric_approx_kl)
                    ]),
                html.Div(
                    style={
                        'flex': '1',
                        'text-align': 'left',
                        'background-color': '#f9f9f9',
                        'padding': '15px',
                        'border-radius': '8px',
                        'margin': '0 0px',
                        'margin-bottom': '10 px',
                        'box-shadow': '0px 2px 4px rgba(0, 0, 0, 0.1)'
                    },
                    children=[

                        html.Span(
                        "Entropy Loss:",
                        id="entropy-loss",
                        style={"cursor": "pointer", "fontSize": "20px"}
                    ),
                    dbc.Tooltip(
                        "Measures the randomness in the agent’s actions. Higher entropy means the agent is exploring more, while lower entropy suggests it is converging on a specific set of actions. In early training, higher entropy is typically desired to encourage exploration. Over time, as the agent learns, entropy decreases as it becomes more confident in its actions.",
                        target="entropy-loss",
                        placement="auto"
                    ),
                dcc.Graph(id='graph-metric-entropy-loss', figure=fig_metric_entropy_loss)
                    ]),
                html.Div(
                    style={
                        'flex': '1',
                        'text-align': 'left',
                        'background-color': '#f9f9f9',
                        'padding': '15px',
                        'border-radius': '8px',
                        'margin': '0 0px',
                        'margin-bottom': '10 px',
                        'box-shadow': '0px 2px 4px rgba(0, 0, 0, 0.1)'
                    },
                    children=[

                        html.Span(
                        "Explained Variance:",
                        id="explained-variance",
                        style={"cursor": "pointer", "fontSize": "20px"}
                    ),
                    dbc.Tooltip(
                        "A measure that indicates how well the value function explains the variance in the actual returns. Higher values (closer to 1) mean the value function is accurately predicting returns, indicating better learning of the value function. Lower values suggest that the value function may not be learning well and may need adjustment.",
                        target="explained-variance",
                        placement="auto"
                    ),
                dcc.Graph(id='graph-metric-explained-variance', figure=fig_metric_explained_variance)
                    ]),
#                html.Div(
#                    style={
#                        'flex': '1',
#                        'text-align': 'left',
#                        'background-color': '#f9f9f9',
#                        'padding': '15px',
#                        'border-radius': '8px',
#                        'margin': '0 0px',
#                        'margin-bottom': '10 px',
#                        'box-shadow': '0px 2px 4px rgba(0, 0, 0, 0.1)'
#                    },
#                    children=[
#
#                        html.Span(
#                        "Loss:",
#                        id="loss",
#                        style={"cursor": "pointer", "fontSize": "20px"}
#                    ),
#                    dbc.Tooltip(
#                        "Reflects the error between the agent's predicted action or value and the actual outcome. Lower loss generally indicates that the model is improving its predictions or actions over time. However, occasional spikes can happen due to policy updates or exploration shifts.",
#                        target="loss",
#                        placement="auto"
#                    ),
#                dcc.Graph(id='graph-metric-loss', figure=fig_metric_loss)
#                    ]),
                html.Div(
                    style={
                        'flex': '1',
                        'text-align': 'left',
                        'background-color': '#f9f9f9',
                        'padding': '15px',
                        'border-radius': '8px',
                        'margin': '0 0px',
                        'margin-bottom': '10 px',
                        'box-shadow': '0px 2px 4px rgba(0, 0, 0, 0.1)'
                    },
                    children=[

                        html.Span(
                        "Policy Gradient Loss",
                        id="policy-gradient-loss",
                        style={"cursor": "pointer", "fontSize": "20px"}
                    ),
                    dbc.Tooltip(
                        "The loss value related specifically to the policy update, typically seen in policy gradient methods. This shows the effectiveness of the policy updates. Large fluctuations in policy gradient loss might indicate instability, and stable values indicate that the policy updates are well-behaved.",
                        target="policy-gradient-loss",
                        placement="auto"
                    ),
                dcc.Graph(id='graph-metric-policy-gradient-loss', figure=fig_metric_policy_gradient_loss)
                    ]),
                html.Div(
                    style={
                        'flex': '1',
                        'text-align': 'left',
                        'background-color': '#f9f9f9',
                        'padding': '15px',
                        'border-radius': '8px',
                        'margin': '0 0 px',
                        'margin-bottom': '10 px',
                        'box-shadow': '0px 2px 4px rgba(0, 0, 0, 0.1)'
                    },
                    children=[

                        html.Span(
                        "Value Loss",
                        id="value-loss",
                        style={"cursor": "pointer", "fontSize": "20px"}
                    ),
                    dbc.Tooltip(
                        "The error associated with the value function (critic) predictions in actor-critic models. This loss should ideally decrease over time as the agent better predicts the expected return. High value loss can suggest issues in the critic network, potentially requiring adjustments to the learning rate or architecture.",
                        target="value-loss",
                        placement="auto"
                    ),
                dcc.Graph(id='graph-metric-value-loss', figure=fig_metric_value_loss)
                    ]),
            ],style={'padding': '20px'}),])
        ])  # End of Agent Tab
    ]),

    # Tabs content placeholder
    html.Div(id='tabs-content-example-graph')
])
from dash import Input, Output, State, callback_context

@app.callback(
    #Output('profiles-time-step', 'children'),
    Output('profiles-time-step-in-profiles', 'children'),
    [Input('button_apply_date_time', 'n_clicks')],
    State('my-date-picker-single', 'date'),
    State('hours', 'value'),
    State('minutes', 'value')


)
def apply_date_time(n_clicks, day, h, m):
    print('Callback triggered')
    print('day:', day)
    if n_clicks != 0 and day is not None:
        # Convert the day to a date object if necessary
        chosen_date = date.fromisoformat(day)
        prof_step = update_profiles_time_step(chosen_date, h, m)
    print('Apply-date-time button has been clicked {} times'.format(n_clicks))
    return prof_step

@app.callback(
    Output('d-t-alert', 'is_open'),
    [Input('my-date-picker-single', 'date'),
     Input('hours', 'value'),
     Input('minutes', 'value'),
     Input('button_apply_date_time', 'n_clicks')],
    [State("d-t-alert", "is_open")]
)
def date_time_alert(day, h, m, n_clicks, is_open):
    # Determine which input triggered the callback
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    # If the button is clicked, close the alert
    if triggered_id == 'button_apply_date_time' and n_clicks:
        return False

    # If there is any change in date, hours, or minutes, show the alert
    if triggered_id in ['my-date-picker-single', 'hours', 'minutes']:
        return True

    # Return the current state if no recognized input triggered the callback
    return is_open


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
    print("range1:",fig.layout.xaxis.range)



    if fig.layout.xaxis.range is None:
        fig.layout.xaxis.range =["2016-01-01 00:00","2016-12-31 23:45"]
    target_date_time = str(fig.layout.xaxis.range[0])
    id_date_time = date_time_2016['Date Time'].astype(str).str[:9] == target_date_time[:9]
    id_date_time = date_time_2016[id_date_time].index
    # Check if x-axis range is defined
    print("id_date_time",id_date_time)
    if not id_date_time.empty:
        xstrat = id_date_time[0]
    else:
        xstrat = 0

    if xstrat + time_interval > 96 * 365:
        xstrat = 0
    # Update x-axis range
    new_start = str(date_time_2016['Date Time'][xstrat])
    new_end = str(date_time_2016['Date Time'][xstrat + time_interval])
    fig.update_xaxes(range=[new_start, new_end])

    print("id:", id_date_time)
    print("range2:", fig.layout.xaxis.range)

    print(xstrat,",",xstrat + time_interval)  # Debugging print statement
    print("date_time_2016['Date Time'][xstrat]:", date_time_2016['Date Time'][xstrat])
    print("date_time_2016['Date Time'][xstrat + time_interval]:",date_time_2016['Date Time'][xstrat + time_interval])
    return fig

@app.callback(
    Output('computed-table', 'data'),
    [Input('button_update_kpi_bf', 'n_clicks'), Input('button_update_kpi_af', 'n_clicks')],
    [State('computed-table', 'data')]
)
def update_kpi(bf_clicks, af_clicks, data_table):
    print('Callback triggered')

    # Ensure `data_table` is properly initialized
    if data_table is None:
        print("data_table is None. Initializing with default values.")
        data_table = [
            {'KPIs': 'Number of Congested Lines', 'bf-op': '', 'af-op': ''},
            {'KPIs': 'Line with the Highest Line Loading', 'bf-op': '', 'af-op': ''},
            {'KPIs': 'Highest Line Loading(%)', 'bf-op': '', 'af-op': ''},
            {'KPIs': 'Average Line Loading of Congested Lines(%)', 'bf-op': '', 'af-op': ''}
        ]

    # Determine which button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        return data_table  # No button was clicked; return the data as is
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Default KPI values
    num_cong, lin_max, ll_max, avg_ll_congested = 'NA', 'NA', 'NA', 'NA'

    # Update "Before Optimization" values
    if triggered_id == 'button_update_kpi_bf' and bf_clicks and bf_clicks > 0:
        num_cong, lin_max, ll_max, avg_ll_congested = update_kpi_step(True)
        lin_max_display = 'Line ' + str(lin_max)
        try:
            data_table[0]['bf-op'] = num_cong
            data_table[1]['bf-op'] = lin_max_display
            data_table[2]['bf-op'] = ll_max
            data_table[3]['bf-op'] = avg_ll_congested
        except Exception as e:
            print("Error updating bf-op data_table:", e)
            for row in data_table:
                row['bf-op'] = 'NA'

    # Update "After Optimization" values
    elif triggered_id == 'button_update_kpi_af' and af_clicks and af_clicks > 0:
        #num_cong, lin_max, ll_max, avg_ll_congested = update_kpi_step(GlobalState.action)
        #lin_max_display = 'Line ' + str(lin_max)
        #try:
        data_table[0]['af-op'] = GlobalState.num_congested_op
        data_table[1]['af-op'] = 'Line '+ str(GlobalState.line_max_op)
        data_table[2]['af-op'] = "{:.1f}".format(100*GlobalState.lineload_max_op)
        data_table[3]['af-op'] = "{:.1f}".format(100*GlobalState.avg_ll_congested_op)
        #except Exception as e:
        #    print("Error updating af-op data_table:", e)
        #    for row in data_table:
        #        row['af-op'] = 'NA'

    print('Updated data_table:', data_table)
    return data_table


@app.callback(
    Output("collapse-gap", "is_open"),
    [Input("toggle-gap", "n_clicks")],
    [State("collapse-gap", "is_open")]
)
def toggle_box_gap(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("collapse-table", "is_open"),
    [Input("toggle-table", "n_clicks")],
    [State("collapse-table", "is_open")]
)
def toggle_box_gap(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    [Output("output", "children"), Output("loading-output", "children"), Output('switching-table', 'data')],
    [Input("button_run_op", "n_clicks")],[State('switching-table', 'data')]
)
def handle_button_click(n_clicks, sw_table):
    if sw_table is None:  # Initialize sw_table if it is None
        sw_table = []
    if n_clicks and n_clicks > 0:
        # Run optimization and display its result
        print('Run Optimization')
        result = Env.run_opt()  # Call your function
        print('Result from run_opt:', result)
        GlobalState.action, GlobalState.num_congested_op, GlobalState.line_max_op, GlobalState.lineload_max_op, GlobalState.avg_ll_congested_op, sw_id,  sw_bus, sw_type, sw_substation, sw_subnet, sw_action = result
        # Simulate a loading process
        while len(sw_table) < len(sw_substation):
            sw_table.append({"Substation": "", "Sub-network":"", "Bus-1": "", "Switch-ID": "", "Type of Switch": "", "Switching Action": ""})

        while len(sw_table) > len(sw_substation):
            sw_table.pop()  # Removes the last row in sw_table
        # Update sw_table with values from sw_substation
        for i, (switch_id, switch_type, bus, substation, subnet, action_closed) in enumerate(zip(sw_id, sw_type, sw_bus, sw_substation, sw_subnet, sw_action)):
            sw_table[i]["Switch-ID"] = switch_id
            sw_table[i]["Type of Switch"] = switch_type
            sw_table[i]["Bus-1"] = bus
            #sw_table[i]["Substation"] = substation
            sw_table[i]["Substation"] = '-' if pd.isna(substation) else substation
            # if DC Open/Close if CB On/Off
            if switch_type == 'CB':
                sw_table[i]["Switching Action"] = 'On' if action_closed == 1 else 'Off'
            else:
            ## switch_type == 'LBS'
                sw_table[i]["Switching Action"] = 'Close' if action_closed == 1 else 'Open'

            sw_table[i]["Subnet"] = subnet


                # Print updated sw_table
        print('sw_table_len',len(sw_table))

        return '', "Optimization complete", sw_table

    # Default messages when optimization hasn't started yet
    return "Optimization not started yet", "Click to run optimization", sw_table

@app.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    Output("modal_rw_vol", "is_open"),
    [Input("open_rw_vol", "n_clicks"), Input("close_rw_vol", "n_clicks")],
    [State("modal_rw_vol", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    Output("modal_rw_loss", "is_open"),
    [Input("open_rw_loss", "n_clicks"), Input("close_rw_loss", "n_clicks")],
    [State("modal_rw_loss", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

# Run app
if __name__ == '__main__':
    app.run_server()