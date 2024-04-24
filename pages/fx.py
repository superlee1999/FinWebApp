from dash import html, callback, ctx, Input, Output, State, dash_table, register_page, dcc
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import dash_draggable
import datetime as dt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from webapp_utils import utils
import pandas as pd


register_page(__name__, path='/')


fx_spot_matrix_contents = html.Div([
    dbc.Row(children=[
        dbc.Col(),
        dbc.Col(
            dbc.InputGroup([
                dbc.RadioItems(id='fx-spot-matrix-type', options=['LastPrice', '%Change'], value='LastPrice', inline=True, style={'width': '200px', 'background-color': '#787878', 'color': 'black'})
            ],
            style={
                'justify-content': 'end',
                'margin-right': '10px'
            })
        )
    ]),
    dbc.Row(dbc.Spinner(
        html.Div([
            dcc.Graph(id='fx-spot-matrix-graph', figure=utils.blank_fig(), responsive=True)
        ])
    ))
])


fx_spot_time_series_contents = html.Div([
    dbc.Row(children=[
        dbc.Col(children=[
            dbc.InputGroup([
                dcc.Dropdown(
                    id='fx-selected-pair',
                    options=sorted(utils.DataStore.fx_g10),
                    value=sorted(utils.DataStore.fx_g10)[0],
                    multi=False,
                    placeholder='Select fx pair',
                    style={'width': '150px', 'margin-right': '30px'}
                ),
                dbc.InputGroupText('Regression Independent Pair:', style={'background-color': '#787878', 'color': 'black'}),
                dcc.Dropdown(
                    id='fx-independent-pair',
                    options=sorted(utils.DataStore.fx_g10),
                    multi=False,
                    placeholder='Select fx pair',
                    style={'width': '150px'}
                ),
                dbc.InputGroupText('Lag:', style={'background-color': '#787878', 'color': 'black'}),
                dcc.Input(id='fx-independent-pair-lag', placeholder='lag', type='number', value=0, min=0, style={'width': '50px'}),
                dbc.InputGroupText('Regression On:', style={'background-color': '#787878', 'color': 'black'}),
                dcc.Dropdown(
                    id='fx-regression-on',
                    options=['Value', '%Change'],
                    multi=False,
                    value='Value',
                    style={'width': '150px', 'margin-right': '20px'}
                ),
            ]),
        ], width='auto'),
        dbc.Col(
            dbc.InputGroup([
                dbc.Checklist(id='fx-spot-moving-average', options=['DisplayMovingAverage'], style={'background-color': '#787878', 'color': 'black', 'margin-right': '10px'}),
                dcc.Input(id='fx-spot-moving-average-window', placeholder='window size', type='number', value=50, min=5, style={'width': '150px', 'background-color': '#787878', 'color': 'black', 'margin-right': '20px'})
            ],
            style={
                'justify-content': 'end',
                'margin-right': '10px'
            })
        )
    ]),
    dbc.Row(dbc.Spinner(
        html.Div([
            dcc.Graph(id='fx-spot-time-series-graph', figure=utils.blank_fig(), responsive=True)
        ]),
    )),
])


fx_regression_contents = html.Div([
    dbc.Row(dbc.Spinner(
        html.Div([
            dcc.Graph(id='fx-spot-regression-graph', figure=utils.blank_fig(), responsive=True)
        ])
    ))
])


def set_fx_spot_layout(fx_spot_matrix_contents, fx_spot_time_series_contents, fx_regression_contents):
    contents_layout = html.Div([
        dash_draggable.ResponsiveGridLayout(
            id='drag-fx-spot',
            layouts={
                "lg": [
                    {
                        "i": "1",
                        "x": 0, "y": 0, "w": 12, "h": 15
                    },
                    {
                        "i": "2",
                        "x": 0, "y": 15, "w": 12, "h": 15
                    },
                    {
                        "i": "3",
                        "x": 0, "y": 30, "w": 12, "h": 15
                    }
                ],
                "md": [
                    {
                        "i": "1",
                        "x": 0, "y": 0, "w": 10, "h": 15
                    },
                    {
                        "i": "2",
                        "x": 0, "y": 15, "w": 10, "h": 15
                    },
                    {
                        "i": "3",
                        "x": 0, "y": 30, "w": 10, "h": 15
                    }
                ],
                "sm": [
                    {
                        "i": "1",
                        "x": 0, "y": 0, "w": 6, "h": 15
                    },
                    {
                        "i": "2",
                        "x": 0, "y": 15, "w": 6, "h": 15
                    },
                    {
                        "i": "3",
                        "x": 0, "y": 30, "w": 6, "h": 15
                    }
                ]
            },
            children=[
                dbc.Tabs(id='1',
                         active_tab='fx-spot-matrix',
                         children=[dbc.Tab(tab_id='fx-spot-matrix', label='Spot Matrix', children=fx_spot_matrix_contents)],
                         persistence=True,
                         persistence_type='session'),
                dbc.Tabs(id='2',
                         active_tab='fx-spot-time-series',
                         children=[dbc.Tab(tab_id='fx-spot-time-series', label='Spot Time Series', children=fx_spot_time_series_contents)],
                         persistence=True,
                         persistence_type='session'),
                dbc.Tabs(id='3',
                         active_tab='fx-regression',
                         children=[dbc.Tab(tab_id='fx-regression', label='Spot Time Series Regression', children=fx_regression_contents)],
                         persistence=True,
                         persistence_type='session')
            ]
        )
    ])
    return contents_layout


layout = html.Div(children=[
    html.Hr(),
    dbc.Row(children=[
        dbc.InputGroup([
            dbc.InputGroupText('Start Date:'),
            dcc.DatePickerSingle(
                id='start-date',
                date=dt.date(2022, 1, 1),
                display_format='YYYY-MM-DD',
                persistence=True,
                persistence_type='session'
            ),
            dbc.InputGroupText('End Date:'),
            dcc.DatePickerSingle(
                id='end-date',
                date=utils.DataStore.get_last_date(),
                display_format='YYYY-MM-DD',
                persistence=True,
                persistence_type='session'
            ),
            dbc.Button(id='load-data',
                       children='Load Data',
                       style={'width': '150px', 'textAlign': 'center', 'margin-left': '10px', 'display': 'inline-block'})
            ],
            style={'justify-content': 'end', 'margin-right': '10px'})
    ]),
    dbc.Card([
        dbc.CardHeader(
            dbc.Tabs(
                id='fx-tabs',
                children=[
                    dbc.Tab(tab_id='fx-spot', label='Spot', tab_style={'width': '250px', 'textAlign': 'center'}),
                ],
                active_tab='fx-spot',
                persistence=True,
                persistence_type='session'
            )
        ),
        dbc.CardBody(
            id='fx-contents',
            children=set_fx_spot_layout(fx_spot_matrix_contents, fx_spot_time_series_contents, fx_regression_contents)
        )
    ])
    ],
    style={'font-size': '90%'}
)


@callback(
    output=Output('fx-spot-matrix-graph', 'figure'),
    inputs=[State('start-date', 'date'), State('end-date', 'date'),
            Input('fx-spot-matrix-type', 'value'), Input('load-data', 'n_clicks')]
)
def display_fx_spot_matrix(start_date, end_date, spot_matrix_type, load_data_clicked):
    if ctx.triggered_id is None:
        raise PreventUpdate

    start_date = dt.date.fromisoformat(start_date)
    end_date = dt.date.fromisoformat(end_date)

    fig = go.Figure()
    if spot_matrix_type=='LastPrice':
        df_spot, d = utils.DataStore.get_fx_spot_matrix(fx_pairs=utils.DataStore.fx_g10, date=end_date)
        df_change, d = utils.DataStore.get_fx_spot_pct_change_matrix(fx_pairs=utils.DataStore.fx_g10, date=end_date)
        vals = df_change.fillna(0).values
        center = (0 - vals[:].min()) / (vals[:].max() - vals[:].min())
        fig.add_trace(go.Heatmap(z=df_change*100,
                                 x=df_change.columns,
                                 y=df_change.index,
                                 text=df_spot,
                                 hovertemplate='%{x}%{y}:%{text:.4f}<extra></extra>',
                                 texttemplate='%{text:.4f}',
                                 colorscale=[(0, 'rgb(255, 0, 0, 0)'), (center, 'rgb(0, 0, 0)'), (1, 'rgb(0, 255, 0)')]))
    elif spot_matrix_type=='%Change':
        df_change, d = utils.DataStore.get_fx_spot_pct_change_matrix(fx_pairs=utils.DataStore.fx_g10, date=end_date)
        vals = df_change.fillna(0).values
        center = (0 - vals[:].min()) / (vals[:].max() - vals[:].min())
        fig.add_trace(go.Heatmap(z=df_change*100,
                                 x=df_change.columns,
                                 y=df_change.index,
                                 text=df_change*100,
                                 hovertemplate='%{x}%{y}:%{text:.4f}<extra></extra>',
                                 texttemplate='%{text:.4f}',
                                 colorscale=[(0, 'rgb(255, 0, 0, 0)'), (center, 'rgb(0, 0, 0)'), (1, 'rgb(0, 255, 0)')]))
    else:
        raise PreventUpdate

    fig.update_layout(
        title={
            'text': f'{end_date.isoformat()}',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis={
            'side': 'top'
        }
    )

    return fig


@callback(
    output=[Output('fx-spot-time-series-graph', 'figure'), Output('fx-spot-regression-graph', 'figure')],
    inputs=[State('start-date', 'date'), State('end-date', 'date'),
            Input('fx-selected-pair', 'value'), Input('fx-spot-moving-average', 'value'), Input('fx-spot-moving-average-window', 'value'),
            Input('fx-independent-pair', 'value'), Input('fx-independent-pair-lag', 'value'), Input('fx-regression-on', 'value'),
            Input('load-data', 'n_clicks')]
)
def display_fx_spot_time_series(start_date, end_date, fx_pair, display_moving_average, moving_average_window, fx_independent_pair, fx_independent_pair_lag, regression_on, load_data_clicked):
    if ctx.triggered_id is None:
        raise PreventUpdate

    if fx_pair is None:
        raise PreventUpdate
    if fx_independent_pair_lag is None:
        fx_independent_pair_lag = 0

    start_date = dt.date.fromisoformat(start_date)
    end_date = dt.date.fromisoformat(end_date)
    reg_fig = utils.blank_fig()

    df_spots = utils.DataStore.get_fx_spots(fx_pairs=[fx_pair], start_date=start_date, end_date=end_date)
    spot_last = df_spots.iloc[-1][fx_pair]
    spot_high = df_spots.loc[:, fx_pair].max()
    spot_high_date = df_spots.loc[:, fx_pair].idxmax()
    spot_low = df_spots.loc[:, fx_pair].min()
    spot_low_date = df_spots.loc[:, fx_pair].idxmin()
    spot_average = df_spots.loc[:, fx_pair].mean()

    spot_fig = make_subplots(specs=[[{"secondary_y": True}]]) if fx_independent_pair else make_subplots(specs=[[{"secondary_y": False}]])
    spot_fig.add_trace(go.Scatter(x=df_spots.index, y=df_spots[fx_pair], line=dict(color='blue'), name=f'{fx_pair} spot'), secondary_y=False)
    spot_fig.update_yaxes(title_text=f'{fx_pair}', secondary_y=False)
    spot_fig.add_annotation(text=f'{fx_pair}<br>'
                                 f'{"Last":<10}{spot_last:>30.2f}<br>'
                                 f'{"High":<10}{spot_high_date.date().isoformat():>10}{spot_high:>10.2f}<br>'
                                 f'{"Low":<10}{spot_low_date.date().isoformat():>10}{spot_low:>10.2f}<br>'
                                 f'{"Mean":<10}{spot_average:>27.2f}',
                            align='left',
                            showarrow=False,
                            xref='paper',
                            yref='paper',
                            x=0,
                            y=1.3,
                            bordercolor='black',
                            borderwidth=1)

    if display_moving_average and display_moving_average[0]=='DisplayMovingAverage' and moving_average_window is not None:
        df_moving_average = df_spots[fx_pair].rolling(moving_average_window).mean()
        spot_fig.add_trace(go.Scatter(x=df_moving_average.index, y=df_moving_average, line=dict(color='blue', dash='dash'), name=f'{fx_pair} {moving_average_window} days MAVG'), secondary_y=False)

    # independent pair
    if fx_independent_pair and fx_independent_pair!=fx_pair:
        df_other_spots = utils.DataStore.get_fx_spots(fx_pairs=[fx_independent_pair], start_date=start_date, end_date=end_date)
        other_spot_last = df_other_spots.iloc[-1][fx_independent_pair]
        other_spot_high = df_other_spots.loc[:, fx_independent_pair].max()
        other_spot_high_date = df_other_spots.loc[:, fx_independent_pair].idxmax()
        other_spot_low = df_other_spots.loc[:, fx_independent_pair].min()
        other_spot_low_date = df_other_spots.loc[:, fx_independent_pair].idxmin()
        other_spot_average = df_other_spots.loc[:, fx_independent_pair].mean()

        spot_fig.add_trace(go.Scatter(x=df_other_spots.index, y=df_other_spots[fx_independent_pair], line=dict(color='red'), name=f'{fx_independent_pair} spot'), secondary_y=True)
        spot_fig.update_yaxes(title_text=f'{fx_independent_pair}', secondary_y=True)
        spot_fig.add_annotation(text=f'{fx_independent_pair}<br>'
                                     f'{"Last":<10}{other_spot_last:>30.2f}<br>'
                                     f'{"High":<10}{other_spot_high_date.date().isoformat():>10}{other_spot_high:>10.2f}<br>'
                                     f'{"Low":<10}{other_spot_low_date.date().isoformat():>10}{other_spot_low:>10.2f}<br>'
                                     f'{"Mean":<10}{other_spot_average:>27.2f}',
                                align='left',
                                showarrow=False,
                                xref='paper',
                                yref='paper',
                                x=0.9,
                                y=1.3,
                                bordercolor='black',
                                borderwidth=1)

        if display_moving_average and display_moving_average[0]=='DisplayMovingAverage' and moving_average_window is not None:
            df_other_moving_average = df_other_spots[fx_independent_pair].rolling(moving_average_window).mean()
            spot_fig.add_trace(go.Scatter(x=df_other_moving_average.index, y=df_other_moving_average, line=dict(color='red', dash='dash'), name=f'{fx_independent_pair} {moving_average_window} days MAVG'), secondary_y=True)

        # regression plot
        # reg_results = utils.simple_linear_regression(df_other_spots, df_spots)
        if regression_on:
            df_dep = utils.DataStore.get_fx_spots(fx_pairs=[fx_pair], start_date=utils.DataStore.get_first_date(), end_date=end_date)
            df_indep = utils.DataStore.get_fx_spots(fx_pairs=[fx_independent_pair], start_date=utils.DataStore.get_first_date(), end_date=end_date)
            if fx_independent_pair_lag and fx_independent_pair_lag > 0:
                if regression_on=='Value':
                    df_indep = df_indep.shift(fx_independent_pair_lag).loc[start_date:end_date]
                    df_dep = df_dep.loc[start_date:end_date]
                else:
                    df_indep = df_indep.pct_change().shift(fx_independent_pair_lag).loc[start_date:end_date] * 100
                    df_dep = df_dep.pct_change().loc[start_date:end_date] * 100
            else:
                if regression_on=='Value':
                    df_indep = df_indep.loc[start_date:end_date]
                    df_dep = df_dep.loc[start_date:end_date]
                else:
                    df_indep = df_indep.pct_change().loc[start_date:end_date] * 100
                    df_dep = df_dep.pct_change().loc[start_date:end_date] * 100
            reg_df = pd.concat([df_indep, df_dep], axis=1)
            reg_fig = px.scatter(data_frame=reg_df, x=fx_independent_pair, y=fx_pair, trendline='ols', marginal_x="histogram", marginal_y="histogram")
            reg_results = px.get_trendline_results(reg_fig).loc[0, 'px_fit_results']
            reg_fig.add_annotation(text=f'{"nobs: "}{reg_results.nobs:.0f}<br>'
                                        f'{"lag: "}{fx_independent_pair_lag:.0f}',
                                   align='left',
                                   showarrow=False,
                                   xref='paper',
                                   yref='paper',
                                   x=0,
                                   y=1.1,
                                   bordercolor='black',
                                   borderwidth=1)

    return spot_fig, reg_fig