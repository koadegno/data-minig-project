from dash import Dash, dcc, html, Input, Output
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc

# CSV with 'cluster' column
df = pd.read_csv('data_with_clusters.csv', sep=';')
df['timestamps_UTC'] = pd.to_datetime(df['timestamps_UTC'])

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])


config = dbc.Card(
    [
        html.Div(
            [
                dcc.DatePickerRange(
                            id="date-filter",
                            start_date=df['timestamps_UTC'].min(),
                            end_date=df['timestamps_UTC'].max(),
                            display_format='YYYY-MM-DD',
                        ),
            ],
        ),
        html.Div(
            [
                dbc.Label("Features"),
                dcc.Dropdown(
                            id="variable-dropdown",
                            options=[
                                {"label": col, "value": col}
                                for col in df.columns if col not in ["timestamps_UTC"]
                            ],
                            value=["cluster"],
                            multi=True,
                        ),
            ]
        ),
        html.Div(
            [
                dcc.Graph(id="cluster-pie-chart"),
            ]
        ),
    ],
    body=True,
)

map = dbc.Card(
    [
        html.Div(
            [
                dcc.Graph(id="map-graph")
            ],
        ),
    ],
    body = True,
)

features_graph = dbc.Card(
    [
        html.Div(
            [
                html.Div(id="feature-graphs")
            ],
        ),
    ],
    body = True,
)

app.layout = dbc.Container(
    [
        html.H1("Cool Train", style={'text-align':'center', 'color':'white'}),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(config, md=4, style={"padding-top":"1rem"}),
                dbc.Col(map, md=8, style={"padding-top":"1rem"}),
                dbc.Col(features_graph, md=12, style={"padding-top":"1rem"}),
            ],
            align="center",
        ),
    ],
    fluid=True,
    style={'background-color':'rgb(0, 75, 147)', 'min-height': '100vh', 'min-width' : '100vw'}
)

@app.callback(
    Output("map-graph", "figure"),
    Output("cluster-pie-chart", "figure"),
    Output("feature-graphs", "children"),
    Input("variable-dropdown", "value"),
    Input("date-filter", "start_date"),
    Input("date-filter", "end_date"),
)
def update_graph(selected_variable, start_date, end_date):
    hover = ["mapped_veh_id"]
    hover.extend(selected_variable)
    
    # Filter df according to selected dates
    filtered_df = df[(df['timestamps_UTC'] >= start_date) & (df['timestamps_UTC'] <= end_date)]
    
    fig_map = px.scatter_mapbox(
        filtered_df,
        lat="lat",
        lon="lon",
        hover_data=hover,
        color="cluster",
        zoom=7,
        title="Map",
    )
    fig_map.update_layout(mapbox_style="open-street-map", height=565)

    # Cluster synth
    cluster_counts = filtered_df['cluster'].value_counts()
    fig_pie_chart = px.pie(
        names=cluster_counts.index,
        values=cluster_counts.values,
        title="Anomalies proportion",
    )
    # Do not render subplot if there is no selected feature
    if not selected_variable:
        return fig_map, fig_pie_chart, None
    
    # Subplots
    subplots = make_subplots(rows=len(selected_variable), cols=1, subplot_titles=selected_variable, row_heights=[350] * len(selected_variable))
    for i, var in enumerate(selected_variable):
        # Create histogram with different color for each variable
        hist, bins = np.histogram(filtered_df[var], bins=50)
        # Cycle through Plotly colors
        color = px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]  
        # Mean
        mean_line = go.Scatter(x=[filtered_df[var].mean(), filtered_df[var].mean()], y=[0, max(hist)], mode="lines", name="Mean")
        # Median
        median_line = go.Scatter(x=[np.median(filtered_df[var]), np.median(filtered_df[var])], y=[0, max(hist)], mode="lines", name="Median")
        # Add histogram on subplot
        bar_trace = go.Bar(x=bins[:-1], y=hist, width=(bins[1]-bins[0]), marker=dict(color=color), name=f"{var}")
        subplots.add_trace(bar_trace, row=i+1, col=1)
        # Add mean & median on subplot
        subplots.add_trace(mean_line, row=i+1, col=1)
        subplots.add_trace(median_line, row=i+1, col=1)
        # Add Y-axis label
        subplots.update_yaxes(title_text="Quantity", row=i+1, col=1)
        subplots.update_layout(title_text="Features", height=350 * len(selected_variable))
    return fig_map, fig_pie_chart, dcc.Graph(figure=subplots)

if __name__ == "__main__":
    app.run_server(debug=True)
