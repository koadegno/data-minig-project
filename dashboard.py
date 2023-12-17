from dash import Dash, dcc, html, Input, Output
import plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
import plotly.tools
import plotly.subplots


# CSV with 'cluster' column
# df = pd.read_csv("data_with_clusters.csv", sep=";")
# df = pd.read_csv(
#     "chucks\chuck_0.csv",
#     sep=";",
# )

# df["timestamps_UTC"] = pd.to_datetime(df["timestamps_UTC"])
# df = df[~df["timestamps_UTC"].dt.year.isin([2022])]

# np.random.seed(42)
# df["cluster"] = np.random.choice([0, 1], size=len(df), replace=True)
'''df = pd.read_csv(
    "results_norma\cluster_0_ar41_with_isolation_forest_cluster.csv",
    sep=",",
).rename(columns={"cluster_0": "cluster"})
df["timestamps_UTC"] = pd.to_datetime(df["timestamps_UTC"])
df = df[~df["timestamps_UTC"].dt.year.isin([2022])]'''

df = pd.read_csv(
    "ar41_for_ulb_all_outliers.csv",
    sep=";",
)
df["timestamps_UTC"] = pd.to_datetime(df["timestamps_UTC"])

thresholds = {"RS_E_InAirTemp_PC1": 65, "RS_E_InAirTemp_PC2": 65, "RS_E_WatTemp_PC1": 100, "RS_E_WatTemp_PC2": 100, "RS_T_OilTemp_PC1": 115, "RS_T_OilTemp_PC2": 115}

features_list = [
    "RS_E_InAirTemp_PC1",
    "RS_E_InAirTemp_PC2",
    "RS_E_OilPress_PC1",
    "RS_E_OilPress_PC2",
    "RS_E_RPM_PC1",
    "RS_E_RPM_PC2",
    "RS_E_WatTemp_PC1",
    "RS_E_WatTemp_PC2",
    "RS_T_OilTemp_PC1",
    "RS_T_OilTemp_PC2",
    "temperature",
    "precipitation",
    "windspeed_10m",
    "sum_pollen",
]

TRAIN_1_COLOR = "blue"
TRAIN_2_COLOR = "orange"

veh_id_unique = np.sort(df["mapped_veh_id"].unique())


app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])


config = dbc.Card(
    [
        html.Div(
            [
                dcc.DatePickerRange(
                    id="date-filter",
                    start_date=df["timestamps_UTC"].min(),
                    end_date=df["timestamps_UTC"].max(),
                    display_format="YYYY-MM-DD",
                ),
            ],
        ),
        
        html.Div(
            [
                dbc.Label("Reasons"),
                dcc.Dropdown(
                    id="reason-dropdown",
                    options=[{'label': reason, 'value': reason} for reason in df['reason'].unique()],
                    multi=False,
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
            [dcc.Graph(id="map-graph")],
        ),
    ],
    body=True,
)

features_graph = dbc.Card(
    [ 
        html.Div(
        [
            dbc.Label("Features"),
            dcc.Dropdown(
                id="variable-dropdown",
                options=[{"label": col, "value": col} for col in df.columns if col not in ["timestamps_UTC", "reason"]],
                multi=True,
            ),
        ]
        ),
        html.Div(
            [html.Div(id="feature-graphs")],
        ),
    ],
    body=True,
)


first_page_layout = dbc.Container(
    [
        html.H1("Cool Train", style={"text-align": "center", "color": "white"}),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(config, md=4, style={"padding-top": "1rem"}),
                dbc.Col(map, md=8, style={"padding-top": "1rem"}),
                dbc.Col(features_graph, md=12, style={"padding-top": "1rem"}),
            ],
            align="center",
        ),
    ],
    fluid=True,
    style={"background-color": "rgb(0, 75, 147)", "min-height": "100vh", "min-width": "100vw"},
)


# Creation of the second page with the map and drop-down menus for vehicle tracking
second_page_layout = dbc.Container(
    [
        html.H1("Tracking Page", style={"text-align": "center", "color": "white"}),
        html.Hr(),
        html.Div(
            [
                html.Div(
                    [
                        dcc.DatePickerSingle(
                            id="date-picker-single-1",
                            min_date_allowed=df["timestamps_UTC"].min(),
                            max_date_allowed=df["timestamps_UTC"].max(),
                            initial_visible_month=df["timestamps_UTC"].max(),
                            date=df["timestamps_UTC"].max(),
                            display_format="YYYY-MM-DD",
                            style={"width": "300px"},
                        ),
                        dcc.Dropdown(
                            id="vehicle-dropdown-1",
                            options=[{"label": veh_id, "value": veh_id} for veh_id in veh_id_unique],
                            value=veh_id_unique[0],
                            style={"width": "300px"},
                            clearable=True,
                            searchable=True,
                            placeholder="Select a vehicle...",
                        ),
                    ],
                    style={"margin-bottom": "20px", "margin-left": "20px"},
                ),
                html.Div(
                    [
                        dcc.DatePickerSingle(
                            id="date-picker-single-2",
                            min_date_allowed=df["timestamps_UTC"].min(),
                            max_date_allowed=df["timestamps_UTC"].max(),
                            initial_visible_month=df["timestamps_UTC"].max(),
                            date=df["timestamps_UTC"].max(),
                            display_format="YYYY-MM-DD",
                            style={"width": "300px"},
                        ),
                        dcc.Dropdown(
                            id="vehicle-dropdown-2",
                            options=[{"label": veh_id, "value": veh_id} for veh_id in veh_id_unique],
                            value=veh_id_unique[1],
                            style={"width": "300px"},  # Adjust the width of the dropdown as needed
                        ),
                    ],
                    style={"margin-bottom": "20px", "margin-left": "20px"},
                ),
                html.Div(
                    [
                        dcc.Graph(id="tracking-map"),
                    ],
                ),
                html.Div(
                    id="plots-container",  # Container for both Plotly and Matplotlib plots
                    className="plots-container",
                ),
            ],
        ),
    ],
    fluid=True,
    style={"background-color": "rgb(0, 75, 147)", "min-height": "100vh", "min-width": "100vw"},
)

app.layout = html.Div(
    [
        dcc.Tabs(
            [
                dcc.Tab(label="Main Page", children=first_page_layout),
                dcc.Tab(label="Tracking Page", children=second_page_layout),
            ]
        ),
    ]
)


@app.callback(
    Output("map-graph", "figure"),
    Output("cluster-pie-chart", "figure"),
    Output("feature-graphs", "children"),
    Input("variable-dropdown", "value"),
    Input("reason-dropdown", "value"),
    Input("date-filter", "start_date"),
    Input("date-filter", "end_date"),
)
def update_graph(selected_features, selected_reason, start_date, end_date):
    # Filter df according to selected dates
    filtered_df = df[(df["timestamps_UTC"] >= start_date) & (df["timestamps_UTC"] <= end_date)]
    filtered_df = filtered_df[filtered_df['reason'] == selected_reason]

    # Check if filtered_df is empty
    if not filtered_df.empty:
        # Create a map centered on Belgium
        fig_map = px.scatter_mapbox(
            filtered_df,
            lat="lat",
            lon="lon",
            hover_data=["mapped_veh_id"],
            color="reason",
            zoom=7,
            center=dict(lat=50.8503, lon=4.3517),  # Coordinates for Belgium
            title="Map",
            opacity=0.5,
            mapbox_style="carto-positron",
        )
        fig_map.update_layout(height=565, showlegend=False)
    else:
        # Create an empty map with a message
        fig_map = px.scatter_mapbox(
            lat=[0], lon=[0],  
            zoom=5,
            center=dict(lat=50.8503, lon=4.3517),
            title="Map",
            opacity=0.5,
            mapbox_style="carto-positron",
        )
        fig_map.update_layout(height=565, showlegend=False)
        fig_map.update_layout(annotations=[
            dict(
                text="No data available",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5
            )
        ])
    # Cluster synth
    cluster_counts = df["reason"].value_counts()
    fig_pie_chart = px.pie(
        names=cluster_counts.index,
        values=cluster_counts.values,
        title="Anomalies proportion",
    )
    # Do not render subplot if there is no selected feature
    if not selected_features:
        return fig_map, fig_pie_chart, None

    # Subplots
    subplots = make_subplots(rows=len(selected_features), cols=1, subplot_titles=selected_features, row_heights=[350] * len(selected_features))
    for i, var in enumerate(selected_features):
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
        subplots.update_layout(title_text="Features", height=350 * len(selected_features))
    return fig_map, fig_pie_chart, dcc.Graph(figure=subplots)


@app.callback(
    Output("tracking-map", "figure"),
    Input("date-picker-single-1", "date"),
    Input("vehicle-dropdown-1", "value"),
    Input("date-picker-single-2", "date"),
    Input("vehicle-dropdown-2", "value"),
)
def update_tracking_map(date_1, vehicle_1, date_2, vehicle_2):
    filtered_df_1 = df[(df["timestamps_UTC"].dt.date == pd.to_datetime(date_1).date()) & (df["mapped_veh_id"] == vehicle_1)]
    filtered_df_2 = df[(df["timestamps_UTC"].dt.date == pd.to_datetime(date_2).date()) & (df["mapped_veh_id"] == vehicle_2)]

    fig = px.scatter_mapbox(
        filtered_df_1,
        lat="lat",
        lon="lon",
        hover_data={"timestamps_UTC": True},
        custom_data=["timestamps_UTC"],
        mapbox_style="carto-positron",
        color_discrete_sequence=[TRAIN_1_COLOR],
        zoom=10,
        opacity=0.6,
    )

    fig.add_trace(
        px.scatter_mapbox(
            filtered_df_2,
            lat="lat",
            lon="lon",
            hover_data={"timestamps_UTC": True},
            custom_data=["timestamps_UTC"],
            mapbox_style="carto-positron",
            color_discrete_sequence=[TRAIN_2_COLOR],
            zoom=10,
        ).data[0]
    )

    fig.update_traces(
        hovertemplate="<b>Latitude</b>: %{lat}<br><b>Longitude</b>: %{lon}<br><b>Date</b>: %{customdata[0]}<extra></extra>",
    )

    fig.update_layout(
        title="Scatter Mapbox with Date Information",
        hovermode="closest",
    )

    return fig


def create_features_plots_for_train1_and_train2(df_train_1, df_train_2, features_list, thresholds):
    plotly_plots = []
    df_train_1 = df_train_1.sort_values(["timestamps_UTC"])
    df_train_2 = df_train_2.sort_values(["timestamps_UTC"])
    train_1_id = None
    train_2_id = None
    if not df_train_1.empty:
        train_1_id = df_train_1["mapped_veh_id"].unique()[0]
    if not df_train_2.empty:
        train_2_id = df_train_2["mapped_veh_id"].unique()[0]

    time_values_train_1 = [(t.hour * 60 + t.minute) for t in df_train_1["timestamps_UTC"].dt.time]
    time_values_train_2 = [(t.hour * 60 + t.minute) for t in df_train_2["timestamps_UTC"].dt.time]

    for feature in features_list:
        fig = px.line()
        fig.add_scatter(x=time_values_train_2, y=df_train_2[feature], mode="lines", name=f"{feature} - Train {train_2_id}", line=dict(color=TRAIN_2_COLOR))
        fig.add_scatter(x=time_values_train_1, y=df_train_1[feature], mode="lines", name=f"{feature} - Train {train_1_id}", line=dict(color=TRAIN_1_COLOR))

        plotly_plots.append(fig)

    return plotly_plots


# Callback to update plots for selected trains and dates
@app.callback(
    Output("plots-container", "children"),
    Input("date-picker-single-1", "date"),
    Input("vehicle-dropdown-1", "value"),
    Input("date-picker-single-2", "date"),
    Input("vehicle-dropdown-2", "value"),
)
def update_plots(date_1, vehicle_1, date_2, vehicle_2):
    filtered_df_1 = df[(df["timestamps_UTC"].dt.date == pd.to_datetime(date_1).date()) & (df["mapped_veh_id"] == vehicle_1)]
    filtered_df_2 = df[(df["timestamps_UTC"].dt.date == pd.to_datetime(date_2).date()) & (df["mapped_veh_id"] == vehicle_2)]

    plotly_plots = create_features_plots_for_train1_and_train2(filtered_df_1, filtered_df_2, features_list, thresholds)

    subplots = make_subplots(rows=len(plotly_plots), subplot_titles=features_list, cols=1, row_heights=[350] * len(plotly_plots))

    # Combine plots for both trains
    for idx, feature in enumerate(features_list):
        for trace in plotly_plots[idx].data:
            subplots.add_trace(trace, row=idx + 1, col=1)

        subplots.update_xaxes(
            tickvals=list(range(0, 24 * 60, 60)),
            ticktext=[f"{h:02d}:00" for h in range(24)],
            tickangle=45,
            tickmode="array",
            row=idx + 1,
            col=1,
        )
        if feature in thresholds:
            threshold = [thresholds[feature]] * len(range(0, 24 * 60, 60))

            h_line = go.Scatter(x=list(range(0, 24 * 60, 60)), y=threshold, mode="lines", line_color="green", name=f"Threshold {feature}: {threshold[0]}", opacity=0.5)
            subplots.add_trace(h_line, row=idx + 1, col=1)

        subplots.update_yaxes(title_text="Values", row=idx + 1, col=1)
    subplots.update_layout(title_text="Features", height=350 * len(plotly_plots))

    return dcc.Graph(figure=subplots)


if __name__ == "__main__":
    app.run_server(debug=True, threaded=True)
