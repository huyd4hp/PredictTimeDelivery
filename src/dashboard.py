from dash import Dash, html, dcc, dash_table
import plotly.express as px
import pandas as pd
import os
from dash.dependencies import Input, Output, State
import numpy as np
import statistics as st

app = Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Dashboard - EDA"

data = pd.read_csv(os.getcwd() + '/dataset/dataset.csv')
data_clean = pd.read_csv(os.getcwd() + '/dataset/dataset_clean.csv')

del data["Unnamed: 0"]
del data["ID"]
del data["Delivery_person_ID"]

data['Vehicle_condition'] = data['Vehicle_condition'].astype(object)

numeric_features_dataset = data.select_dtypes(include=[np.number])
numeric_features_dataset_clean = data_clean.select_dtypes(include=[np.number])
categorical_features_dataset = data.select_dtypes(include=[object])
categorical_features_dataset_clean = data_clean.select_dtypes(include=[object])


def build_tabs():
    return html.Div(
        id="tabs",
        className="tabs",
        children=[
            dcc.Tabs(
                id="app-tabs",
                value="tab1",
                className="custom-tabs",
                children=[
                    dcc.Tab(
                        id="Dataset-tab",
                        label="Dataset",
                        value="tab1",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                    dcc.Tab(
                        id="Dataset-clean-tab",
                        label="Dataset Clean",
                        value="tab2",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                ],
            )
        ],
    )


def category_calculate_statistics(dataset):
    mode = []
    isNull = []
    for column in categorical_features_dataset.columns:
        mode.append(st.mode(dataset[column]))
        isNull.append("{:.2f}".format(dataset[column].isnull().sum() / len(dataset) * 100))
    df = pd.DataFrame({
        "Category feature": categorical_features_dataset.columns,
        "Mode": mode,
        "Missing value (%)": isNull
    })
    return df


def numeric_calculate_statistics(dataset):
    median = []
    mean = []
    std = []
    var = []
    isNull = []
    min = []
    max = []
    for column in numeric_features_dataset.columns:
        median.append("{:.2f}".format(dataset[column].median()))
        mean.append("{:.2f}".format(dataset[column].mean()))
        min.append("{:.2f}".format(dataset[column].min()))
        max.append("{:.2f}".format(dataset[column].max()))
        std.append("{:.2f}".format(dataset[column].std()))
        var.append("{:.2f}".format(dataset[column].var()))
        isNull.append("{:.2f}".format(dataset[column].isnull().sum() / len(dataset) * 100))
    df = pd.DataFrame({
        "Numeric feature": numeric_features_dataset.columns,
        "Median": median,
        "Mean": mean,
        "Min": min,
        "Max": max,
        "Variance": var,
        "Standard deviation": std,
        "Missing value (%)": isNull,
    })
    return df


numeric_features_statistics_dataset = numeric_calculate_statistics(data)
category_features_statistics_dataset = category_calculate_statistics(data)
fig_target_dispersion = px.histogram(data, x='Time_taken_(min)', title='Target feature dispersion')
fig_restaurant_mapbox = px.density_mapbox(data, lat='Restaurant_latitude', lon='Restaurant_longitude', radius=3,
                                          center=dict(lat=0, lon=0), zoom=1, mapbox_style="stamen-terrain",
                                          title='Restaurant location distribution')
fig_delivery_mapbox = px.density_mapbox(data, lat='Delivery_location_latitude', lon='Delivery_location_longitude',
                                        radius=3, center=dict(lat=0, lon=0), zoom=1, mapbox_style="stamen-terrain",
                                        title='Delivery location distribution')


def build_tab_1():
    return [
        html.Div(
            children=[
                html.Div(style={"display": 'flex', 'justify-content': 'space-around', 'height': '370px',
                                'margin-bot': '5px'},
                         children=[
                             html.Div(style={'width': '980px'}, children=[
                                 html.P(style={'font-size': '16px', 'text-align': 'center'},
                                        children="Statistical table of numeric features"),
                                 dash_table.DataTable(
                                     style_data={
                                         'whiteSpace': 'normal',
                                     },
                                     data=numeric_features_statistics_dataset.to_dict('records'),
                                     columns=[{"name": i, "id": i} for i in
                                              numeric_features_statistics_dataset.columns]),
                             ]),
                             html.Div(style={'width': '650px'}, children=[
                                 html.P(style={'font-size': '16px', 'text-align': 'center'},
                                        children="Statistical table of category features"),
                                 dash_table.DataTable(category_features_statistics_dataset.to_dict('records'),
                                                      [{"name": i, "id": i} for i in
                                                       category_features_statistics_dataset.columns]),
                             ]),
                         ]),
                html.Div(
                    style={"display": 'flex', 'justify-content': 'space-around'},
                    children=[
                        html.Div(
                            style={'width': '600px', 'margin-top': '20px'},
                            children=[
                                dcc.Graph(
                                    figure=fig_restaurant_mapbox, style={'height': '470px'}
                                ),
                            ]
                        ),
                        html.Div(
                            style={'width': '600px', 'margin-top': '20px'},
                            children=[
                                dcc.Graph(
                                    figure=fig_delivery_mapbox, style={'height': '470px'}
                                ),
                            ]
                        ),
                        html.Div(
                            style={'width': '750px', 'margin-top': '20px'},
                            children=[
                                dcc.Graph(
                                    figure=fig_target_dispersion, style={'height': '470px'}
                                ),
                            ]
                        )
                    ]
                )
            ]),
    ]


data_gb = data_clean[["Road_traffic_density", "Hour_order", "Time_taken_(min)"]]
data_groupby = data_gb.groupby(["Road_traffic_density", "Hour_order"], as_index=False).mean()
data_pivot = data_groupby.pivot(index="Hour_order", columns="Road_traffic_density")
fig_pivot = px.imshow(data_pivot.values, labels=dict(x="Road traffic", y="Hour order", color="Time"),
                x=['High', 'Jam', 'Low', 'Medium'], y=data_pivot.index, title='Group feature hour order and road traffic density')
fig_pivot.update_xaxes(side="bottom")

fig_road_traffic_density = px.histogram(data_clean, x="Road_traffic_density", title='Road traffic density dispersion')
fig_box = px.box(data_clean, x="Road_traffic_density", y="Time_taken_(min)", title='Correlation between road traffic density and delivery time')
fig_hour_order = px.histogram(data_clean, x="Hour_order", title='Hour order dispersion')
fig = px.box(data_clean, x="Hour_order", y="Time_taken_(min)", title='Correlation between hour order and delivery time')
fig_multiple_deliveries = px.histogram(data_clean, x="Multiple_deliveries", title='Multiple deliveries dispersion')
fig_box_multiple_deliveries = px.box(data_clean, x="Multiple_deliveries", y="Time_taken_(min)", title='Correlation between multiple deliveries and delivery time')

for i in range(len(data_clean['Delivery_person_ratings'])):
    data_clean['Delivery_person_ratings'] = data_clean['Delivery_person_ratings'].astype(int)
fig_box_delivery_person_ratings = px.box(data_clean, x="Delivery_person_ratings", y="Time_taken_(min)", title='Correlation between delivery person ratings and delivery time')

df_corr = data_clean.corr(numeric_only=True).round(2)
mask = np.zeros_like(df_corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
df_corr_viz = df_corr.mask(mask).dropna(how='all').dropna('columns', how='all')
fig_corr_heatmap = px.imshow(df_corr_viz, text_auto=True, title='Heatmap represents the correlation between numeric features')


def build_tab_2():
    return [
        html.Div(
            children=[
                html.Div(
                    style={'display': 'flex', 'justify-content': 'space-between'},
                    children=[
                        html.Div(
                            style={'display': 'flex'},
                            children=[
                                html.Div(style={'width': '480px'},
                                         children=[dcc.Graph(
                                             figure=fig_hour_order, style={'height': '310px'}
                                         )]),
                                html.Div(style={'width': '520px', 'margin-left': '-30px'},
                                         children=[dcc.Graph(
                                             figure=fig, style={'height': '320px'}
                                         )])
                            ]
                        ),
                        html.Div(
                            style={'display': 'flex'},
                            children=[
                                html.Div(style={'width': '430px'},
                                         children=[dcc.Graph(
                                             figure=fig_road_traffic_density, style={'height': '310px'}
                                         )]),
                                html.Div(style={'width': '500px', 'margin-left': '-30px'},
                                         children=[dcc.Graph(
                                             figure=fig_box, style={'height': '320px'}
                                         )])
                            ]
                        ),
                    ]
                ),
                html.Div(
                    style={'display': 'flex', 'justify-content': 'space-between', 'margin-top': '-30px', 'margin-right': '150px'},
                    children=[
                        html.Div(
                            children=[
                                html.Div(
                                    style={'display': 'flex', 'width': '1100px'},
                                    children=[
                                        html.Div(
                                            style={'width': '500px'},
                                            children=[dcc.Graph(
                                                figure=fig_pivot, style={'height': '310px'}
                                            )]
                                        ),
                                        html.Div(
                                            style={'width': '550px'},
                                            children=[dcc.Graph(
                                                figure=fig_box_delivery_person_ratings, style={'height': '310px'}
                                            )]
                                        )
                                    ]
                                ),
                                html.Div(
                                    style={'display': 'flex', 'width': '1100px', 'margin-top': '-30px'},
                                    children=[
                                        html.Div(
                                            style={'width': '500px'},
                                            children=[dcc.Graph(
                                                figure=fig_multiple_deliveries, style={'height': '310px'}
                                            )]
                                        ),
                                        html.Div(
                                            style={'width': '550px'},
                                            children=[dcc.Graph(
                                                figure=fig_box_multiple_deliveries, style={'height': '310px'}
                                            )]
                                        )
                                    ]
                                )
                            ]
                        ),
                        html.Div(
                            children=[
                                dcc.Graph(
                                    figure=fig_corr_heatmap, style={'height': '580px'}
                                )
                            ]
                        )
                    ]
                )
            ]
        )
    ]


app.layout = html.Div(
    id="big-app-container",
    children=[
        html.Div(
            id="app-container",
            children=[
                build_tabs(),
                html.Div(id="app-content"),
            ],
        ),
    ],
)


@app.callback(
    [Output("app-content", "children")],
    [Input("app-tabs", "value")],
)
def render_tab_content(tab_switch):
    if tab_switch == "tab1":
        return build_tab_1()
    return build_tab_2()


if __name__ == '__main__':
    app.run_server(debug=True)
