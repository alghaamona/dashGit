#By Mona
# ---------------------------------------------------------Importing ------------------------------------------------
import plotly.express as px
import dash_bootstrap_components as dbc
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
from pyorbital.orbital import Orbital

# ---------------------------------------------------------Reading file ------------------------------------------------

data = pd.read_csv("dataset.csv", encoding='unicode_escape')

# -----------------------------------------------------Data Processing-----------------------------------
# generate invoice month for each line purchase equal to the first day of the month when the purchase was made


data['InvoiceMonth'] = pd.to_datetime(data['InvoiceDate']).to_numpy().astype('datetime64[M]')

# first invoice month for every customer
data['CohortMonth'] = data.groupby('CustomerID')['InvoiceMonth'].transform('min')

# drop null values
data.dropna(inplace=True)

data.head()
# drop NaN values
data.dropna()

# compute year and month from Invoice Date
invoice_year = data['InvoiceMonth'].dt.year.astype('int')
invoice_mon = data['InvoiceMonth'].dt.month.astype('int')

# compute year and month from Cohort Date
cohort_year = data['CohortMonth'].dt.year.astype('int')
cohort_mon = data['CohortMonth'].dt.month.astype('int')

# find the differences
diff_year = invoice_year - cohort_year
diff_mon = invoice_mon - cohort_mon

# calculate the cohort index for each invoice
data['CohortIndex'] = diff_year * 12 + diff_mon + 1
data.head()
# group by cohort month and index and find number of unique customers for each grouping
grouped = data.groupby(['CohortMonth', 'CohortIndex', ])['CustomerID'].apply(pd.Series.nunique) \
    .reset_index()
# pivot the data with cohort month as rows and Cohort Index as columns
grouped = grouped.pivot(index='CohortMonth', columns='CohortIndex', values='CustomerID')
grouped
# divide each column by value of the first(cohort size) to find retention rate
size = grouped.iloc[:, 0]
retention_table = grouped.divide(size, axis=0)

# compute the percentage
retention_table.round(3) * 100

# -------------------------------- Graph --------------------------------------------------------
a = px.imshow(
    retention_table, text_auto=".2%", color_continuous_scale="blues", range_color=[0, 1]
).update_xaxes(side="top", dtick=1).update_yaxes(dtick="M1")
a

# --------------------------------------------- App ----------------------------------------------
satellite = Orbital('TERRA')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

df = pd.read_csv("dataset.csv", encoding='unicode_escape')

# save the values of contries
input_selection = df.Country.unique()

app.layout = html.Div(
    html.Div([

        dbc.Row([
            dbc.Col([
                html.H1("e-commerce Dashboard ", style={'textAlign': 'center', 'color': 'Black'})
            ], width=12)
        ], align='center'),
        html.Br(),
        dbc.Row([
            dbc.Col([
                html.Label('Please Select The Country You Want ', style={'color': 'Black'}),
                dcc.Dropdown(
                    id='dropDown',
                    options=[{'label': x, 'value': x} for x in sorted(input_selection)],
                    value=input_selection[0],
                ),
            ], width=12, align='center'),
        ]),
        dcc.Graph(id='heatmap'),

    ])
)


# Multiple components can update everytime interval gets fired.
@app.callback(
       Output(component_id='heatmap', component_property='figure'),
       Input(component_id='dropDown', component_property='value'))
def update_graph_live(n):
    print(n)
    dff = df.copy()

    dff = dff[dff["Country"] == n]
    # group by cohort month and index and find number of unique customers for each grouping
    grouped = dff.groupby(['CohortMonth', 'CohortIndex', ])['CustomerID'].apply(pd.Series.nunique) \
          .reset_index()
    # pivot the data with cohort month as rows and Cohort Index as columns
    grouped = grouped.pivot(index='CohortMonth', columns='CohortIndex', values='CustomerID')
    # divide each column by value of the first(cohort size) to find retention rate
    size = grouped.iloc[:, 0]
    retention_table = grouped.divide(size, axis=0)
    # compute the percentage
    retention_table.round(3) * 100

    fig_map = px.imshow(
        retention_table, text_auto=".2%", color_continuous_scale="blues", range_color=[0, 1]
    ).update_xaxes(side="top", dtick=1).update_yaxes(dtick="M1")

    fig = go.Figure(fig_map)

    return fig


if __name__ == '__main__':
    app.run_server()
