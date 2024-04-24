from dash import Dash, html, page_registry, page_container
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import ThemeSwitchAIO


theme_switch = ThemeSwitchAIO(
    aio_id='theme',
    themes=[dbc.themes.CYBORG, dbc.themes.LITERA],
    switch_props={'persistence': True, 'persistence_type': 'session'}
)
button_style = {'font-size': '90%', 'width': '200px', 'textAlign': 'center', 'height': '50px'}

app = Dash(__name__,
           title='Data Analytics',
           use_pages=True,
           suppress_callback_exceptions=True,
           external_stylesheets=[dbc.themes.CYBORG, dbc.icons.BOOTSTRAP])

# main page
app.layout = html.Div(
    children=[
        theme_switch,
        html.H2('Data Analytics'),
        html.Div(
            [
                dbc.ButtonGroup([dbc.Button(page['title'].upper(), href=page['relative_path'], style=button_style) for page in page_registry.values()])
            ]
        ),
        page_container
    ]
)

if __name__=='__main__':
    app.run_server(debug=True)

