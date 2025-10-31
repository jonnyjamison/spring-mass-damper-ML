import numpy as np
import requests
from dash import Dash, dcc, html, Output, Input
import plotly.graph_objs as go
from scipy.integrate import odeint


API_URL = "http://localhost:8080/predict"  # Update if running elsewhere


def physics_model(m, c, k, t):
    """Analytical solution for damped oscillator using ODE solver."""
    def equation(x, t):
        return [x[1], -(c / m) * x[1] - (k / m) * x[0]]

    x0 = [1, 0]  # Initial position=1, velocity=0
    sol = odeint(equation, x0, t)
    return sol[:, 0]


def request_ml_prediction(m, c, k):
    """Request prediction from FastAPI endpoint."""
    response = requests.post(API_URL, json={"m": m, "c": c, "k": k})
    return response.json()["displacement"]


# Initialize Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.H2("Spring–Mass–Damper System Comparison"),
    dcc.Graph(id="displacement-graph"),
    
    html.Div([
        html.Label("Mass (m)", style={"margin-top": "20px"}),
        dcc.Slider(id="mass", min=0.5, max=5, step=0.1, value=1),
        html.Label("Damping (c)", style={"margin-top": "20px"}),
        dcc.Slider(id="damping", min=0.1, max=2, step=0.1, value=0.3),
        html.Label("Stiffness (k)", style={"margin-top": "20px"}),
        dcc.Slider(id="stiffness", min=0.5, max=10, step=0.5, value=2.5),
    ], style={"width": "80%", "margin": "auto"}),
])


@app.callback(
    Output("displacement-graph", "figure"),
    [Input("mass", "value"), Input("damping", "value"), Input("stiffness", "value")]
)
def update_graph(m, c, k):
    t = np.linspace(0, 10, 1000)

    # True physics
    y_true = physics_model(m, c, k, t)

    # ML surrogate
    y_pred = request_ml_prediction(m, c, k)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y_true, mode="lines", name="Physics Model", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=t, y=y_pred, mode="lines", name="ML Surrogate"))
    
    fig.update_layout(
        title="Displacement over Time",
        xaxis_title="Time (s)",
        yaxis_title="Displacement (m)",
        template="plotly_white"
    )
    return fig


if __name__ == "__main__":
    app.run(debug=True)
