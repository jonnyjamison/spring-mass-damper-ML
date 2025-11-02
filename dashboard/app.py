import numpy as np
import requests
from dash import Dash, dcc, html, Output, Input
import plotly.graph_objs as go
from scipy.integrate import odeint

# URL of FastAPI ML model (running in Docker)
API_URL = "http://localhost:8080/predict"


def physics_model(m, c, k, t):
    """Analytical solution for damped oscillator using ODE solver."""
    def equation(x, _):
        return [x[1], -(c / m) * x[1] - (k / m) * x[0]]

    x0 = [1.0, 0.0]  # Initial displacement and velocity
    sol = odeint(equation, x0, t)
    return sol[:, 0]


def request_ml_prediction(m, c, k):
    """Request prediction from FastAPI endpoint."""
    response = requests.post(API_URL, json={"m": m, "c": c, "k": k})
    if response.status_code != 200:
        raise RuntimeError(f"API error: {response.status_code}")
    return response.json()["displacement"]


# Initialize Dash app
app = Dash(__name__)
app.title = "Spring–Mass–Damper Simulator"


# Custom label + slider component
def labeled_slider(label, slider_id, min_v, max_v, step_v, value_v):
    return html.Div(
        [
            html.Label(label, style={"font-weight": "bold", "margin-bottom": "-15px"}),
            dcc.Slider(
                id=slider_id,
                min=min_v,
                max=max_v,
                step=step_v,
                value=value_v,
                tooltip={"placement": "bottom", "always_visible": True},
                marks=None,
            ),
        ],
        style={"margin-bottom": "25px"},
    )


# Main layout
app.layout = html.Div(
    [
        html.H1(
            "Spring–Mass–Damper Simulation Dashboard",
            style={
                "text-align": "center",
                "color": "#2c3e50",
                "margin-top": "10px",
                "font-family": "Arial, sans-serif",
            },
        ),
        html.P(
            "Physics model vs. Machine Learning model",
            style={"text-align": "center", "font-style": "italic", "margin-bottom": "30px"},
        ),
        html.Div(
            [
                html.H4("Model Parameters", style={"text-align": "center"}),
                labeled_slider("Mass (m)", "mass", 0.5, 5, 0.1, 1),
                labeled_slider("Damping (c)", "damping", 0.1, 2, 0.1, 0.3),
                labeled_slider("Stiffness (k)", "stiffness", 0.5, 10, 0.5, 2.5),
            ],
            style={
                "padding": "25px 15px",
                "border": "1px solid #e1e1e1",
                "border-radius": "8px",
                "background-color": "#f9f9f9",
                "margin": "20px auto",
                "width": "60%",
                "box-shadow": "2px 2px 12px rgba(0, 0, 0, 0.08)",
            },
        ),
        html.Div(
            dcc.Graph(id="displacement-graph"),
            style={"margin-top": "20px"},
        ),
        html.Footer(
            [
                "© 2025 Jonny Jamison | ",
                html.A(
                    "View on GitHub",
                    href="https://github.com/jonnyjamison/spring-mass-damper-ML",
                    target="_blank",
                    style={"color": "#2980b9", "text-decoration": "none"},
                ),
            ],
            style={
                "text-align": "center",
                "margin-top": "40px",
                "color": "#7f8c8d",
                "font-size": "14px",
            },
        ),
    ]
)


@app.callback(
    Output("displacement-graph", "figure"),
    [Input("mass", "value"), Input("damping", "value"), Input("stiffness", "value")],
)
def update_displacement_graph(m, c, k):
    t = np.linspace(0, 10, 1000)
    y_true = physics_model(m, c, k, t)
    y_pred = request_ml_prediction(m, c, k)

    # Displacement graph
    fig_disp = go.Figure()
    fig_disp.add_trace(
        go.Scatter(x=t, y=y_true, mode="lines", name="Physics Model", line=dict(dash="dash"))
    )
    fig_disp.add_trace(
        go.Scatter(x=t, y=y_pred, mode="lines", name="ML Surrogate", line=dict(width=2))
    )
    fig_disp.update_layout(
        title="Displacement over Time",
        xaxis_title="Time (s)",
        yaxis_title="Displacement (m)",
        template="plotly_white",
        legend=dict(x=0.7, y=1.1),
    )

    return fig_disp


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=True)
