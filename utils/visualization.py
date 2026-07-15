"""
utils/visualization.py - Plotly chart builders for the dashboard
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


# ─────────────── COLOUR PALETTE ───────────────
PINK  = "#E91E8C"
BLUE  = "#3D5AF1"
CYAN  = "#00BCD4"
GREEN = "#43A047"
RED   = "#E53935"
AMBER = "#FB8C00"
PURPLE= "#7B1FA2"
BG    = "#0E1117"
CARD  = "#1A1D27"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG,
    plot_bgcolor=CARD,
    font=dict(color="#FFFFFF", family="Inter, sans-serif"),
    margin=dict(l=20, r=20, t=40, b=20),
)


def pie_chart(labels: list, values: list, title: str, colors: list = None) -> go.Figure:
    """Donut-style pie chart."""
    if colors is None:
        colors = [PINK, BLUE, CYAN, GREEN, AMBER, PURPLE]
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors[:len(labels)], line=dict(color=BG, width=2)),
        textinfo="label+percent",
        textfont=dict(size=13),
        hoverinfo="label+value+percent",
    ))
    fig.update_layout(title=dict(text=title, font=dict(size=16, color=PINK)), **PLOTLY_LAYOUT)
    return fig


def bar_chart(x: list, y: list, title: str, color: str = PINK, orientation: str = "v") -> go.Figure:
    """Gradient bar chart."""
    if orientation == "v":
        fig = go.Figure(go.Bar(x=x, y=y, marker_color=color,
                                marker_line=dict(width=0),
                                text=y, textposition="outside"))
    else:
        fig = go.Figure(go.Bar(x=y, y=x, orientation="h",
                                marker_color=color,
                                text=y, textposition="outside"))
    fig.update_layout(title=dict(text=title, font=dict(size=16, color=PINK)),
                      xaxis=dict(gridcolor="#2A2D3E"),
                      yaxis=dict(gridcolor="#2A2D3E"),
                      **PLOTLY_LAYOUT)
    return fig


def timeline_chart(df: pd.DataFrame) -> go.Figure:
    """Line chart showing prediction count over time."""
    if df.empty:
        return go.Figure()
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    trend = df.groupby("date").size().reset_index(name="count")

    fig = go.Figure(go.Scatter(
        x=trend["date"], y=trend["count"],
        mode="lines+markers",
        line=dict(color=PINK, width=3),
        marker=dict(size=8, color=CYAN, line=dict(color=PINK, width=2)),
        fill="tozeroy",
        fillcolor=f"rgba(233,30,140,0.12)",
    ))
    fig.update_layout(
        title=dict(text="📈 Prediction Trend Over Time", font=dict(size=16, color=PINK)),
        xaxis=dict(gridcolor="#2A2D3E"),
        yaxis=dict(gridcolor="#2A2D3E"),
        **PLOTLY_LAYOUT
    )
    return fig


def module_bar_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart of predictions by module."""
    if df.empty:
        return go.Figure()
    counts = df["module"].value_counts()
    colors = [PINK, BLUE, CYAN]
    fig = go.Figure(go.Bar(
        x=counts.index.tolist(),
        y=counts.values.tolist(),
        marker_color=colors[:len(counts)],
        text=counts.values.tolist(),
        textposition="outside"
    ))
    fig.update_layout(
        title=dict(text="📊 Predictions by Module", font=dict(size=16, color=PINK)),
        xaxis=dict(gridcolor="#2A2D3E"),
        yaxis=dict(gridcolor="#2A2D3E"),
        **PLOTLY_LAYOUT
    )
    return fig


def gauge_chart(value: float, title: str = "Confidence") -> go.Figure:
    """Gauge/speedometer chart for confidence score."""
    color = RED if value >= 0.75 else (AMBER if value >= 0.50 else GREEN)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(value * 100, 1),
        number=dict(suffix="%", font=dict(size=32, color=color)),
        title=dict(text=title, font=dict(size=16, color="#FFFFFF")),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#555"),
            bar=dict(color=color),
            bgcolor="#1A1D27",
            bordercolor="#2A2D3E",
            steps=[
                dict(range=[0, 50], color="#1E2732"),
                dict(range=[50, 75], color="#1A2535"),
                dict(range=[75, 100], color="#2A1520"),
            ],
            threshold=dict(line=dict(color=RED, width=4), thickness=0.75, value=75),
        ),
    ))
    fig.update_layout(paper_bgcolor=BG, font=dict(color="#FFFFFF"),
                      margin=dict(l=20, r=20, t=50, b=20))
    return fig


def confusion_matrix_chart(cm: list, labels: list) -> go.Figure:
    """Heatmap for confusion matrix."""
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=[[0, "#1A1D27"], [1, PINK]],
        text=cm, texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(
        title=dict(text="Confusion Matrix", font=dict(size=16, color=PINK)),
        xaxis=dict(title="Predicted"),
        yaxis=dict(title="Actual"),
        **PLOTLY_LAYOUT
    )
    return fig


def metric_card_html(label: str, value, icon: str = "📊", color: str = "#E91E8C") -> str:
    """Return HTML for a metric card."""
    return f"""
    <div style='background:linear-gradient(135deg,#1A1D27,#0E1117);
                border:1px solid {color}44; border-radius:16px; padding:20px 24px;
                text-align:center; box-shadow:0 4px 20px {color}22; margin:6px 0;'>
        <div style='font-size:2rem;'>{icon}</div>
        <div style='font-size:2.2rem; font-weight:800; color:{color};'>{value}</div>
        <div style='color:#aaa; font-size:0.9rem; margin-top:4px;'>{label}</div>
    </div>"""
