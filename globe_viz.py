"""
globe_viz.py — Interactive 3D globe visualisation

Renders magnetic residuals, detected anomaly clusters, orbital groundtrack,
and priority downlink flags on an interactive Plotly globe.

Output: self-contained HTML file (no server required).
"""

import os
import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("[globe_viz] WARNING: plotly not installed. Globe rendering skipped.")


# ── Colour palette ────────────────────────────────────────────────────────

PRIORITY_COLOURS = {
    "CRITICAL": "#FF2D55",
    "HIGH":     "#FF9F0A",
    "MEDIUM":   "#30D158",
    "NOMINAL":  "#636366",
}


def _residual_colorscale():
    return [
        [0.0,  "#0a1628"],
        [0.2,  "#0d3b6e"],
        [0.4,  "#1a6aad"],
        [0.5,  "#e8e8e8"],
        [0.6,  "#c0392b"],
        [0.8,  "#7b0000"],
        [1.0,  "#3d0000"],
    ]


def render_globe(swarm_df: pd.DataFrame,
                 anomaly_df: pd.DataFrame,
                 decisions_df: pd.DataFrame,
                 output_path: str = "output/globe.html"):
    """
    Build and save an interactive Plotly globe figure.

    Layers:
      1. Orbital groundtrack (thin line)
      2. Residual field colour map (scatter_geo)
      3. Anomaly cluster markers (sized by score)
      4. Priority downlink flags (star markers)
      5. Known anomaly catalog (reference rings)
    """
    if not PLOTLY_AVAILABLE:
        print("[globe_viz] Plotly not available — skipping render.")
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig = go.Figure()

    # ── Layer 1: Orbital groundtrack ─────────────────────────────
    fig.add_trace(go.Scattergeo(
        lat       = swarm_df["lat"],
        lon       = swarm_df["lon"],
        mode      = "lines",
        line      = dict(width=1.5, color="rgba(100,180,255,0.35)"),
        name      = "Swarm Groundtrack",
        hoverinfo = "skip",
    ))

    # ── Layer 2: Residual field heatmap ──────────────────────────
    r_norm  = swarm_df["residual"]
    r_abs   = r_norm.abs().max()
    r_clamp = r_norm.clip(-r_abs, r_abs)

    fig.add_trace(go.Scattergeo(
        lat    = swarm_df["lat"],
        lon    = swarm_df["lon"],
        mode   = "markers",
        marker = dict(
            size          = 6,
            color         = r_clamp,
            colorscale    = _residual_colorscale(),
            cmin          = -r_abs,
            cmax          =  r_abs,
            colorbar      = dict(
                title     = "Residual (nT)",
                thickness = 18,
                len       = 0.55,
                x         = 1.01,
                tickfont  = dict(color="white", size=11),
                titlefont = dict(color="white", size=12),
            ),
            opacity       = 0.85,
        ),
        text      = [f"Residual: {v:.1f} nT<br>Lat: {la:.2f}° Lon: {lo:.2f}°"
                     for v, la, lo in zip(r_clamp, swarm_df["lat"], swarm_df["lon"])],
        hoverinfo = "text",
        name      = "Field Residual",
    ))

    # ── Layer 3: Anomaly clusters ─────────────────────────────────
    if not anomaly_df.empty:
        fig.add_trace(go.Scattergeo(
            lat    = anomaly_df["lat"],
            lon    = anomaly_df["lon"],
            mode   = "markers",
            marker = dict(
                size    = (anomaly_df["score"] * 18).clip(6, 18),
                color   = anomaly_df["score"],
                colorscale = "Plasma",
                cmin    = 0.4,
                cmax    = 1.0,
                symbol  = "circle",
                line    = dict(color="white", width=0.5),
                opacity = 0.9,
            ),
            text      = [f"Anomaly Cluster {int(c)}<br>Score: {s:.2f}<br>"
                          f"Residual: {r:.1f} nT"
                         for c, s, r in zip(
                             anomaly_df["cluster"],
                             anomaly_df["score"],
                             anomaly_df["residual"])],
            hoverinfo = "text",
            name      = "Anomaly Clusters",
        ))

    # ── Layer 4: Priority downlink flags ─────────────────────────
    if not decisions_df.empty:
        priority_flagged = decisions_df[decisions_df["downlink_priority"]]
        for _, row in priority_flagged.iterrows():
            colour = PRIORITY_COLOURS.get(row["priority_label"], "#FFFFFF")
            fig.add_trace(go.Scattergeo(
                lat       = [row["center_lat"]],
                lon       = [row["center_lon"]],
                mode      = "markers+text",
                marker    = dict(
                    size    = 20,
                    color   = colour,
                    symbol  = "star",
                    line    = dict(color="white", width=1.5),
                ),
                text      = [row["priority_label"]],
                textfont  = dict(color=colour, size=9),
                textposition = "top center",
                hovertext = (f"<b>{row['priority_label']}</b><br>"
                             f"Cluster {int(row['cluster_id'])}<br>"
                             f"Peak: {row['peak_residual_nT']:.1f} nT<br>"
                             f"Score: {row['anomaly_score']:.2f}<br>"
                             f"Near: {row['nearest_catalog']}<br>"
                             f"{row['rationale'][:80]}"),
                hoverinfo = "text",
                name      = f"{row['priority_label']} flag",
                showlegend= True,
            ))

    # ── Layout ────────────────────────────────────────────────────
    fig.update_geos(
        projection_type       = "natural earth",
        showland              = True,
        landcolor             = "#1a1f2e",
        showocean             = True,
        oceancolor            = "#0d1520",
        showlakes             = True,
        lakecolor             = "#0d1520",
        showcountries         = True,
        countrycolor          = "rgba(255,255,255,0.2)",
        showcoastlines        = True,
        coastlinecolor        = "rgba(255,255,255,0.4)",
        showframe             = False,
        bgcolor               = "#0a0e1a",
        lataxis_showgrid      = True,
        lonaxis_showgrid      = True,
        lataxis_gridcolor     = "rgba(255,255,255,0.08)",
        lonaxis_gridcolor     = "rgba(255,255,255,0.08)",
        lataxis_range         = [-90, 90],
        lonaxis_range         = [-180, 180],
    )

    # ── Layer 5: Known anomaly reference rings ────────────────────
    from onboard_logic import KNOWN_ANOMALIES
    for _, row in KNOWN_ANOMALIES.iterrows():
        # Draw a small reference circle
        angles = np.linspace(0, 2*np.pi, 40)
        ring_lat = row["lat"] + row["radius_deg"] * np.sin(angles) * 0.6
        ring_lon = row["lon"] + row["radius_deg"] * np.cos(angles)
        fig.add_trace(go.Scattergeo(
            lat       = ring_lat.tolist() + [ring_lat[0]],
            lon       = ring_lon.tolist() + [ring_lon[0]],
            mode      = "lines",
            line      = dict(color="rgba(255,200,50,0.25)", width=1, dash="dot"),
            name      = f"Known: {row['name']}",
            hoverinfo = "name",
            showlegend= False,
        ))
        fig.add_trace(go.Scattergeo(
            lat       = [row["lat"]],
            lon       = [row["lon"]],
            mode      = "text",
            text      = [row["name"]],
            textfont  = dict(color="rgba(255,200,50,0.5)", size=8),
            hoverinfo = "skip",
            showlegend= False,
        ))

    fig.update_layout(
        title = dict(
            text    = ("QOMAD — Quantum-Inspired Orbital Magnetic Anomaly Detector<br>"
                       "<sup>ESA Swarm Pass | CHAOS-7 Core Field Subtracted | "
                       "NV-Centre Quantum Filter Applied</sup>"),
            x       = 0.5,
            font    = dict(color="white", size=15),
        ),
        paper_bgcolor = "#0a0e1a",
        plot_bgcolor  = "#0a0e1a",
        font          = dict(color="white"),
        margin        = dict(l=0, r=0, t=80, b=0),
        legend        = dict(
            bgcolor     = "rgba(10,14,26,0.8)",
            bordercolor = "rgba(255,255,255,0.2)",
            borderwidth = 1,
            font        = dict(color="white", size=10),
        ),
        height = 800,
        updatemenus = [dict(
            type       = "buttons",
            direction  = "right",
            x          = 0.5,
            xanchor    = "center",
            y          = -0.02,
            yanchor    = "top",
            pad        = dict(r=10, t=10),
            bgcolor    = "rgba(20,30,50,0.9)",
            bordercolor= "rgba(255,255,255,0.2)",
            font       = dict(color="white", size=11),
            buttons    = [
                dict(label="🌍 Globe",
                     method="relayout",
                     args=[{"geo.projection.type": "orthographic"}]),
                dict(label="🗺 Natural Earth",
                     method="relayout",
                     args=[{"geo.projection.type": "natural earth"}]),
                dict(label="📐 Mercator",
                     method="relayout",
                     args=[{"geo.projection.type": "mercator"}]),
                dict(label="⬆ North Polar",
                     method="relayout",
                     args=[{"geo.projection.type": "azimuthal equal area",
                            "geo.projection.rotation.lat": 90}]),
            ]
        )],
    )

    # ── Annotation: sensitivity panel ────────────────────────────
    fig.add_annotation(
        x=0.01, y=0.01, xref="paper", yref="paper",
        text=("🛰 <b>Swarm Alpha</b> | ⚛ NV-Centre Magnetometer<br>"
              "Sensitivity: <b>~1 fT/√Hz</b> vs 100 pT/√Hz classical<br>"
              "Anomaly clusters shown by ML score (Isolation Forest)"),
        align="left",
        showarrow=False,
        font=dict(color="rgba(200,200,200,0.85)", size=10),
        bgcolor="rgba(10,14,26,0.7)",
        bordercolor="rgba(255,255,255,0.2)",
        borderwidth=1,
    )

    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    print(f"[globe_viz] Globe saved → {output_path}  ({os.path.getsize(output_path)//1024} KB)")