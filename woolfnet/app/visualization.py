"""Plotly figure builders for the embedding-exploration UI."""

import numpy as np
import plotly.graph_objects as go

from woolfnet.analysis import display_token

DEFAULT_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
    "#7f7f7f",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#9edae5",
]

QUERY_COLOR = "gold"
NEIGHBOR_COLOR = "rgb(255,140,0)"


def build_embedding_plot(
    projection: np.ndarray,
    tokens: list[str],
    cluster_ids: np.ndarray | None = None,
    membership_strengths: np.ndarray | None = None,
    query_pieces: list[str] = (),
    neighbor_tokens: list[str] = (),
    title: str = "",
    leading_space: str = "",
    palette: list[str] | None = None,
) -> go.Figure:
    """2D or 3D scatter of a UMAP projection, with optional cluster colouring and query overlays.

    When ``membership_strengths`` is given (1-D float array, same length as ``tokens``),
    per-cluster points use it as per-point opacity, clamped to ``[0.2, 1.0]`` for visibility."""
    n_dims = projection.shape[1]
    if n_dims not in (2, 3):
        raise ValueError(f"projection must be 2D or 3D, got shape {projection.shape}")

    palette = palette or DEFAULT_PALETTE
    labels = np.array([display_token(t, leading_space=leading_space) for t in tokens])
    tokens_arr = np.array(tokens)

    query_mask = np.isin(tokens_arr, list(query_pieces))
    neighbor_mask = np.isin(tokens_arr, list(neighbor_tokens)) & ~query_mask
    background_mask = ~query_mask & ~neighbor_mask

    point_opacity: np.ndarray | None = None
    if membership_strengths is not None:
        point_opacity = 0.2 + 0.8 * np.clip(np.asarray(membership_strengths), 0.0, 1.0)

    fig = go.Figure()

    if cluster_ids is None:
        _scatter(
            fig,
            projection[background_mask],
            labels[background_mask],
            n_dims,
            color="rgba(120,120,120,0.45)",
            size=4,
            opacity=1.0,
            name="vocabulary",
            showlegend=False,
        )
    else:
        noise_mask = background_mask & (cluster_ids == -1)
        if noise_mask.any():
            _scatter(
                fig,
                projection[noise_mask],
                labels[noise_mask],
                n_dims,
                color="rgba(120,120,120,0.35)",
                size=3,
                opacity=1.0,
                name="noise",
                showlegend=False,
            )
        for cluster_id in np.unique(cluster_ids):
            if cluster_id == -1:
                continue
            cluster_mask = background_mask & (cluster_ids == cluster_id)
            if not cluster_mask.any():
                continue
            opacity = point_opacity[cluster_mask] if point_opacity is not None else 0.65
            _scatter(
                fig,
                projection[cluster_mask],
                labels[cluster_mask],
                n_dims,
                color=palette[int(cluster_id) % len(palette)],
                size=4 if n_dims == 2 else 3,
                opacity=opacity,
                name=f"cluster {int(cluster_id)}",
                outline=True,
            )

    if neighbor_mask.any():
        _scatter(
            fig,
            projection[neighbor_mask],
            labels[neighbor_mask],
            n_dims,
            color=NEIGHBOR_COLOR,
            size=11 if n_dims == 2 else 6,
            opacity=1.0,
            symbol="diamond",
            name="neighbors",
            with_text=True,
            outline=True,
        )
    if query_mask.any():
        _scatter(
            fig,
            projection[query_mask],
            labels[query_mask],
            n_dims,
            color=QUERY_COLOR,
            size=16 if n_dims == 2 else 9,
            opacity=1.0,
            symbol="star" if n_dims == 2 else "diamond",
            name="query",
            with_text=True,
            outline=True,
        )

    _apply_layout(fig, title, n_dims)
    return fig


def build_drift_plot(
    projection_base: np.ndarray,
    projection_finetuned: np.ndarray,
    tokens: list[str],
    drift_magnitudes: np.ndarray,
    top_n_arrows: int = 30,
    title: str = "",
    leading_space: str = "",
) -> go.Figure:
    """2D drift plot for a base/fine-tuned pair, coloured by per-token drift magnitude."""
    if projection_base.shape != projection_finetuned.shape:
        raise ValueError("base and fine-tuned projections must have the same shape")
    if projection_base.shape[1] != 2:
        raise NotImplementedError("drift plot is 2D only for now")

    labels = [display_token(t, leading_space=leading_space) for t in tokens]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=projection_base[:, 0],
            y=projection_base[:, 1],
            mode="markers",
            marker=dict(
                size=4, color=drift_magnitudes, colorscale="Hot", opacity=0.45, showscale=False
            ),
            hovertext=labels,
            hoverinfo="text",
            name="base",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=projection_finetuned[:, 0],
            y=projection_finetuned[:, 1],
            mode="markers",
            marker=dict(
                size=5,
                color=drift_magnitudes,
                colorscale="Hot",
                opacity=0.85,
                colorbar=dict(title="drift"),
                showscale=True,
            ),
            hovertext=labels,
            hoverinfo="text",
            name="fine-tuned",
        )
    )
    for i in np.argsort(drift_magnitudes)[::-1][:top_n_arrows]:
        fig.add_annotation(
            ax=projection_base[i, 0],
            ay=projection_base[i, 1],
            x=projection_finetuned[i, 0],
            y=projection_finetuned[i, 1],
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            text=labels[i],
            font=dict(size=10, color="black"),
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="rgba(30,30,30,0.7)",
            standoff=2,
        )
    _apply_layout(fig, title, n_dims=2)
    return fig


def _scatter(
    fig: go.Figure,
    coords: np.ndarray,
    text: np.ndarray,
    n_dims: int,
    color: str,
    size: int,
    opacity: float | np.ndarray,
    name: str,
    symbol: str = "circle",
    with_text: bool = False,
    outline: bool = False,
    showlegend: bool = True,
) -> None:
    marker = dict(size=size, color=color, opacity=opacity, symbol=symbol)
    if outline:
        marker["line"] = dict(
            width=1 if with_text else 0.4,
            color="black" if with_text else "white",
        )
    common = dict(
        mode="markers+text" if with_text else "markers",
        marker=marker,
        text=list(text) if with_text else None,
        textposition="top center" if with_text else None,
        textfont=dict(size=12, color="black") if with_text else None,
        hovertext=list(text),
        hoverinfo="text",
        name=name,
        showlegend=showlegend,
    )
    if n_dims == 2:
        fig.add_trace(go.Scatter(x=coords[:, 0], y=coords[:, 1], **common))
    else:
        fig.add_trace(go.Scatter3d(x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], **common))


def _apply_layout(fig: go.Figure, title: str, n_dims: int) -> None:
    axis_off = dict(showgrid=False, zeroline=False, showticklabels=False, title=None)
    common = dict(
        title=dict(text=title, font=dict(size=15)) if title else None,
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        height=700,
        margin=dict(l=20, r=20, t=60 if title else 20, b=20),
        legend=dict(itemsizing="constant", font=dict(size=11)),
    )
    if n_dims == 2:
        fig.update_layout(xaxis=axis_off, yaxis=axis_off, **common)
    else:
        fig.update_layout(scene=dict(xaxis=axis_off, yaxis=axis_off, zaxis=axis_off), **common)
