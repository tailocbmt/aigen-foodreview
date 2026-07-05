import plotly.graph_objects as go


def add_pareto_line(fig, xs, ys):
    """Connects all target models sequentially from 28M to 83M to 138M."""
    # Sort together by parameter size (x) to ensure chronological line rendering
    pts = sorted(zip(xs, ys), key=lambda p: p[0])
    fx = [p[0] for p in pts]
    fy = [p[1] for p in pts]

    fig.add_trace(go.Scatter(
        x=fx, y=fy, mode="lines",
        line=dict(color="rgba(128,128,128,0.5)", width=1.5, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ))


def save_fig(fig, basename, caption):
    """Save figure as HTML (always) and optionally as PNG via plotly+matplotlib."""
    # Always save interactive HTML — works everywhere
    fig.write_html(f"{basename}.html", include_plotlyjs="cdn")
    print(f"{basename}.html")


def _save_via_matplotlib(fig, basename):
    """Quick-and-dirty rasterization using matplotlib."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # Map Plotly symbols → matplotlib markers
    PLOTLY_TO_MPL_MARKER = {
        "circle": "o", "circle-open": "o",
        "square": "s", "square-open": "s",
        "diamond": "D", "diamond-open": "D",
        "cross": "P", "x": "X",
        "triangle-up": "^", "triangle-down": "v",
        "star": "*", "star-open": "*", "star-square-open": "*",
        "star-triangle-up": "*",
        "hexagram": "h", "pentagon": "p",
        "bowtie": "d",  # fallback shape
    }

    mpl_fig, ax = plt.subplots(figsize=(12, 7))
    for trace in fig.data:
        if hasattr(trace, 'mode') and trace.mode and "markers" in trace.mode:
            # Extract color and symbol from the trace
            color = None
            mpl_marker = "o"
            size = 50
            if hasattr(trace, 'marker') and trace.marker:
                color = trace.marker.color
                if trace.marker.symbol:
                    sym = trace.marker.symbol
                    if isinstance(sym, (list, tuple)):
                        sym = sym[0]
                    mpl_marker = PLOTLY_TO_MPL_MARKER.get(sym, "o")
                if trace.marker.size:
                    s = trace.marker.size
                    if isinstance(s, (list, tuple)):
                        s = max(s)
                    size = s ** 2 * 0.5  # scale for matplotlib

            facecolor = color
            edgecolor = color
            # Handle "open" symbols
            if hasattr(trace, 'marker') and trace.marker and trace.marker.symbol:
                sym = trace.marker.symbol
                if isinstance(sym, str) and "open" in sym:
                    facecolor = "none"

            ax.scatter(trace.x, trace.y, label=trace.name,
                       s=size, alpha=0.85, marker=mpl_marker,
                       c=facecolor if facecolor != "none" else None,
                       facecolors=facecolor if facecolor == "none" else None,
                       edgecolors=edgecolor)

            # Map annotations to Matplotlib fallback
            for x, y, txt in zip(trace.x, trace.y, trace.text):
                ax.annotate(txt, (x, y), textcoords="offset points",
                            xytext=(0, 10), ha='center', fontsize=8)

        elif hasattr(trace, 'mode') and trace.mode and "lines" in trace.mode:
            ax.plot(trace.x, trace.y, linestyle=":",
                    color="gray", alpha=0.6, linewidth=1.5)
        elif hasattr(trace, "type") and trace.type == "bar":
            ax.bar(trace.x, trace.y, label=trace.name, alpha=0.8,
                   color=trace.marker.color if trace.marker else None)

    if fig.layout.xaxis.type == "log":
        ax.set_xscale("log")
    ax.set_xlabel(fig.layout.xaxis.title.text or "")
    ax.set_ylabel(fig.layout.yaxis.title.text or "")
    if fig.layout.yaxis.range:
        ax.set_ylim(fig.layout.yaxis.range)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(fig.layout.title.text.split("<br>")[
                 0] if fig.layout.title.text else "")
    ax.grid(True, which="both", alpha=0.4)
    mpl_fig.tight_layout()
    mpl_fig.savefig(f"{basename}.pdf", dpi=150)
    plt.close(mpl_fig)


# ── model database ─────────────────────────────────────────────────────────────
# EVONS
NAME = "Improved EVONS"
MODELS = [
    ("ResNet-18 + TinyBERT",   "ResNet-18 + TinyBERT",  28.0,
     91.09, 95.50, 96.56, 97.52, 96.31, "Table Data"),
    ("ResNet-18 + DistilBERT", "ResNet-18 + DistilBERT",  83.0,
     59.10, 79.22, 84.68, 78.07, 81.07, "Table Data"),
    ("ResNet-50 + BERT-base",  "ResNet-50 + BERT-base", 138.6,
     92.88, 96.42, 95.15, 97.51, 97.04, "Table Data"),
]
# AIGEN FOODREVIEW
# NAME = "AiGen-FoodReview"
# MODELS = [
#     ("ResNet-18 + TinyBERT",   "ResNet-18 + TinyBERT",  28.0,
#      67.43, 0.0, 78.01, 48.80, 49.25, "Table Data"),
#     ("ResNet-18 + DistilBERT", "ResNet-18 + DistilBERT",  83.0,
#      50.37, 0.0, 62.65, 51.12, 35.87, "Table Data"),
#     ("ResNet-50 + BERT-base",  "ResNet-50 + BERT-base", 138.6,
#      91.79, 0.0, 90.46, 93.69, 92.04, "Table Data"),
# ]

FAMILY_COLOR = {
    "ResNet-18 + TinyBERT": "#1f77b4",
    "ResNet-18 + DistilBERT": "#37860c",
    "ResNet-50 + BERT-base": "#f00707"
}

MARKER_SYMBOL = {
    "ResNet-18 + TinyBERT": "circle",
    "ResNet-18 + DistilBERT": "square",
    "ResNet-50 + BERT-base": "star"
}

LEGEND = dict(
    yanchor="bottom", y=0.01,
    xanchor="right", x=0.99,
    bgcolor="rgba(255,255,255,0.8)"
)

TARGET_METRIC = "macro_f1"
METRIC_LABELS = {
    "subset_acc": "Subset Acc. (%)",
    "multilabel_acc": "Multi-label Acc. (%)",
    "precision": "Precision (%)",
    "recall": "Recall (%)",
    "macro_f1": "Macro-F1 (%)"
}

data_b = []
for item in MODELS:
    name, family, params = item[0], item[1], item[2]
    metrics = {
        "subset_acc": item[3],
        "multilabel_acc": item[4],
        "precision": item[5],
        "recall": item[6],
        "macro_f1": item[7]
    }
    src = item[8]

    y_val = metrics[TARGET_METRIC]
    if params is not None and y_val is not None:
        data_b.append((name, family, params, y_val, metrics, src))

fig1 = go.Figure()

for fam in sorted(set(f for _, f, *_ in data_b)):
    pts = [item for item in data_b if item[1] == fam]
    col = FAMILY_COLOR.get(fam, "#AAAAAA")
    sym = MARKER_SYMBOL.get(fam, "circle")

    fig1.add_trace(go.Scatter(
        x=[item[2] for item in pts],
        y=[item[3] for item in pts],
        mode="markers+text",
        text=[item[0] for item in pts],
        textposition="top center",
        name=fam,
        marker=dict(color=col, symbol=sym, size=14,
                    line=dict(width=1.5, color="white")),
        customdata=[[
            item[0],
            item[4]["subset_acc"],
            item[4]["multilabel_acc"],
            item[4]["precision"],
            item[4]["recall"],
            item[4]["macro_f1"],
            item[5]
        ] for item in pts],
        hovertemplate=(
            "<b>%{customdata[0]}</b><br><br>"
            "Params: %{x:.1f}M<br>"
            "Subset Acc: %{customdata[1]:.2f}%<br>"
            "Multi-label Acc: %{customdata[2]:.2f}%<br>"
            "Precision: %{customdata[3]:.2f}%<br>"
            "Recall: %{customdata[4]:.2f}%<br>"
            "Macro-F1: %{customdata[5]:.2f}%<br><br>"
            "Source: %{customdata[6]}<extra></extra>"
        ),
    ))

# Sequential path tracking connection call
add_pareto_line(fig1,
                [item[2] for item in data_b],
                [item[3] for item in data_b])

fig1.update_xaxes(title_text="Params (M)", type="log",
                  tickvals=[20, 30, 50, 80, 100, 150],
                  ticktext=["20", "30", "50", "80", "100", "150"])

fig1.update_yaxes(title_text=METRIC_LABELS[TARGET_METRIC], range=[70, 101])

fig1.update_layout(
    title=dict(text=(
        f"{NAME} - {METRIC_LABELS[TARGET_METRIC]} vs Parameters (M)"
        "<br><span style='font-size:14px;font-weight:normal;color:gray;'>"
        "Dotted Line = Parameter Sequence Path Trace"
        "</span>"
    )),
    legend=LEGEND,
    template="plotly_white"
)

fig1.update_traces(cliponaxis=False)

# Render files
save_fig(fig1, "pareto_acc_params", "Model Evaluation Step Profile")
_save_via_matplotlib(fig1, "pareto_acc_params")
