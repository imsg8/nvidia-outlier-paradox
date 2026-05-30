import matplotlib.pyplot as plt
import matplotlib as mpl

# ─── Consistent Color Palette ─────────────────────────────────────────
COLORS = {
    "train":      "#1f77b4",   # blue
    "test":       "#ff7f0e",   # orange
    "predicted":  "#2ca02c",   # green
    "val":        "#d62728",   # red
    "residual":   "#9467bd",   # purple
    "zero_line":  "#e74c3c",   # bright red
    "highlight":  "#17becf",   # cyan
    "fill":       "#aec7e8",   # light blue
}


def set_plot_style():
    """
    Standardizes matplotlib plot styles for research paper quality:
    - Sets figure size and DPI
    - Sets font sizes for titles, labels, ticks, legends
    - Enables grid with subtle alpha
    - Sets consistent line widths
    - Uses tight layout automatically
    """
    plt.rcParams['figure.figsize'] = (16, 10)
    plt.rcParams['figure.dpi'] = 300

    # Title and label fonts
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 16

    # Tick fonts
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    # Legend font
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['legend.framealpha'] = 0.9
    plt.rcParams['legend.edgecolor'] = '#cccccc'

    # Line width and grid
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.color'] = '#cccccc'

    # Tight layout automatically
    plt.rcParams['figure.autolayout'] = True

    # Spine styling
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

    print("Plot style standardized successfully!")


def add_metric_box(ax, metrics_dict, loc="upper left"):
    """
    Add a translucent text box with key metrics to a plot axis.

    Args:
        ax: matplotlib Axes object
        metrics_dict: dict like {"R2": 0.996, "RMSE": 10.75}
        loc: location string - "upper left", "upper right", "lower left", "lower right"
    """
    text = "\n".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                      for k, v in metrics_dict.items()])

    # Map location string to coordinates
    loc_map = {
        "upper left":  (0.02, 0.98, "top", "left"),
        "upper right": (0.98, 0.98, "top", "right"),
        "lower left":  (0.02, 0.02, "bottom", "left"),
        "lower right": (0.98, 0.02, "bottom", "right"),
    }
    x, y, va, ha = loc_map.get(loc, loc_map["upper left"])

    ax.text(
        x, y, text,
        transform=ax.transAxes,
        fontsize=11, verticalalignment=va, horizontalalignment=ha,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.85),
        family="monospace",
    )
