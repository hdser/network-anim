import sys
import argparse
import pathlib
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from tqdm import tqdm

# Color map for forks (consistent colors)
# Using Plotly's default qualitative sequence for consistency
# Updated based on user's provided script
FORK_COLOR_MAP = {fork: color for fork, color in zip(
    ['Pectra', 'Deneb', 'Capella', 'Bellatrix', 'Altair'],
    px.colors.qualitative.Plotly
)}

# Helper function to get color for a fork
def get_fork_color(fork_name):
    # Fallback for unknown forks - cycle through qualitative colors
    if fork_name not in FORK_COLOR_MAP:
        # Simple hash-based color assignment for potentially new forks
        # Using a different sequence for fallbacks
        color_palette = px.colors.qualitative.Alphabet
        hash_val = hash(fork_name)
        color_index = hash_val % len(color_palette)
        # print(f"Assigning fallback color {color_palette[color_index]} to new fork: {fork_name}")
        return color_palette[color_index]
    return FORK_COLOR_MAP.get(fork_name, px.colors.qualitative.Plotly[-1]) # Default if lookup fails after check

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the DataFrame: converts date, extracts hour."""
    if df.empty:
        return df

    # Ensure 'date' column exists and convert to datetime
    if "date" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'date' column.")
    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception as e:
        raise ValueError(f"Failed to parse 'date' column: {e}")

    # Ensure 'cnt' column exists and is numeric
    if "cnt" not in df.columns:
         raise ValueError("Input DataFrame must contain a 'cnt' column.")
    try:
        df["cnt"] = pd.to_numeric(df["cnt"])
    except Exception as e:
        raise ValueError(f"Failed to convert 'cnt' column to numeric: {e}")

    # Ensure 'fork' and 'country' columns exist
    if "fork" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'fork' column.")
    if "country" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'country' column.")

    # Extract hour (as string for categorical axis/animation frame)
    # Sort by date first to ensure correct hour sequence
    df = df.sort_values("date")
    df["hour"] = df["date"].dt.strftime('%Y-%m-%d %H:00') # Use full datetime for unique hours

    return df


def build_overall_forks_figure_updated(df: pd.DataFrame) -> go.Figure:
    """
    Create a stacked area chart showing fork distribution over time (hourly),
    with a toggle for absolute counts vs. fractions. Ensures distinct colors
    and removes background.
    """
    if df.empty:
        tqdm.write("[!] Input data is empty, cannot build overall forks figure.")
        return go.Figure()

    # Aggregate data: sum counts per hour and fork
    hourly_fork_counts = df.groupby(["hour", "fork"])["cnt"].sum().reset_index()

    # Pivot for stacked area chart (absolute counts)
    pivot_abs = hourly_fork_counts.pivot(index="hour", columns="fork", values="cnt").fillna(0)

    # Calculate fractions
    hourly_total = hourly_fork_counts.groupby("hour")["cnt"].sum()
    # Avoid division by zero if an hour has zero total counts
    hourly_total = hourly_total.replace(0, 1) # Treat 0 total as 1 for fraction calculation (0/1=0)
    hourly_fork_counts["total_cnt"] = hourly_fork_counts["hour"].map(hourly_total)
    hourly_fork_counts["fraction"] = (hourly_fork_counts["cnt"] / hourly_fork_counts["total_cnt"]) * 100
    hourly_fork_counts["fraction"] = hourly_fork_counts["fraction"].fillna(0) # Handle potential NaNs

    # Pivot for stacked area chart (fractions)
    pivot_frac = hourly_fork_counts.pivot(index="hour", columns="fork", values="fraction").fillna(0)

    # Ensure consistent fork order and handle potential new forks
    all_forks_data = df['fork'].unique()
    all_forks_pivot = pivot_abs.columns.tolist()
    all_forks = sorted(list(set(all_forks_data) | set(all_forks_pivot))) # Combine and sort

    pivot_abs = pivot_abs.reindex(columns=all_forks, fill_value=0)
    pivot_frac = pivot_frac.reindex(columns=all_forks, fill_value=0)

    # Create figure
    fig = go.Figure()
    n_forks = len(all_forks)

    # Add traces for absolute counts (initially visible)
    for fork in all_forks:
        fork_color = get_fork_color(fork)
        fig.add_trace(go.Scatter(
            x=pivot_abs.index,
            y=pivot_abs[fork],
            mode='lines',
            # Explicitly set line color AND fill color
            line=dict(width=0.5, color=fork_color),
            fillcolor=fork_color,
            stackgroup='one', # Define stack group for absolute counts
            name=f"{fork} (Count)",
            legendgroup="count", # Group legends
            legendgrouptitle_text="Absolute Count",
            hovertemplate=(
                f"<b>Hour:</b> %{{x}}<br>"
                f"<b>Fork:</b> {fork}<br>"
                f"<b>Count:</b> %{{y}}<extra></extra>"
            ),
            visible=True # Visible by default
        ))

    # Add traces for fractions (initially hidden)
    for fork in all_forks:
        fork_color = get_fork_color(fork)
        fig.add_trace(go.Scatter(
            x=pivot_frac.index,
            y=pivot_frac[fork],
            mode='lines',
            # Explicitly set line color AND fill color
            line=dict(width=0.5, color=fork_color),
            fillcolor=fork_color,
            stackgroup='two', # Different stack group for fractions
            name=f"{fork} (Fraction)",
            legendgroup="fraction", # Group legends
            legendgrouptitle_text="Fraction (%)",
            hovertemplate=(
                f"<b>Hour:</b> %{{x}}<br>"
                f"<b>Fork:</b> {fork}<br>"
                f"<b>Fraction:</b> %{{y:.2f}}%<extra></extra>"
            ),
            visible=False # Hidden by default
        ))

    # Configure layout and add toggle buttons
    fig.update_layout(
        title="Overall Fork Distribution Over Time (Hourly)",
        xaxis_title="Hour",
        yaxis_title="Node Count", # Initial Y-axis title
        hovermode="x unified",
        legend=dict(tracegroupgap=20), # Add gap between legend groups
        # --- Remove background ---
        plot_bgcolor='white', # Set plot area background to white
        paper_bgcolor='white', # Set paper background to white
        # -------------------------
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                active=0, # Corresponds to the first button ('Absolute Counts')
                x=0.5,
                xanchor="center",
                y=1.15,
                yanchor="top",
                buttons=list([
                    dict(label="Absolute Counts",
                         method="update",
                         args=[{"visible": [True] * n_forks + [False] * n_forks}, # Show count traces, hide fraction traces
                               {"yaxis.title": "Node Count", "yaxis.ticksuffix": None}]), # Update y-axis
                    dict(label="Fractions (%)",
                         method="update",
                         args=[{"visible": [False] * n_forks + [True] * n_forks}, # Hide count traces, show fraction traces
                               {"yaxis.title": "Fraction (%)", "yaxis.ticksuffix": "%"}]), # Update y-axis
                ]),
            )
        ]
    )
    # Set initial y-axis range slightly above 0 if using counts
    fig.update_yaxes(rangemode='tozero')

    return fig


def build_top_countries_figure_updated(df: pd.DataFrame, top_n: int = 10, frame_duration: int = 300) -> go.Figure:
    """
    Create an animated horizontal stacked bar chart showing fork distribution
    for the top N countries over time (hourly). Removes background.
    """
    if df.empty:
        tqdm.write("[!] Input data is empty, cannot build top countries figure.")
        return go.Figure()

    # Calculate total count per country to determine top N
    country_totals = df.groupby("country")["cnt"].sum().sort_values(ascending=False)

    # Identify top N countries and 'Others'
    if len(country_totals) > top_n:
        top_countries = country_totals.head(top_n).index.tolist()
        # Create 'country_display' column
        df['country_display'] = df['country'].apply(lambda x: x if x in top_countries else 'Others')
        # Define category order: Top N descending, then 'Others'
        category_order = top_countries[::-1] + ['Others'] # Reverse for plotting top-down
    else:
        # If fewer than top_n countries, show all
        top_countries = country_totals.index.tolist()
        df['country_display'] = df['country']
        category_order = top_countries[::-1] # Reverse for plotting top-down


    # Aggregate data per hour, country_display, and fork
    anim_data = df.groupby(["hour", "country_display", "fork"])["cnt"].sum().reset_index()

    # Ensure all combinations of hour, country_display, fork exist for smooth animation
    all_hours = sorted(df['hour'].unique()) # Ensure hours are sorted for consistent frame order
    all_display_countries = category_order # Use the sorted order
    all_forks = sorted(df['fork'].unique()) # Sort forks for consistent color mapping/legend order
    multi_index = pd.MultiIndex.from_product(
        [all_hours, all_display_countries, all_forks],
        names=['hour', 'country_display', 'fork']
    )

    # Reindex the aggregated data, filling missing counts with 0
    anim_data_full = anim_data.set_index(['hour', 'country_display', 'fork']).reindex(multi_index, fill_value=0).reset_index()

    # --- Create Animated Plot ---
    tqdm.write("[+] Creating animated bar chart...")
    try:
        fig = px.bar(
            anim_data_full,
            x="cnt",
            y="country_display",
            color="fork",
            orientation='h',
            animation_frame="hour",
            animation_group="country_display", # Ensures bars transition smoothly for each country
            title=f"Hourly Fork Distribution in Top {top_n} Countries (+ Others)",
            labels={"cnt": "Node Count", "country_display": "Country", "fork": "Fork"},
            category_orders={
                "country_display": category_order, # Set fixed order for y-axis
                "fork": all_forks # Ensure consistent fork legend order
                },
            color_discrete_map=FORK_COLOR_MAP, # Apply consistent colors
            height=max(600, len(category_order) * 40) # Adjust height based on number of bars
        )

        # Improve layout and animation settings
        fig.update_layout(
            xaxis_title="Total Node Count",
            yaxis_title="Country / Region",
            legend_title="Fork",
            barmode='stack',
            yaxis={'categoryorder': 'array', 'categoryarray': category_order}, # Ensure order is respected
            # --- Remove background ---
            plot_bgcolor='white', # Set plot area background to white
            paper_bgcolor='white', # Set paper background to white
            # -------------------------
            # Ensure slider steps match hours correctly
            sliders=[dict(
                active=0,
                currentvalue={"prefix": "Hour: "},
                pad={"t": 50},
                steps=[dict(label=str(hour),
                            method="animate",
                            args=[[str(hour)], # Frame name must match animation_frame values
                                  dict(mode="immediate",
                                       frame=dict(duration=frame_duration, redraw=True),
                                       transition=dict(duration=max(0, frame_duration - 50))) # Slight overlap
                                  ])
                       for hour in all_hours] # Use sorted hours for steps
            )],
             updatemenus=[dict(
                type='buttons',
                showactive=False,
                buttons=[dict(label='Play',
                              method='animate',
                              args=[None, dict(frame=dict(duration=frame_duration, redraw=True),
                                               fromcurrent=True,
                                               transition=dict(duration=max(0, frame_duration - 50), easing='linear'))]), # Smooth transition
                         dict(label='Pause',
                              method='animate',
                              args=[[None], dict(mode='immediate')]) # Pause immediately
                        ]
            )]
        )

        # Ensure x-axis adjusts range per frame for better visibility
        # Calculate max count per country *across all forks* for each hour
        max_hourly_country_sum = anim_data_full.groupby(['hour', 'country_display'])['cnt'].sum().max()
        fig.update_xaxes(range=[0, max_hourly_country_sum * 1.1])

        # Apply hover template
        fig.update_traces(hovertemplate="<b>Country:</b> %{y}<br><b>Fork:</b> %{fullData.name}<br><b>Count:</b> %{x}<extra></extra>")

        tqdm.write("[✓] Animation created.")
        return fig

    except Exception as e:
        tqdm.write(f"[!] Error creating animated plot: {e}")
        import traceback
        traceback.print_exc()
        return go.Figure() # Return empty figure on error

def create_combined_dashboard(fig1: go.Figure, fig2: go.Figure) -> str:
    """Combines two Plotly figures into a single HTML dashboard."""

    # Generate HTML for each figure separately
    # Use plotly.js from CDN for the first plot, not for the second.
    html1 = fig1.to_html(full_html=False, include_plotlyjs='cdn')
    html2 = fig2.to_html(full_html=False, include_plotlyjs=False)

    # Combine HTML with basic structure
    combined_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8" />
        <title>Fork Distribution Dashboard</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; background-color: #f4f4f4; }} /* Optional: Light grey page bg */
            .plotly-graph-div {{
                margin-bottom: 40px !important; /* Add space between plots */
                background-color: white !important; /* Ensure container div is white if needed */
                box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1); /* Optional: Add subtle shadow */
                border-radius: 8px; /* Optional: Rounded corners */
                overflow: hidden; /* Ensures shadow respects border-radius */
             }}
            h1 {{ text-align: center; color: #333; }}
        </style>
    </head>
    <body>
        <h1>Fork Distribution Dashboard</h1>
        <div>{html1}</div>
        <div>{html2}</div>
    </body>
    </html>
    """
    return combined_html

# --- CLI and Main Execution ---

def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an HTML dashboard visualizing fork distribution from CSV data."
    )
    parser.add_argument(
        "-i", "--input",
        type=pathlib.Path,
        required=True,
        help="Input CSV file path (e.g., fork_data.csv). Needs columns: date, country, fork, cnt",
    )
    parser.add_argument(
        "-o", "--output",
        type=pathlib.Path,
        required=True,
        help="Output HTML file path (e.g., forks_dashboard.html).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top countries to display individually in the second plot.",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=500, # Default animation speed (milliseconds per frame)
        help="Animation speed (frame duration) in milliseconds for the country plot (lower is faster). Default: 500.",
    )

    # Basic validation
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Input file not found: {args.input}")
    if args.output.suffix.lower() != ".html":
        print(f"[Warning] Output file does not end with .html. Saving as {args.output.with_suffix('.html')}")
        args.output = args.output.with_suffix(".html")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    return args

def main():
    args = cli()
    in_path = args.input
    out_path = args.output

    # Load data
    print(f"[•] Loading data from {in_path}...")
    try:
        df = pd.read_csv(in_path)
        print(f"[✓] Loaded {len(df)} rows.")
    except Exception as e:
        print(f"[ERROR] Failed to load input CSV: {e}")
        sys.exit(1)

    # Preprocess data
    print("[•] Preprocessing data...")
    try:
        df = preprocess_data(df)
        if df.empty:
             print("[Warning] Preprocessing resulted in empty DataFrame.")
        else:
            print(f"[✓] Data preprocessed. Date range: {df['date'].min()} to {df['date'].max()}")
    except Exception as e:
        print(f"[ERROR] Failed during preprocessing: {e}")
        sys.exit(1)

    # Build figures
    print(f"[•] Building overall fork distribution visualization...")
    try:
        overall_fig = build_overall_forks_figure_updated(df)
        if not overall_fig.data and not df.empty: # Check if figure is empty, ignore if input df was empty
             raise ValueError("Figure creation returned empty figure despite non-empty input.")
        print("[✓] Overall forks figure built.")
    except Exception as e:
        print(f"[ERROR] Failed to build overall fork figure: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"[•] Building top {args.top} countries fork distribution visualization...")
    try:
        countries_fig = build_top_countries_figure_updated(df, top_n=args.top, frame_duration=args.speed)
        if not countries_fig.data and not df.empty: # Check if figure is empty
             raise ValueError("Figure creation returned empty figure despite non-empty input.")
        print("[✓] Top countries figure built.")
    except Exception as e:
        print(f"[ERROR] Failed to build countries figure: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Create combined dashboard
    print(f"[•] Creating combined dashboard...")
    try:
        # Only proceed if figures were likely created successfully
        if (overall_fig.data or df.empty) and (countries_fig.data or df.empty):
             dashboard_html = create_combined_dashboard(overall_fig, countries_fig)
             print("[✓] Combined dashboard created.")
        else:
             print("[!] Skipping dashboard creation due to empty figures.")
             dashboard_html = "<html><body>Error: Could not generate plots.</body></html>" # Placeholder error message
    except Exception as e:
        print(f"[ERROR] Failed to create combined dashboard: {e}")
        import traceback
        traceback.print_exc()
        dashboard_html = "<html><body>Error: Could not generate dashboard HTML.</body></html>" # Placeholder error message
        # Optionally: sys.exit(1) here? Depends if user wants a partial file or nothing.

    # Save to file
    try:
        out_path = out_path.with_suffix(".html") # Ensure .html extension
        with open(str(out_path), 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        print(f"[✓] Dashboard saved to {out_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save output file to {out_path}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()