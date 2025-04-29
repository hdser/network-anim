#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import reverse_geocoder as rg
from tqdm import tqdm

# Optional GIF support — only imported when requested/used in _save_gif
try:
    import imageio.v3 as iio
except ImportError:
    iio = None # Will be checked in _save_gif

# ──────────────────────────────────────────────────────────────────────────────
# Geo helpers (fast, single‑process, deduplicated, robust fallback)
# ──────────────────────────────────────────────────────────────────────────────

def infer_countries(
    df: pd.DataFrame,
    *,
    lat_col: str = "lat",
    lon_col: str = "long",
) -> pd.DataFrame:
    """Add ISO‑3166 country codes, trying fastest methods first.

    Workflow
    --------
    1. **Deduplicate** coordinates → query unique points only.
    2. Try ultra‑fast KD‑tree query with `workers=0` (vectorised, single-thread).
    3. If unavailable or fails, try `workers=1` (single-process, non-vectorised).
    4. If that also fails, try default parallel search (can have issues).
    5. If that also fails, fall back to fully sequential RGeocoder look‑ups.
    """
    if lat_col not in df.columns or lon_col not in df.columns:
         raise ValueError(f"Missing required columns: '{lat_col}' or '{lon_col}'")

    # Handle potential NaN/missing values before processing
    df_valid = df[[lat_col, lon_col]].dropna()
    if df_valid.empty:
        tqdm.write("[!] No valid coordinates found to infer countries.")
        df = df.copy()
        df["country"] = "Unknown" # Assign a default value
        return df

    lat = df_valid[lat_col].to_numpy()
    lon = df_valid[lon_col].to_numpy()
    coords = np.column_stack((lat, lon))

    unique_coords, inverse_indices = np.unique(coords, axis=0, return_inverse=True)
    unique_list: List[Tuple[float, float]] = [tuple(p) for p in unique_coords]

    tqdm.write(
        f"[•] Inferring country codes for {len(unique_list):,} unique valid coordinates "
        f"(out of {len(df):,} total rows)..."
    )

    # Use mode=1 to get dict results consistently, needed for fallback
    # We'll extract 'cc' later
    results: Optional[List[Dict[str, Any]]] = None

    # Check if the 'workers' parameter is supported in this version
    workers_supported = False
    rg_signature = str(rg.search.__code__.co_varnames)
    if "workers" in rg_signature:
        workers_supported = True
    
    # --- Method 1: Try optimized methods if available, otherwise go straight to default ---
    if workers_supported:
        try:
            tqdm.write("[•] Trying rg.search(..., workers=0)")
            results = rg.search(unique_list, mode=1, workers=0)
            tqdm.write("[✓] rg.search(workers=0) succeeded.")
        except (TypeError, RuntimeError) as e:
            tqdm.write(f"[!] Optimized search failed: {type(e).__name__}: {e}")
            results = None
            
        if results is None:
            try:
                tqdm.write("[•] Trying rg.search(..., workers=1)")
                results = rg.search(unique_list, mode=1, workers=1)
                tqdm.write("[✓] rg.search(workers=1) succeeded.")
            except (TypeError, RuntimeError) as e1:
                tqdm.write(f"[!] Single-process search failed: {type(e1).__name__}: {e1}")
                results = None
    
    # --- Method 2: Try default parallel (Standard approach) ---
    if results is None:
        try:
            tqdm.write("[•] Using default rg.search(...)")
            results = rg.search(unique_list, mode=1)
            tqdm.write("[✓] Default rg.search() succeeded.")
        except Exception as e2:
            tqdm.write(f"[!] Default search failed: {type(e2).__name__}: {e2}")
            results = None
    
    # --- Method 3: Sequential fallback (Slow but reliable) ---
    if results is None:
        try:
            tqdm.write("[•] Falling back to sequential geocoding")
            geo = rg.RGeocoder(mode=1, verbose=False)
            results = [geo.query([c])[0] for c in tqdm(unique_list, desc="    sequential")]
            tqdm.write("[✓] Sequential geocoding succeeded.")
        except Exception as e3:
            tqdm.write(f"[!] Sequential geocoding failed: {type(e3).__name__}: {e3}")
            results = None


    # Process results if any method succeeded
    df = df.copy() # Work on a copy
    if results is not None:
        # Extract country codes ('cc') from the list of result dictionaries
        # Handle potential missing 'cc' key defensively, although unlikely with mode=1
        codes = np.array([r.get("cc", "??") for r in results], dtype="U2") # '??' for unexpected missing code

        # Create a temporary series matching the original valid coordinates' index
        country_series = pd.Series(codes[inverse_indices], index=df_valid.index)

        # Map the results back to the original DataFrame, leaving NaNs where coords were missing
        df["country"] = country_series
        df["country"] = df["country"].fillna("Unknown") # Fill NaNs from original missing coords
    else:
        # If all methods failed, assign 'Unknown'
        tqdm.write("[!] All geocoding methods failed. Assigning 'Unknown' country.")
        df["country"] = "Unknown"

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Calculate country statistics
# ──────────────────────────────────────────────────────────────────────────────

def calculate_country_stats(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Calculate country statistics for consistent ordering.
    
    Returns:
    - Enhanced DataFrame with sorted_order column
    - Dictionary of country to max node count (for sorting)
    """
    df = df.copy()
    
    # Calculate max node count per country across all hours for sorting
    country_max_counts = df.groupby('country')['cnt'].sum().to_dict()
    
    # Create a sort order based on country max totals (descending)
    sorted_countries = sorted(
        country_max_counts.keys(), 
        key=lambda c: country_max_counts[c], 
        reverse=True
    )
    country_order = {c: i for i, c in enumerate(sorted_countries)}
    df['sorted_order'] = df['country'].map(country_order)
    
    return df, country_max_counts


# ──────────────────────────────────────────────────────────────────────────────
# Plotly figure builder
# ──────────────────────────────────────────────────────────────────────────────

def _pad_first_frame(df: pd.DataFrame, size_epsilon: float = 0.0001) -> pd.DataFrame:
    """Ensure all countries appear in the first frame for consistent legend/color.

    Adds dummy entries with near-zero size for countries not present initially.
    """
    if "hour" not in df.columns:
         # Ensure 'hour' column exists (might not if build_figure is called directly)
         if "date" not in df.columns or not pd.api.types.is_datetime64_any_dtype(df["date"]):
              raise ValueError("DataFrame needs a datetime 'date' column to create 'hour' for padding.")
         df = df.copy()
         df["hour"] = df["date"].dt.strftime("%Y-%m-%d %H:%M")
    else:
         df = df.copy() # Still copy to avoid modifying original df


    if df.empty or df["hour"].isna().all():
        tqdm.write("[!] Cannot pad first frame: No valid 'hour' data found.")
        return df

    first_hour = df["hour"].min()
    present_countries = set(df.loc[df["hour"] == first_hour, "country"].unique())
    all_countries = set(df["country"].unique())
    missing_countries = all_countries - present_countries

    if missing_countries:
        # Create one dummy row per missing country for the first hour
        # Take the first occurrence of each missing country in the *entire* dataset
        # to get valid lat/lon for the dummy point.
        dummy_rows = (
            df[df["country"].isin(missing_countries)]
            .sort_values("date") # Use original date to get first instance
            .drop_duplicates("country")
            .assign(
                hour=first_hour,
                cnt=size_epsilon # Make size effectively zero
            )
        )
        if not dummy_rows.empty:
             tqdm.write(f"[•] Padding first frame with {len(dummy_rows)} dummy entries for consistent legend.")
             df = pd.concat([df, dummy_rows], ignore_index=True)
             # Sort by hour again to ensure proper animation sequence
             df = df.sort_values("hour", ignore_index=True)

    return df


def build_figure(df: pd.DataFrame, *, title: str | None = None, frame_duration: int = 300):
    """Return a Plotly figure with one frame per hour.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data with required columns
    title : str, optional
        Title for the plot
    frame_duration : int, default=300
        Duration of each frame in milliseconds (lower value = faster animation)
    """
    if df.empty:
        tqdm.write("[!] Input data is empty, cannot build figure.")
        # Return an empty figure object or handle as appropriate
        import plotly.graph_objects as go
        return go.Figure()

    df = df.copy()
    # Ensure 'hour' column exists for animation frame
    if "date" not in df.columns or not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise ValueError("DataFrame needs a datetime 'date' column to create 'hour' for animation.")
    df["hour"] = df["date"].dt.strftime("%Y-%m-%d %H:%M")

    # Pad the first frame *after* creating 'hour' and *before* plotting
    df = _pad_first_frame(df)
    
    # Calculate country stats for consistent ordering
    df, country_max_counts = calculate_country_stats(df)
    
    # Calculate hourly totals for each frame
    hourly_totals = {}
    for hour in df['hour'].unique():
        hour_data = df[df['hour'] == hour]
        country_hour_totals = hour_data.groupby('country')['cnt'].sum().to_dict()
        total_nodes_hour = int(hour_data['cnt'].sum())
        hourly_totals[hour] = {
            'country_totals': country_hour_totals,
            'total_nodes': total_nodes_hour
        }

    # Check for essential columns for plotting
    required_cols = ["lat", "long", "cnt", "country", "hour"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame is missing required columns for plotting: {missing_cols}")

    # Ensure numeric types where needed by Plotly
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['long'] = pd.to_numeric(df['long'], errors='coerce')
    df['cnt'] = pd.to_numeric(df['cnt'], errors='coerce')

    # Drop rows where essential numeric columns became NaN after coercion
    initial_rows = len(df)
    df = df.dropna(subset=['lat', 'long', 'cnt'])
    if len(df) < initial_rows:
        tqdm.write(f"[!] Dropped {initial_rows - len(df)} rows with invalid numeric data (lat/long/cnt).")

    if df.empty:
        tqdm.write("[!] Data is empty after cleaning, cannot build figure.")
        import plotly.graph_objects as go
        return go.Figure()

    # Create the scatter geo plot
    fig = px.scatter_geo(
        df,
        lat="lat",
        lon="long",
        size="cnt",
        color="country",
        hover_name="country",
        hover_data={"country": False,  # Hide duplicate country in hover
                    "cnt": True,      # Keep showing current count
                    "sorted_order": False},  # Hide the sort order in hover
        animation_frame="hour",
        projection="natural earth",
        size_max=30,
        title=title or "Network evolution over time",
        category_orders={"country": sorted(country_max_counts.keys(), 
                                          key=lambda c: country_max_counts[c], 
                                          reverse=True)}
    )
    
    # Customize hover template 
    for trace in fig.data:
        if hasattr(trace, "hovertemplate"):
            trace.hovertemplate = trace.hovertemplate.replace("cnt=", "Current Nodes: ")
    
    # Update frame layouts with current node counts for each country and hour
    first_hour = df['hour'].min()
    max_total = max(data['total_nodes'] for data in hourly_totals.values())
    
    # Initial legend names for the base figure
    for i, trace in enumerate(fig.data):
        if hasattr(trace, "name") and trace.name in country_max_counts:
            country = trace.name
            if first_hour in hourly_totals and country in hourly_totals[first_hour]['country_totals']:
                node_count = int(hourly_totals[first_hour]['country_totals'][country])
                trace.name = f"{country} ({node_count:,})"
            else:
                trace.name = f"{country} (0)"
    
    # Update title and add annotation
    first_hour_total = hourly_totals[first_hour]['total_nodes'] if first_hour in hourly_totals else 0
    fig.update_layout(
        title=f"Network evolution over time (Current: {first_hour_total:,} nodes)",
        margin=dict(l=0, r=0, t=50, b=10),
        legend_title_text='Country Ranking',
    )
    
    # Update each frame with the appropriate hour's node counts
    for i, frame in enumerate(fig.frames):
        hour = frame.name
        
        # Update frame title and annotation with current hour's total
        if hour in hourly_totals:
            current_total = hourly_totals[hour]['total_nodes']
            
            # Update title for this frame
            if frame.layout is None:
                frame.layout = {}
            frame.layout.title = f"Network evolution over time (Nodes Crawled Hourly: {current_total:,})"
            
            # Update the trace names for this frame to show current node counts
            current_country_totals = hourly_totals[hour]['country_totals']
            for trace in frame.data:
                if hasattr(trace, "name"):
                    parts = trace.name.split(" (")
                    if len(parts) > 0:
                        country = parts[0]
                        if country in current_country_totals:
                            node_count = int(current_country_totals[country])
                            trace.name = f"{country} ({node_count:,})"
                        else:
                            trace.name = f"{country} (0)"
    
    # Set animation speed for HTML output with play and pause buttons
    # Apply the animation speed to both initial and play button animations
    animation_settings = {
        "frame": {"duration": frame_duration, "redraw": True},
        "fromcurrent": True,
        "transition": {"duration": 0},
        "mode": "immediate"
    }
    
    # We can't set transition and duration directly on frames, 
    # so we'll rely on the updatemenus and auto-play script instead

    # Create the buttons for play/pause with the proper animation settings
    fig.layout.updatemenus = [
        {
            "type": "buttons",
            "showactive": False,
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, animation_settings]
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [
                        [None],  # Pause at the current frame
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }
                    ]
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 10},
            "x": 0.1,
            "y": 0
        }
    ]
    
    # Set the default animation settings
    fig.layout.sliders[0].active = 0
    fig.layout.sliders[0].pad = {"b": 10, "t": 50}
    fig.layout.sliders[0].currentvalue = {
        "prefix": "Time: ",
        "visible": True,
        "xanchor": "right"
    }
    fig.layout.sliders[0].transition = {"duration": 0}
    
    # Configure animation to play automatically with the correct speed
    # This is handled when writing the HTML file with auto_play=True
    
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# GIF helper
# ──────────────────────────────────────────────────────────────────────────────

def _save_gif(fig, path: pathlib.Path, fps: int = 10):
    """Render frames with **Kaleido + imageio**."""
    if iio is None:  # Check if imageio was imported successfully
        sys.exit("[ERROR] imageio not installed – add it via 'pip install imageio'")

    path = path.with_suffix(".gif")
    # Check if figure has frames
    if not fig.frames:
         tqdm.write("[!] Figure has no animation frames. Cannot create GIF.")
         return

    with tempfile.TemporaryDirectory() as tmpdir:
        frames = []
        tmp_path = pathlib.Path(tmpdir)
        tqdm.write(f"[•] Rendering {len(fig.frames)} frames for GIF...")
        for k, frame in enumerate(tqdm(fig.frames, desc="    Frames", unit="frame")):
            # Create a figure reflecting the state of the current frame
            # Update layout first, then data
            frame_fig = go.Figure(layout=frame.layout)
            frame_fig.add_traces(frame.data)

             # Apply the base figure's layout properties not overridden by the frame
            frame_fig.update_layout(fig.layout)
            # Crucially update the specific frame's layout properties
            frame_fig.update_layout(frame.layout)


            png_path = tmp_path / f"frame_{k:05}.png"
            try:
                # Use frame_fig which contains data + layout for this specific frame
                frame_fig.write_image(str(png_path), scale=2, engine="kaleido")
                frames.append(iio.imread(png_path))
            except Exception as e:
                 tqdm.write(f"\n[!] Error rendering frame {k}: {e}. Skipping frame.")
                 # Decide if you want to continue without the frame or stop
                 # continue

        if not frames:
             tqdm.write("[!] No frames were successfully rendered. GIF not created.")
             return

        tqdm.write(f"[•] Writing GIF with {len(frames)} frames...")
        # Use duration (milliseconds per frame) instead of fps directly for imageio v3
        duration_ms = 1000 / fps  # Convert fps to milliseconds per frame
        iio.imwrite(path, frames, duration=duration_ms, loop=0)
        print(f"[✓] Animated GIF saved to {path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI glue
# ──────────────────────────────────────────────────────────────────────────────

def cli():
    p = argparse.ArgumentParser(
        description="Animate network evolution from a geolocation CSV (country inferred)."
    )
    p.add_argument("-i", "--input", required=True, help="CSV with date,lat,long,cnt")
    p.add_argument("-o", "--output", default="network.html", help="Output .html or .gif")
    p.add_argument("--fps", type=int, default=10, help="Frames per second for GIF output (higher = faster)")
    p.add_argument("--speed", type=int, default=300, 
                   help="Animation speed for HTML output (lower value = faster, default: 300ms per frame)")
    p.add_argument("--title", default="", help="Plot title override")
    args = p.parse_args()

    csv_path = pathlib.Path(args.input).expanduser().resolve()
    out_path = pathlib.Path(args.output).expanduser().resolve()

    if not csv_path.is_file():
         print(f"[ERROR] Input CSV not found: {csv_path}")
         sys.exit(1)

    print(f"[•] Reading input CSV: {csv_path}")
    try:
        # Specify low_memory=False for potentially mixed types if needed
        df = pd.read_csv(csv_path, parse_dates=["date"])#, low_memory=False)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {e}")
        sys.exit(1)

    # --- Infer Countries ---
    try:
        df = infer_countries(df)
    except Exception as e:
         print(f"[ERROR] Failed during country inference: {e}")
         # Optionally continue without country inference or exit
         # df['country'] = 'Unknown' # Example fallback
         sys.exit(1) # Exit if country inference is critical

    # --- Build Figure ---
    # Determine title
    plot_title = args.title or f"Network Evolution ({csv_path.stem})"
    print(f"[•] Building Plotly figure...")
    try:
        # Pass frame_duration parameter to control HTML animation speed
        fig = build_figure(df, title=plot_title, frame_duration=args.speed)
        # Check if the figure is valid (e.g., has data)
        if not fig.data:
             print("[!] No data to plot after processing. Output will be empty.")
             # Optionally exit or create an empty file
             # sys.exit(0)


    except Exception as e:
        print(f"[ERROR] Failed to build Plotly figure: {e}")
        import traceback
        traceback.print_exc() # Print detailed error
        sys.exit(1)

    tqdm.write(f"[✓] Figure built. Total rows plotted: {len(df):,} | Unique Countries: {df['country'].nunique()}")


    # --- Save Output ---
    try:
        if out_path.suffix.lower() == ".gif":
            _save_gif(fig, out_path, fps=args.fps)
        else:
            # Ensure output path is .html if not .gif
            out_path = out_path.with_suffix(".html")
            # Add a custom HTML file with embedded JavaScript to control animation speed
            html_content = fig.to_html(include_plotlyjs='cdn')
            
            # Add JavaScript to automatically set the animation speed on load
            speed_script = f"""
            <script>
            document.addEventListener('DOMContentLoaded', function() {{
                // Set animation options with the specified frame duration
                var frameOpts = {{
                    frame: {{ duration: {args.speed}, redraw: true }},
                    transition: {{ duration: 0 }},
                    mode: "immediate"
                }};
                
                // Get the figure ID
                var figureEl = document.querySelector('.plotly-graph-div');
                if (figureEl) {{
                    var figureId = figureEl.id;
                    
                    // Start the animation after a short delay to ensure everything is loaded
                    setTimeout(function() {{
                        Plotly.animate(figureId, null, frameOpts);
                    }}, 500);
                }}
            }});
            </script>
            """
            
            # Insert the speed script before the closing body tag
            html_content = html_content.replace('</body>', f'{speed_script}</body>')
            
            # Write the modified HTML file
            with open(str(out_path), 'w') as f:
                f.write(html_content)
            
            print(f"[✓] Interactive HTML saved to {out_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save output file to {out_path}: {e}")
        import traceback
        traceback.print_exc() # Print detailed error
        sys.exit(1)


if __name__ == "__main__":
    cli()