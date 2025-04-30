# Network-Evolution-Anim

Create a world-map animation (HTML or GIF) from a CSV that lists
geolocated node counts by hour.

**[View Interactive Animation](https://hdser.github.io/network-anim/)** ← Click to view the interactive HTML animation

## 1. Quick start

```bash
git clone <this-repo-url> network-evolution-anim
cd network-evolution-anim
python -m venv .venv    # or micromamba create -n netanim python=3.11
source .venv/bin/activate
pip install -r requirements.txt

# HTML output (recommended – interactive, zoomable, draggable)
python animate_network.py -i nodes_geo.csv -o network.html

# Control HTML animation speed (lower value = faster animation)
python animate_network.py -i nodes_geo.csv -o network.html --speed 100

# GIF output (smaller file, viewable anywhere)
python animate_network.py -i nodes_geo.csv -o network.gif --fps 4

# Faster GIF animation (higher fps = faster animation)
python animate_network.py -i nodes_geo.csv -o network.gif --fps 15
```

## 2. Command-line options

| Option | Description |
|--------|-------------|
| `-i, --input` | CSV file with date,lat,long,cnt columns (required) |
| `-o, --output` | Output file path (.html or .gif) |
| `--fps` | Frames per second for GIF output (higher = faster, default: 10) |
| `--speed` | Animation speed for HTML output in milliseconds (lower = faster, default: 300) |
| `--title` | Custom title for the animation |

## 3. Features

### Key visualization features:

- **Automatic Country Detection**: Automatically detects countries from lat/long coordinates
- **Interactive Controls**: Play/pause buttons for HTML output
- **Node Count Statistics**: 
  - Countries in legend are sorted by total node count (descending)
  - Each country label includes its total node count
  - Overall total node count displayed in annotation
  - Hover information shows current node count for each point
- **Adjustable Speed**: Control animation speed for both HTML and GIF outputs
- **Multiple Output Formats**: Generate interactive HTML or shareable GIF animations

### Input requirements:

The input CSV should have these columns:
- `date`: Timestamp (will be parsed as datetime)
- `lat`: Latitude coordinate
- `long`: Longitude coordinate
- `cnt`: Node count at this location and time


# Fork Distribution Visualization

This extension to the Network Evolution Animation project visualizes the distribution of different forks over time, providing both overall percentages and breakdown by top countries.

## 1. Quick Start

```bash
# Make sure you have the required dependencies installed
pip install -r requirements.txt

# Generate the forks dashboard
python visualize_forks.py -i forks_data.csv -o forks.html

# Control animation speed (lower value = faster animation)
python visualize_forks.py -i forks_data.csv -o forks.html --speed 200

# Change number of top countries displayed
python visualize_forks.py -i forks_data.csv -o forks.html --top 15
```

## 2. Command-line Options

| Option | Description |
|--------|-------------|
| `-i, --input` | CSV file with date,country,fork,cnt columns (required) |
| `-o, --output` | Output HTML dashboard file path |
| `--top` | Number of top countries to display (default: 10) |
| `--speed` | Animation speed in milliseconds (lower = faster, default: 300) |

## 3. Features

### Visualization components:

1. **Overall Fork Distribution**:
   - Shows the percentage distribution of different forks over time
   - Bars are colored by fork type
   - Animation shows how the distribution evolves hourly
   - Hover information includes both percentage and actual node count

2. **Top Countries Fork Distribution**:
   - Displays the top countries based on total node count 
   - Shows stacked bars with each fork's contribution
   - Animation shows the evolution hourly
   - Countries are ordered by total node count (descending)

### Input Requirements:

The input CSV file should have these columns:
- `date`: Timestamp (will be parsed as datetime)
- `country`: Two-letter country code (ISO 3166-1 alpha-2)
- `fork`: Name of the fork
- `cnt`: Number of nodes for this country/fork combination at this time

## 4. Integration

This visualization integrates with the existing Network Evolution Animation project:
- Updated index.html includes links to both visualizations
- Consistent styling across visualizations
- Compatible with the same infrastructure and deployment approach


## License

This project is licensed under the [MIT License](LICENSE).