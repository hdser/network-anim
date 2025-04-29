# Network-Evolution-Anim

Create a world-map animation (HTML or GIF) from a CSV that lists
geolocated node counts by hour.

![Network Animation Preview](https://via.placeholder.com/800x400?text=Network+Animation+Preview)

**[View Interactive Animation](https://hdser.github.io/network-evolution-anim/network.html)** ← Click to view the interactive HTML animation

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

## 4. Viewing Interactive HTML Animations

### Option 1: GitHub Pages (Recommended)
To set up GitHub Pages for your interactive animations:

1. In your repository, go to Settings → Pages
2. Set source to "main" branch and "/docs" folder
3. Generate your HTML animations to the "docs" directory:
   ```bash
   python animate_network.py -i nodes_geo.csv -o docs/network.html
   ```
4. Push to GitHub and your animation will be available at `https://yourusername.github.io/network-evolution-anim/network.html`

### Option 2: Use GIF for GitHub README
If you need the animation visible directly in the README, use the GIF output:

```bash
python animate_network.py -i nodes_geo.csv -o network.gif --fps 10
```

Then add to your README:
```markdown
![Network Animation](./network.gif)
```