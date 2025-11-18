# BPPSO Assignment 1 - Process Mining Analysis

This project contains process mining analysis scripts and Jupyter notebooks for analyzing the BPI Challenge 2017 event log using pm4py (Process Mining for Python).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Dataset Requirements](#dataset-requirements)
- [Running Scripts](#running-scripts)
- [Running Notebooks](#running-notebooks)
- [Output Locations](#output-locations)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- **Python 3.8 or higher** (Python 3.9+ recommended)
- **Git** (for cloning the repository)
- **Graphviz** (optional, for process model visualization)
  - Windows: Download from [Graphviz website](https://graphviz.org/download/) or install via `choco install graphviz`
  - Linux: `sudo apt-get install graphviz` (Ubuntu/Debian) or `sudo yum install graphviz` (RHEL/CentOS)
  - macOS: `brew install graphviz`

## Installation



### Setup

1. **Create a virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Upgrade pip:**
   ```bash
   python -m pip install --upgrade pip
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install pm4py from local directory:**
   ```bash
   pip install -e pm4py
   ```

### Verifying Installation

After installation, verify that pm4py is installed correctly:
```bash
python -c "import pm4py; print(pm4py.__version__)"
```

## Project Structure

```
BPPSO - Assignment 1/
├── Dataset/                          # Event log files (not in repository)
│   └── BPI Challenge 2017.xes       # Main dataset (you need to provide this)
├── Results/                          # Generated outputs
│   ├── Advanced_Analysis/           # Advanced analysis results
│   ├── Attribute_Visualizations/    # Attribute visualization plots
│   └── Models/                      # Process models
├── src/                              # Source code
│   ├── advanced analysis/            # Advanced analysis notebooks
│   │   ├── advanced_analysis_data_drift.ipynb
│   │   ├── advanced_analysis_feature_importance.ipynb
│   │   └── advanced_analysis_variant_clustering.ipynb
│   ├── process model/                # Process discovery notebooks
│   │   ├── basic_models.ipynb
│   │   └── model_gauntlet.ipynb
│   ├── Visualization/                # Visualization scripts
│   ├── basic_analysis.py             # Basic event log statistics
│   ├── a_concept_trace_coverage.py   # Trace coverage analysis
│   └── ...                           # Other analysis scripts
├── pm4py/                            # Local pm4py installation
├── requirements.txt                  # Python dependencies
├── setup_venv.bat                    # Windows setup script
├── setup_venv.sh                     # Linux/macOS setup script
└── README.md                         # This file
```

## Dataset Requirements

**Important:** The dataset is not included in the repository due to size constraints.

You need to:
1. Download the **BPI Challenge 2017** event log from the [BPI Challenge website](https://www.win.tue.nl/bpi/2017/challenge)
2. Place the file `BPI Challenge 2017.xes` in the `Dataset/` directory
3. Ensure the file path is: `Dataset/BPI Challenge 2017.xes`

The scripts and notebooks expect this exact file name and location.

## Running Scripts

**Important:** Always activate your virtual environment before running scripts:
```bash
# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

All scripts should be run from the **project root directory**.

### Basic Analysis Scripts

#### Basic Event Log Statistics
```bash
python src/basic_analysis.py
```
Generates basic statistics about the event log (number of cases, events, variants, durations, etc.)

#### Trace Coverage Analysis
```bash
python src/a_concept_trace_coverage.py
```
Analyzes trace coverage and generates a coverage report.

#### Column Information
```bash
python src/colum_info.py
```
Prints all columns and their values from the event log.

#### Lifecycle Analysis
```bash
python src/lifecycle_transition_analysis.py
python src/variants_lifecycle_analysis.py
```
Analyze lifecycle transitions and variants.

#### Visualization Scripts
```bash
# Case arrival rate visualization
python src/Visualization/case_arrival_rate_visualization.py

# Attribute visualizations
python src/Visualization/attribute_visualizations.py

# Activity prefix pie chart
python src/Visualization/activity_prefix_piechart.py

# Case duration boxplot
python src/Visualization/case_duration_boxplot.py

# Case length boxplot
python src/Visualization/case_length_boxplot.py

# Event attribute barplots
python src/Visualization/event_attribute_barplots.py

# Resource distribution
python src/Visualization/resource_distribution_create_application.py

# Resource usage visualization
python src/Visualization/resource_usage_visualization.py
```

### Other Analysis Scripts

```bash
# Check constant attributes
python src/check_constant_attributes.py

# Generate lifecycle combined XES
python src/generate_lifecycle_combined_xes.py

# Log filtering
python src/log_filter.py

# Print most used variants
python src/print_most_used_variants.py

# Process discovery with lifecycle
python src/process model/process_discovery_lifecycle.py
```

## Running Notebooks

### Starting Jupyter Notebook

1. **Activate your virtual environment:**
   ```bash
   # Windows
   venv\Scripts\activate

   # Linux/macOS
   source venv/bin/activate
   ```

2. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

3. **Navigate to the notebook you want to run:**
   - Advanced Analysis: `src/advanced analysis/`
   - Process Models: `src/process model/`

### Advanced Analysis Notebooks

 Located in `src/advanced analysis/`:

- **`advanced_analysis_data_drift.ipynb`**: Analyzes data drift over time
- **`advanced_analysis_feature_importance.ipynb`**: Feature importance analysis using SHAP
- **`advanced_analysis_variant_clustering.ipynb`**: Clustering of process variants
- **`advanced_analysis_activity_lifecycle_kmeans.ipynb`**: Simple k-means on activity-lifecycle combos

### Process Model Notebooks

Located in `src/process model/`:

- **`basic_models.ipynb`**: Basic process discovery models
- **`model_gauntlet.ipynb`**: Comprehensive model comparison

**Note:** Notebooks use relative paths like `../../Dataset/` and `../../Results/`, so they should be run from their respective directories or with the project root as the working directory.

## Output Locations

All generated outputs are saved in the `Results/` directory:

 - **`Results/Advanced_Analysis/`**: Outputs from advanced analysis notebooks
   - `activity_lifecycle_kmeans/`: Activity+lifecycle k-means clustering
   - `data_drift/`: Data drift analysis results
   - `feature_importance/`: Feature importance analysis results
   - `variant_clustering/`: Variant clustering results
- **`Results/Attribute_Visualizations/`**: Visualization plots
- **`Results/Models/`**: Generated process models (BPMN files, images)
- **`Results/Model_Comparisons.md`**: Model comparison documentation


## Notes

- **pm4py Local Installation**: This project includes pm4py as a local directory. It's installed in editable mode (`-e` flag) so changes to pm4py code are immediately available.
- **Virtual Environment**: Always work within the virtual environment to avoid dependency conflicts.
- **Dataset Size**: The BPI Challenge 2017 dataset is large (~500MB). Ensure you have sufficient disk space.
- **Execution Time**: Some analysis scripts may take several minutes to run depending on your hardware.

## License

This project is for educational purposes as part of the BPPSO Assignment 1. The pm4py library is licensed under AGPL 3.0.
