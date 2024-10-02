# Step 1: Install Necessary Libraries
Before implementing the Env class, ensure that the required libraries are installed:
pip install stable-baselines3, pip install gym


# Step 2: Import Required Modules
This script is located in the src/Environment/ directory and is responsible for loading power network data from SimBench. It performs the following functions:

# Step 3: Define the Custom Environment Class
This script is located in the src/Environment/ directory and is responsible for loading power network data from SimBench. It performs the following functions:


Imports necessary libraries:

pandapower for power network analysis.
numpy, pandas, and matplotlib for data manipulation and visualization.
simbench for accessing standard benchmark grid datasets.
Loads SimBench network data:

Retrieves specific SimBench codes and loads the corresponding network.
Processes and extracts profiles:

Extracts power profiles for various network elements such as loads (p_mw, q_mvar), static generators (p_mw), generators (p_mw), and storage (p_mw).
Data preparation:

The extracted data is stored in variables like load_p, sgen_p, gen_mw, which are then used in the dashboard for visualization.

# src\GUI\Dashboard_Data\Dashboard.py
This script is located in the src/GUI/ directory and is responsible for setting up and running a Dash web application that visualizes the power network data. Key features include:

Interactive Dashboard:

The dashboard allows users to select different time intervals (e.g., year, month, week, day) to adjust the scope of the visualization.
Graphical Visualization:

The main component of the dashboard is a line graph showing:
Demand: Total load demand over time.
RES Production: Power generated from renewable energy sources.
Conventional Generation: Power generated from conventional sources.
Dynamic Data Updates:

The graph updates dynamically based on the selected time interval, providing an interactive exploration of the power data.