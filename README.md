# TI-4550
This repository contains code for simulating heat pump control scenarios using reinforcement learning, developed as part of the TIØ4550 project thesis. Follow the steps below to set up the boptestgym environment and run simulations.

## Setup Instructions for the project.

The `boptestgym` environment is used to simulate heat pump optimization scenarios for reinforcement learning tasks. Follow these steps to set up the environment:

### Prerequisites
Ensure you have the following installed on your system:
- **Python** (version 3.8 or higher)
- **Docker** (for running BOPTEST test cases)
- **pip** (Python package manager)
- **git** (to clone the repository)
- A **Unix-based system** (Linux/MacOS recommended; Windows with WSL may also work but no guarantees!)

### Installation Steps

1. Clone the Repository

git clone [https://github.com/your-username/TI-4550.git](https://github.com/hanskrio/TI-4550.git)

cd TI-4550

2. Install Required Python Packages

- Set up a virtual environment (recommended) and install dependencies:
python -m venv venv
source venv/bin/activate
- On Windows, use `venv\Scripts\activate`

3. Set Up BOPTEST

Follow the official instructions to set up BOPTEST:
https://github.com/ibpsa/project1-boptest

4. Set Up boptestgym

## Quick-Start (running BOPTEST locally)
In my opinion, it is necessary to set up BOPTEST locally. The BOPTEST-Service code from https://github.com/ibpsa/project1-boptest-gym is useful for exploration but has limitations. For example, the DQN model parameters (e.g., batch_size=24 and buffer_size=365*24) are unusually small, and the simulation runs for only 24 hours. These settings are generally are not suitable research and experimentation.

1. Create a conda environment from the environment.yml file provided (instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)).
2. Run a BOPTEST case with the building emulator model to be controlled (instructions [here](https://github.com/ibpsa/project1-boptest/blob/master/README.md)).
3. Set up a like url = 'http://127.0.0.1:5000' from your machine to the BOPTEST server.

### Or, follow the instructions for setting up `boptestgym`from:
https://github.com/ibpsa/project1-boptest-gym

6. Run Your First Simulation

Execute a test script to ensure everything is set up correctly:
python scripts/test_boptestgym.py

Troubleshooting
- Docker connection issues: Ensure Docker is running, and the container is active.
- Port conflicts: Check if port 5000 is already in use. Modify the `-p` option in the `docker run` command if necessary (e.g., `-p 5001:5000`).
- Dependency errors: Ensure all packages in `requirements.txt` are installed without errors.

