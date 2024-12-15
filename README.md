# TI-4550
This repository contains code for simulating heat pump control scenarios using reinforcement learning, developed as part of the TIØ4550 project thesis. Follow the steps below to set up the boptestgym environment.

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

- git clone [https://github.com/your-username/TI-4550.git](https://github.com/hanskrio/TI-4550.git)

- cd TI-4550

2. Install Required Python Packages

- Set up a virtual environment (recommended) and install dependencies:
python -m venv venv
source venv/bin/activate
- On Windows, use `venv\Scripts\activate`

3. Set Up BOPTEST

Follow the official instructions to set up BOPTEST:
https://github.com/ibpsa/project1-boptest

4. Set Up boptestgym (see below)

#### Quick-Start (running BOPTEST locally)
Setting up BOPTEST locally provides significant advantages for research and experimentation. 

Note: While the BOPTEST-Service code from [https://github.com/ibpsa/project1-boptest-gym](https://github.com/ibpsa/project1-boptest-gym) serves as a useful starting point for exploration, it has several technical constraints. The implemented DQN model parameters, including the batch size of 24 and buffer size of 365*24, are substantially smaller than typically recommended values for reinforcement learning research. Additionally, the 24-hour simulation duration limits the ability to evaluate long-term control strategies and system behaviors. For research applications and experimentation, these default configurations require modification to align with theory.

1. Create a conda environment from the environment.yml file provided (instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)).
2. Run a BOPTEST case with the building emulator model to be controlled (instructions [here](https://github.com/ibpsa/project1-boptest/blob/master/README.md)).
3. Set up an url = 'http://127.0.0.1:5000' from your machine to the BOPTEST server. To identify if port 5000 is already in use, you can use the following command in your terminal: `lsof -i :5000`


#### Or, follow the instructions for setting up `boptestgym`from:
https://github.com/ibpsa/project1-boptest-gym

6. Train RL agents or run simulations with already trained agents from this repository :) 

Execute a test script to ensure everything is set up correctly:
TBD

Troubleshooting
- Docker connection issues: Ensure Docker is running, and the container is active.
- Port conflicts: Check if port 5000 is already in use. Modify the `-p` option in the `docker run` command if necessary (e.g., `-p 5001:5000`).
- Dependency errors: Ensure all packages in `requirements.txt` are installed without errors.

