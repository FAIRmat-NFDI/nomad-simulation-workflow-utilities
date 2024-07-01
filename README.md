# NOMAD-Simulation-Workflow-Utilities
Utilities for interfacing with NOMAD within simulation workflows, including python API functions for uploading to NOMAD and querying the repository as well as automated generation of NOMAD custom workflow yaml file ([NOMAD workflow data models](https://github.com/nomad-coe/nomad-schema-plugin-simulation-workflow)).

## Getting started

### Install the dependencies

Clone the repository and enter in the directory:
```sh
git clone https://github.com/FAIRmat-NFDI/nomad-simulation-workflow-utilities.git
cd nomad-simulation-workflow-utilities
```

Create a new virtual environment with Python 3.10 in the root folder of the project:
```sh
python3.10 -m venv .pyenv
source .pyenv/bin/activate
```

Make sure to have pip upgraded to its latest version:
```sh
pip install --upgrade pip
```

We recommend using `uv` for pip installing all packages due to its improved performance and speed when installing Pypi packages:
```sh
pip install uv
```

Install the development packages as well as specify the URL of the `nomad-lab` package using the `--index-url` flag:
```sh
uv pip install '.[dev]' --index-url https://gitlab.mpcdf.mpg.de/api/v4/projects/2187/packages/pypi/simple
```

In order to develop the code, add the editable (`e`) flag to the installation step:
```sh
uv pip install -e '.[dev]' --index-url https://gitlab.mpcdf.mpg.de/api/v4/projects/2187/packages/pypi/simple
```

**Note!**
Until we have an official pypi NOMAD release with the plugins functionality. Make
sure to include NOMAD's internal package registry (via `--index-url` in the above command).

### Run the tests

You can run local tests using the `pytest` package:

```sh
python -m pytest -sv tests
```

where the `-s` and `-v` options toggle the output verbosity.

Our CI/CD pipeline produces a more comprehensive test report using `coverage` and `coveralls` packages. We suggest you to generate your own coverage reports locally by doing:

```sh
uv pip install coverage coveralls
python -m pytest --cov=src  tests
```

### Run linting and auto-formatting

We use [Ruff](https://docs.astral.sh/ruff/) for auto-formatting our Python modules. This package is included as part of the `[dev]` dependencies of the project, and can be run in the terminal:

```sh
ruff check .
ruff format . --check
```

Ruff auto-formatting is also a part of the GitHub workflow actions. Make sure that before you make a Pull Request, `ruff format . --check` runs in your local without any errors otherwise the workflow action will fail.

### Using VSCode for debugging

I recommend to use VSCode as your visual editor for debugging. The project comes with a `settings.json` file which already takes care of formatting everytime you save changes in a file (using [Ruff](https://docs.astral.sh/ruff/)).

You can also create your own personal debugger `launch.json`. As an example, this is my current debug file:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "nomad-simulation-workflow-utilities python",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/debug.py",
            "python": "${workspaceFolder}/.pyenv/bin/python",
            "console": "integratedTerminal",
            "justMyCode": true,
            "env": {
                "PYTHONDONTWRITEBYTECODE": "1"
            }
        },
        {
            "name": "nomad-simulation-workflow-utilities pytest",
            "type": "debugpy",
            "request": "launch",
            "module": "pytest",
            "args": [
                "-sv",
                "${workspaceFolder}/tests"
            ],
            "python": "${workspaceFolder}/.pyenv/bin/python",
            "console": "integratedTerminal",
            "justMyCode": true,
            "env": {
                "PYTHONDONTWRITEBYTECODE": "1"
            }
        },
    ]
}
```

Feel free to modify the relative paths under `"program"` or `"args"` respectively.