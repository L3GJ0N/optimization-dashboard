# Gradient Descent Analysis Dashboard

An interactive visualization tool for understanding gradient descent optimization in 2D and 3D. The dashboard allows you to:

- Visualize different optimization functions
- Analyze gradient descent steps
- Interact with the optimization process
- View multiple visualization perspectives

## Setup

### Prerequisites

- [uv](https://astral.sh/uv) package manager

### Installing uv

Install the uv package manager using the official installer:

```bash
# On Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

For more installation options, visit the [uv documentation](https://github.com/astral-sh/uv).

### Installation

1. Clone the repository

```bash
git clone https://github.com/yourusername/gradient-descent-analysis.git
cd gradient-descent-analysis
```

1. Running the application

The simplest way to run the application is using uv run:

```bash
uv run gd-analyzer
```

### Command Line Options

- `--port`: Specify custom port (default: 8050)
- `-d, --debug`: Run in debug mode

### Examples

```bash
# Run with custom port
uv run gd-analyzer --port 8080

# Run with debug mode enabled
uv run gd-analyzer --debug

# Combine options
uv run gd-analyzer --port 8080 --debug
```

After starting, open your web browser and navigate to:

- Default mode: `http://localhost:8050`
- Custom port: `http://localhost:<port>`

## Project Structure

```
fachvortrag/
├── src/                          # Source code
│   └── gradient_descent/         # Main package
│       ├── optimization/         # Optimization logic
│       ├── ui/                   # UI related code
│       └── utils/                # Utilities
├── tests/                        # Test directory
└── assets/                       # Static assets
```

## Local development

### Install development environment

```bash
uv sync --extra dev
```

### Run tests

```bash
uv run pytest tests
```

**NOTE**: To be able to run the e2e tests you may need to install the playwright dependencies once

```bash
source .venv/bin/activate
playwright install
```

## License

### MIT License

Copyright (c) 2025 Dominik Mueller

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
