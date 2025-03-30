# Gradient Descent Analysis Dashboard

An interactive visualization tool for understanding gradient descent optimization in 2D and 3D. The dashboard allows you to:
- Visualize different optimization functions
- Analyze gradient descent steps
- Interact with the optimization process
- View multiple visualization perspectives

## Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/gradient-descent-analysis.git
cd gradient-descent-analysis
```

2. Create a virtual environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Running the Application

The dashboard can be started with default settings using:
```bash
python main.py
```

### Command Line Options

- `--port`: Specify custom port (default: 8050)
- `-d, --debug`: Run in debug mode

Example with custom port and debug mode:
```bash
python main.py --port 8080 -d
```

After starting, open your web browser and navigate to:
- Default mode: `http://localhost:8050`
- Custom port: `http://localhost:<port>`

## Development

To run the application in development mode with hot reloading:
```bash
python main.py -d
```

## License

[Add your license information here]
