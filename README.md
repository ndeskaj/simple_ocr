# Simple OCR Neural Network

A simple Optical Character Recognition (OCR) neural network that can recognize handwritten digits.

## Getting Started

### Prerequisites
- Python 3.x
- NumPy
- Web browser

### Installation

1. Clone the repository
2. Make sure you have the required dependencies installed:
```
pip install numpy
```

### Running the Application

1. Start the server:
```
python server.py
```

2. Open your web browser and navigate to:
```
http://localhost:8000
```

## How to Use

1. **Draw a digit**: Use your mouse to draw a digit (0-9) in the black canvas.
2. **Train the network**: Enter the digit you drew in the text field and click "Train Network".
3. **Test recognition**: Draw a digit and click "Test Recognition" to see if the neural network can recognize it.
4. **Clear the canvas**: Click "Clear Canvas" to start over.

![alt text](https://github.com/ndeskaj/simple_ocr/blob/main/images/screenshot.png)


## How It Works

This project implements a simple neural network with:
- Input layer: 400 nodes (20x20 pixel grid)
- Hidden layer: 15 nodes
- Output layer: 10 nodes (one for each digit 0-9)

The network is trained with labeled examples from the included dataset and can be further trained with user inputs.

## Files

- `server.py`: HTTP server that handles requests and interfaces with the OCR neural network
- `ocr.py`: Implementation of the neural network
- `ocr.html`, `ocr.css`, `ocr.js`: Web interface
- `data.csv`, `dataLabels.csv`: Training data
