import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from ocr import OCRNeuralNetwork
import numpy as np
import os
import traceback

HOST_NAME = 'localhost'
PORT_NUMBER = 8000
HIDDEN_NODE_COUNT = 15

# load data from csv files
data_matrix = np.loadtxt(open('data.csv', 'rb'), delimiter=',')
data_labels = np.loadtxt(open('dataLabels.csv', 'rb'))

# convert to python lists
data_matrix = data_matrix.tolist()
data_labels = data_labels.tolist()

# Global neural network instance
nn = OCRNeuralNetwork(HIDDEN_NODE_COUNT, data_matrix, data_labels, list(range(5000)))

# Debug function to visualize a digit
def debug_print_digit(image_data):
    if not image_data or len(image_data) != 400:
        print("Invalid image data")
        return
    
    # Convert to 20x20 grid
    for i in range(0, 400, 20):
        row = image_data[i:i+20]
        print(''.join(['█' if p == 1 else ' ' for p in row]))
    print("\n")

# Function to reset neural network
def reset_neural_network():
    global nn
    # Delete the saved neural network file if it exists
    if os.path.exists(OCRNeuralNetwork.NN_FILE_PATH):
        os.remove(OCRNeuralNetwork.NN_FILE_PATH)
        print(f"Deleted neural network file: {OCRNeuralNetwork.NN_FILE_PATH}")
    
    # Reinitialize the neural network
    nn = OCRNeuralNetwork(HIDDEN_NODE_COUNT, data_matrix, data_labels, list(range(5000)), use_file=False)
    return True

class JSONHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            
            # Serve the HTML file
            with open('ocr.html', 'rb') as file:
                self.wfile.write(file.read())
            return
        elif self.path == '/ocr.js':
            self.send_response(200)
            self.send_header("Content-type", "application/javascript")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            
            with open('ocr.js', 'rb') as file:
                self.wfile.write(file.read())
            return
        elif self.path == '/ocr.css':
            self.send_response(200)
            self.send_header("Content-type", "text/css")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            
            with open('ocr.css', 'rb') as file:
                self.wfile.write(file.read())
            return
        else:
            # Show a basic info page for other paths
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b"<html><head><title>OCR Neural Network</title></head>")
            self.wfile.write(b"<body><h1>OCR Neural Network Server</h1>")
            self.wfile.write(b"<p>Server is running. Visit the <a href='/'>home page</a> to use the OCR demo.</p>")
            self.wfile.write(b"</body></html>")
            return

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        return

    def do_POST(self):
        response_code = 200
        response = ""
        
        var_len = int(self.headers.get('Content-Length', 0))
        content = self.rfile.read(var_len)
        
        try:
            payload = json.loads(content)
        
            if payload.get('train'):
                try:
                    print("Received training data:")
                    for i, train_item in enumerate(payload['trainArray']):
                        print(f"Item {i+1}:")
                        if 'y0' in train_item:
                            non_zero = sum(1 for p in train_item['y0'] if p == 1)
                            print(f"  Data points: {len(train_item['y0'])}, Non-zero: {non_zero}")
                        if 'label' in train_item:
                            print(f"  Label: {train_item['label']}")
                        debug_print_digit(train_item.get('y0', []))
                    
                    nn.train(payload['trainArray'])
                    nn.save()
                    response = {
                        "train": True,
                        "status": "success"
                    }
                except Exception as e:
                    print(f"Training error: {str(e)}")
                    traceback.print_exc()
                    response_code = 500
                    response = {
                        "error": "Training failed",
                        "details": str(e)
                    }
            elif payload.get('predict'):
                try:
                    # Debug: Print received image data
                    print("Received image for prediction:")
                    debug_print_digit(payload['image'])
                    
                    # Count non-zero pixels
                    non_zero = sum(1 for p in payload['image'] if p == 1)
                    print(f"Non-zero pixels: {non_zero}/400")
                    
                    # Calculate confidence scores
                    digit, confidence_scores = nn.predict_with_confidence(payload['image'])
                    print(f"Prediction result: {digit}")
                    
                    # Sort confidence scores
                    sorted_confidence = sorted(confidence_scores.items(), 
                                              key=lambda x: x[1], 
                                              reverse=True)
                    
                    response = {
                        "type": "test",
                        "result": digit,
                        "confidence": confidence_scores[digit],
                        "all_scores": confidence_scores
                    }
                except Exception as e:
                    print(f"Prediction error: {str(e)}")
                    response_code = 500
                    response = {
                        "error": "Prediction failed",
                        "details": str(e)
                    }
            elif payload.get('reset'):
                # Reset the neural network
                success = reset_neural_network()
                response = {
                    "reset": True,
                    "status": "success" if success else "failed"
                }
            else:
                response_code = 400
                response = {
                    "error": "Invalid request. Use 'train', 'predict', or 'reset' operations."
                }
        except json.JSONDecodeError:
            response_code = 400
            response = {
                "error": "Invalid JSON payload"
            }
        except Exception as e:
            print(f"Server error: {str(e)}")
            response_code = 500
            response = {
                "error": "Server error",
                "details": str(e)
            }

        self.send_response(response_code)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        
        if response:
            self.wfile.write(json.dumps(response).encode('utf-8'))
        return
    
if __name__ == '__main__':
    server_class = HTTPServer
    httpd = server_class((HOST_NAME, PORT_NUMBER), JSONHandler)
    
    print(f"Server started at http://{HOST_NAME}:{PORT_NUMBER}")
    print("Press Ctrl+C to stop the server")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    else:
        print("Unexpected server exception occurred.")
    finally:
        httpd.server_close()
        print("Server stopped.")
