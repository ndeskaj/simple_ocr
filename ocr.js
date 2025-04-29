var ocrDemo = {
    CANVAS_WIDTH: 200,
    TRANSLATED_WIDTH: 20,
    PIXEL_WIDTH: 10, // TRANSLATED_WIDTH = CANVAS_WIDTH / PIXEL_WIDTH

    BATCH_SIZE: 1,

    // Server Variables
    PORT: 8000,
    HOST: "http://localhost",

    // Colors
    BLACK: "#000000",
    BLUE: "#0000ff",
    WHITE: "#ffffff",

    trainArray: [],
    trainingRequestCount: 0,

    onLoadFunction: function() {
        this.resetCanvas();
        this.clearResult();
    },

    resetCanvas: function() {
        var canvas = document.getElementById('canvas');
        var ctx = canvas.getContext('2d');

        this.data = [];
        ctx.fillStyle = this.BLACK;
        ctx.fillRect(0, 0, this.CANVAS_WIDTH, this.CANVAS_WIDTH);
        var matrixSize = 400;
        while (matrixSize--) this.data.push(0);
        this.drawGrid(ctx);

        canvas.onmousemove = function(e) { this.onMouseMove(e, ctx, canvas) }.bind(this);
        canvas.onmousedown = function(e) { this.onMouseDown(e, ctx, canvas) }.bind(this);
        canvas.onmouseup = function(e) { this.onMouseUp(e, ctx) }.bind(this);
        
        this.clearResult();
    },
    
    clearResult: function() {
        var resultDiv = document.getElementById('result');
        if (resultDiv) {
            resultDiv.innerHTML = '';
        }
    },
    
    showResult: function(message, isError) {
        var resultDiv = document.getElementById('result');
        if (resultDiv) {
            resultDiv.innerHTML = message;
            resultDiv.className = isError ? 'error' : 'success';
        }
    },
    
    drawGrid: function(ctx){
        for (var x = this.PIXEL_WIDTH, y = this.PIXEL_WIDTH;
                 x < this.CANVAS_WIDTH; x += this.PIXEL_WIDTH, 
                 y += this.PIXEL_WIDTH) {
                    ctx.strokeStyle = this.BLUE;
                    ctx.beginPath();
                    ctx.moveTo(x, 0);
                    ctx.lineTo(x, this.CANVAS_WIDTH);
                    ctx.stroke();

                    ctx.beginPath();
                    ctx.moveTo(0, y);
                    ctx.lineTo(this.CANVAS_WIDTH, y);
                    ctx.stroke();
        }
    },

    onMouseMove: function(e, ctx, canvas) {
        if (!canvas.isDrawing) {
            return;
        }
        this.fillSquare(ctx, 
            e.clientX - canvas.offsetLeft, 
            e.clientY - canvas.offsetTop);
    },

    onMouseDown: function(e, ctx, canvas) {
        canvas.isDrawing = true;
        this.fillSquare(ctx, 
            e.clientX - canvas.offsetLeft, 
            e.clientY - canvas.offsetTop);
    },

    onMouseUp: function(e) {
        canvas.isDrawing = false;
    },

    fillSquare: function(ctx, x, y){
        var xPixel = Math.floor(x / this.PIXEL_WIDTH);
        var yPixel = Math.floor(y / this.PIXEL_WIDTH);
        
        // Make sure we're within bounds
        if (xPixel < 0 || xPixel >= this.TRANSLATED_WIDTH || 
            yPixel < 0 || yPixel >= this.TRANSLATED_WIDTH) {
            return;
        }

        ctx.fillStyle = this.WHITE;
        ctx.fillRect(xPixel * this.PIXEL_WIDTH, yPixel * this.PIXEL_WIDTH, 
            this.PIXEL_WIDTH, this.PIXEL_WIDTH);
            
        // Update the data array (1 = white, 0 = black)
        var dataIndex = yPixel * this.TRANSLATED_WIDTH + xPixel;
        if (dataIndex >= 0 && dataIndex < 400) {
            this.data[dataIndex] = 1;
        }
    },

    train: function() {
        var digitalVal = document.getElementById('digit').value;
        if (!digitalVal || digitalVal === '' || this.data.indexOf(1) < 0){
            this.showResult("Please draw a digit and enter its value (0-9) to train the network", true);
            return;
        }
        
        if (!/^[0-9]$/.test(digitalVal)) {
            this.showResult("Please enter a single digit (0-9)", true);
            return;
        }
        
        this.trainArray.push({"y0": this.data, "label": parseInt(digitalVal)});
        this.trainingRequestCount++;

        if (this.trainingRequestCount == this.BATCH_SIZE){
            this.showResult("Sending training data to server...", false);
            var json = {
                trainArray: this.trainArray,
                train: true
            };

            this.sendData(json);
            this.trainingRequestCount = 0;
            this.trainArray = [];
        }
    },

    test: function() {
        if (this.data.indexOf(1) < 0){
            this.showResult("Please draw a digit to test the network", true);
            return;
        }

        this.showResult("Analyzing digit...", false);
        var json = {
            image: this.data,
            predict: true
        };
        this.sendData(json);
    },

    receiveResponse: function(xmlHttp) {
        if (xmlHttp.status != 200){
            this.showResult("Server error: " + xmlHttp.status, true);
            return; 
        }
        var responseJSON = JSON.parse(xmlHttp.responseText);
        if (xmlHttp.responseText && responseJSON.type == "test"){
            // Create detailed result HTML
            var resultHTML = "<div class='prediction-main'>The neural network predicts: <span class='prediction-digit'>" + responseJSON.result + "</span></div>";
            
            // Add confidence percentage
            if (responseJSON.confidence) {
                var confidence = Math.round(responseJSON.confidence * 100);
                resultHTML += "<div class='confidence'>Confidence: " + confidence + "%</div>";
            }
            
            // Add all scores if available
            if (responseJSON.all_scores) {
                resultHTML += "<div class='all-scores'><h4>All confidence scores:</h4><table>";
                
                // Sort digits by confidence
                var sortedScores = [];
                for (var digit in responseJSON.all_scores) {
                    sortedScores.push({
                        digit: digit,
                        score: responseJSON.all_scores[digit]
                    });
                }
                
                sortedScores.sort(function(a, b) {
                    return b.score - a.score;
                });
                
                // Add rows for each digit
                for (var i = 0; i < sortedScores.length; i++) {
                    var item = sortedScores[i];
                    var percentage = Math.round(item.score * 100);
                    var barWidth = Math.max(1, percentage);
                    
                    resultHTML += "<tr>" +
                        "<td class='digit'>" + item.digit + "</td>" +
                        "<td class='score-bar'><div class='bar' style='width: " + barWidth + "%;'></div></td>" +
                        "<td class='score-value'>" + percentage + "%</td>" +
                    "</tr>";
                }
                
                resultHTML += "</table></div>";
            }
            
            this.showResult(resultHTML, false);
        } else if (responseJSON.train) {
            this.showResult("Training successful! Network updated.", false);
            this.resetCanvas();
        }
    }, 

    onError: function(e) {
        this.showResult("Error connecting to server: " + e.target.statusText, true);
    },

    sendData: function(json){
        var xmlHttp = new XMLHttpRequest();
        xmlHttp.open("POST", this.HOST + ":" + this.PORT);
        xmlHttp.onload = function() {
            this.receiveResponse(xmlHttp);
        }.bind(this);
        xmlHttp.onerror = function() {
            this.onError(xmlHttp);
        }.bind(this);
        var msg = JSON.stringify(json);
        xmlHttp.setRequestHeader('Content-length', msg.length);
        xmlHttp.setRequestHeader("Connection", "close");
        xmlHttp.setRequestHeader('Content-Type', 'application/json');
        xmlHttp.send(msg);
    },
    
    resetNeuralNetwork: function() {
        if (!confirm("Are you sure you want to reset the neural network? This will delete all training data and reset to initial state.")) {
            return;
        }
        
        this.showResult("Resetting neural network...", false);
        
        var json = {
            reset: true
        };
        
        var xmlHttp = new XMLHttpRequest();
        xmlHttp.open("POST", this.HOST + ":" + this.PORT);
        xmlHttp.onload = function() {
            if (xmlHttp.status == 200) {
                var response = JSON.parse(xmlHttp.responseText);
                if (response.reset && response.status === "success") {
                    this.showResult("Neural network has been reset successfully!", false);
                    this.resetCanvas();
                } else {
                    this.showResult("Error resetting neural network: " + response.error, true);
                }
            } else {
                this.showResult("Server error: " + xmlHttp.status, true);
            }
        }.bind(this);
        xmlHttp.onerror = function() {
            this.onError(xmlHttp);
        }.bind(this);
        xmlHttp.setRequestHeader('Content-Type', 'application/json');
        xmlHttp.send(JSON.stringify(json));
    }
}