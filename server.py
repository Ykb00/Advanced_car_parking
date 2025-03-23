# Updated server.py to receive and display streams from main.py

from flask import Flask, Response, render_template, jsonify
import cv2
import numpy as np
import threading
import time
import socket
import pickle
import struct
import io

app = Flask(__name__)

# Global variables
frame_buffer = None
parking_stats = {
    'total_spaces': 0,
    'free_spaces': 0,
    'occupied_spaces': 0,
    'occupancy_rate': 0
}
stream_thread = None
streaming_active = False
client_connected = False

# Configuration options
STREAMING_HOST = '192.168.137.1'
STREAMING_PORT = 9999
JPEG_QUALITY = 70

def receive_stream():
    """Receive processed frames and stats from main.py"""
    global frame_buffer, parking_stats, client_connected
    
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    while True:
        try:
            # Connect to the main.py stream server
            print("Attempting to connect to main.py stream...")
            client_socket.connect((STREAMING_HOST, STREAMING_PORT))
            client_connected = True
            print("Connected to main.py stream server")
            
            data = b""
            payload_size = struct.calcsize("L")
            
            while True:
                # Receive message size
                while len(data) < payload_size:
                    packet = client_socket.recv(4096)
                    if not packet:
                        raise ConnectionError("Connection closed")
                    data += packet
                
                # Extract message size
                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("L", packed_msg_size)[0]
                
                # Receive full message data
                while len(data) < msg_size:
                    packet = client_socket.recv(4096)
                    if not packet:
                        raise ConnectionError("Connection closed")
                    data += packet
                
                # Extract frame and stats data
                frame_data = data[:msg_size]
                data = data[msg_size:]
                
                # Deserialize the data
                received_data = pickle.loads(frame_data)
                
                # Update frame buffer and stats
                encoded_frame = received_data['frame']
                nparr = np.frombuffer(encoded_frame, np.uint8)
                frame_buffer = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Update stats
                parking_stats = received_data['stats']
        
        except (ConnectionRefusedError, ConnectionError) as e:
            print(f"Stream connection error: {e}")
            client_connected = False
            
            # Wait before attempting to reconnect
            time.sleep(5)
            
        except Exception as e:
            print(f"Unexpected error in stream: {e}")
            client_connected = False
            
            # Wait before attempting to reconnect
            time.sleep(5)

def generate_frames():
    """Generate frames for the web client"""
    global frame_buffer, streaming_active
    
    streaming_active = True
    last_frame_time = time.time()
    
    while streaming_active:
        current_time = time.time()
        if current_time - last_frame_time < 0.04:  # ~25 FPS
            time.sleep(0.005)  # Small sleep to reduce CPU usage
            continue
            
        if frame_buffer is not None:
            # Encode the frame as JPEG
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            _, buffer = cv2.imencode('.jpg', frame_buffer, encode_params)
            frame_bytes = buffer.tobytes()
            
            # Yield the frame in the format required by HTTP streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
            last_frame_time = current_time
        else:
            # If no frame is available, yield a simple blank frame with status
            blank_frame = np.zeros((540, 960, 3), dtype=np.uint8)
            
            # Add text about connection status
            status_text = "Connecting to main.py stream..." if not client_connected else "Waiting for video..."
            cv2.putText(blank_frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            _, buffer = cv2.imencode('.jpg', blank_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.5)  # Longer sleep when no frame is available
    
    # Reset flag when client disconnects
    streaming_active = False

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    """Return parking statistics as JSON"""
    return jsonify(parking_stats)

@app.route('/connection_status')
def connection_status():
    """Return connection status to main.py stream"""
    return jsonify({
        'connected': client_connected
    })

def create_templates():
    """Create the necessary templates folder and HTML files"""
    import os
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create index.html - using the same template as in the original server.py
    with open('templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Parking Space Monitor</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f4f4f4;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            padding: 20px;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .video-container {
            width: 100%;
            text-align: center;
            margin-bottom: 20px;
        }
        .video-feed {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
        }
        .stats-container {
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            margin-top: 20px;
        }
        .stat-box {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            margin: 10px;
            min-width: 200px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }
        .total {
            color: #007bff;
        }
        .free {
            color: #28a745;
        }
        .occupied {
            color: #dc3545;
        }
        .rate {
            color: #6610f2;
        }
        .controls {
            text-align: center;
            margin-top: 15px;
        }
        button {
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 10px 15px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 4px;
        }
        footer {
            text-align: center;
            margin-top: 20px;
            color: #6c757d;
        }
        #connection-status {
            text-align: center;
            margin-top: 5px;
            font-weight: bold;
        }
        .connected {
            color: green;
        }
        .disconnected {
            color: red;
        }
        @media (max-width: 768px) {
            .stat-box {
                min-width: 120px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Parking Space Monitor</h1>
        
        <div class="video-container">
            <img id="video-feed" src="{{ url_for('video_feed') }}" class="video-feed" alt="Parking Video Feed" onerror="handleVideoError()">
            <div id="connection-status" class="connected">Connected to Server</div>
            <div id="stream-status" class="disconnected">Connecting to Stream...</div>
        </div>
        
        <div class="stats-container">
            <div class="stat-box">
                <h3>Total Spaces</h3>
                <div class="stat-value total" id="total-spaces">-</div>
            </div>
            <div class="stat-box">
                <h3>Free Spaces</h3>
                <div class="stat-value free" id="free-spaces">-</div>
            </div>
            <div class="stat-box">
                <h3>Occupied Spaces</h3>
                <div class="stat-value occupied" id="occupied-spaces">-</div>
            </div>
            <div class="stat-box">
                <h3>Occupancy Rate</h3>
                <div class="stat-value rate" id="occupancy-rate">-</div>
            </div>
        </div>
        
        <div class="controls">
            <button id="refresh-stream" onclick="refreshStream()">Refresh Stream</button>
        </div>
        
        <footer>
            Real-time Parking Detection System
        </footer>
    </div>

    <script>
        // Variables to track connection status
        let connectionLost = false;
        let retryCount = 0;
        const maxRetries = 3;
        
        // Function to update the statistics
        function updateStats() {
            fetch('/stats')
                .then(response => {
                    if (response.ok) {
                        return response.json();
                    }
                    throw new Error('Network response was not ok');
                })
                .then(data => {
                    document.getElementById('total-spaces').textContent = data.total_spaces;
                    document.getElementById('free-spaces').textContent = data.free_spaces;
                    document.getElementById('occupied-spaces').textContent = data.occupied_spaces;
                    document.getElementById('occupancy-rate').textContent = data.occupancy_rate + '%';
                    
                    // Reset connection status if it was previously lost
                    if (connectionLost) {
                        connectionLost = false;
                        updateConnectionStatus(true);
                    }
                })
                .catch(error => {
                    console.error('Error fetching stats:', error);
                    connectionLost = true;
                    updateConnectionStatus(false);
                });
                
            // Check connection status to main.py stream
            fetch('/connection_status')
                .then(response => response.json())
                .then(data => {
                    const streamStatusElement = document.getElementById('stream-status');
                    if (data.connected) {
                        streamStatusElement.textContent = 'Connected to Stream';
                        streamStatusElement.className = 'connected';
                    } else {
                        streamStatusElement.textContent = 'Not Connected to Stream';
                        streamStatusElement.className = 'disconnected';
                    }
                })
                .catch(error => {
                    console.error('Error checking stream connection:', error);
                });
        }
        
        // Function to handle video errors
        function handleVideoError() {
            retryCount++;
            if (retryCount <= maxRetries) {
                console.log(`Stream error, attempting reconnect (${retryCount}/${maxRetries})...`);
                setTimeout(refreshStream, 2000);
            } else {
                updateConnectionStatus(false);
                console.error('Failed to reconnect to video stream after multiple attempts');
            }
        }
        
        // Function to refresh the video stream
        function refreshStream() {
            const videoFeed = document.getElementById('video-feed');
            videoFeed.src = "{{ url_for('video_feed') }}?" + new Date().getTime(); // Add cache-busting parameter
            retryCount = 0;
            updateConnectionStatus(true);
        }
        
        // Function to update connection status display
        function updateConnectionStatus(connected) {
            const statusElement = document.getElementById('connection-status');
            if (connected) {
                statusElement.textContent = 'Connected to Server';
                statusElement.className = 'connected';
            } else {
                statusElement.textContent = 'Disconnected - Click Refresh Stream';
                statusElement.className = 'disconnected';
            }
        }

        // Update stats initially and then every 1.5 seconds
        updateStats();
        setInterval(updateStats, 1500);
        
        // Add event listener to check if page is visible and pause/resume accordingly
        document.addEventListener('visibilitychange', function() {
            const videoFeed = document.getElementById('video-feed');
            if (document.hidden) {
                // Page is hidden, pause video stream to save resources
                videoFeed.style.display = 'none';
            } else {
                // Page is visible again, resume video stream
                videoFeed.style.display = 'block';
                refreshStream(); // Reconnect the stream
            }
        });
    </script>
</body>
</html>
        ''')

def start_server():
    """Start the server"""
    global stream_thread
    
    # Create the necessary template files
    create_templates()
    
    # Start the stream receiving thread
    stream_thread = threading.Thread(target=receive_stream)
    stream_thread.daemon = True
    stream_thread.start()
    
    # Run the Flask app with optimized settings
    app.run(host='0.0.0.0', port=3000, debug=False, threaded=True)

if __name__ == '__main__':
    start_server()