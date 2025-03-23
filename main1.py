import cv2
import numpy as np
import threading
import socket
import pickle
import struct
from functions import find_polygon_center, save_object, load_object, is_point_in_polygon, get_label_name
from ultralytics import YOLO
import time

# Configuration for streaming
STREAMING_HOST = '0.0.0.0'
STREAMING_PORT = 9999
JPEG_QUALITY = 70
FRAME_WIDTH = 960
FRAME_HEIGHT = 540

# Flag to control streaming
streaming_enabled = True

def start_stream_server():
    """Start a socket server to stream processed frames to server.py"""
    global streaming_enabled
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((STREAMING_HOST, STREAMING_PORT))
    server_socket.listen(5)
    
    print(f"Stream server started at {STREAMING_HOST}:{STREAMING_PORT}")
    
    while streaming_enabled:
        try:
            client_socket, addr = server_socket.accept()
            print(f"Connection from {addr}")
            handle_client(client_socket)
        except Exception as e:
            print(f"Stream server error: {e}")
    
    server_socket.close()

def handle_client(client_socket):
    """Handle client connection for streaming"""
    global processed_frame, streaming_stats
    
    try:
        while streaming_enabled:
            if processed_frame is not None and streaming_stats is not None:
                # Prepare data packet with frame and stats
                data = {
                    'stats': streaming_stats
                }
                
                # Encode frame as JPEG
                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                _, encoded_frame = cv2.imencode('.jpg', processed_frame, encode_params)
                
                # Create data packet
                data['frame'] = encoded_frame.tobytes()
                
                # Serialize data
                data_bytes = pickle.dumps(data)
                
                # Send message size followed by data
                message_size = struct.pack("L", len(data_bytes))
                client_socket.sendall(message_size + data_bytes)
                
                # Rate limiting to ~25 FPS
                time.sleep(0.04)
    except Exception as e:
        print(f"Error streaming to client: {e}")
    finally:
        client_socket.close()

# Global variables to store processed frame and stats
processed_frame = None
streaming_stats = None

# Start streaming server in a separate thread
streaming_thread = threading.Thread(target=start_stream_server)
streaming_thread.daemon = True
streaming_thread.start()

def auto_detect_parking_spaces(frame, model):
    """Automatically detect parking spaces from vehicles in the frame"""
    # Detect vehicles in the frame
    results = model(frame, device='cpu')[0]
    auto_spaces = []
    
    box_width, box_height = 80, 160  # Default values
    
    for detection in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        label_name = get_label_name(class_id)
        
        # Check if it's a vehicle
        if label_name in ["car", "van", "truck", "bus", "bicycle", "tricycle", "awning-tricycle", "motor"]:
            # Create a polygon from the bounding box
            # Add some margin to make the spaces slightly larger than the vehicles
            margin = 5
            space_polygon = [
                (int(x1-margin), int(y1-margin)),
                (int(x1-margin), int(y2+margin)),
                (int(x2+margin), int(y2+margin)),
                (int(x2+margin), int(y1-margin))
            ]
            auto_spaces.append(space_polygon)
            
            # Update box size for manually added boxes based on average vehicle size
            box_width = int((x2 - x1) * 1.1)  # 10% larger than detected car
            box_height = int((y2 - y1) * 1.1)
    
    return auto_spaces, box_width, box_height

# Main processing code
def main():
    global processed_frame, streaming_stats
    
    # Load a pretrained YOLOv8n model
    model = YOLO("Models/yolov8m mAp 48/weights/best.pt")

    # List to store points
    polygon_data = load_object()
    points = []

    # Variables for modes
    MODE_DRAW_POLYGON = 0
    MODE_REMOVE_BOX = 1
    MODE_ADD_BOX = 2
    current_mode = MODE_DRAW_POLYGON

    # Size for manually added boxes
    box_width = 80
    box_height = 160

    # Template for adding new polygons
    template_polygon = []  # Will store the shape of the last drawn polygon

    def draw_polygon(event, x, y, flags, param):
        nonlocal current_mode, polygon_data, points, template_polygon
        
        if event == cv2.EVENT_LBUTTONUP:
            if current_mode == MODE_DRAW_POLYGON:
                points.append((x, y))
            elif current_mode == MODE_REMOVE_BOX:
                # Check if clicked point is inside any polygon
                to_remove = []
                for i, polygon in enumerate(polygon_data):
                    if is_point_in_polygon((x, y), polygon):
                        to_remove.append(i)
                
                # Remove polygons from highest index to lowest to avoid index shifting
                for i in sorted(to_remove, reverse=True):
                    polygon_data.pop(i)
                
                if to_remove:
                    save_object(polygon_data)
                    print(f"Removed {len(to_remove)} polygons")
            elif current_mode == MODE_ADD_BOX:
                if template_polygon:
                    # Calculate the center of the template polygon
                    template_center_x = sum(p[0] for p in template_polygon) / len(template_polygon)
                    template_center_y = sum(p[1] for p in template_polygon) / len(template_polygon)
                    
                    # Calculate offset for each point from the center of the template
                    new_polygon = []
                    for px, py in template_polygon:
                        # Calculate offset from template center
                        offset_x = px - template_center_x
                        offset_y = py - template_center_y
                        
                        # Apply offset to clicked point to create new polygon point
                        new_point = (int(x + offset_x), int(y + offset_y))
                        new_polygon.append(new_point)
                    
                    polygon_data.append(new_polygon)
                    save_object(polygon_data)
                    print(f"Added new parking space at ({x}, {y}) with the same shape as template")
                else:
                    print("No template polygon available. Draw and save a polygon first.")

    # Create a window and bind the function to it
    cap = cv2.VideoCapture("Media/video4.mp4")
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", draw_polygon)

    mouse_x, mouse_y = -1, -1

    # Function to track mouse position
    def mouse_move(event, x, y, flags, param):
        nonlocal mouse_x, mouse_y
        if event == cv2.EVENT_MOUSEMOVE:
            mouse_x, mouse_y = x, y
        # Also handle the draw_polygon functionality
        draw_polygon(event, x, y, flags, param)

    # Update mouse callback to track movement
    cv2.setMouseCallback("image", mouse_move)

    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop the video if it ends
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        mask_1 = np.zeros_like(frame)
        mask_2 = np.zeros_like(frame)
        
        results = model(frame, device='cpu')[0]
        polygon_data_copy = polygon_data.copy()
        
        for detection in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            label_name = get_label_name(class_id)
            
            if label_name in ["bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]:
                car_polygon = [(int(x1), int(y1)), (int(x1), int(y2)), (int(x2), int(y2)), (int(x2), int(y1))]
                
                for cou, i in enumerate(polygon_data_copy):
                    poligon_center = find_polygon_center(i)
                    is_present = is_point_in_polygon(poligon_center, car_polygon)
                    if is_present:
                        cv2.fillPoly(mask_1, [np.array(i)], (0, 0, 255))
                        polygon_data_copy.remove(i)
        
        # Update stats for streaming
        total_spaces = len(polygon_data)
        free_spaces = len(polygon_data_copy)
        streaming_stats = {
            'total_spaces': total_spaces,
            'free_spaces': free_spaces,
            'occupied_spaces': total_spaces - free_spaces,
            'occupancy_rate': round((total_spaces - free_spaces) / total_spaces * 100, 1) if total_spaces > 0 else 0
        }
        
        cv2.putText(frame,
                    f'Total space : {total_spaces}',
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (8, 210, 255),
                    2,
                    cv2.LINE_4)
        
        cv2.putText(frame,
                    f'Free space : {free_spaces}',
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (8, 210, 90),
                    3,
                    cv2.LINE_4)
        
        # Display current mode
        mode_text = "Mode: "
        if current_mode == MODE_DRAW_POLYGON:
            mode_text += "Draw Polygon (Default)"
        elif current_mode == MODE_REMOVE_BOX:
            mode_text += "Remove Box"
        elif current_mode == MODE_ADD_BOX:
            if template_polygon:
                mode_text += "Add Box (Template Available)"
            else:
                mode_text += "Add Box (No Template)"
        
        cv2.putText(frame,
                    mode_text,
                    (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_4)
        
        for i in polygon_data_copy:
            cv2.fillPoly(mask_2, [np.array(i)], (0, 255, 255))
        
        frame = cv2.addWeighted(mask_1, 0.2, frame, 1, 0)
        frame = cv2.addWeighted(mask_2, 0.2, frame, 1, 0)
        
        # Draw the points of the current polygon
        if current_mode == MODE_DRAW_POLYGON:
            for x, y in points:
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        
        # Draw a preview box for add box mode
        if current_mode == MODE_ADD_BOX and template_polygon and mouse_x >= 0 and mouse_y >= 0:
            # Calculate the center of the template polygon
            template_center_x = sum(p[0] for p in template_polygon) / len(template_polygon)
            template_center_y = sum(p[1] for p in template_polygon) / len(template_polygon)
            
            # Draw preview polygon
            preview_points = []
            for px, py in template_polygon:
                # Calculate offset from template center
                offset_x = px - template_center_x
                offset_y = py - template_center_y
                
                # Apply offset to mouse position
                preview_x = int(mouse_x + offset_x)
                preview_y = int(mouse_y + offset_y)
                preview_points.append((preview_x, preview_y))
            
            # Draw the preview polygon
            preview_polygon = np.array(preview_points, np.int32)
            cv2.polylines(frame, [preview_polygon], True, (0, 255, 0), 2)
        
        # Store the processed frame for streaming
        processed_frame = frame.copy()
        
        cv2.imshow("image", frame)
        
        wail_key = cv2.waitKey(1)
        if wail_key == ord("s") or wail_key == ord("S"):
            if current_mode == MODE_DRAW_POLYGON and len(points) > 0:
                # Save the current polygon as the template for future use
                template_polygon = points.copy()
                
                polygon_data.append(points)
                points = []
                save_object(polygon_data)
                print(f"Saved polygon with {len(template_polygon)} points as template")
        elif wail_key == ord("r") or wail_key == ord("R"):
            try:
                polygon_data.pop()
                save_object(polygon_data)
            except:
                pass
        elif wail_key == ord("c") or wail_key == ord("C"):  # Clear all polygons
            polygon_data = []
            save_object(polygon_data)
            print("All parking spaces cleared")
        elif wail_key == ord("a") or wail_key == ord("A"):  # Auto-detect parking spaces
            auto_spaces, new_box_width, new_box_height = auto_detect_parking_spaces(frame, model)
            polygon_data = auto_spaces
            box_width, box_height = new_box_width, new_box_height
            save_object(polygon_data)
            print(f"Auto-detected {len(polygon_data)} parking spaces")
        elif wail_key == ord("d") or wail_key == ord("D"):  # Switch to Draw Polygon mode
            current_mode = MODE_DRAW_POLYGON
            points = []  # Clear current points
            print("Switched to Draw Polygon mode")
        elif wail_key == ord("x") or wail_key == ord("X"):  # Switch to Remove Box mode
            current_mode = MODE_REMOVE_BOX
            points = []  # Clear current points
            print("Switched to Remove Box mode - click to remove parking spaces")
        elif wail_key == ord("b") or wail_key == ord("B"):  # Switch to Add Box mode
            current_mode = MODE_ADD_BOX
            points = []  # Clear current points
            if template_polygon:
                print(f"Switched to Add Box mode - click to add parking spaces with same shape as template ({len(template_polygon)} points)")
            else:
                print("Switched to Add Box mode - No template available. Draw and save a polygon first.")
        elif wail_key & 0xFF == ord("q") or wail_key & 0xFF == ord("Q"):
            break

    # Clean up before exit
    streaming_enabled = False
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()