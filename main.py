#original code

import cv2
import numpy as np
from functions import find_polygon_center, save_object, load_object, is_point_in_polygon, get_label_name
from ultralytics import YOLO

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
    global current_mode, polygon_data, points, template_polygon
    
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

def auto_detect_parking_spaces(frame):
    # Detect vehicles in the frame
    results = model(frame, device='cpu')[0]
    auto_spaces = []
    
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
            global box_width, box_height
            box_width = int((x2 - x1) * 1.1)  # 10% larger than detected car
            box_height = int((y2 - y1) * 1.1)
    
    return auto_spaces

# Create a black image, a window and bind the function to window
# cap = cv2.VideoCapture("Media/carPark.mp4")
# cap = cv2.VideoCapture("Media/3858833-hd_1920_1080_24fps.mp4")
cap = cv2.VideoCapture("Media/video4.mp4")
cv2.namedWindow("image")
cv2.setMouseCallback("image", draw_polygon)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (1280, 720))
    mask_1 = np.zeros_like(frame)
    mask_2 = np.zeros_like(frame)
    
    results = model(frame, device='cpu')[0]
    polygon_data_copy = polygon_data.copy()
    
    for detection in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        label_name = get_label_name(class_id)
        
        if label_name == "bicycle" or label_name == "car" or label_name == "van" or label_name == "truck" or label_name == "tricycle" or label_name == "awning-tricycle" or label_name == "bus" or label_name == "motor":
            car_polygon = [(int(x1), int(y1)), (int(x1), int(y2)), (int(x2), int(y2)), (int(x2), int(y1))]
            # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            # frame = cv2.circle(frame, car_polygon[0], 1, (255, 255, 0), 3)
            # frame = cv2.circle(frame, car_polygon[1], 1, (0, 255, 255), 3)
            # frame = cv2.circle(frame, car_polygon[2], 1, (0, 0, 255), 3)
            # frame = cv2.circle(frame, car_polygon[3], 1, (255, 0, 0), 3)
            
            for cou, i in enumerate(polygon_data_copy):
                poligon_center = find_polygon_center(i)
                # frame = cv2.circle(frame, poligon_center, 1, (255, 0, 255), 3) # center point of polygon
                is_present = is_point_in_polygon(poligon_center, car_polygon)
                if is_present == True:
                    cv2.fillPoly(mask_1, [np.array(i)], (0, 0, 255))
                    polygon_data_copy.remove(i)
    
    cv2.putText(frame,
                f'Total space : {len(polygon_data)}',
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (8, 210, 255),
                2,
                cv2.LINE_4)
    
    cv2.putText(frame,
                f'Free space : {len(polygon_data_copy)}',
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
    if current_mode == MODE_ADD_BOX and template_polygon:
        # Get mouse position
        mouse_x, mouse_y = -1, -1  # Default values
        try:
            # Get mouse position - this might not work on all systems
            x, y = cv2.getMousePos("image")
            mouse_x, mouse_y = x, y
        except:
            # If getMousePos doesn't work, we can't show the preview
            pass
        
        if mouse_x >= 0 and mouse_y >= 0:
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
        polygon_data = auto_detect_parking_spaces(frame)
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

cap.release()
cv2.destroyAllWindows()