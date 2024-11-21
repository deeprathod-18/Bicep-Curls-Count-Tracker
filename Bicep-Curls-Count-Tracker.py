import cv2
import mediapipe as mp
import numpy as np
import time

# initialize mediapipe shit
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# initialize camera
cap = cv2.VideoCapture(0)

# get captured video's dimensions
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# set window to fullscreen
cv2.namedWindow("Bicep Curl Counter", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Bicep Curl Counter", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# get screen dimensions
screen = cv2.getWindowImageRect("Bicep Curl Counter")
screen_width, screen_height = screen[2], screen[3]

# set new width and height by scaling the video
scale = min(screen_width / original_width, screen_height / original_height)
new_width = int(original_width * scale)
new_height = int(original_height * scale)

# calculate padding
pad_left = (screen_width - new_width) // 2
pad_top = (screen_height - new_height) // 2

# get new positions for text and progress bars
left_text_pos = (pad_left // 2, pad_top + 30)
right_text_pos = (screen_width - pad_left // 2 - 200, pad_top + 30)
left_bar_pos = (pad_left // 2, pad_top + 60)
right_bar_pos = (screen_width - pad_left // 2 - 200, pad_top + 60)

# init variables
left_counter = 0
right_counter = 0
left_stage = None
right_stage = None

# Calibration variables
calibration_time = 5
start_time = None
calibration_stage = "start"
left_angles = []
right_angles = []

# Colors
BLUE = (255, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    
    return angle

def draw_progress_bar(image, percentage, position, size=(200, 30), color=BLUE):
    percentage = max(0, min(percentage, 1))
    start_point = position
    end_point = (position[0] + size[0], position[1] + size[1])
    cv2.rectangle(image, start_point, end_point, WHITE, 2)
    filled_end_point = (int(position[0] + size[0] * percentage), position[1] + size[1])
    cv2.rectangle(image, start_point, filled_end_point, color, -1)

def draw_text_with_outline(image, text, position, font_scale=0.7, color=WHITE, thickness=2):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness * 3, cv2.LINE_AA)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def is_arm_visible(shoulder, elbow, wrist):
    return all(0 <= coord <= 1 for point in [shoulder, elbow, wrist] for coord in point)

def is_wrist_pronated(wrist, pinky, index):
    wrist_to_pinky = np.array(pinky) - np.array(wrist)
    wrist_to_index = np.array(index) - np.array(wrist)
    cross_product = np.cross(wrist_to_pinky, wrist_to_index)
    return cross_product > 0  # Positive for pronated wrist (may need to be reversed depending on coordinate system)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize frame while maintaining aspect ratio
    resized_frame = cv2.resize(frame, (new_width, new_height))
    
    # Create a black canvas of screen size
    canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    
    # Place the resized frame on the canvas
    canvas[pad_top:pad_top+new_height, pad_left:pad_left+new_width] = resized_frame
    
    # Convert the BGR image to RGB
    image = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # Make detection
    results = pose.process(image)
    
    # Convert back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    try:
        landmarks = results.pose_landmarks.landmark
        
        # Left Arm
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        left_pinky = [landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].y]
        left_index = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y]
        
        # Right Arm
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        right_pinky = [landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].y]
        right_index = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y]
        
        # Check arm visibility and wrist pronation
        left_arm_visible = is_arm_visible(left_shoulder, left_elbow, left_wrist)
        right_arm_visible = is_arm_visible(right_shoulder, right_elbow, right_wrist)
        left_wrist_pronated = is_wrist_pronated(left_wrist, left_pinky, left_index)
        right_wrist_pronated = is_wrist_pronated(right_wrist, right_pinky, right_index)
        
        # Calculate angles for visible arms
        if left_arm_visible:
            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        if right_arm_visible:
            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # Calibration phase
        if calibration_stage == "start":
            draw_text_with_outline(image, "Calibration: Extend arms fully", (pad_left // 2, screen_height - pad_top - 60))
            draw_text_with_outline(image, "Press 'c' to start", (pad_left // 2, screen_height - pad_top - 30))
            if cv2.waitKey(10) & 0xFF == ord('c'):
                calibration_stage = "max"
                start_time = time.time()
        
        elif calibration_stage == "max":
            elapsed_time = time.time() - start_time
            remaining_time = max(0, calibration_time - elapsed_time)
            draw_text_with_outline(image, f"Calibrating maximum angle: {remaining_time:.1f}s", (pad_left // 2, screen_height - pad_top - 30))
            if left_arm_visible:
                left_angles.append(left_angle)
            if right_arm_visible:
                right_angles.append(right_angle)
            if elapsed_time >= calibration_time:
                max_left_angle = np.mean(left_angles) if left_angles else None
                max_right_angle = np.mean(right_angles) if right_angles else None
                left_angles.clear()
                right_angles.clear()
                calibration_stage = "min"
                start_time = time.time()
        
        elif calibration_stage == "min":
            elapsed_time = time.time() - start_time
            remaining_time = max(0, calibration_time - elapsed_time)
            draw_text_with_outline(image, "Curl arms fully", (pad_left // 2, screen_height - pad_top - 60))
            draw_text_with_outline(image, f"Calibrating minimum angle: {remaining_time:.1f}s", (pad_left // 2, screen_height - pad_top - 30))
            if left_arm_visible:
                left_angles.append(left_angle)
            if right_arm_visible:
                right_angles.append(right_angle)
            if elapsed_time >= calibration_time:
                min_left_angle = np.mean(left_angles) if left_angles else None
                min_right_angle = np.mean(right_angles) if right_angles else None
                calibration_stage = "done"
        
        elif calibration_stage == "done":
            # Visualize angles for visible arms
            if left_arm_visible:
                cv2.putText(image, f"L: {left_angle:.2f}", 
                            (pad_left + 10, pad_top + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2, cv2.LINE_AA)
            if right_arm_visible:
                cv2.putText(image, f"R: {right_angle:.2f}", 
                            (screen_width - pad_left - 100, pad_top + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2, cv2.LINE_AA)
            
            # Curl counter logic and visual feedback
            if left_arm_visible and max_left_angle is not None and min_left_angle is not None:
                left_percentage = (left_angle - min_left_angle) / (max_left_angle - min_left_angle)
                left_color = BLUE
                if left_angle > (max_left_angle - 10) and left_wrist_pronated:
                    left_stage = "down"
                    left_color = RED
                if left_angle < (min_left_angle + 10) and left_stage == "down":
                    left_stage = "up"
                    left_counter += 1
                    left_color = GREEN
                
                # Draw left arm line with color feedback
                cv2.line(image, tuple(np.multiply(left_shoulder, [image.shape[1], image.shape[0]]).astype(int)),
                         tuple(np.multiply(left_elbow, [image.shape[1], image.shape[0]]).astype(int)), left_color, 3)
                cv2.line(image, tuple(np.multiply(left_elbow, [image.shape[1], image.shape[0]]).astype(int)),
                         tuple(np.multiply(left_wrist, [image.shape[1], image.shape[0]]).astype(int)), left_color, 3)
                
                # Draw left progress bar
                draw_progress_bar(image, left_percentage, left_bar_pos, color=left_color)
                
                # Display left curl counter
                draw_text_with_outline(image, f"Left Reps: {left_counter}", left_text_pos, color=YELLOW)
            
            if right_arm_visible and max_right_angle is not None and min_right_angle is not None:
                right_percentage = (right_angle - min_right_angle) / (max_right_angle - min_right_angle)
                right_color = BLUE
                if right_angle > (max_right_angle - 10) and right_wrist_pronated:
                    right_stage = "down"
                    right_color = RED
                if right_angle < (min_right_angle + 10) and right_stage == "down":
                    right_stage = "up"
                    right_counter += 1
                    right_color = GREEN
                
                # Draw right arm line with color feedback
                cv2.line(image, tuple(np.multiply(right_shoulder, [image.shape[1], image.shape[0]]).astype(int)),
                         tuple(np.multiply(right_elbow, [image.shape[1], image.shape[0]]).astype(int)), right_color, 3)
                cv2.line(image, tuple(np.multiply(right_elbow, [image.shape[1], image.shape[0]]).astype(int)),
                         tuple(np.multiply(right_wrist, [image.shape[1], image.shape[0]]).astype(int)), right_color, 3)
                
                # Draw right progress bar
                draw_progress_bar(image, right_percentage, right_bar_pos, color=right_color)
                
                # Display right curl counter
                draw_text_with_outline(image, f"Right Reps: {right_counter}", right_text_pos, color=YELLOW)
    
    except:
        pass
    
    # Render detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    cv2.imshow('Bicep Curl Counter', image)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()