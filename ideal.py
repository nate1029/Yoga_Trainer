import cv2
import mediapipe as mp
import math
from PIL import Image

mp_pose = mp.solutions.pose

# Function to calculate the angle between three points
def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return round(ang + 360 if ang < 0 else ang)

# Function to calculate joint angles from pose landmarks
def calculate_angles(poseLandmarks):
    angles = {
        'right_elbow': getAngle(poseLandmarks[16], poseLandmarks[14], poseLandmarks[12]),
        'right_shoulder': getAngle(poseLandmarks[14], poseLandmarks[12], poseLandmarks[24]),
        'left_shoulder': getAngle(poseLandmarks[13], poseLandmarks[11], poseLandmarks[23]),
        'right_knee': getAngle(poseLandmarks[24], poseLandmarks[26], poseLandmarks[28]),
        # Add more joints as needed
    }
    return angles

# Load and process image for joint detection and angle calculation
def process_image_to_ideal(image_path):
    # Load the image
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    
    # Initialize MediaPipe Pose
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        # Convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image to extract pose landmarks
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            poseLandmarks = [(int(lm.x * w), int(lm.y * h)) for lm in results.pose_landmarks.landmark]
            
            # Calculate the user's joint angles
            angles = calculate_angles(poseLandmarks)
            
            return angles
        else:
            return "Pose not detected"

# Example Usage
if __name__ == '__main__':
    # Set the image path
    image_path = 'tree.jpg'  # Replace with the actual image path
    
    # Process the image to detect angles and register them as ideal values
    ideal_angles = process_image_to_ideal(image_path)
    
    # Output the ideal angles
    if isinstance(ideal_angles, str):
        print(ideal_angles)
    else:
        print("Ideal angles registered from the image:")
        for joint, angle in ideal_angles.items():
            print(f"{joint}: {angle}°")
