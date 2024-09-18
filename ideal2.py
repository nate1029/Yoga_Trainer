import math
import cv2
import mediapipe as mp

# Initialize Mediapipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate angle between three points
def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return round(ang + 360 if ang < 0 else ang)

# Function to extract angles for all major joints
def extract_angles(pose_landmarks, image_width, image_height):
    joints = {}
    
    # Get pixel coordinates of relevant landmarks
    right_shoulder = (int(pose_landmarks[12].x * image_width), int(pose_landmarks[12].y * image_height))
    right_elbow = (int(pose_landmarks[14].x * image_width), int(pose_landmarks[14].y * image_height))
    right_wrist = (int(pose_landmarks[16].x * image_width), int(pose_landmarks[16].y * image_height))

    left_shoulder = (int(pose_landmarks[11].x * image_width), int(pose_landmarks[11].y * image_height))
    left_elbow = (int(pose_landmarks[13].x * image_width), int(pose_landmarks[13].y * image_height))
    left_wrist = (int(pose_landmarks[15].x * image_width), int(pose_landmarks[15].y * image_height))

    right_hip = (int(pose_landmarks[24].x * image_width), int(pose_landmarks[24].y * image_height))
    right_knee = (int(pose_landmarks[26].x * image_width), int(pose_landmarks[26].y * image_height))
    right_ankle = (int(pose_landmarks[28].x * image_width), int(pose_landmarks[28].y * image_height))

    left_hip = (int(pose_landmarks[23].x * image_width), int(pose_landmarks[23].y * image_height))
    left_knee = (int(pose_landmarks[25].x * image_width), int(pose_landmarks[25].y * image_height))
    left_ankle = (int(pose_landmarks[27].x * image_width), int(pose_landmarks[27].y * image_height))

    # Calculate angles
    joints['right_elbow'] = getAngle(right_wrist, right_elbow, right_shoulder)
    joints['left_elbow'] = getAngle(left_wrist, left_elbow, left_shoulder)

    joints['right_shoulder'] = getAngle(right_elbow, right_shoulder, right_hip)
    joints['left_shoulder'] = getAngle(left_elbow, left_shoulder, left_hip)

    joints['right_hip'] = getAngle(right_shoulder, right_hip, right_knee)
    joints['left_hip'] = getAngle(left_shoulder, left_hip, left_knee)

    joints['right_knee'] = getAngle(right_hip, right_knee, right_ankle)
    joints['left_knee'] = getAngle(left_hip, left_knee, left_ankle)

    return joints

# Load and process image
def process_image(image_path, pose_name):
    # Load image using OpenCV and check if image is loaded successfully
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}. Please check the file path.")
        return None

    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process image using Mediapipe Pose
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            h, w, _ = image.shape
            joints = extract_angles(results.pose_landmarks.landmark, w, h)
            return {pose_name: joints}
        else:
            print("No pose landmarks detected.")
            return None

# Example usage
image_path = "C:/Users/Naiteek/Downloads/YogaGuru3/tree.jpg"  # Provide the full path to your image
pose_name = "Tree"  # Replace with the appropriate pose name
ideal_angles = process_image(image_path, pose_name)

if ideal_angles:
    print("Ideal angles from the image:")
    print(ideal_angles)
