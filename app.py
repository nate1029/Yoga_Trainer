import streamlit as st
import numpy as np
import math, pickle
from PIL import Image
import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Import model
load_model = pickle.load(open('YogaModel.pkl', 'rb'))

# Define ideal angles for each yoga pose
ideal_angles = {
    'Tree': {
        'right_elbow': 144,
        'right_shoulder': 179,
        'left_shoulder': 171,
        'right_knee': 180,
        # Add more joint angles based on the ideal Tree Pose posture
    },
    'Mountain': {
        'right_elbow': 169,
        'right_shoulder': 269,
        'left_shoulder': 104,
        'right_knee': 114,
        # Add more joint angles based on the ideal Mountain Pose posture
    },
    'Warrior2': {
        'right_elbow': 175,
        'right_shoulder': 171,
        'left_shoulder': 194,
        'right_knee': 16,
        # Add more joint angles based on the ideal Warrior2 Pose posture
    }
}

# Function to calculate the angle between three points
def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return round(ang + 360 if ang < 0 else ang)

# Function to calculate the feature list (angles)
def feature_list(poseLandmarks, posename):
    return [
        getAngle(poseLandmarks[16], poseLandmarks[14], poseLandmarks[12]),
        getAngle(poseLandmarks[14], poseLandmarks[12], poseLandmarks[24]),
        getAngle(poseLandmarks[13], poseLandmarks[11], poseLandmarks[23]),
        getAngle(poseLandmarks[15], poseLandmarks[13], poseLandmarks[11]),
        getAngle(poseLandmarks[12], poseLandmarks[24], poseLandmarks[26]),
        getAngle(poseLandmarks[11], poseLandmarks[23], poseLandmarks[25]),
        getAngle(poseLandmarks[24], poseLandmarks[26], poseLandmarks[28]),
        getAngle(poseLandmarks[23], poseLandmarks[25], poseLandmarks[27]),
        getAngle(poseLandmarks[26], poseLandmarks[28], poseLandmarks[32]),
        getAngle(poseLandmarks[25], poseLandmarks[27], poseLandmarks[31]),
        getAngle(poseLandmarks[0], poseLandmarks[12], poseLandmarks[11]),
        getAngle(poseLandmarks[0], poseLandmarks[11], poseLandmarks[12]),
        posename
    ]

# Function to compare user's angles with ideal angles
def compare_with_ideal(poseLandmarks, pose_name):
    angles = {
        'right_elbow': getAngle(poseLandmarks[16], poseLandmarks[14], poseLandmarks[12]),
        'right_shoulder': getAngle(poseLandmarks[14], poseLandmarks[12], poseLandmarks[24]),
        'left_shoulder': getAngle(poseLandmarks[13], poseLandmarks[11], poseLandmarks[23]),
        'right_knee': getAngle(poseLandmarks[24], poseLandmarks[26], poseLandmarks[28]),
        # Add more joints
    }
    
    ideal = ideal_angles[pose_name]
    feedback = {}
    
    for joint, angle in angles.items():
        ideal_angle = ideal[joint]
        deviation = abs(angle - ideal_angle)
        
        if deviation > 5:  # You can adjust this tolerance level
            feedback[joint] = f"{angle}° (Adjust by {deviation}° to match ideal {ideal_angle}°)"
        else:
            feedback[joint] = f"{angle}° (Ideal: {ideal_angle}°)"
    
    return feedback

# Set the layout width wide 
st.set_page_config(layout="wide")

# Sidebar for selecting the pose
st.sidebar.title('YogaGuru')
app_mode = st.sidebar.selectbox('Select The Pose', ['Tree', 'Mountain', 'Warrior2'])

# Display based on selected pose
if app_mode:
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.write(f"{app_mode} Pose")
        image = Image.open(f'{app_mode.lower()}.jpg')
        st.image(image, caption=f'{app_mode} Pose')
    
    with col2:
        st.write("Webcam Live Feed")
        button = st.empty()
        start = button.button('Start')
        
        if start:
            stop = button.button('Stop')
            visible_message = st.empty()
            FRAME_WINDOW = st.image([])
            accuracytxtbox = st.empty()
            cap = cv2.VideoCapture(0)
            
            # Create placeholders for each joint angle feedback
            feedback_boxes = {joint: st.empty() for joint in ideal_angles[app_mode]}
            
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                while cap.isOpened():
                    ret, frame = cap.read()
                    h, w, c = frame.shape
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = pose.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                    # Render detections
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
                    
                    FRAME_WINDOW.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    
                    poseLandmarks = []
                    if results.pose_landmarks:
                        for lm in results.pose_landmarks.landmark:
                            poseLandmarks.append((int(lm.x * w), int(lm.y * h)))
                    
                    if len(poseLandmarks) == 0:
                        visible_message.text("Body Not Visible")
                        accuracytxtbox.text('')
                        continue
                    else:
                        visible_message.text("")
                        
                        # Calculate accuracy
                        d = feature_list(poseLandmarks, 1)
                        rt_accuracy = int(round(load_model.predict(np.array(d).reshape(1, -1))[0], 0))
                        
                        # Output accuracy
                        if rt_accuracy < 75:
                            accuracytxtbox.text(f"Accuracy : Not so Good {rt_accuracy}")
                        elif 75 <= rt_accuracy < 85:
                            accuracytxtbox.text(f"Accuracy : Good {rt_accuracy}")
                        elif 85 <= rt_accuracy < 95:
                            accuracytxtbox.text(f"Accuracy : Very Good {rt_accuracy}")
                        elif 95 <= rt_accuracy < 100:
                            accuracytxtbox.text(f"Accuracy : Near to perfection {rt_accuracy}")
                        else:
                            accuracytxtbox.text(f"Accuracy : You reached your goal perfection 100")
                        
                        # Compare with ideal angles and update feedback in place
                        feedback = compare_with_ideal(poseLandmarks, app_mode)
                        for joint, text in feedback.items():
                            feedback_boxes[joint].text(f"{joint}: {text}")
                        
                    if stop:
                        cap.release()
                        cv2.destroyAllWindows()
else:
    st.write('Your Camera is Not Detected !')
