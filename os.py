import cv2
import mediapipe as mp
import pyautogui
import math
import numpy as np

class HandOSController:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Get screen size for mapping coordinates
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open camera")
        
        # For smoothing cursor movement
        self.prev_x, self.prev_y = 0, 0
        self.smoothing_factor = 0.5
        
        # Click state tracking
        self.left_click_active = False
        self.right_click_active = False
        
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def is_index_up(self, landmarks):
        """Check if index finger is extended"""
        # Check if index finger tip is above all index finger joints
        index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        index_mcp = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        
        return index_tip.y < index_pip.y and index_tip.y < index_mcp.y
    
    def is_middle_up(self, landmarks):
        """Check if middle finger is extended"""
        middle_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_pip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        middle_mcp = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        
        return middle_tip.y < middle_pip.y and middle_tip.y < middle_mcp.y
    
    def is_ring_up(self, landmarks):
        """Check if ring finger is extended"""
        ring_tip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        ring_pip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_PIP]
        ring_mcp = landmarks[self.mp_hands.HandLandmark.RING_FINGER_MCP]
        
        return ring_tip.y < ring_pip.y and ring_tip.y < ring_mcp.y
    
    def is_pinky_up(self, landmarks):
        """Check if pinky finger is extended"""
        pinky_tip = landmarks[self.mp_hands.HandLandmark.PINKY_TIP]
        pinky_pip = landmarks[self.mp_hands.HandLandmark.PINKY_PIP]
        pinky_mcp = landmarks[self.mp_hands.HandLandmark.PINKY_MCP]
        
        return pinky_tip.y < pinky_pip.y and pinky_tip.y < pinky_mcp.y
    
    def is_thumb_up(self, landmarks):
        """Check if thumb is extended"""
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = landmarks[self.mp_hands.HandLandmark.THUMB_IP]
        thumb_mcp = landmarks[self.mp_hands.HandLandmark.THUMB_MCP]
        
        return thumb_tip.x > thumb_ip.x and thumb_tip.x > thumb_mcp.x
    
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Flip the frame horizontally for a mirror view
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe Hands
            results = self.hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Get hand landmarks
                    landmarks = hand_landmarks.landmark
                    
                    # Get index finger tip coordinates
                    index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    height, width, _ = frame.shape
                    
                    # Check finger states
                    index_up = self.is_index_up(landmarks)
                    middle_up = self.is_middle_up(landmarks)
                    ring_up = self.is_ring_up(landmarks)
                    pinky_up = self.is_pinky_up(landmarks)
                    thumb_up = self.is_thumb_up(landmarks)
                    
                    # Cursor movement: Only index finger up
                    if index_up and not middle_up and not ring_up and not pinky_up:
                        # Convert coordinates to screen coordinates
                        x = int(index_tip.x * self.screen_width)
                        y = int(index_tip.y * self.screen_height)
                        
                        # Smooth the movement
                        smooth_x = self.prev_x + self.smoothing_factor * (x - self.prev_x)
                        smooth_y = self.prev_y + self.smoothing_factor * (y - self.prev_y)
                        
                        # Move the cursor
                        pyautogui.moveTo(smooth_x, smooth_y)
                        
                        # Update previous coordinates
                        self.prev_x, self.prev_y = smooth_x, smooth_y
                        
                        # Display mode
                        cv2.putText(frame, "MOVE", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Left click: Index and middle fingers up
                    if index_up and middle_up and not ring_up and not pinky_up:
                        if not self.left_click_active:
                            pyautogui.click()
                            self.left_click_active = True
                            cv2.putText(frame, "LEFT CLICK", (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        self.left_click_active = False
                    
                    # Right click: Pinky finger up
                    if pinky_up and not index_up and not middle_up and not ring_up:
                        if not self.right_click_active:
                            pyautogui.rightClick()
                            self.right_click_active = True
                            cv2.putText(frame, "RIGHT CLICK", (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        self.right_click_active = False
            
            # Display the frame
            cv2.imshow('Hand OS Controller', frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = HandOSController()
    controller.run()
