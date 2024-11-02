import cv2
import pickle
import numpy as np
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# Load model
model_dict = pickle.load(open('./mymodel.p', 'rb'))
model = model_dict['model']

# Initialize Mediapipe Hand model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define Video Processor
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        H, W, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        data_aux = []
        x_, y_ = [], []

        results = self.hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                # Define bounding box
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) + 10
                y2 = int(max(y_) * H) + 10

                # Make prediction
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = prediction[0]

                # Draw bounding box and prediction text
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
                cv2.putText(img, str(predicted_character), (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit App
def main():
    st.title("Real-time Hand Sign Detection App")

    st.header("Live Feed")

    webrtc_streamer(
        key="hand-sign-detection",
        video_processor_factory=VideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False}
    )

if __name__ == "__main__":
    main()
