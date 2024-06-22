import streamlit as st
from ultralytics import YOLO
import cvzone
import cv2
import numpy as np
from PIL import Image
import math
import os
import tempfile
# # Load YOLO model
model = YOLO('D:/virtual/test/fire.pt')

# # Reading the classes
classnames = ['fire']

# Create a Streamlit app
st.title("Forest Fire Detection")
st.write("Select an option to predict:")

# Add a selectbox to choose between uploading an image, uploading a video, and using the webcam
option = st.selectbox("Options", ["Upload Image", "Upload Video"])


def process_frame(frame):
    result = model(frame, stream=True)

    # Getting bbox,confidence and class names informations to work with
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence > 50:
                x1,y1,x2,y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),5)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                   scale=1.5,thickness=2)
                
    return frame

# If the user chooses to upload an image
if option == "Upload Image":
    # Create a file uploader
    uploaded_file = st.file_uploader("Select an image", type=["jpg", "jpeg", "png"])

    # If an image is uploaded
    if uploaded_file is not None:
        try:
            # Read the image using PIL
            image = Image.open(uploaded_file)
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (640, 480))  # Resize to 640x480

            # Process the frame and display
            frame = process_frame(frame)
            st.image(frame, channels="BGR", caption="Output Image")
            
        except Exception as e:
            st.error(f"An error occurred: {e}")


elif option == "Upload Video":
        # Create a file uploader for videos
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

        if uploaded_file is not None:
            try:
                # Save the uploaded video to a temporary file
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())

                cap = cv2.VideoCapture(tfile.name)
                
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                progress_bar = st.progress(0)

                # Process and display frames from the video
                frame_display = st.empty()
                current_frame = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = cv2.resize(frame, (640, 480))
                    processed_frame = process_frame(frame)

                    # Display the processed frame in Streamlit
                    frame_display.image(processed_frame, channels="BGR")
                    
                    # Update progress bar
                    current_frame += 1
                    progress_bar.progress(current_frame / frame_count)

                cap.release()

            except Exception as e:
                st.error(f"An error occurred: {e}")


# # If the user chooses to use the webcam
# elif option == "Use Webcam":
#     st.write("Press 'Start' to begin webcam feed and 'Stop' to end it.")
    
#     start_button = st.button("Start")
#     stop_button = st.button("Stop")
    
#     if start_button:
#         FRAME_WINDOW = st.image([])
#         cap = cv2.VideoCapture(0)
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 st.error("Failed to capture image from webcam.")
#                 break

#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame = process_frame(frame)

#             # Display the output image with bounding boxes
#             FRAME_WINDOW.image(frame)
           
#             if stop_button:
#                 cap.release()
#                 break
#         cv2.destroyAllWindows()



