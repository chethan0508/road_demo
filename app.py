import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile

# Load trained model
model = YOLO("Model/best.pt")  # change train2 if needed

st.title("Roadside Pothole Detection System")

option = st.radio("Choose input type:", ["Image", "Video"])

if option == "Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="Input Image", channels="BGR")

        results = model(img, conf=0.1)
        for r in results:
            output = r.plot()

        with col2:
            st.image(output, caption="Detected Output", channels="BGR")

else:
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=0.1)
            for r in results:
                frame = r.plot()

            stframe.image(frame, channels="BGR")


        cap.release()
