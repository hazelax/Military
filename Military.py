import streamlit as st
from PIL import Image
import cv2
import numpy as np

# File upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run detection
    results = model.predict(source=uploaded_file)
    boxes = results[0].boxes

    output_image = np.array(image.copy())

    # Draw detections
    for box in boxes:
        cls_id = int(box.cls)
        label = results[0].names[cls_id]
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # get box coordinates

        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output_image, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Print detection info
        st.write(f"Detected **{label}** with confidence **{conf:.2f}**")

    st.image(output_image, caption="Detection Results", use_column_width=True)
