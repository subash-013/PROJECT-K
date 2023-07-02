import streamlit as st
import numpy as np
import tensorflow as tf

def FindSymbol(img):
    traffic_model = tf.keras.models.load_model("sign_detector.h5")
    img = tf.keras.preprocessing.image.load_img(img)
    img = img.resize((30,30))
    img = np.array(img)
    img = np.expand_dims(img,axis=0)
    img = np.array(img)
    result_vector = traffic_model.predict(img)
    result = np.argmax(result_vector,axis=1)
    return result

traffic_symbols = {
        0: 'Speed limit (20km/h)',
        1: 'Speed limit (30km/h)',
        2: 'Speed limit (50km/h)',
        3: 'Speed limit (60km/h)',
        4: 'Speed limit (70km/h)',
        5: 'Speed limit (80km/h)',
        6: 'End of speed limit (80km/h)',
        7: 'Speed limit (100km/h)',
        8: 'Speed limit (120km/h)',
        9: 'No passing',
        10: 'No passing for vehicles over 3.5 metric tons',
        11: 'Right-of-way at the next intersection',
        12: 'Priority road',
        13: 'Yield',
        14: 'Stop',
        15: 'No vehicles',
        16: 'Vehicles over 3.5 metric tons prohibited',
        17: 'No entry',
        18: 'General caution',
        19: 'Dangerous curve to the left',
        20: 'Dangerous curve to the right',
        21: 'Double curve',
        22: 'Bumpy road',
        23: 'Slippery road',
        24: 'Road narrows on the right',
        25: 'Road work',
        26: 'Traffic signals',
        27: 'Pedestrians',
        28: 'Children crossing',
        29: 'Bicycles crossing',
        30: 'Beware of ice/snow',
        31: 'Wild animals crossing',
        32: 'End of all speed and passing limits',
        33: 'Turn right ahead',
        34: 'Turn left ahead',
        35: 'Ahead only',
        36: 'Go straight or right',
        37: 'Go straight or left',
        38: 'Keep right',
        39: 'Keep left',
        40: 'Roundabout mandatory',
        41: 'End of no passing',
        42: 'End of no passing by vehicles over 3.5 metric tons'
        }

st.title("TRAFFIC SIGN PREDICTOR")
upload_img = st.file_uploader(label="Upload The Image Here")
if upload_img is not None:
    st.image(upload_img,caption="Uploaded Image")
    if st.button("Predict"):
        st.write("**The Predicted Sign:-**")
        class_label = FindSymbol(upload_img)
        st.markdown(traffic_symbols[class_label[0]])