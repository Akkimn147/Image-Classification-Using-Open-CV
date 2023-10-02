import streamlit as st
from PIL import Image
import io
import base64
import util
from util import classify_image, get_b64_test_image_for_simrat, get_cropped_image_if_2_eyes, get_cv2_image_from_base64_string, cv2, w2d

import os
import tempfile
import streamlit as st
from PIL import Image
import joblib
import numpy as np

loaded_model = joblib.load('saved_model.pkl')
# List of image file paths
image_paths = ['E:\\Learn\Data S\\Image Data set\\Image Classification\\server\\static\\imagesofact\\1.jpg', 
               'E:\\Learn\Data S\\Image Data set\\Image Classification\\server\\static\\imagesofact\\2.jpg', 
               'E:\\Learn\Data S\\Image Data set\\Image Classification\\server\\static\\imagesofact\\3.jpg',
               'E:\\Learn\Data S\\Image Data set\\Image Classification\\server\\static\\imagesofact\\4.jpg',
               'E:\\Learn\Data S\\Image Data set\\Image Classification\\server\\static\\imagesofact\\5.jpg',
               'E:\\Learn\Data S\\Image Data set\\Image Classification\\server\\static\\imagesofact\\6.jpg',
               'E:\\Learn\Data S\\Image Data set\\Image Classification\\server\\static\\imagesofact\\7.jpg']
names = ["Pooje Hegde", "Angelina Jolie", "Hande Ercel", "Jethalal Gada", "John Statham", "Milana Nagraj", "Simrat Kaur"]
image_width = 100  # Adjust this value as needed

# def image_to_base64(uploaded_image):
#     if uploaded_image is not None:
#         image_bytes = uploaded_image.read()
#         base64_string = base64.b64encode(image_bytes).decode()
#         return base64_string
#     return None

# Create a Streamlit column layout for displaying images in a row
columns = st.columns(len(image_paths))

# Loop through the image paths and display each image in a column
for i, image_path in enumerate(image_paths):
    with columns[i]:
        image = Image.open(image_path)
        st.image(image, caption=f'{names[i]}', width=image_width)

temp_dir = tempfile.mkdtemp()
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
file_path = ''

if uploaded_file is not None:
    # Save the uploaded file to the temporary directory
    with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.read())
    
    # Get the complete file path as a string
    file_path = os.path.join(temp_dir, uploaded_file.name)

    # Display the file pathf
    # st.write(f"File path: {file_path}")
    if st.checkbox("Show file path"):
        st.write(f"File path: {file_path}")
    
if file_path:
    column = st.columns(1)
    im = Image.open(file_path)
    st.image(im, width=image_width)
    
if file_path:
    img = cv2.imread(file_path)
    scalled_raw_img = cv2.resize(img, (32, 32))
    img_har = w2d(img,'db1',5)
    scalled_img_har = cv2.resize(img_har, (32, 32))
    combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1),scalled_img_har.reshape(32*32,1)))
    final_image = np.array(combined_img).astype(float)
    final_image = final_image.reshape(1, 4096)

                    
    
if st.button('Classify'):
    names= classify_image(None, file_path)
    names
    
if __name__ == "__main__":
    util.load_saved_artifacts()
    