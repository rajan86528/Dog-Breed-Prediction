import pickle
import streamlit as st
from PIL import Image, ImageOps
from classifier import image_classification
import matplotlib.pyplot as plt
import numpy as np





# AND in st.sidebar!
with st.sidebar:
      st.subheader("Choose a dog image...")
      uploaded_file = st.file_uploader("", type=["jpg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded image.', use_column_width=True)
    st.write("")
    with st.spinner('Identifying...'):
        label = image_classification(image,"new20220912-14371662993433-all-images-Adam.h5")

    btn = st.button("See Results!!")
    if btn :
      st.info(label)
