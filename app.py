# Importing dependencies
from skimage.metrics import structural_similarity as compare_ssim
import streamlit as st
from PIL import Image
import numpy as np
import cv2

def getRes(im1,im2):
# im1="./input1.png"
# im2="./input2.png"

    imageA = im1
    imageB = im2

    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    print("SSIM: {}".format(score))

    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    for c in contours:
        area = cv2.contourArea(c)
        if area > 40:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(imageB, (x, y), (x + w, y + h), (0,0,0), 2)

    # cv2.imshow('diff', contours)
    # cv2.imshow('thresh', thresh)
    # cv2.waitKey(1)
    cv2.waitKey()
    st.write(f"Similarity Score: {score}")
    return imageB

def main():
    st.set_page_config(page_title="FindDifferece", page_icon="🔍")
    st.header("Find difference between two images 🖼️")
    im1=st.file_uploader("Upload image 1", type=["png", "jpg", "jpeg"])
    im2=st.file_uploader("Upload image 2", type=["png", "jpg", "jpeg"])
    if st.button("Process"):
        im1 = Image.open(im1)
        im2 = Image.open(im2)
        im1 = np.array(im1)
        im2 = np.array(im2)
        st.image(im1, caption='Image 1')
        st.image(im2, caption='Image 2')
        im=getRes(im1, im2)
        st.image(im, caption='Difference in Images')

if __name__ == "__main__":
    main()