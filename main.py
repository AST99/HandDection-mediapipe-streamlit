import cv2
import streamlit as st
import mediapipe as mp
import cv2 as cv
import numpy as np
import tempfile
import time
from PIL import Image
import mode as m

DEMO_IMAGE = 'demo/demo.jpg'
DEMO_VIDEO = 'demo/demo.mp4'

# Basic App Scaffolding
st.title('Face Mesh App using Streamlit')

## Add Sidebar and Main Window style
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html=True,
)

## Create Sidebar
st.sidebar.title('Hand detection Sidebar')

## Define available pages in selection box
app_mode = st.sidebar.selectbox(
    'App Mode',
    ['À propos du projet','Alphabet','Compteur de doigts','Selfi']
)

# Resize Images to fit Container
@st.cache()
# Get Image Dimensions
def image_resize(image, width=None, height=None, inter=cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    dim = None
    # grab the image size
    (h,w) = image.shape[:2]

    if width is None and height is None:
        return image
    # calculate the ratio of the height and construct the
    # dimensions
    if width is None:
        r = width/float(w)
        dim = (int(w*r),height)
    # calculate the ratio of the width and construct the
    # dimensions
    else:
        r = width/float(w)
        dim = width, int(h*r)

    # Resize image
    resized = cv.resize(image,dim,interpolation=inter)

    return resized


# About Page

if app_mode == 'À propos du projet':
    st.markdown('''
                ## Face Mesh \n
                In this application we are using **MediaPipe** for creating a Face Mesh. **StreamLit** is to create 
                the Web Graphical User Interface (GUI) \n
    ''')

    ## Add Sidebar and Window style
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width: 350px
            margin-left: -350px
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Image Page

elif app_mode == 'Alphabet':

  st.sidebar.markdown('---')

  ## Add Sidebar and Window style
  st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width: 350px
            margin-left: -350px
        }
        </style>
        """,
        unsafe_allow_html=True,
  )
    
  st.sidebar.image('image_2.png')
  m.alphabet()
  



# Video Page

elif app_mode == 'Compteur de doigts':

    st.set_option('deprecation.showfileUploaderEncoding', False)

    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=1)

    st.sidebar.markdown('---')

    ## Add Sidebar and Window style
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width: 350px
            margin-left: -350px
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    m.modeSimple()



elif app_mode == 'Selfi':

  st.set_option('deprecation.showfileUploaderEncoding', False)

  drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=1)

  st.sidebar.markdown('---')

  ## Add Sidebar and Window style
  st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width: 350px
            margin-left: -350px
        }
        </style>
        """,
        unsafe_allow_html=True,
  )
  m.selfi()