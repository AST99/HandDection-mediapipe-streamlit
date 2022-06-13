import cv2
import streamlit as st
import mediapipe as mp
import cv2 as cv
import numpy as np
import mode as m

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,960)

## Create Sidebar
st.sidebar.title('Hand detection Sidebar')


st.title('Projet réaliser par TOURE Abou-bakar Sidik et BOMI Gael Victoire')

## Define available pages in selection box
app_mode = st.sidebar.selectbox(
    'Mode',
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
    st.sidebar.empty()
    st.header('Application de détection des mains avec Streamlit.\n')
    
    st.text('Dans cette application permet la détection des mains avec MediaPipe.\n')
    st.text(" L'application à des fonctionnalités tels que:\n -la prise de photo,\n -compteur de doights\n -l'alphabet en langue des signes Américain.")
    
    st.text('\nStreamLit permet de créer l\'interface utilisateur graphique Web (GUI)')
    cap.release()
    cv2.destroyAllWindows()

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

elif app_mode == 'Alphabet':
  cv2.destroyAllWindows()
  st.sidebar.empty()
  st.sidebar.markdown('---')
  st.empty()
  
  st.sidebar.image('media/image_2.png')
  m.alphabet()

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
    
elif app_mode == 'Compteur de doigts':
    st.empty()
    st.sidebar.empty()
    st.set_option('deprecation.showfileUploaderEncoding', False)

    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=1)

    st.sidebar.markdown('---')

    st.image("media/Hand_GesturesCount.png")

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
    st.empty()
    st.set_option('deprecation.showfileUploaderEncoding', False)

    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=1)
    st.image('media/qqs.png')
    
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
    st.sidebar.markdown('---')
    st.sidebar.markdown('Photos prises')
    m.selfi()
    


  