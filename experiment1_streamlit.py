import streamlit as st

#Using Streamlit's Navigation Bar to allow user switch between loading their own image and trying out pre-loaded images
pg = st.navigation([st.Page("experiment1_streamlit_preloaded.py",title="Preloaded images"), st.Page("experiment1_streamlit_userupload.py",title="Custom Upload")])
pg.run()
