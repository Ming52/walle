#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st


# In[ ]:


import os


# In[ ]:


from fastai.vision.all import *


# In[ ]:


import pathlib


# In[ ]:


#temp = pathlib.PosixPath


# In[ ]:


#pathlib.PosixPath = pathlib.WindowsPath


# In[ ]:


#path = os.path.dirname(os.path.abspath(__file__))


# In[ ]:


model_path = os.path.join(path, 'export.pkl')


# In[ ]:


learn_inf = load_learner(model_path)


# In[ ]:


#pathlib.PosixPath = temp


# In[ ]:


uploaded_file = st.file_uploader("Choose an image...",
                                type=["jpg","png","jpeg"])


# In[ ]:


if uploaded_file is not None:
    img = PILImage.create(uploaded_file)
    st.image(img.to_thumb(500,500),caption='Your Image')
    pred,pred_idx,probs = learn_inf.predict(img)
    st.write(f'Prediction:{pred};Probability:{probs[pred_idx]:.04f}')

