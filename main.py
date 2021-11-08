# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import streamlit as st
import pickle
import numpy as np
import pandas as pd

#import the model
pipe = pickle.load(open('clf.pkl','rb'))
data = pickle.load(open('data.pkl','rb'))

#import the data


st.title("Predict 'Heating Load' and 'Cooling Load' ")

#'Relative Compactness'
rela_comp = st.selectbox('Relative Compactness',data['Relative Compactness'].unique())

#'Surface Area'
surf_area = st.selectbox('Surface Area',data['Surface Area'].unique())

#'Wall Area'
wall_area = st.selectbox('Wall Area',data['Wall Area'].unique())

#'Roof Area'
roof_area = st.selectbox('Roof Area',data['Roof Area'].unique())

#'Overall Height'
ovl_hgt = st.selectbox('Overall Height',data['Overall Height'].unique())

#'Orientation'
orientation = st.selectbox('Orientation',data['Orientation'].unique())

#'Glazing Area'
gl_area = st.selectbox('Glazing Area',data['Glazing Area'].unique())

#'Glazing Area Distribution'
gl_area_dist = st.selectbox('Glazing Area Distribution',data['Glazing Area Distribution'].unique())

if st.button('Predict HEATING LOAD and COOLING LOAD'):
    query = np.array([rela_comp,surf_area,wall_area,roof_area,ovl_hgt,orientation,gl_area,gl_area_dist])
    query = query.reshape(1,8)
    st.title(str('The Heating Load is ') + str(round(pipe.predict(query)[0,0],2)))
    st.title(str('The Cooling Load is ') + str(round(pipe.predict(query)[0,1],2)))



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
