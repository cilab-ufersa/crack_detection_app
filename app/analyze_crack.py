import pyecharts as pe 
from pyecharts import options as opts
import streamlit as st
from streamlit_echarts import st_echarts
from utils import *
import cv2


def show():
    st.set_page_config(
    page_title="Crackit",
    page_icon="🧱",
    layout="wide"
    )
    first_col, second_col = st.columns([1, 1])

    image = cv2.imread(st.query_params.get('image'))

    x_values, y_values = white_pixels(image)

    points = []	

    for i in range(len(x_values)):
        points.append([x_values[i], y_values[i]])
    
    st.session_state['image'] = points

    st.session_state['image'] = [[int(x), int(y)] for x, y in st.session_state['image']]


    if 'clicked_points' not in st.session_state:
        st.session_state['clicked_points'] = []
        
    if 'submit' not in st.session_state:
        st.session_state['submit'] = False

    sessions = st.session_state['clicked_points']
    size = len(sessions)

    last = []
    penultimate = []
    st.session_state['submit'] = False
    if size % 2 == 0 and size != 0:
        
        last = sessions[-1]
        penultimate = sessions[-2]
        
        st.session_state['clicked_points'] = []
        st.session_state['submit'] = True
        
    with first_col:
        st.header("Crackit: Surface Crack Analysis 📉")
        
        st.write("""
                In this section, you can analyze the crack by selecting two points on the image. The app will calculate the line equation that passes through the points and the angle of the line with the x-axis. This can be useful to determine the orientation of the crack.
                
                    1. Click on the image to select the first point.
                    2. Click on the image to select the second point.
                    3. The app will calculate the line equation and the angle of the line and display the results.
                    4. Click on the Clean button to start over or the Back to Home Page button to return to the home page.
                
                """)
        if st.session_state['submit'] == True:
        
            point1 = sessions[0]
            point2 = sessions[1]
            m, b = calculate_line_equation(point1, point2)
            st.info(f"The line equation that passes through the points is $y = {-m:.2f}x {'+' if b > 0 else ''} {b:.2f}$ and your angle with the x-axis is ${np.degrees(np.arctan(-m)):.2f}°$")
            
            if st.button("Clean", help= "Click to clean the points and start over"):
                pass

        if st.button("Back to Home Page", help= "Click to go back to the home page"):
            st.query_params.update({"page": "home"})
            st.rerun()
    

    options = {
        "xAxis": {
                "min":0,
                "max": 224
            },
        "yAxis": {
                "min":0,
                "max": 224,
                "inverse": True
            },
        "series": [
            {
                "symbolSize": 15,
                "data": st.session_state['image'],
                "type": "scatter",
            },
            {
                "symbolSize": 20,
                "data": [
                    last,
                    penultimate
                    ],
                "type": "line",
                "itemStyle": {
                    "color": "red"
                }
            },
        ],
    }
    
    events = {
        "click": "function(params) { console.log(params.name); return params.value }",
        "dblclick":"function(params) { return [params.type, params.name, params.value] }"
    }

    with second_col:
        values = st_echarts(options=options,width="100%", height="650%", events=events)
        
        if values:
            st.session_state['clicked_points'].append(values)
        
    
        if len(st.session_state['clicked_points']) == 2:
            st.rerun()

        