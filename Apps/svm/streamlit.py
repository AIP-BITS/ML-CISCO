import streamlit as st
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

st.title("This is some title")
st.header("Header")
st.subheader("Sub Header")
st.text("Some Text")
st.markdown("Some markdown")

st.header("# Writing a dataframe")
df = pd.DataFrame({'col1': [1,2,3],'col2':[4,5,6]})
st.write(df)

st.header("# Styling dataframe")
df = pd.DataFrame(
    np.random.randn(10, 20),
    columns=('col %d' % i for i in range(20)))

st.dataframe(df.style.highlight_max(axis=0))

#write(mpl_fig) : Displays a Matplotlib figure.

#write(altair) : Displays an Altair chart.

#write(keras) : Displays a Keras model.

#write(graphviz) : Displays a Graphviz graph.

#write(plotly_fig) : Displays a Plotly figure.

#write(bokeh_fig) : Displays a Bokeh figure.

st.header("# Graphviz chart")
st.graphviz_chart('''
    digraph {
        run -> intr
        intr -> runbl
        runbl -> run
        run -> kernel
        kernel -> zombie
        kernel -> sleep
        kernel -> runmem
        sleep -> swap
        swap -> runswap
        runswap -> new
        runswap -> runmem
        new -> runmem
        sleep -> runmem
    }
''')

st.header("Interactive widgets")
# Interactive widgets

st.subheader("button")
if st.button('Say hello'):
    st.write('Why hello there')
else:
    st.write('Goodbye')


st.subheader("checkbox")
agree = st.checkbox('I agree')
if agree:
    st.write('Great!')

st.subheader("radio")    
genre = st.radio(
    "What's your favorite movie genre",
    ('Comedy', 'Drama', 'Documentary'))

if genre == 'Comedy':
    st.write('You selected comedy.')
else:
    st.write("You didn't select comedy.")

st.subheader("selectbox")      
option = st.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone'))

st.write('You selected:', option)

st.subheader("multiselect")  
options = st.multiselect(
    'What are your favorite colors',
    ['Green', 'Yellow', 'Red', 'Blue'],
    ['Yellow', 'Red'])

st.write('You selected:', options)

st.subheader("Different Slider")
age = st.slider('How old are you?', 0, 130, 25)
st.write("I'm ", age, 'years old')

values = st.slider(
    'Select a range of values',
    0.0, 100.0, (25.0, 75.0))
st.write('Values:', values)


from datetime import time
appointment = st.slider(
    "Schedule your appointment:",
    value=(time(11, 30), time(12, 45)))
st.write("You're scheduled for:", appointment)

from datetime import datetime
start_time = st.slider(
    "When do you start?",
    value=datetime(2020, 1, 1, 9, 30),
    format="MM/DD/YY - hh:mm")
st.write("Start time:", start_time)

st.subheader("select_slider")
color = st.select_slider(
    'Select a color of the rainbow',
    options=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'])
st.write('My favorite color is', color)

start_color, end_color = st.select_slider(
    'Select a range of color wavelength',
    options=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'],
    value=('red', 'blue'))
st.write('You selected wavelengths between', start_color, 'and', end_color)

st.subheader("single-line text input")
title = st.text_input('Movie title', 'Life of Brian')
st.write('The current movie title is', title)

st.subheader("Numeric input")
number = st.number_input('Insert a number')
st.write('The current number is ', number)

st.subheader("Multi line text input")
txt = st.text_area('Text to analyze', '''
    It was the best of times, it was the worst of times, it was
    the age of wisdom, it was the age of foolishness, it was
    the epoch of belief, it was the epoch of incredulity, it
    was the season of Light, it was the season of Darkness, it
    was the spring of hope, it was the winter of despair, (...)
    ''')
st.write('Sentiment:', 'Positive/Negative')

# d = st.date_input(
    # "When's your birthday",
    # datetime.date(2019, 7, 6))
# st.write('Your birthday is:', d)

# t = st.time_input('Set an alarm for', datetime.time(8, 45))
# st.write('Alarm is set for', t)

st.subheader("Upload a single file of a particular type")
uploaded_file = st.file_uploader("Choose a single JPG file", type="jpg")
if uploaded_file is not None:
    st.write("filename:", uploaded_file.name)

st.subheader("Upload multiple file")
uploaded_files = st.file_uploader("Upload multiple files", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    st.write("filename:", uploaded_file.name)

st.subheader("Color Picker")    
color = st.color_picker('Pick A Color', '#00f900')
st.write('The current color is', color)

st.sidebar.subheader("Widgets on sidebar")
add_selectbox = st.sidebar.selectbox(
    "How would you like to be contacted?",
    ("Email1", "Home phone1", "Mobile phone1")
)

age = st.sidebar.slider('How new are you?', 0, 130, 25)
st.write("I'm ", age, 'years old')


st.subheader("Container, placed in row format")
container = st.beta_container()
age1 = container.slider('How new1 are you?', 0, 130, 25)
#container.write("I'm ", age, 'years old')
# You can call any Streamlit command, including custom components:
container.bar_chart(np.random.randn(50, 3))

st.write("This is outside the container")

st.subheader("Container, placed in column format")
col1, col2 = st.beta_columns([3, 1])
data = np.random.randn(10, 1)

col1.subheader("A wide column with a chart")
col1.line_chart(data)

col2.subheader("A narrow column with the data")
col2.write(data)



st.subheader("Expander")
st.line_chart({"data": [1, 5, 2, 6, 2, 1]})
with st.beta_expander("See explanation"):
    st.write("""
        The chart above shows some numbers I picked for you.
        I rolled actual dice for these, so they're *guaranteed* to
        be random.
    """)
    st.image("https://static.streamlit.io/examples/dice.jpg")
    
st.subheader("Display progress and status")
import time
my_bar = st.progress(0)
for percent_complete in range(100):
    time.sleep(0.1)
    my_bar.progress(percent_complete + 1)
    
    
with st.spinner('Wait for it...'):
    time.sleep(5)
st.success('Done!')

st.balloons()

st.error('This is an error')
st.warning('This is a warning')
st.info('This is a purely informational message')
st.success('This is a success message!')
e = RuntimeError('This is an exception of type RuntimeError')
st.exception(e)

import time

with st.empty():
    for seconds in range(60):
        st.write(f"⏳ {seconds} seconds have passed")
        time.sleep(1)
    st.write("✔️ 1 minute over!")

