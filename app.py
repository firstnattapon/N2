import pandas as pd
pd.set_option("display.precision", 8)
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# sns.set_style("whitegrid")

# genre = st.radio(
#     "What's your favorite movie genre",
#     ('Comedy', 'Drama', 'Documentary'))

# if genre == 'Comedy':
#     st.write('You selected comedy.')
# else:
#     st.write("You didn't select comedy.")

c_1 =  st.slider('c_1', min_value=0, max_value=12, value=5, step=None, format=None, key=None)
c_2 =  st.slider('c_2', min_value=0, max_value=12, value=5, step=None, format=None, key=None)

x = ["A", "K", "Q", "J","T", "9", "8" , "7" , "6" , "5" , "4" , "3" , "2"]
y = ["A", "K", "Q", "J","T", "9", "8" , "7" , "6" , "5" , "4" , "3" , "2"]

A = 0.7  ; B = 0.6  ; C = 0.5  ; D= 0.4 ; E = 0.3 ; F = 0.2  ;  G  =0.1  ; H = 0.0
data = np.array([
[A, A, B, C, D, E, F, F, F, F, F, F, F],
[A, A, D, E, G, G, G, G, G, G, G, G, G],
[B, D, A, E, G, G, G, G, G, G, G, G, G],
[C, E, E, A, G, G, G, G, G, G, G, G, G],
[D, G, G, F, B, G, G, G, G, G, G, G, G],
[E, G, G, H, F, B, G, G, G, G, G, G, G],
[F, G, G, H, H, F, C, G, G, G, G, G, G],
[F, G, G, H, H, H, F, C, G, G, G, G, G],
[F, G, G, H, H, H, H, F, D, G, G, G, G],
[F, G, G, H, H, H, H, H, F, D, G, G, G],
[F, G, G, H, H, H, H, H, H, F, E, G, G],
[F, G, G,H, H, H, H, H, H, H, G, E,  G],
[F, G, G, H, H, H, H, H, H, H, H, G, E],
])

data[[c_1] ,[c_2]] = 1.
fig, ax = plt.subplots(figsize=(5 , 5))
im = ax.imshow(data)
ax.set_xticks(np.arange(len(x)))
ax.set_yticks(np.arange(len(y)))
ax.xaxis.tick_top()
ax.set_xticklabels(x)
ax.set_yticklabels(y)
fig.tight_layout()
st.pyplot()
