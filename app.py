import pandas as pd
pd.set_option("display.precision", 8)
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# sns.set_style("whitegrid")

# genre = st.radio("c_1",(x))

# if genre == 'Comedy':
#     st.write('You selected comedy.')
# else:
#     st.write("You didn't select comedy.")


x = ["A", "K", "Q", "J","T", "9", "8" , "7" , "6" , "5" , "4" , "3" , "2"]
y = ["A", "K", "Q", "J","T", "9", "8" , "7" , "6" , "5" , "4" , "3" , "2"]

c_1 = st.sidebar.radio("c_1",(x))
c_2 = st.sidebar.radio("c_2",(x))
st.sidebar.write('_'*20)

# Z = {"A" : 0.7  , "B" : 0.6 , "C" : 0.5 , "D" : 0.4 , "E" : 0.3 , "F" :0.2 , "G" :0.1 ,  "H" : 0.0}
# A = Z['A']  ; B = Z['B'] ; C = Z['C']  ; D= Z['D'] ; E = Z['E'] ; F = Z['F']  ; G = Z['G'] ; H = Z['H']

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

# data[[Z[c_1]] ,[[Z[c_2]] = 1.
fig, ax = plt.subplots(figsize=(3 , 3))
im = ax.imshow(data , cmap='Greys')
ax.set_xticks(np.arange(len(x)))
ax.set_yticks(np.arange(len(y)))
ax.xaxis.tick_top()
ax.set_xticklabels(x)
ax.set_yticklabels(y)
fig.tight_layout()
st.pyplot()
