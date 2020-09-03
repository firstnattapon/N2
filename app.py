import pandas as pd
pd.set_option("display.precision", 8)
import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# sns.set_style("whitegrid")

@st.cache(suppress_st_warning=True)
def preflop():
    df = pd.read_pickle('./preflop.pickle')
    df = df[['Human', 'EV' , 'y']]
    df = df.set_index(['y'])
    df['EV'] = df['EV'] *100
    return  df

x = ["A", "K", "Q", "J","T", "9", "8" , "7" , "6" , "5" , "4" , "3" , "2"]
# Suit = st.sidebar.radio("Suit",("o" , "s"))
# c_1 = st.sidebar.selectbox("c_1",(x))
# c_2 = st.sidebar.selectbox("c_2",(x))

Suit = st.radio("Suit",("o" , "s"))
c_1 = st.selectbox("c_1",(x))
c_2 = st.selectbox("c_2",(x))
h = c_1 + c_2 + Suit
df = preflop()
df = df[df['Human'] == h]
st.write(df)

if df.index == 1:
    st.markdown("![OY7T8b.png](https://sv1.picz.in.th/images/2020/09/04/OY7T8b.png)")
if df.index == 2:
    st.markdown("![OY7E7P.png](https://sv1.picz.in.th/images/2020/09/04/OY7E7P.png)")
if df.index == 3:
    st.markdown("![OY7Ogt.jpg](https://sv1.picz.in.th/images/2020/09/04/OY7Ogt.jpg)")
if df.index == 4:
    st.markdown("![OY7jFl.png](https://sv1.picz.in.th/images/2020/09/04/OY7jFl.png)")
if df.index == 5:
    st.markdown("![OY7Dlv.jpg](https://sv1.picz.in.th/images/2020/09/04/OY7Dlv.jpg)")
if df.index == 6:
    st.markdown("![OY7otk.jpg](https://sv1.picz.in.th/images/2020/09/04/OY7otk.jpg)")
if df.index == 7:
    st.markdown("![OY7b2e.jpg](https://sv1.picz.in.th/images/2020/09/04/OY7b2e.jpg)")
    
# Z = {"A":0, "K":1, "Q":2, "J":3,"T":4, "9":5, "8":6, "7":7, "6":8, "5":9, "4":10, "3":11, "2":12}
# A = 0.7  ; B = 0.6 ; C =  0.5 ; D= 0.4  ; E = 0.3 ; F = 0.2  ; G = 0.1  ; H =  0.0

# data = np.array([
# [A, A, B, C, D, E, F, F, F, F, F, F, F],
# [A, A, D, E, G, G, G, G, G, G, G, G, G],
# [B, D, A, E, G, G, G, G, G, G, G, G, G],
# [C, E, E, A, G, G, G, G, G, G, G, G, G],
# [D, G, G, F, B, G, G, G, G, G, G, G, G],
# [E, G, G, H, F, B, G, G, G, G, G, G, G],
# [F, G, G, H, H, F, C, G, G, G, G, G, G],
# [F, G, G, H, H, H, F, C, G, G, G, G, G],
# [F, G, G, H, H, H, H, F, D, G, G, G, G],
# [F, G, G, H, H, H, H, H, F, D, G, G, G],
# [F, G, G, H, H, H, H, H, H, F, E, G, G],
# [F, G, G,H, H, H, H, H, H, H, G, E,  G],
# [F, G, G, H, H, H, H, H, H, H, H, G, E],
# ])

# data[[c_1] ,[c_2]] = 1.
# df = pd.DataFrame(data)
# c_1 = Z[c_1]
# c_2 = Z[c_2]
# data[[c_1] ,[c_2]] = 1.
# fig, ax = plt.subplots(figsize=(5 , 5))
# im = ax.imshow(data ,)
# ax.set_xticks(np.arange(len(x)))
# ax.set_yticks(np.arange(len(y)))
# ax.xaxis.tick_top()
# ax.set_xticklabels(x)
# ax.set_yticklabels(y)
# fig.tight_layout()
# st.pyplot()
