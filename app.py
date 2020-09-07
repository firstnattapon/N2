# import pandas as pd
# import streamlit as st
# import seaborn as sns
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import json
# import hiplot as hip
# pd.set_option('precision', 3)
# # sns.set_style("whitegrid")

# @st.cache(suppress_st_warning=True)
# def preflop():
#     df = pd.read_pickle('./preflop.pickle')
#     df = df[['Human', 'EV' , 'y']]
#     df = df.set_index(['y'])
#     df['top_range'] = abs(df['EV'] - 1.)
#     return  df

# x = ["A", "K", "Q", "J","T", "9", "8" , "7" , "6" , "5" , "4" , "3" , "2"]
# Suit = st.radio("Suit",("o" , "s"))
# c_1 = st.radio("c_1",(x))
# c_2 = st.radio("c_2",(x))
# st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
# st.write("_"*50)
# h = c_1 + c_2 + Suit
# df = preflop()
# df = df[df['Human'] == h]

# if df.index == 1:
#     st.markdown("![OY7T8b.png](https://sv1.picz.in.th/images/2020/09/04/OY7T8b.png)")
# if df.index == 2:
#     st.markdown("![OY7E7P.png](https://sv1.picz.in.th/images/2020/09/04/OY7E7P.png)")
# if df.index == 3:
#     st.markdown("![OY7Ogt.jpg](https://sv1.picz.in.th/images/2020/09/04/OY7Ogt.jpg)")
# if df.index == 4:
#     st.markdown("![OY7jFl.png](https://sv1.picz.in.th/images/2020/09/04/OY7jFl.png)")
# if df.index == 5:
#     st.markdown("![OY7Dlv.jpg](https://sv1.picz.in.th/images/2020/09/04/OY7Dlv.jpg)")
# if df.index == 6:
#     st.markdown("![OY7otk.jpg](https://sv1.picz.in.th/images/2020/09/04/OY7otk.jpg)")
# if df.index == 7:
#     st.markdown("![OY7b2e.jpg](https://sv1.picz.in.th/images/2020/09/04/OY7b2e.jpg)")
# # st.markdown("![faae325e00926b7bfbea492651688358.jpg](https://www.img.in.th/images/faae325e00926b7bfbea492651688358.jpg)")

# pd.set_option('precision', 3)
# data = pd.read_pickle('./preflop.pickle')
# # data = data.reset_index()
# data = data[['Human' , 'Ang_Card' , 'Suited' , 'EV' ,  'y']]
# data = data.to_dict('r')
# xp = hip.Experiment.from_iterable(data)
# # Display with `display_st` instead of `display`
# ret_val = xp.display_st(key="hip")
# st.markdown("hiplot returned " + json.dumps(ret_val))

# # Z = {"A":0, "K":1, "Q":2, "J":3,"T":4, "9":5, "8":6, "7":7, "6":8, "5":9, "4":10, "3":11, "2":12}
# # A = 0.7  ; B = 0.6 ; C =  0.5 ; D= 0.4  ; E = 0.3 ; F = 0.2  ; G = 0.1  ; H =  0.0

# # data = np.array([
# # [A, A, B, C, D, E, F, F, F, F, F, F, F],
# # [A, A, D, E, G, G, G, G, G, G, G, G, G],
# # [B, D, A, E, G, G, G, G, G, G, G, G, G],
# # [C, E, E, A, G, G, G, G, G, G, G, G, G],
# # [D, G, G, F, B, G, G, G, G, G, G, G, G],
# # [E, G, G, H, F, B, G, G, G, G, G, G, G],
# # [F, G, G, H, H, F, C, G, G, G, G, G, G],
# # [F, G, G, H, H, H, F, C, G, G, G, G, G],
# # [F, G, G, H, H, H, H, F, D, G, G, G, G],
# # [F, G, G, H, H, H, H, H, F, D, G, G, G],
# # [F, G, G, H, H, H, H, H, H, F, E, G, G],
# # [F, G, G,H, H, H, H, H, H, H, G, E,  G],
# # [F, G, G, H, H, H, H, H, H, H, H, G, E],
# # ])

# # data[[c_1] ,[c_2]] = 1.
# # df = pd.DataFrame(data)
# # c_1 = Z[c_1]
# # c_2 = Z[c_2]
# # data[[c_1] ,[c_2]] = 1.
# # fig, ax = plt.subplots(figsize=(5 , 5))
# # im = ax.imshow(data ,)
# # ax.set_xticks(np.arange(len(x)))
# # ax.set_yticks(np.arange(len(y)))
# # ax.xaxis.tick_top()
# # ax.set_xticklabels(x)
# # ax.set_yticklabels(y)
# # fig.tight_layout()
# # st.pyplot()

#___________________________________________________________________________________________________

import pandas as pd
import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import hiplot as hip
from pokereval.card import Card as c
from pokereval.hand_evaluator import HandEvaluator as ev
pd.set_option('precision', 3)
# sns.set_style("whitegrid")

def  fig ():
    values = np.array([[1, 1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 6],
        [1, 1, 4, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7],
        [2, 4, 1, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7],
        [3, 5, 5, 1, 5, 7, 7, 7, 7, 7, 7, 7, 7],
        [4, 7, 7, 6, 2, 5, 7, 7, 7, 7, 7, 7, 7],
        [5, 7, 7, 8, 6, 2, 5, 7, 7, 7, 7, 7, 7],
        [6, 7, 7, 8, 8, 6, 3, 5, 7, 7, 7, 7, 7],
        [6, 7, 7, 8, 8, 8, 6, 3, 6, 7, 7, 7, 7],
        [6, 7, 7, 8, 8, 8, 8, 6, 4, 6, 7, 7, 7],
        [6, 7, 7, 8, 8, 8, 8, 8, 6, 4, 6, 7, 7],
        [6, 7, 7, 8, 8, 8, 8, 8, 8, 6, 4, 7, 7],
        [6, 7, 7, 8, 8, 8, 8, 8, 8, 8, 7, 5, 7],
        [6, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 7, 5]])

    keys = np.array([['AAp', 'AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A7s', 'A6s','A5s', 'A4s', 'A3s', 'A2s'],
            ['AKo', 'KKp', 'KQs', 'KJs', 'KTs', 'K9s', 'K8s', 'K7s', 'K6s','K5s', 'K4s', 'K3s', 'K2s'],
            ['AQo', 'KQo', 'QQp', 'QJs', 'QTs', 'Q9s', 'Q8s', 'Q7s', 'Q6s','Q5s', 'Q4s', 'Q3s', 'Q2s'],
            ['AJo', 'KJo', 'QJo', 'JJp', 'JTs', 'J9s', 'J8s', 'J7s', 'J6s','J5s', 'J4s', 'J3s', 'J2s'],
            ['ATo', 'KTo', 'QTo', 'JTo', 'TTp', 'T9s', 'T8s', 'T7s', 'T6s','T5s', 'T4s', 'T3s', 'T2s'],
            ['A9o', 'K9o', 'Q9o', 'J9o', 'T9o', '99p', '98s', '97s', '96s','95s', '94s', '93s', '92s'],
            ['A8o', 'K8o', 'Q8o', 'J8o', 'T8o', '98o', '88p', '87s', '86s','85s', '84s', '83s', '82s'],
            ['A7o', 'K7o', 'Q7o', 'J7o', 'T7o', '97o', '87o', '77p', '76s','75s', '74s', '73s', '72s'],
            ['A6o', 'K6o', 'Q6o', 'J6o', 'T6o', '96o', '86o', '76o', '66p','65s', '64s', '63s', '62s'],
            ['A5o', 'K5o', 'Q5o', 'J5o', 'T5o', '95o', '85o', '75o', '65o','55p', '54s', '53s', '52s'],
            ['A4o', 'K4o', 'Q4o', 'J4o', 'T4o', '94o', '84o', '74o', '64o','54o', '44p', '43s', '42s'],
            ['A3o', 'K3o', 'Q3o', 'J3o', 'T3o', '93o', '83o', '73o', '63o','53o', '43o', '33p', '32s'],
            ['A2o', 'K2o', 'Q2o', 'J2o', 'T2o', '92o', '82o', '72o', '62o','52o', '42o', '32o', '22p']])

    fig, ax = plt.subplots( 1 , 2 , figsize=(15,15))
    ax[0].matshow(values, cmap='ocean');
    for (i, j), z in np.ndenumerate(values):
        alpha = ['A', 'K', 'Q', 'J' , 'T' , '9'  , '8' , '7' , '6' , '5'  , '4' , '3' , '2']
        ax[0].text(j, i, '{:0.0f}'.format(z), ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
        ax[0].set_xticks(list(range(len(alpha))))
        ax[0].set_xticklabels(alpha)
        ax[0].set_yticks(list(range(len(alpha))))
        ax[0].set_yticklabels(alpha)
    ax[1].matshow(values, cmap='ocean');
    for (i, j), z in np.ndenumerate(keys):
        ax[1].text(j, i, '{}'.format(z), ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
        ax[1].set_xticks(list(range(len(alpha))))
        ax[1].set_xticklabels(alpha)
        ax[1].set_yticks(list(range(len(alpha))))
        ax[1].set_yticklabels(alpha)
    plt.show()


def  class_preflop (x ):
    values = np.array([[1, 1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 6],
        [1, 1, 4, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7],
        [2, 4, 1, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7],
        [3, 5, 5, 1, 5, 7, 7, 7, 7, 7, 7, 7, 7],
        [4, 7, 7, 6, 2, 5, 7, 7, 7, 7, 7, 7, 7],
        [5, 7, 7, 8, 6, 2, 5, 7, 7, 7, 7, 7, 7],
        [6, 7, 7, 8, 8, 6, 3, 5, 7, 7, 7, 7, 7],
        [6, 7, 7, 8, 8, 8, 6, 3, 6, 7, 7, 7, 7],
        [6, 7, 7, 8, 8, 8, 8, 6, 4, 6, 7, 7, 7],
        [6, 7, 7, 8, 8, 8, 8, 8, 6, 4, 6, 7, 7],
        [6, 7, 7, 8, 8, 8, 8, 8, 8, 6, 4, 7, 7],
        [6, 7, 7, 8, 8, 8, 8, 8, 8, 8, 7, 5, 7],
        [6, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 7, 5]])

    keys = np.array([['AAp', 'AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A7s', 'A6s','A5s', 'A4s', 'A3s', 'A2s'],
            ['AKo', 'KKp', 'KQs', 'KJs', 'KTs', 'K9s', 'K8s', 'K7s', 'K6s','K5s', 'K4s', 'K3s', 'K2s'],
            ['AQo', 'KQo', 'QQp', 'QJs', 'QTs', 'Q9s', 'Q8s', 'Q7s', 'Q6s','Q5s', 'Q4s', 'Q3s', 'Q2s'],
            ['AJo', 'KJo', 'QJo', 'JJp', 'JTs', 'J9s', 'J8s', 'J7s', 'J6s','J5s', 'J4s', 'J3s', 'J2s'],
            ['ATo', 'KTo', 'QTo', 'JTo', 'TTp', 'T9s', 'T8s', 'T7s', 'T6s','T5s', 'T4s', 'T3s', 'T2s'],
            ['A9o', 'K9o', 'Q9o', 'J9o', 'T9o', '99p', '98s', '97s', '96s','95s', '94s', '93s', '92s'],
            ['A8o', 'K8o', 'Q8o', 'J8o', 'T8o', '98o', '88p', '87s', '86s','85s', '84s', '83s', '82s'],
            ['A7o', 'K7o', 'Q7o', 'J7o', 'T7o', '97o', '87o', '77p', '76s','75s', '74s', '73s', '72s'],
            ['A6o', 'K6o', 'Q6o', 'J6o', 'T6o', '96o', '86o', '76o', '66p','65s', '64s', '63s', '62s'],
            ['A5o', 'K5o', 'Q5o', 'J5o', 'T5o', '95o', '85o', '75o', '65o','55p', '54s', '53s', '52s'],
            ['A4o', 'K4o', 'Q4o', 'J4o', 'T4o', '94o', '84o', '74o', '64o','54o', '44p', '43s', '42s'],
            ['A3o', 'K3o', 'Q3o', 'J3o', 'T3o', '93o', '83o', '73o', '63o','53o', '43o', '33p', '32s'],
            ['A2o', 'K2o', 'Q2o', 'J2o', 'T2o', '92o', '82o', '72o', '62o','52o', '42o', '32o', '22p']])
    n =[]
    for i in np.ravel(keys):
        x_1 = i[0] ;  x_2 = i[1] ;  x_3 = i[2] ;
        y  =    {'2':2/14 , '3':3/14, '4':4/14, '5':5/14, '6':6/14,'7':7/14,'8':8/14,'9':9/14,'T':10/14, 'J':11/14,'Q':12/14,'K':13/14,'A':14/14 , 'o':-1,'p':1,'s':2}
        z_1 = y[x_1] ; z_2 = y[x_2] ; z_3 = y[x_3]
        z  =  (z_1 + z_2 + z_3) / 3
        n.append(z)
    n = np.reshape(n  , (-1 , 13))
    dictionary = dict(zip(n.reshape(-1), values.reshape(-1)))
    return dictionary[x]
 

def  op_preflop (class_preflop  , position , action):
    if class_preflop  == 1 :
        if position == 'U_HJ':
            if action == 'UN_OPENED'    : return  'OPEN'
            elif  action == 'LIMPERS'      : return  'ISO'
            elif  action == 'ONE_RAISE'  : return  'SHOVE'
        if position == 'C_B':
            if action == 'UN_OPENED'    : return  'OPEN'
            elif  action == 'LIMPERS'      : return  'ISO'
            elif  action == 'ONE_RAISE'  : return  'SHOVE'
        if position == 'BL':
            if action == 'UN_OPENED'    : return  'OPEN'
            elif  action == 'LIMPERS'      : return  'SHOVE'
            elif  action == 'ONE_RAISE'  : return  'SHOVE'
        if position == 'VS_3BET':
            if action == 'UN_OPENED'    : return  'SHOVE'
            elif  action == 'LIMPERS'      : return  'SHOVE'
            elif  action == 'ONE_RAISE'  : return  'SHOVE'
        if position == 'VS_STEAL':
            if action == 'UN_OPENED'    : return  'SHOVE'
            elif  action == 'LIMPERS'      : return  'SHOVE'
            elif  action == 'ONE_RAISE'  : return  'SHOVE'

    elif class_preflop == 2 :
        if position == 'U_HJ':
            if action == 'UN_OPENED'    : return  'OPEN'
            elif  action == 'LIMPERS'      : return  'ISO'
            elif  action == 'ONE_RAISE'  : return  'PER>20%'
        if position == 'C_B':
            if action == 'UN_OPENED'    : return  'OPEN'
            elif  action == 'LIMPERS'      : return  'SHOVE'
            elif  action == 'ONE_RAISE'  : return  'PER>10%'
        if position == 'BL':
            if action == 'UN_OPENED'    : return  'OPEN'
            elif  action == 'LIMPERS'      : return  'SHOVE'
            elif  action == 'ONE_RAISE'  : return  'SHOVE'
        if position == 'VS_3BET':
            if action == 'UN_OPENED'    : return  '3BET>9%'
            elif  action == 'LIMPERS'      : return  '3BET>9%'
            elif  action == 'ONE_RAISE'  : return  '3BET>9%'
        if position == 'VS_STEAL':
            if action == 'UN_OPENED'    : return  'SHOVE'
            elif  action == 'LIMPERS'      : return  'SHOVE'
            elif  action == 'ONE_RAISE'  : return  'SHOVE'

    elif class_preflop == 3 :
        if position == 'U_HJ':
            if action == 'UN_OPENED'    : return  'OPEN'
            elif  action == 'LIMPERS'      : return  'ISO'
            elif  action == 'ONE_RAISE'  : return  'FOLD'
        if position == 'C_B':
            if action == 'UN_OPENED'    : return  'OPEN'
            elif  action == 'LIMPERS'      : return  'SHOVE'
            elif  action == 'ONE_RAISE'  : return  'PER>30%'
        if position == 'BL':
            if action == 'UN_OPENED'    : return  'OPEN'
            elif  action == 'LIMPERS'      : return  'SHOVE'
            elif  action == 'ONE_RAISE'  : return  'PER>20%'
        if position == 'VS_3BET':
            if action == 'UN_OPENED'    : return  '3BET>25%'
            elif  action == 'LIMPERS'      : return  '3BET>25%'
            elif  action == 'ONE_RAISE'  : return  '3BET>25%'
        if position == 'VS_STEAL':
            if action == 'UN_OPENED'    : return  'SHOVE'
            elif  action == 'LIMPERS'      : return  'SHOVE'
            elif  action == 'ONE_RAISE'  : return  'SHOVE'

    elif class_preflop == 4 :
        if position == 'U_HJ':
            if action == 'UN_OPENED'    : return  'OPEN'
            elif  action == 'LIMPERS'      : return  'FOLD'
            elif  action == 'ONE_RAISE'  : return  'FOLD'
        if position == 'C_B':
            if action == 'UN_OPENED'    : return  'OPEN'
            elif  action == 'LIMPERS'      : return  'ISO'
            elif  action == 'ONE_RAISE'  : return  'FOLD'
        if position == 'BL':
            if action == 'UN_OPENED'    : return  'OPEN'
            elif  action == 'LIMPERS'      : return  'SHOVE'
            elif  action == 'ONE_RAISE'  : return  'PER>30%'
        if position == 'VS_3BET':
            if action == 'UN_OPENED'    : return  '3BET>30%'
            elif  action == 'LIMPERS'      : return  '3BET>30%'
            elif  action == 'ONE_RAISE'  : return  '3BET>30%'
        if position == 'VS_STEAL':
            if action == 'UN_OPENED'    : return  'PER>10%'
            elif  action == 'LIMPERS'      : return  'PER>10%'
            elif  action == 'ONE_RAISE'  : return  'PER>10%'

    elif class_preflop == 5 :
        if position == 'U_HJ':
            if action == 'UN_OPENED'    : return  'FOLD'
            elif  action == 'LIMPERS'      : return  'FOLD'
            elif  action == 'ONE_RAISE'  : return  'FOLD'
        if position == 'C_B':
            if action == 'UN_OPENED'    : return  'OPEN'
            elif  action == 'LIMPERS'      : return  'ISO'
            elif  action == 'ONE_RAISE'  : return  'FOLD'
        if position == 'BL':
            if action == 'UN_OPENED'    : return  'OPEN'
            elif  action == 'LIMPERS'      : return  'LIMP'
            elif  action == 'ONE_RAISE'  : return  'FOLD'
        if position == 'VS_3BET':
            if action == 'UN_OPENED'    : return  'FOLD'
            elif  action == 'LIMPERS'      : return  'FOLD'
            elif  action == 'ONE_RAISE'  : return  'FOLD'
        if position == 'VS_STEAL':
            if action == 'UN_OPENED'    : return  'PER>20%'
            elif  action == 'LIMPERS'      : return  'PER>20%'
            elif  action == 'ONE_RAISE'  : return  'PER>20%'

    elif class_preflop == 6 :
        if position == 'U_HJ':
            if action == 'UN_OPENED'    : return  'FOLD'
            elif  action == 'LIMPERS'      : return  'FOLD'
            elif  action == 'ONE_RAISE'  : return  'FOLD'
        if position == 'C_B':
            if action == 'UN_OPENED'    : return  'OPEN'
            elif  action == 'LIMPERS'      : return  'ISO_FISH'
            elif  action == 'ONE_RAISE'  : return  'FOLD'
        if position == 'BL':
            if action == 'UN_OPENED'    : return  'OPEN'
            elif  action == 'LIMPERS'      : return  'LIMP'
            elif  action == 'ONE_RAISE'  : return  'FOLD'
        if position == 'VS_3BET':
            if action == 'UN_OPENED'    : return  'FOLD'
            elif  action == 'LIMPERS'      : return  'FOLD'
            elif  action == 'ONE_RAISE'  : return  'FOLD'
        if position == 'VS_STEAL':
            if action == 'UN_OPENED'    : return  'PER>30%'
            elif  action == 'LIMPERS'      : return  'PER>30%'
            elif  action == 'ONE_RAISE'  : return  'PER>30%'

    elif class_preflop == 7 :
        if position == 'U_HJ':
            if action == 'UN_OPENED'    : return  'FOLD'
            elif  action == 'LIMPERS'      : return  'FOLD'
            elif  action == 'ONE_RAISE'  : return  'FOLD'
        if position == 'C_B':
            if action == 'UN_OPENED'    : return  'FOLD'
            elif  action == 'LIMPERS'      : return  'FOLD'
            elif  action == 'ONE_RAISE'  : return  'FOLD'
        if position == 'BL':
            if action == 'UN_OPENED'    : return  'OPEN'
            elif  action == 'LIMPERS'      : return  'FOLD'
            elif  action == 'ONE_RAISE'  : return  'FOLD'
        if position == 'VS_3BET':
            if action == 'UN_OPENED'    : return  'FOLD'
            elif  action == 'LIMPERS'      : return  'FOLD'
            elif  action == 'ONE_RAISE'  : return  'FOLD'
        if position == 'VS_STEAL':
            if action == 'UN_OPENED'    : return  'FOLD'
            elif  action == 'LIMPERS'      : return  'FOLD'
            elif  action == 'ONE_RAISE'  : return  'FOLD'
    else:
        return 'FOLD'

def  gen():
    s_card   =   ['2' , '3' , '4' , '5' , '6' , '7' , '8' , '9' , 'T' , 'J' , 'Q' , 'K' , 'A']
    s        =   ['o' , 's']
    n_card   =   [ 2 ,  3 ,  4, 5 , 6 ,  7 , 8 , 9  , 10 , 11 , 12 , 13 , 14] 
    n        =   [ 2 , -1 ]
    card_s   =  [[c_1 , c_2 , str(np.where(c_1 != c_2 , s_s ,'p'))] for c_1  in  s_card  for c_2   in  s_card  for s_s in s ]
    card_n   =  [[n_1 , n_2 , int(np.where(n_1 != n_2 , n_s , 1))]  for n_1 in  n_card for n_2   in n_card  for n_s in n ]
    df_1     = pd.DataFrame(card_s , columns =['s_card1' , 's_card2' , 's_suited' ] )
    df_2     = pd.DataFrame(card_n  , columns=['n_card1' , 'n_card2' , 'n_suited'] )
    df       = pd.concat(objs=[df_1 , df_2 ] , axis=1 )
    df['index']  = (df.s_card1 + df.s_card2 + df.s_suited)
    df['ev']     =  df.apply(lambda x : np.where( x.s_suited == 's' ,  ev.evaluate_hand([c(x.n_card1 , 1), c(x.n_card2, 1)] ) ,
                                                ev.evaluate_hand([c(x.n_card1 , 1), c(x.n_card2, 2)] ) )  , axis=1)
    df['avg_card']        = df.apply(lambda x : ((x.n_card1/14 )+(x.n_card2/14) + (x.n_suited)) /3 , axis=1)
    df['class_preflop']   = df.apply(lambda x : class_preflop(x['avg_card'])  , axis=1)
    df          =  pd.concat([df]*5, ignore_index=False)
    position    = np.repeat(['U_HJ' , 'C_B' ,  'BL' , 'VS_3BET' , 'VS_STEAL'] , 338)
    df['position'] = position
    df             =  pd.concat([df]*3, ignore_index=False)
    action         = np.repeat(['UN_OPENED' , 'LIMPERS' , 'ONE_RAISE'] , 1690)
    df['action']   = action
    df['output_preflop'] = df.apply(lambda x : op_preflop(x.class_preflop , x.position  , x.action) , axis=1)

    
    
if __name__ == '__main__':
    @st.cache(suppress_st_warning=True)
    def preflop():
        df = pd.read_pickle('./preflop.pickle')
        return  df

    x = ["A", "K", "Q", "J","T", "9", "8" , "7" , "6" , "5" , "4" , "3" , "2"]
    suit = st.radio("suit",("O" , "P" ,"S"))
    c_1 = st.radio("c_1",(x))
    c_2 = st.radio("c_2",(x))
    action = st.radio("action",("UN_OPENED" , "LIMPERS" ,"ONE_RAISE"))
    position = st.radio("position",("U_HJ" , "C_B" , "BL" , "VS_3BET" , "VS_STEAL"))
    st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    y     = {'2':2/14 , '3':3/14, '4':4/14, '5':5/14, '6':6/14,'7':7/14,'8':8/14,'9':9/14,'T':10/14, 'J':11/14,'Q':12/14,'K':13/14,'A':14/14 , 'O':-1,'P':1,'S':2}
    z_1  = y[c_1]
    z_2  = y[c_2]
    z_3  = y[suit]
    z  =  (z_1 + z_2 + z_3) / 3
    
    df = preflop()
    # df = df[df['Human'] == h]
    data = df[['s_card1' , 's_card2' , 's_suited' , 'ev' ,  'class_preflop']]
    data = data.to_dict('r')
    xp = hip.Experiment.from_iterable(data)
    ret_val = xp.display_st(key="hip")
    st.markdown("hiplot returned " + json.dumps(ret_val))
