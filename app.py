
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
import SessionState 
pd.set_option('precision', 3)
# sns.set_style("whitegrid")

if __name__ == '__main__':
    @st.cache(suppress_st_warning=True)
    def preflop():
        df = pd.read_pickle('./preflop.pickle')
        x = ["A", "K", "Q", "J","T", "9", "8" , "7" , "6" , "5" , "4" , "3" , "2"]
        return  df , x
    
    session = SessionState.get(run_id=0)
    df , x = preflop()    
    c_1 = st.radio("c_1",(x), key=session.run_id)
    c_2 = st.radio("c_2",(x), key=session.run_id)   
    suit = st.radio("suit",("P" , "O" ,"S") , index= 0 if c_1==c_2 else 1 , key=session.run_id)
    action = st.radio("action",("LIMPERS" , "UN_OPENED" ,"ONE_RAISE"), key=session.run_id)
    position = st.radio("position",("U_HJ" , "C_B" , "BL" , "VS_3BET" , "VS_STEAL"), key=session.run_id)
    st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    y  = {'2':2 , '3':3, '4':4, '5':5, '6':6 ,'7':7, '8':8, '9':9 ,'T':10, 'J':11, 'Q':12 ,'K':13 , 'A':14 , 'O':-1 ,'P':0 , 'S':1}
    n_card1 = y[c_1] ; n_card2  = y[c_2] ; n_suited  = y[suit]
    ev_c= np.where(n_suited == 1  ,  ev.evaluate_hand([c(n_card1 , 1), c(n_card2, 1)] ) ,
                   ev.evaluate_hand([c(n_card1 , 1), c(n_card2, 2)]))

    df  = df[df['ev'] == ev_c]
    df  = df[df['position'] == position ]
    df  = df[df['action'] == action ]
    df_o = df.output_preflop.to_numpy()
    df_c = df.class_preflop.to_numpy()
    code = '''{}  >  {}  >  {} > {} > {}'''.format((c_1+c_2+suit) , position , action , df_c[-1] , df_o[-1] )
    st.code(code, language='python')
    if st.button("{})  {}".format( df_c[-1] , df_o[-1])):
        session.run_id += 1
        st.write("_"*20)
    
#     if st.checkbox("plot", value = 0): 
#         st.markdown("![90dbb9ae25a0542d8876a74da01477a6.png](https://www.img.in.th/images/90dbb9ae25a0542d8876a74da01477a6.png)")
#         st.markdown("[![a607ec3f270aa7e759b723d935c5947a.png](https://www.img.in.th/images/a607ec3f270aa7e759b723d935c5947a.png)")

#     if st.checkbox("hiplot_preflop" , value = 0): 
#         if st.button("{}".format('Reset')):
#             session.run_id += 1
#         df = preflop()
#         data = df[['index', 'n_card1' , 'n_card2' , 's_suited'  , 'class_preflop', 'position' , 'action' , 'output_preflop' ,'ev']]
#         data = data.to_dict('r')
#         xp = hip.Experiment.from_iterable(data)
#         ret_val = xp.display_st(key=session.run_id)
#         st.markdown("hiplot returned " + json.dumps(ret_val))
        
#_______________________________________________________________________________________________________
        
    @st.cache(suppress_st_warning=True)
    def postflop():
        df = pd.read_pickle('./postflop.pickle')
        p = df.position.unique()
        b = df.board.unique()
        h = df.hit.unique()
        return  df , p , b , h       
    
    df , p , b , h  = postflop()
    op_p = st.radio('position',p)
    op_b = st.radio('board',b)
#     op_h = st.radio('hit',h)
    op_h= st.checkbox('hit' ,h)

    
    code = '''{}  >  {}  >  {}  '''.format(op_p , op_b , op_h )
    st.code(code, language='python')
    
#     if st.button("{})  {}".format( df_c[-1] , df_o[-1])):
#         session.run_id += 1
    st.write("_"*20)
    
    
    
    
    
    
#     if st.checkbox("hiplot_postflop" , value = 0): 
#         if st.button("{}".format('Reset')):
#             session.run_id += 1
#         df_2 = postflop()
#         data_2 = df_2
#         data_2 = data_2.to_dict('r')
#         xp = hip.Experiment.from_iterable(data_2)
#         ret_val = xp.display_st(key=session.run_id)
#         st.markdown("hiplot returned " + json.dumps(ret_val))
        

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

