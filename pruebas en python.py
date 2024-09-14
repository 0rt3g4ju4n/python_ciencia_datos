#%%
import pandas as pd
#%%
def lista(x): 
  a1 = []
  a2 = []
  a3 = []
  a4 = []
  a5 = []
  a6 = []
  a7 = []
  a8 = []
  a9 = []
  for A in range(0, x):
      for B in range(0, x):
          for C in range(0, x):
              for D in range(0, x):
                  for E in range(0, x):
                      for F in range(0, x):
                          for G in range(0, x):
                              for H in range(0, x):
                                  for I in range(0, x):
                                      a1.append(A)
                                      a2.append(B)
                                      a3.append(C)
                                      a4.append(D)
                                      a5.append(E)
                                      a6.append(F)
                                      a7.append(G)
                                      a8.append(H)
                                      a9.append(I)
  df = pd.DataFrame()
  df['A1'] = a1
  df['A2'] = a2
  df['A3'] = a3
  df['A4'] = a4
  df['A5'] = a5
  df['A6'] = a6
  df['A7'] = a7
  df['A8'] = a8
  df['A9'] = a9
  df.to_csv('Lista_datos.csv')
  return {'Lista de valores':df}    
                                    
lista(5)