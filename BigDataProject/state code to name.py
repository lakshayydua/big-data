import pandas as pd

df=pd.read_csv('/home/ldua/Desktop/BigDataProject/Output/42401/42401.csv')

num_rows = df.shape[0]

map_code = dict()
map_code['1'] = "Alabama",
map_code['2'] = "Alaska"
for i in range(num_rows):
    df.loc[i,"State Code"] = map_code[str(df.loc[i,"State Code"])]

    break