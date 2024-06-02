import pandas as pd

df1 = pd.read_csv("/hy-tmp/wav_file.csv")["name"]
df2 = pd.read_csv("/hy-tmp/metadata_daicwoz_crop_resample.csv")["name"]
# print(df2[~df2.apply(tuple, 1).isin(df1.apply(tuple, 1))])

print(df1[~df1.apply(tuple, 1).isin(df2.apply(tuple, 1))])
