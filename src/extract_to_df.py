import os
import pandas as pd

def extract_to_df(path, name):
    filenames = list(path.glob(r'**/*.jpg'))
    labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filenames))
    coordinates = list(map(lambda x: os.path.split(
        os.path.split(x)[1])[1][:-4], filenames))
    latitudes = [x.split(',')[1] for x in coordinates]
    longitudes = [x.split(',')[0] for x in coordinates]
    
    path = pd.Series(filenames, name = 'Path').astype(str)
    latitude = pd.Series(latitudes, name = 'Latitude').astype(float)
    longitude = pd.Series(longitudes, name = 'Longtitue').astype(float)
    label = pd.Series(labels, name= 'Label').astype(str)
    df = pd.concat([path, latitude, longitude, label], axis = 1)
    df.to_csv(f'Dataframes/{name}.csv', index= False)
    return df