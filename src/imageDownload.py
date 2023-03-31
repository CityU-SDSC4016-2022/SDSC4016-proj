import pandas as pd
import requests
image_feature_file = './image_file2/'
df = pd.read_json('meta_Movies_and_TV.json', lines=True)
#%%

for i in df[['asin','imageURLHighRes']].iloc:
    if type(i['imageURLHighRes']) == list:
        if i['imageURLHighRes'] != []:
            try:
                response = requests.get(i['imageURLHighRes'][0])
                with open(image_feature_file+f"{i['asin']}.jpg", "wb") as f:
                    f.write(response.content)
            except Exception as e:
                print(e)