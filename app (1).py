import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import seaborn as sns
# from tqdm import tqdm
import pickle
# sns.set()
data = pd.read_csv("dataset.csv")
data.head(5)

data.isnull()

data.isnull().sum()

data = data.dropna(axis = 0)
data.isnull().sum()

data = data.drop_duplicates('track_name', keep='last')


data.info()

df_num = data.drop(columns = {'track_name','album_name','artists','track_id','explicit','track_genre'})
df_num.info()

df_num.corr()

from sklearn.preprocessing import MinMaxScaler
for col in df_num.columns:
  MinMaxScaler(col)
df_num.head()

import joblib
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 10)
cluster = kmeans.fit_predict(df_num)
df_num['Cluster_no'] = cluster
MinMaxScaler(df_num['Cluster_no'])  

df_num['name'] = data['track_name']


c1 = df_num[df_num['Cluster_no'] == 4]


def abc(song):
  temp = df_num.loc[df_num['name'] == song, 'Cluster_no']
  print(temp)
  res = df_num[df_num['Cluster_no'] == int(temp)]
  res = res.result_df = res.drop_duplicates()
  # print(res['name'].head(5))
  # return res['name'].tolist()
  result=res['name'].head(5)
  return result.tolist()
  


# def abc(song):
#   temp = df_num.loc[df_num['name'] == song, 'Cluster_no']
#   res = df_num[df_num['Cluster_no'] == int(temp)]
#   res = res.result_df = res.drop_duplicates()
#   return res['name'].tolist()






from flask import Flask , render_template,request
import pickle
import numpy as np
import sklearn
app = Flask(__name__)

# model=pickle.load(open("model.pkl","rb"))
@app.route('/')
def hello_world():
    return render_template('search.html')


# @app.route('/predict' , methods = (['GET','POST']))
# def predict():
#     print(request.form)
#     features = request.form['q']
#     # final = [np.array(features)]
#     print(features)
#     prediction = abc(features)
#     print(prediction)
#     return render_template('search.html',prediction_text = "The songs are {}".format(prediction))
#     # return render_template('search.html', prediction = prediction)

@app.route('/predict', methods=['POST'])
def predict():
    features = request.form['q']
    prediction = abc(features)
    # prediction_text = ", ".join(prediction)
    prediction_text = "<br>".join([f"<li>{p}</li>" for p in prediction])
    return render_template('search.html', prediction_text=prediction_text)


if __name__ == "__main__":
    app.run(debug=True)