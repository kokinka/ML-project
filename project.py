import datetime
import ipaddress

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

with open('facebook-jaroslavakokavcova/security_and_login_information/account_activity.json') as data_file:
    loaded = pd.read_json(data_file)
data = []
for x in loaded['account_activity']:
    x['timestamp'] = datetime.datetime.fromtimestamp(x['timestamp'])
    dataln = {'weekday': x['timestamp'].weekday(), 'hour': x['timestamp'].time().hour,
              'ip': int(ipaddress.IPv4Address(x['ip_address']))}
    data.append(dataln)
df = pd.DataFrame(data=data)
plt.figure(1)
df.groupby('weekday').ip.count().plot.bar()
plt.savefig('weekdays.pdf')

plt.figure(2)
time_weight = df.groupby('hour').ip.count().plot(kind='bar')
plt.savefig('hours.pdf')

x = df.values
scaler = StandardScaler()
scaled = scaler.fit_transform(x)
pca = PCA(n_components=2)
transformed = pca.fit_transform(scaled)
plt.figure(3)
plt.scatter(transformed[:, 0], transformed[:, 1])
plt.savefig('visualisation.pdf')

plt.figure(4)
clustering = DBSCAN().fit(scaled)
labels = clustering.labels_
unique_labels = set(labels)
for k in unique_labels:
    if k == -1:
        col = 'black'
    else:
        col = 'red'
    class_member_mask = (labels == k)
    xy = transformed[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', color=col)
plt.savefig('outliers.pdf')
plt.show()
