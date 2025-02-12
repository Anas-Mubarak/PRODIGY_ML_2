import pandas as pnd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler  
from sklearn.cluster import KMeans

#reading using pandas
try:
    tds = pnd.read_csv("Mall_Customers.csv")
except FileNotFoundError:
    print("No such file exist please put the Mall_Customers.csv File in the folder with main.py")

allspend = tds[["Age","Annual Income (k$)","Spending Score (1-100)"]]

#Scaling
sl = StandardScaler()
sd = sl.fit_transform(allspend)

#Calculating K using elbow method inertia has the sse after we use k so we store it in ssel
ssel = []
for k in range(1,20):
    km = KMeans(n_clusters=k)
    km.fit(sd)
    ssel.append(km.inertia_)

#plotting the k and sse uncomment and find best k using elbow
'''plt.plot(range(1,20),ssel)
plt.grid()
plt.show()'''

#clustering and predicting
reg = KMeans(n_clusters=5)
tl = reg.fit_predict(sd)

#adding a clusters column to training dataset and initializing 3d plotting
tds['clusters'] = tl
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(5):
    cluster_data = tds[tds['clusters'] == i]
    ax.scatter(cluster_data['Age'], cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'], label=f'Cluster {i}')

ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')
ax.set_title('K-Means Clustering (Age, Income, Spending Score)')
ax.legend()
plt.show()
