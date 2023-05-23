'''
The HTRU_2 dataset describes several celestial objects and the idea is to be able to classify if an object is a pulsar star or not.
'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D

# Load data
data = pd.read_csv('HTRU_2.csv', names=['mean_ip', 'sd_ip', 'ec_ip', 
                                        'sw_ip', 'mean_dm', 'sd_dm', 
                                        'ec_dm', 'sw_dm', 'pulsar'])

# Preview of the data
# print(data.head())

# Split features from labels
#features = data[[col for col in data.columns if col != 'pulsar']]
features = data.drop('pulsar', axis=1)
labels = data['pulsar']

# Scale data
robust_data = RobustScaler().fit_transform(features)

# Instantiate PCA without specifying number of components
pca_all = PCA()

# Fit to scale data
pca_all.fit(robust_data)

# Save cumulative explained variance
cum_var = (np.cumsum(pca_all.explained_variance_ratio_))
n_comp = [i for i in range(1, pca_all.n_components_ + 1)]

# Plot cumulative variance
ax = sns.pointplot(x=n_comp, y=cum_var)
ax.set(xlabel='number of principal components', ylabel='cumulative explained variance')
plt.show()

# Instantiate PCA with 3 components
pca_3 = PCA(3)

# Fit to scaled data
pca_3.fit(robust_data)

# Transform scaled data
data_3pc = pca_3.transform(robust_data)

# Render the 3D plot
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(data_3pc[:,0], data_3pc[:,1], data_3pc[:,2], c=labels,
            cmap=plt.cm.Set1, edgecolors='k', s=25, label=data['pulsar'])

ax.legend(['non-pulsars'], fontsize='large')

ax.set_title('First three PCA directions')
ax.set_xlabel('1st principal component')
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd principal component")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd principal component")
ax.w_zaxis.set_ticklabels([])

plt.show()

# Instantiate PCA with 2 components
pca_2 = PCA(2)

# Fit and transform scaled data
pca_2.fit(robust_data)
data_2pc = pca_2.transform(robust_data)

# Render the 2D plot
ax = sns.scatterplot(x=data_2pc[:,0], 
                     y=data_2pc[:,1], 
                     hue=labels,
                     palette=sns.color_palette("muted", n_colors=2))

ax.set(xlabel='1st principal component', ylabel='2nd principal component', title='First two PCA directions')
plt.show()
