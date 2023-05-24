import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits

from sklearn.decomposition import TruncatedSVD

# Loard the digits dataset
digits = load_digits()

# plot first digit
image = digits.data[0].reshape((8,8))
plt.matshow(image, cmap='gray')
plt.show()

# save data into x variable

x = digits.data

# normalize pixel valus
x = x/255

# Print shapes of dataset and data points
print(f'Digits data has shape {x.shape}')
print(f'Each data point has shape {x[0].shape}')

# Plot 1st digit to check normalization
image = x[0].reshape((8,8))
plt.matshow(image, cmap='gray')
plt.show()

# Instantiate truncated SVD with (original dimension - 1) components
org_dim = x.shape[1]
tsvd = TruncatedSVD(org_dim - 1)
tsvd.fit(x)

# Save cumulative explained variance
cum_var = (np.cumsum(tsvd.explained_variance_ratio_))
n_comp = [i for i in range(1, org_dim)]

# Plot cumulative variance
ax = sns.scatterplot(x=n_comp, y=cum_var)
ax.set(xlabel='number of  components', ylabel='cumulative explained variance')
plt.show()

print(f"Explained variance with 5 components: {float(cum_var[4:5])*100:.2f}%")

# Instantiate a truncated SVD with 5 components
tsvd = TruncatedSVD(n_components=5)

# Get the transformed data
x_tsvd = tsvd.fit_transform(x)

#print shapes of dataset and data points
print(f"Original data points have shape {x[0].shape}\n")
print(f"Transformed data points have shape {x_tsvd[0].shape}\n")

# See data reduced
image_reduced_5 = tsvd.inverse_transform(x_tsvd[0].reshape(1, -1))
image_reduced_5 = image_reduced_5.reshape((8, 8))
plt.matshow(image_reduced_5, cmap = 'gray')
plt.show()

# Using more components
def image_given_components(n_components, verbose=True):
    tsvd = TruncatedSVD(n_components=n_components)
    X_tsvd = tsvd.fit_transform(x)
    if verbose:
        print(f"Explained variance with {n_components} components: {float(tsvd.explained_variance_ratio_.sum())*100:.2f}%\n")
    image = tsvd.inverse_transform(X_tsvd[0].reshape(1, -1))
    image = image.reshape((8, 8))
    return image

image_reduced_32 = image_given_components(32)
plt.matshow(image_reduced_32, cmap = 'gray')
plt.show()

fig = plt.figure()

# Original image
ax1 = fig.add_subplot(1,4,1)
ax1.matshow(image, cmap = 'gray')
ax1.title.set_text('Original')
ax1.axis('off') 

# Using 32 components
ax2 = fig.add_subplot(1,4,2)
ax2.matshow(image_reduced_32, cmap = 'gray')
ax2.title.set_text('32 components')
ax2.axis('off') 

# Using 5 components
ax3 = fig.add_subplot(1,4,3)
ax3.matshow(image_reduced_5, cmap = 'gray')
ax3.title.set_text('5 components')
ax3.axis('off') 

# Using 1 components
ax4 = fig.add_subplot(1,4,4)
ax4.matshow(image_given_components(1), cmap = 'gray') # Change this parameter to see other representations
ax4.title.set_text('1 component')
ax4.axis('off')

plt.tight_layout()
plt.show()