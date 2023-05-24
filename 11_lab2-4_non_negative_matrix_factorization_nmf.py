'''
NMF expresses samples as combinations of interpretable parts. For example, 
it represents documents as combinations of topics, and images in terms of commonly occurring visual patterns. 
NMF, like PCA, is a dimensionality reduction technique. In contrast to PCA, however, NMF models are interpretable. 
This means NMF models are easier to understand and much easier for us to explain to others. NMF can't be applied to 
every dataset, however. It requires the sample features be non-negative, so greater than or equal to 0.

To test NMF you will use the 20newsgroups dataset which comprises around 12000 newsgroups posts on 20 topics. 
'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.datasets import fetch_20newsgroups

# Download data
data = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))

# Get the actual text data from the sklearn bunch
data = data.get('data')

print(f'Data has {len(data)} elements.')
print(f'First 2 elements:')
for n, d in enumerate(data[:2], start=1):
    print('===='*10)
    print(f'Element number {n}:\n{d}')

# Instantiate vectorizer setting dimensionality of data
# The stop_words param refer to words that dont add much value to the content 
# of the document and must be ommited
vectorizer = TfidfVectorizer(max_features=500, stop_words='english')

# Vectorize original data
vect_data = vectorizer.fit_transform(data)

# Print dimensionality
print(f'Data has shape {vect_data.shape} after vectorization.')
print(f'Each data point has shape {vect_data[0].shape} after vectorization.')

# Desired number of components
n_comp = 5

# Instantiate NMF with the desired number of components
nmf = NMF(n_components = n_comp, random_state = 42)

# Apply NMF to the vectorized data
nmf.fit(vect_data)

reduced_vect_data = nmf.transform(vect_data)

# Print dimensionality
print(f"Data has shape {reduced_vect_data.shape} after NMF.")
print(f"Each data point has shape {reduced_vect_data[0].shape} after NMF.")

# Save feature names for plotting
feature_names = vectorizer.get_feature_names_out()

# Define function for plotting top 20 words for each topic
def plot_words_for_topics(n_comp, nmf, feature_names):
  fig, axes = plt.subplots(((n_comp-1)//5)+1, 5, figsize=(25, 15))
  axes = axes.flatten()

  for num_topic, topic in enumerate(nmf.components_, start=1):

    # Plot only the top 20 words

    # Get the top 20 indexes
    top_indexes = np.flip(topic.argsort()[-20:])

    # Get the corresponding feature name
    top_features = [feature_names[i] for i in top_indexes]

    # Get the importance of each word
    importance = topic[top_indexes]

    # Plot a barplot
    ax = axes[num_topic-1]
    ax.barh(top_features, importance, color="green")
    ax.set_title(f"Topic {num_topic}", {"fontsize": 20})
    ax.invert_yaxis()
    ax.tick_params(labelsize=15)

  plt.tight_layout()
  plt.show()

# Run the function
plot_words_for_topics(n_comp, nmf, feature_names)

# The following function condenses the previously used code so you can play trying out with different number of components
def try_NMF(n_comp):
    nmf = NMF(n_components=n_comp, random_state=42)
    nmf.fit(vect_data)
    feature_names = vectorizer.get_feature_names_out()
    plot_words_for_topics(n_comp, nmf, feature_names)

# Try different values!
try_NMF(20)