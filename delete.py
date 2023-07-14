import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

X = np.array([[1, 2], [3, 4], [8, 9]])
Y = np.array([[7, 8], [9, 10]])

similarity = cosine_similarity(X, Y)

print(similarity)
print(similarity.max(axis=1))

for v in similarity :
    print(max(v))