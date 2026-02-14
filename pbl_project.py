import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

black_box_model = RandomForestClassifier(n_estimators=100, random_state=42)
black_box_model.fit(X_train, y_train)

predictions = black_box_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Black Box Model Accuracy: {accuracy * 100:.2f}%")

hc = AgglomerativeClustering(n_clusters=5)
hc_labels = hc.fit_predict(X_train)

knn_cluster_finder = KNeighborsClassifier(n_neighbors=5)
knn_cluster_finder.fit(X_train, hc_labels)

instance_index = 0
instance_to_explain = X_test[instance_index].reshape(1, -1)
suggested_cluster = knn_cluster_finder.predict(instance_to_explain)

local_X = X_train[hc_labels == suggested_cluster]
local_y = y_train[hc_labels == suggested_cluster]

local_linear_model = LinearRegression()
local_linear_model.fit(local_X, local_y)

feature_names = iris.feature_names
coefficients = local_linear_model.coef_

results = []
for i in range(5):
    cluster_id = knn_cluster_finder.predict(instance_to_explain)
    local_X_loop = X_train[hc_labels == cluster_id]
    local_y_loop = y_train[hc_labels == cluster_id]
    
    model_loop = LinearRegression()
    model_loop.fit(local_X_loop, local_y_loop)
    
    current_weights = model_loop.coef_
    results.append(current_weights)
    print(f"Iteration {i+1}: Petal Width Weight = {current_weights[3]:.6f}")

is_stable = all(np.array_equal(x, results[0]) for x in results)
print(f"\nStability Check: {'SUCCESS' if is_stable else 'FAILURE'}")

plt.figure(figsize=(10, 6))
colors = ['red' if w < 0 else 'green' for w in coefficients]
plt.barh(feature_names, coefficients, color=colors)
plt.title(f"DLIME Feature Importance (Instance {instance_index})")
plt.xlabel("Feature Weight")
plt.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.show()