
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
import lime
import lime.lime_tabular


iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

black_box_model = RandomForestClassifier(n_estimators=100, random_state=42)
black_box_model.fit(X_train, y_train)


hc = AgglomerativeClustering(n_clusters=5)
hc_labels = hc.fit_predict(X_train)

knn_cluster_finder = KNeighborsClassifier(n_neighbors=5)
knn_cluster_finder.fit(X_train, hc_labels)


idx_val = 0
instance_to_explain_2d = X_test[idx_val].reshape(1, -1)
instance_to_explain_1d = X_test[idx_val]


print("\n--- Testing DLIME (Deterministic) ---")
print("Running DLIME 5 times on the same flower...")


dlime_results = list()

for i in range(5):
    cluster_id = knn_cluster_finder.predict(instance_to_explain_2d)
    local_X = X_train[hc_labels == cluster_id]
    local_y = y_train[hc_labels == cluster_id]
    
    local_model = LinearRegression()
    local_model.fit(local_X, local_y)
    
  
    weights_array = local_model.coef_.tolist()
    petal_width_weight = weights_array.pop(3)
    
    dlime_results.append(petal_width_weight) 
    print(f"DLIME Iteration {i+1}: Petal Width Weight = {petal_width_weight:.6f}")

print("\n--- Testing Standard LIME (Random) ---")
print("Running LIME 5 times on the exact same flower...")

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    mode='classification',
    random_state=None 
)

for i in range(5):
    exp = explainer.explain_instance(instance_to_explain_1d, black_box_model.predict_proba, num_features=1)
    
    lime_explanation_list = exp.as_list()
    top_rule_tuple = lime_explanation_list.pop(0)
    feature_rule, feature_weight = top_rule_tuple
    
    print(f"LIME Iteration {i+1}: Top Feature Rule [{feature_rule}] | Weight = {feature_weight:.6f}")
    
