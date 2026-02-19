DLIME: Deterministic Local Explanations on Iris Dataset

Student Name: Tarang Gupta

Objective: To improve the reliability of XAI (Explainable AI) by implementing a deterministic clustering approach.

Key Features
Accuracy: Achieved 100.00% using a Random Forest "Black Box" model.

Stability: Proven 100% stable results (Weights remained constant at 0.896118 across 5 iterations).

Algorithms: Agglomerative Clustering for deterministic grouping and KNN for local mapping.

How to Run
Install dependencies: pip install -r requirements.txt

Run the script: python pbl_project.py





📊 Project Execution & Results
1. Performance Metrics
Black Box Accuracy: The Random Forest Classifier achieved 100.00% accuracy on the Iris test set.

Local Neighborhood: For the test instance explained (Instance #0), the system identified a deterministic cluster consisting of 27 similar training samples.

2. Feature Importance (DLIME Explanation)
The following weights represent how much each physical trait influenced the model's prediction for the selected flower:

Petal Width (cm): +0.8961 (Most Significant Driver)

Petal Length (cm): +0.3012

Sepal Length (cm): -0.1625

Sepal Width (cm): -0.5885

3. Stability Validation (Deterministic Proof)
This is the core achievement of the project. We ran the explanation process five times for the same instance to verify reproducibility:

Iteration 1: 0.896118

Iteration 2: 0.896118

Iteration 3: 0.896118

Iteration 4: 0.896118

Iteration 5: 0.896118

Result: Stability Check SUCCESS. The results are 100% consistent, proving that DLIME solves the "instability problem" found in standard LIME.
