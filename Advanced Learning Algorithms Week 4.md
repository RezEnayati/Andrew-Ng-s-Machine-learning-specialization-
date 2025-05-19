**Decision Trees**: A decision Tree is a flowchart-like model used in ML for classification and regression tasks. It splits the data into branches based on feature values making a decision at each node, until it reaches a final leaf node. 

**Learning Process:**
Given the machine learning task of determining if an animal is a cat or not, first step for the decision tree is to pick what feature we want as our root node. For example for this problem we can pick ear shape to be the root node. The tee would look something like this: 

While making this tree we had to make some decision:

**Decision 1:** How to choose what features to split on at each node? 
* Goal: Maximize purity (or minimize impurity) meaning at each step try to distinguish as many cats as possible. So we pick the label with the highest purity. 

**Decision 2:** When do you stop splitting? 
* When a node is 100% one class
* When splitting a node will result in the the tree exceeding a maximum depth
* When improvements in purity score are blow a threshold
* When number of examples in a node is below a threshold

**Measuring Purity**
**Entropy** measures the impurity or uncertainty in a dataset.
* **High Entropy:** More mixed (impure) data
* **Low Entropy:** More pure (one dominant class)

Let's say $p_1$ = fraction of examples that are cats and $p_0 = 1 - p_1$. 
The equation for entropy denoted as $H(p_1)$ is:
$$
H(p_1) = -p_1\log_2(p_1) - p_0\log_2(p_0) 
= -p_1\log_2(p_1) - (1-p_1)\log_2(1-p_1)
$$
**Choosing a split: Information Gain**
In a decision tree the choice of what feature to split on will be based on what feature reduces the entropy the most. 

**Formal Definition:** 

$$
Information\ gain = H(p_1^{root}) - (w^{left}H(p_1^{left}) + w^{right}H(p_1^{right}))
$$
We pick the largest gain at each step to achieve more purity. 

**Decision Tree Learning:**
* Start with all examples at the root node 
* Calculate information gain for all possible features, and pick the one with the highest information gain
* Split dataset according to selected feature, and create left and right branches of the tree
* Keep repeating splitting process until stopping criteria is met:
	* When a node is 100% one class
	* When splitting a node will result in the tree exceeding a maximum depth 
	* Information gain from additional splits in less than threshold 
	* When number of examples in a node is below a threshold 
	
Building a decision tree is a recursive algorithm.

---
#### One Hot encoding: 
If a categorical feature can take on $k$ values, create $k$ binary features (0 or 1 valued), and only one feature will take the value 1. 

---
**Regression Tree**
Predicting an animals weights based on Ear shape, Face shape and Whiskers. For the important decision of choosing a split for a decision tree, instead of choosing the highest information gain split we have to choose the split with the lowest variance among the y labels, meaning we want the output that has less variance and a tighter bound. So instead of taking the weighted average of the entropy of the left and right we take the weighted average of the variance. So instead of looking for reduction in information gain we look for reduction in the average weighted variance. 
![[Screenshot 2025-05-19 at 9.10.55 AM.png|600]]

With this technique we can carry out linear regression.

---
#### Ensemble Trees: Using Multiple Decision Trees 
Trees are highly sensitive to small changes of the data, so having more trees will make them less sensitive to changes in the data. An ensemble of trees is like having different people vote for something and we take the majority vote so it gives us a confidence percentage. So every tree is constructed in a different way and gives us a final vote. 

**Sampling with replacement:**
In definition this means choosing a random sample from data, but before taking a newer sample we replace the old sample back in our data set. The way this applies to building an ensemble of tees is as by constructing multiple random training set and put them in the theoretical bag, we create a new random sample of the same training sample and pick another one again and again until we get the same random samples. This let's us construct a new dataset which is different from our original training set. This is the main building block of our tree ensemble. 

**Bag Decision Tree**
Generating a tree sample
Given a training set of size $m$ 
for $b = 1$ to B: 
* Use sampling with replacement to create a new training set of size $m$ 
* Train a decision tree on the new dataset. 

Setting B to a large number is almost always beneficial unless it gets computationally expensive. 

With one simple change to this algorithm can be changed to a **Random Forest Algorithm**. They key idea is that even with random sampling we might get the same choice of feature at the root node. 
To further randomize the feature choice at each node, which will cause the set of trees to become more different from each other so that when we take vote we get varying results. So:
At each node, when choosing a feature to use to split, if $n$ features are available, pick a random subset of $k < n$ features and allow the algorithm to only choose from that subset of features. 
A good choice for $k$ if our $n$ is large is usually $k = \sqrt{n}$  

**XGBoost (Boosted Tree)**
XGBoost is the most common way for implementing tree ensembles in modern ML. 
Given a training set of size $m$ 
for $b = 1$ to B: 
* Use sampling with replacement to create a new training set of size $m$ 
	* But instead of picking from all examples with equal $\frac{1}{m}$ probability, make it more likely to pick misclassified examples from previously trained trees
* Train a decision tree on the new dataset. 

This idea is the same as deliberate practice in real life. 

**XGBoost Features:**
eXtreme Gradient Boosting. 
* Open source implementation of boosted trees
* Fast efficient implementation 
* Good choice for default splitting criteria and criteria for when to stop splitting
* Built in regularization to prevent overfitting 
* Highly competitive algorithm for machine learning competitions 

**Classification**
```
from xgboost import XGBClassifier 

model. = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(x_test)
```

**Regression**
```
from xgboost import XGBRegressor()

model = XGBRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

#### When to use decision Trees?
**Decision Trees and Tree ensembles**
* Works well on tabular (structured) data 
* Not recommended for unstructured data (images, audio, text)
* Fast 
* Small decision trees may be human interpretable

**Neural Networks:**
* Works well on all types of data including tabular (structured) and unstructured data 
* May be slower that decision tree
* Works with transfer learning 
* When building a system of multiple models working together, it might be easier to string together multiple neural networks. 

