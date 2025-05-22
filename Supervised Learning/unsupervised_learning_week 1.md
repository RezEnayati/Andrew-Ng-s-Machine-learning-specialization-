* Clustering 
* Anomaly Detection 
---
**What is Clustering?**
A clustering algorithm looks at a number of data points and automatically finds related or similar data points. 
In supervised learning when we did binary classification with either neural networks or logistic regression our data points had both $x$ label and $y$ label, but in unsupervised learning we don't have target label $y$.
**Clustering:** will look at a dataset and try to find interesting information about them. 
* Used for grouping similar news 
* Market segmentation 
* DNA analysis 
* Astronomical data analysis

**K-means Intuition:**
**Intuition:** 
1. First step is that it will randomly pick two points 
	* These centers are called cluster centroids
2. After the initial guess, it will go through different data examples 
	* It will check weather each point is close to the red cluster centroid or the blue cluster centroid and will assign the points to which ever centroid the data is closer to. 
3. It will look at all the points assigned to a centroid and take an average of them and will move the centroid to wherever the average falls. It does this for all centroids.
4. It then repeats from step 2 again and will move the centroids based on where the data is closer to them. 
5. This ends up moving the centroid the the best location and no further change will happen. 

**K-Means Algorithm:**
Randomly initialize $K$ cluster centroids $\mu_1,\mu_2,\mu_3,...,\mu_K$ ($\mu$ can be whatever dimension the training data has.)
Repeat {
	# Assign points to cluster centroids 
	for i = 1 to m:
		$c^{(i)}$ := index (from $1$ to $K$) of cluster centroid closest to $x^{(i)}$ 
		# Distance in math is called the L2 norm denoted by $\left\| x^{(i)} - \mu_{k} \right\|$  you want to find the value of $k$ that minimizes this. In practice we minimize $\left\| x^{(i)} - \mu_{k} \right\|^2$.
		# Move cluster centroids
	for $k$ = 1 to K:
		$\mu_k$ := average (mean)  of points assigned to cluster $k$. 
*}

K means applies for clusters that are not well separated along with algorithms that are well clustered. 

**Optimization Objective:** 
$c^{(i)}$ = index of cluster (1,2,...,K) to which example $x^{(i)}$ is currently assigned
$\mu_k$ = location of the cluster centroid $K$ 
$\mu_{c^{(i)}}$ = cluster centroid of cluster to which example $x^{(i)}$ has been assigned

**Cost function:** 

$J(c^{(1)},...,c^{(m)}, \mu_1,...,\mu_k) = \frac{1}{m} \sum_{i=1}^m \left\| x^{(i)} - \mu_{k} \right\|^2$   

The name of the cost function is called the **distortion function.**
The algorithm tries to minimize this cost function. 

On every iteration the cost function should go down. The distortion function should never go up, that means that there is a bug in the code, every single step is trying to reduce the cost function $J$.

**Initializing K-means:** The very first step of K-means clustering algorithms, is to choose random locations as the initial guesses for the cluster centroid $\mu_1$ through $\mu_k$, but how do we actually make that random guess?
The algorithm again 

Step 0: Randomly initialize K cluster centroids $\mu_1,\mu_2,...,\mu_k$ 
Repeat {
	Step 1: Assign points to cluster centroids 
	Step 2: Move cluster centroids
}

Choose K < m
The most common way to pick the cluster centroids:
Randomly pick K training examples.
Set $\mu_1, \mu_2,...,\mu_k$ equal to these K examples. 
* This means to take K training examples and place the cluster centroids on top of them. 
* The first random location is important because if the first choice is not a very good one, we might get stuck in local minima. So running k-means multiple times is often a better choice. For every choice of the first stage we can calculate the cost function J and pick the one with the lowest cost function for the initial guess. 

Algorithm:
for 100 random initializations:

for i = 1 to 100 {
	Randomly initialize K-means.
	Run K-means. Get $c^{(1)},...,c^{(m)}, \mu_1,...,\mu_k$ 
	Compute cost function (distortion)
}

Pick the set of clusters that gave the lowest J. 

Running it anywhere from 50-1000 times is common. 

**Choosing the Number of Clusters:** 
The question of choosing the value of K is very ambiguous so there are techniques to automatically choose a good number of clusters.

**Elbow Method:** You would run K-means with a different number of clusters and calculate the cost function J and plot that, then you would choose the number of clusters after the rapid decrease in the cost function. This causes problems because, if the cost function J decreases smoothly, there is no elbow to choose from. 

**In practice:**
Often, you want to get clusters for some later (downstream) purpose. 
Evaluate K-means based on how well it performs on that later purpose. 

**Anomaly Detection:**
These algorithms look at unlabeled datasets of normal events and learns to raise a red flag if there is an unusual or anomalous event.
The most common way to carry out anomaly detection is a technique called density estimation. What it means is that when you are given a training set of these algorithms of these m examples the first thing is to build a model of the probability of x. Meaning you look at the points in the data set and give then a probability of how likely they are to be seen in the dataset. Then we would compute the probability of $x_{test}$ and if the probability of it is small we would flag it as an anomaly. 
**Use cases:**
* Fraud detection in social media accounts or financial institutes.
* Manufacturing to see if product is well enough or not 
* Monitoring computers in data centers

**Gaussian (normal) Distributions**
Say $x$ is a number.
Probability of $x$ is determined by Gaussian with mean $\mu$ and variance $\sigma^2$. 
![[Screenshot 2025-05-21 at 6.37.59 PM.png|400]]
*Also called the bell shape curve*
$f(x) = \frac{1}{\sigma \sqrt{2\pi}} \, e^{-\frac{(x - \mu)^2}{2\sigma^2}}$ this formula gives the probability of a random variable x in a normal distribution. 

The changes in $\sigma$ will result in the plot being thinner or wider, if we increase sigma we get a wider curve and if we decrease sigma we get a thinner curve, when the plot get's thinner it also gets taller and when sigma increases it causes the plot to be shorter this is because the sum of the probabilities will need to add up to one. Changing the mean will move the curve the left or right because the mean gives the middle of the curve. 

$\mu = \frac{1}{m} \sum_{i=1}^{m} x^{(i)}$
$\sigma^2 = \frac{1}{m} \sum_{i=1}^{m} (x^{i} - \mu)^2$
From statistics these are called the **maximum likelihood estimate (MLE)** for $\mu$ and $\sigma$. 
This here is works only if we have one feature, so for practical anomaly detection we usually have a lot of different features. 

**Algorithm for multiple features.**
Training set: ${{\vec{x}^{(1)},\vec{x}^{(2)},...,\vec{x}^{(m)}}}$
Each training example $\vec{x}^{(i)}$ has $n$ features
$p(\vec{x}) = p(x_1;\mu_1,\sigma_1^2 ) \times p(x_2;\mu_2,\sigma_2^2 ) \times p(x_3; \mu_3,\sigma_1^2 )\times...\times p(x_n; \mu_n,\sigma_n^2 )$ 
$p(\vec{x}) = \prod_{j=1}^n p(x_j;\mu_j,\sigma_j^2)$ 

Algorithm:
1. Choose $n$ features $x_i$ that you think might be indicative of anomalous examples. 
2. Fit parameters $\mu_1,...,\mu_n, \sigma_1^2, ..., \sigma_n^2$
3. Given new example $x$, compute $p(x)$.
	* $p(\vec{x}) = \prod_{j=1}^n p(x_j;\mu_j,\sigma_j^2) = \prod_{j=1}^{n} \frac{1}{\sigma \sqrt{2\pi}} \, e^{-\frac{(x - \mu)^2}{2\sigma^2}}$
4. Anomaly if $p(x) < \epsilon$

**How to choose epsilon and evaluating the model:**
It's useful to have some labeled data where we can do cross validation and test example to test the algorithm with it. 
