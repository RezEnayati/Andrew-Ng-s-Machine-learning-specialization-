## Unsupervised Learning Week 2

**Recommender Systems**
Let's say we have a movie streaming website and we want to recommend movies to users. We also allow users to rate movies that they've seen from 0 to 5 stars. 

| **Movie**            | **Alice(1)** | Bob(2) | Carol(3) | Dave(4) |
| -------------------- | ------------ | ------ | -------- | ------- |
| Love at last         | 5            | 5      | 0        | 0       |
| Romance Forever      | 5            | ?      | ?        | 0       |
| Cute Puppies of love | ?            | 4      | 0        | ?       |
| Nonstop car chases   | 0            | 0      | 5        | 4       |
| Swords vs. Karate    | 0            | 0      | 5        | ?       |
$n_u$ = no. of users 
$n_m$ = no. of movies 
$r(i,j) = 1$ if user $j$ has rated movie $i$
$y^{(i,j)}$ = rating given by user j to movie i (defined only if $r(i,j) = 1$)

so in this example 

$n_u = 4$      $r(1,1) = 1$
$n_m = 5$     $r(3,1) = 0$

---
**Using Per Item-Features:** 
How we can develop a recommender system if we had features of each item or in this case features of each movie. To our original chart we can add features to see how much each movie is a romance or an action movie. 

| **Movie**            | **Alice(1)** | Bob(2) | Carol(3) | Dave(4) | $x_1$ romance | $x_2$ action |
| -------------------- | ------------ | ------ | -------- | ------- | ------------- | ------------ |
| Love at last         | 5            | 5      | 0        | 0       | 0.9           | 0            |
| Romance Forever      | 5            | ?      | ?        | 0       | 1.0           | 0.01         |
| Cute Puppies of love | ?            | 4      | 0        | ?       | 0.99          | 0            |
| Nonstop car chases   | 0            | 0      | 5        | 4       | 0.1           | 1.0          |
| Swords vs. Karate    | 0            | 0      | 5        | ?       | 0             | 0.9          |
we can introduce $n$ for the number of features. 
For user 1: Predict rating for movie $i$: $w \cdot x^{(i)} + b$ (similar to linear regression)
$w^{(1)} = [5, 0]$ $b^{(1)} = 0$ $x^{(3)} = [0.99, 0]$ 
$w^{(1)} \cdot x^{(3)} + b^{(1)} = 4.95$ 
By using the parameter $w$ and and the values of x we got for the movie Cute Puppies Of Love, we can estimate what Alice would rate the movie if she watched it and in this case our parameters look pretty accurate because it correctly predicts that she likes more romantic movies.
More generally we assign parameters for each user. 
For user $j$; Predict user j's rating for movie $i$ as $w^{(j)} \cdot x^{(i)} + b^{(j)}$ this is like linear regression except that we are fitting four different linear regression model's for the 4 different users. 

**Cost Function**
Notation:
$r(i,j) = 1$ if user $j$ has rated movie $i$ (0 otherwise)
$y^{(i,j)} =$ rating given by user $j$ on movie $i$ (if defined)
$w^{(j)}, b^{(j)}$ parameters for user $j$
$x^{(i)} =$ feature vector for movie $i$

For every user $j$ and movie $i$, predict rating: $w^{(j)} \cdot x^{(i)} + b^{(j)}$
$m^{(j)} =$ no. of movies rated by user $j$
To learn $w^{(j)}, b^{(j)}$ 

Cost function: same as MSE + L2 
$$
\underset{w^{(j)} b^{(j)}}{min} \ J(w^{(j)}, b^{(j)}) = \frac{1}{2m^{(j)}} \sum_{i;r(i,j)=1}(w^{(j)} \cdot x^{(j)} + b^{(j)} - y^{(i,j)})^2 + \frac{\lambda}{2m^{(j)}}\sum_{k=1}^n(w_k^{(j)})^2
$$
This gives us user $j$'s movie rating. In recommender systems it's convenient to eliminate the $m^{(j)}$ term for the devision and just divide by 2. So to learn parameters $w^{(j)}, b^{(j)}$ for user $j$:
$$
\underset{w^{(j)} b^{(j)}}{min} \ J(w^{(j)}, b^{(j)}) = \frac{1}{2} \sum_{i;r(i,j)=1}(w^{(j)} \cdot x^{(j)} + b^{(j)} - y^{(i,j)})^2 + \frac{\lambda}{2}\sum_{k=1}^n(w_k^{(j)})^2
$$
To learn the parameters for all users:
$$
J\left( \mathbf{w}^{(1)}, \ldots, \mathbf{w}^{(n_u)}, b^{(1)}, \ldots, b^{(n_u)} \right) = 
\frac{1}{2} \sum_{j=1}^{n_u} \sum_{\substack{i:r(i,j)=1}} \left( \mathbf{w}^{(j)} \cdot \mathbf{x}^{(i)} + b^{(j)} - y^{(i,j)} \right)^2 
+ \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^{n} \left( w_k^{(j)} \right)^2
$$
And using gradient descent this will give us a set of parameters for predicting the user's parameters for each of the $n_u$ users. 

---
**Colaborative Filtering Algorithm:**
What if we don't have features $x_1,...,x_n$ to use to predict movies? How can we come up with those features using the data.
Here is the data from before:

| **Movie**            | **Alice(1)** | Bob(2) | Carol(3) | Dave(4) | $x_1$ romance | $x_2$ action |
| -------------------- | ------------ | ------ | -------- | ------- | ------------- | ------------ |
| Love at last         | 5            | 5      | 0        | 0       | ?             | ?            |
| Romance Forever      | 5            | ?      | ?        | 0       | ?             | ?            |
| Cute Puppies of love | ?            | 4      | 0        | ?       | ?             | ?            |
| Nonstop car chases   | 0            | 0      | 5        | 4       | ?             | ?            |
| Swords vs. Karate    | 0            | 0      | 5        | ?       | ?             | ?            |
So we don't know in advance what the features $x_1$ and $x_2$ are. If we pretend to know the features $w$ and $b$ for all the users. What we can do is formulate a cost function for the features $x_1$ and $x_2$ and use gradient descent to find the closest values for the features, we can do this because we have multiple user ratings for a movie. So with pretending that we know the values of $w$ and $b$ for every user we can:

Given the parameters $w^{(1)}, b^{(1)}, w^{(2)}, b^{(2)},...,w^{(n_u)},b^{(n_u)}$:
To learn $x^{(i)}:$ 
$$
J(x^{(i)}) = \frac{1}{2} \sum_{\substack{j:r(i,j)=1}} \left( \mathbf{w}^{(j)} \cdot \mathbf{x}^{(i)} + b^{(j)} - y^{(i,j)} \right)^2 + \frac{\lambda}{2} \sum_{k=1}^{n} \left( x_k^{(i)} \right)^2
$$
To learn $x^{(1)}, x^{(2)}, x^{(3)},..., x^{(n)}$:
$$
J(x^{(1)}, x^{(2)}, x^{(3)},..., x^{(n)})=\frac{1}{2} \sum_{i=1}^{n_m} \sum_{\substack{j:r(i,j)=1}} \left( \mathbf{w}^{(j)} \cdot \mathbf{x}^{(i)} + b^{(j)} - y^{(i,j)} \right)^2 + \frac{\lambda}{2} \sum_{i=1}^{n_m} \sum_{k=1}^{n} \left( x_k^{(i)} \right)^2
$$

this will allow us to learn good features for the movies, but assuming having parameters for the users recall:

Cost function to learn ${w}^{(1)}, \ldots,{w}^{(n_u)}, b^{(1)}, \ldots, b^{(n_u)}$ 
$$
J\left( \mathbf{w}^{(1)}, \ldots, \mathbf{w}^{(n_u)}, b^{(1)}, \ldots, b^{(n_u)} \right) = 
\frac{1}{2} \sum_{j=1}^{n_u} \sum_{\substack{i:r(i,j)=1}} \left( \mathbf{w}^{(j)} \cdot \mathbf{x}^{(i)} + b^{(j)} - y^{(i,j)} \right)^2 
+ \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^{n} \left( w_k^{(j)} \right)^2
$$
To learn $x^{(1)}, x^{(2)}, x^{(3)},..., x^{(n)}$:
$$
J(x^{(1)}, x^{(2)}, x^{(3)},..., x^{(n)}) = \frac{1}{2} \sum_{i=1}^{n_m} \sum_{\substack{j:r(i,j)=1}} \left( \mathbf{w}^{(j)} \cdot \mathbf{x}^{(i)} + b^{(j)} - y^{(i,j)} \right)^2 + \frac{\lambda}{2} \sum_{i=1}^{n_m} \sum_{k=1}^{n} \left( x_k^{(i)} \right)^2
$$
And as you can see the term after the sums is identical, so we can combine these two terms to get the cost function for $x,w,b$:
More generally we can write the sum over all pairs where we do have a movie rating and add the regularization terms. This is a function of $x,w,b$ so minimizing this cost will result in finding the best values for all three parameters. 

$$
J(\mathbf{w}, \mathbf{b}, \mathbf{x}) = 
\frac{1}{2} \sum_{\substack{(i,j):r(i,j)=1}} \left( \mathbf{w}^{(j)} \cdot \mathbf{x}^{(i)} + b^{(j)} - y^{(i,j)} \right)^2 
+ \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^{n} \left( w_k^{(j)} \right)^2 
+ \frac{\lambda}{2} \sum_{i=1}^{n_m} \sum_{k=1}^{n} \left( x_k^{(i)} \right)^2
$$
So we pair over all the movies the users have rated. 

**Gradient Descent For Collaborative Filtering**
We need to optimize with respect to $x$ as well. 
repeat {
$$w_i^{(j)} = w_i^{(j)} - \alpha \frac{\partial}{\partial w_i^{(j)}} J(\mathbf{w}, \mathbf{b}, \mathbf{x}) \\ $$
$$b^{(j)} = b^{(j)} - \alpha \frac{\partial}{\partial b^{(j)}} J(\mathbf{w}, 
\mathbf{b}, \mathbf{x}) \\ $$
$$x_k^{(i)} = x_k^{(i)} - \alpha \frac{\partial}{\partial x_k^{(i)}} J(\mathbf{w}, \mathbf{b}, \mathbf{x}) $$
}
so we update all three parameters for the model. 

This is called **Collaborative Filtering** which is kinda magical but we can do it because multiple users have rated movies.

**Binary Labels: favs, likes, clicks**
How to algorithms to predict binary labels using collaborative filtering:
Let's say the following is likes:

| **Movie**            | **Alice(1)** | Bob(2) | Carol(3) | Dave(4) |
| -------------------- | ------------ | ------ | -------- | ------- |
| Love at last         | 1            | 1      | 0        | 0       |
| Romance Forever      | 1            | ?      | ?        | 0       |
| Cute Puppies of love | ?            | 1      | 0        | ?       |
| Nonstop car chases   | 0            | 0      | 1        | 1       |
| Swords vs. Karate    | 0            | 0      | 1        | ?       |
**Example application:**
1. Did user $j$ purchase an item after being shown?
2. Did the user $j$ fav /like an item?
3. Did user $j$ spend at least 30sec with an item?
4. Did user $j$ click on an item?

Meaning of ratings:
1 -  engaged after being shown item 
0 - did not engage after being shown item 
? - item not yet shown 

**From regression to binary classification:**
- Previously:
	- Predict $y^{(i,j)}$ as $w^{(j)} \cdot x^{(i)} + b^{(j)}$
- For Binary Labels: 
	- Predict that the probability of $y^{(i,j)} = 1$
		is given by $g(w^{(j)} \cdot x^{(i)} + b^{(j)})$
		where $g(z)$: $$g(x) = \frac{1}{1+e^{(z)}}$$
**Cost Function for binary application**
Loss for binary labels: $y^{(i,j)}$: $f_{(\mathbf{w}, \mathbf{b}, \mathbf{x})}(\mathbf{x}) = g\left( \mathbf{w}^{(j)} \cdot \mathbf{x}^{(i)} + b^{(j)} \right)$
Loss For a single example **(Cross-Entropy Loss)**:
$$
L\left( f_{(\mathbf{w}, \mathbf{b}, \mathbf{x})}(\mathbf{x}),\ y^{(i,j)} \right) = 
- y^{(i,j)} \log \left( f_{(\mathbf{w}, \mathbf{b}, \mathbf{x})}(\mathbf{x}) \right)
- \left(1 - y^{(i,j)}\right) \log \left( 1 - f_{(\mathbf{w}, \mathbf{b}, \mathbf{x})}(\mathbf{x}) \right)
$$
For all the parameters including $x$:
$$
J(w,x,b) = \sum_{(i,j):r(i,j)=1} L\left( f_{(\mathbf{w}, \mathbf{b}, \mathbf{x})}(\mathbf{x}),\ y^{(i,j)} \right)
$$
This give us the cost for all the parameters. 

**Implementational Tips:**
**Mean normalizing** will cause the algorithm to run much smoother, say we have the data below.
Notice that user Eve has not rated any movies. 

| **Movie**            | **Alice(1)** | Bob(2) | Carol(3) | Dave(4) | Eve(5) |
| -------------------- | ------------ | ------ | -------- | ------- | ------ |
| Love at last         | 5            | 5      | 0        | 0       | ?      |
| Romance Forever      | 5            | ?      | ?        | 0       | ?      |
| Cute Puppies of love | ?            | 4      | 0        | ?       | ?      |
| Nonstop car chases   | 0            | 0      | 5        | 4       | ?      |
| Swords vs. Karate    | 0            | 0      | 5        | ?       | ?      |
If we minimize the cost function for the given data set below using J:
$$
\frac{1}{2} \sum_{\substack{(i,j):r(i,j)=1}} \left( \mathbf{w}^{(j)} \cdot \mathbf{x}^{(i)} + b^{(j)} - y^{(i,j)} \right)^2 
+ \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^{n} \left( w_k^{(j)} \right)^2 
+ \frac{\lambda}{2} \sum_{i=1}^{n_m} \sum_{k=1}^{n} \left( x_k^{(i)} \right)^2
$$
The parameters for Eve will turn out as $w^{(5)} = [0,0] \ \ b^{(5)} = 0$ because Eve has not rated any movies. So for new users, the algorithm will assume that they have rated 0 for every movie, in this section we will learn how mean normalization can help us solve this problem. 
Say we take all the rating and put them in a matrix, to carry out **mean normalization**.

$$
\begin{bmatrix}
5 & 5 & 0 & 0 & ? \\
5 & ? & ? & 0 & ? \\
? & 4 & 0 & ? & ? \\
0 & 0 & 5 & 4 & ? \\
0 & 0 & 5 & 0 & ?
\end{bmatrix}
$$
And for each movie compute the average so the $\mu$ vector will look like:
$$
\mu = 
\begin{bmatrix}
2.5 \\
2.5 \\
2 \\
2.25 \\
1.25
\end{bmatrix}
$$
So for each rating in the original matrix we can subtract all the value by the mean. which makes our matrix:
$$
\begin{bmatrix}
2.5 & 2.5 & -2.5 & -2.5 & ? \\
2.5 & ? & ? & -2.5 & ? \\
? & 2 & -2 & ? & ? \\
-2.25 & -2.25 & 2.75 & 1.75 & ? \\
-1.25 & -1.25 & 3.75 & -1.25 & ?
\end{bmatrix}
$$
For user $j$, on movie $i$ predict:
$w^{(j)} \cdot x^{(i)} + b^{(j)} + \mu_j$

So for 5 (Eve):
$w^{(5)} = [0,0] \ \ b^{(5)} = 0$ 
$w^{(5)} \cdot x^{(1)} + b^{(5)} + \mu_1 = 2.5$

this will make the optimization a little faster and more reasonable because the initial values in for the parameters of new users will be the mean of other user's ratings.

---
**TensorFlow: Collaborative Filtering**
We can use TensorFlow to take derivatives with respect to variables, this feature is called **Auto Diff** or **Auto Grad**. We can define a custom training loop by. 
The cost function we will use here is a simplified version:
$$J = (wx-1)^2$$
```
w = tf.Variable(3.0) #takes the parameter w and initizlizes it to 3 
x = 1.0
y = 1.0
alpha = 0.01

iterations = 30
for iter in range(iterations):
	# We use TensorFlow's Gradient tape to record the steps
	# Used to compute the cost J, to enable auto differantiation
	with tf.GradientTape():
		fwb = w*x 
		costJ = (fwb - y) ** 2

	# Use the gradient tape to calculte the gradients
	# of the cost with respect to the parameter w.
	[dJdw] = tape.gradient(costK, [w])

	# Run one step of gradient descent by updating 
	# the value of w to reduce the cost.
	w.assign_add(-alpha * dJdw)
```

Once we can calculate the derivatives automatically we can use even more powerful algorithms like the Adam Optimizer. Here is the implementation in TensorFlow

```
# Instansiate an optimizer, learning rate specified
optimizer = keras.optimizers.Adam(learning_rate=1e-1)

iterations = 200

for iter in range(iterations):

	# Use TensorFlow's GradientTape
	# to record the operations used to compute the cost
	with tf.GradientTaoe() as tape:
		# compute the cost (forward pass is included in the cost)
		cost_value = cofiCostFuncV(X, W, b Ynorm, R, num_users, num_movies, 
		lambda)
		
	# Use the gradient tape to automatically retrieve 
	# the gradients of the trainable variables with respect to the loss
	grads = tape.gradient(cost_value, [X,W,b])
	
	# Run one step of gradeint descent by updating 
	# the value of the variables to minimize the loss, zip just reorders the 
	# values
	optimizer.apply_gradients(zip(grads,[X,W,b]))
```

The reason we don't use a neural network is because the collaborative filtering does not fit neatly in the standard neural network types. And we let TensorFlow handle the heavy math work for this. 

---

**Finding Related Items:**
The Collaborative Filtering method, gives us a nice way to find related items, 

The features $x^{(i)}$ of item $i$ are quite hard to interpret, these features do not convey what the movie is about for examples, but these learned features collectively do convey what the movie is like for example. 

To find other items related to it,
* find Item K with $x^{(k)}$ similar to $x^{(i)}$ by:
	* $\sum_{l=1}^{n}(x_l^{(k)} - x_l^{(i)})^{2}$ : this give you the distance between the two items. 

**Limitations of Collaborative Filtering:**
* Cold start problem. How to 
	* rank new items that few users have rated?
	* show something reasonable to new users who have rated few items?

* Use side information about items or users:
	* Item: Genre, movie stars, studio,...
	* User: Demographics (age, gender, location) expressed preferences,...

**Summary:**
The goal of collaborative filtering recommender system is to generate two vectors:
* For each user, a parameter vector that embodies the taste of the user. 
* For each movie, a feature vector of the same size which embodies some description of the movie.
So basically the user vector is multiplied by every movie vector and we add the bias term, this gives a rating the user might potentially give to the movie, we then compare this with the actual rating the user has given this and formulate a cost function which we try to minimize. In the process we update both the user vectors and the movie vectors and reduce the cost. Using the new vectors formed, if we take the new movie vectors and the new user preference vectors and the new bias, we get a potential rating the user might give to a new movie they haven't seen and we only show the user the movies they might rate higher first. 
---
**Collaborative filtering vs Content Based Filtering**

Collaborative Filtering:
* Recommend items to you based on rating of users who have gave similar rating as you, 

Content Based Filtering:
* Recommend items to you based on features of user and item to find a good match, It requires having some features of users and items and uses those to decide which item and users are good match. We continue to use $r(i,j) = 1$ and $y^{(i,j)}$ which is the rating given by user j on item i (if defined). The positive note about content based filtering is that we make good use of both the movie and the user features to make better matches. 

Example of user and item features:

**User Features:** $x_u^{(j)}$  for each user $j$
* Age
* Gender *(1 hot)*
* Movies watched 
* Country *(1 hot, 200)*
* Average rating per genre 

**Movie Features:** $x_m^{(i)}$ for each movie $i$ 
* Year
* Genre/Genres 
* Reviews
* Average rating 

Given these features the task is to figure out if the movie and the user are a good match for each other. 
Previously we were using $W^{(j)} \cdot X^{(i)} + b^{(j)}$, but we will get rid of the b as it does not effect the output of the algorithm, therefore turns into: $W^{(j)} \cdot X^{(i)}$, and instead of writing $w$ we replace it with $V_u^{(j)}$ which is a vector of numbers the user $j$ and we replace $X$ with $V_m^{(i)}$, which is a vector or a list of numbers computed from the features of the movie and $V_u^{(j)}$ is vector calculated from the features of the user. And when we come up and appropriate choice for these vectors then, hopefully the dot product of these will be a good prediction of the user might rate the movie. 

If $V_u$ captures the users's preferences and $V_m$ captures the features of the movies, then the dot product of these would give a sense of how the user might rate the movie. But how do we compute these vectors? An important note is that in contrast to $X_m^{(i)}$ and $X_u^{(j)}$ which can be different sizes, $V_m$ and $V_n$ have to be the same size because we want to take the dot product. 

**Recap:** In collaborative filtering, we have multiple user's give number of rating to multiple users but in content based filtering, we have features of users and features of items and we aim to find the users and the items that match with each other. 

---
**How to compute $V_u$ and $V_m$?** 
A good way to develop a content based filtering is to use **Deep Learning**, the approach here is the same way that commercial systems are built. 
We have to make the below transitions, 
$X_u \rightarrow V_u$  
$X_m \rightarrow V_m$   
For the User Network we can design a Neural Network with 2 hidden layers, with the first one 128 unit, second one 64 units and the output layer as 32 which will give us $V_u$. 
For the Item Network we can design a Neural Network with two hidden layers and one output layer, we want the first hidden layer to have 256 units, the second hidden layer to have 128 units and the output layer to again age 32 units, which has to match the number of the output layer in the User Network. This guarantees that both the user and the item vectors have the same length and we can take the dot product. So hypothetically the two networks can have different number of hidden layers but have to have the same amount of units in the output layer. 

**Prediction:** $V_u \cdot V_m$ 
If we had binary labels, we could have run the dot product through the sigmoid function, like:
$g(V_u \cdot V_m)$ and use this to predict the probability that $y^{(i,j)} = 1$. 

Even though we explained these neural networks separately we can combine them into one, not literally, but we can have the in parallel and make the prediction of the networks the dot product of the two output layers which are $V_u$  and $V_m$. Each of these layers have a different parameters so how do we actually train a network like this? 
What we can do is to construct a cost function, which is similar to the one we saw in collaborative filtering, and we assume that we have some users who have already rated the movies. Therefore the cost function will be:
$$
J = \sum_{(i,j);r(i,j)=1} (v_u^{(j)} \cdot v_m^{(i)} - y^{(i,j)})^2 + NN \ regularization \ term
$$
Assuming that we have some data of some users having rated some movies, we take the sum of the squared difference between the prediction and the actual user rating. So we aim to train the model by minimizing the error of the prediction and this will cause us to get different vectors for the user and the movie. There is no separate procedure to train these networks, we train both at the same time using the optimizer we want like Gradient Descent or Adam. 

After training the model, we can use this to find similar items. 

* $v_u^{(j)}$ is a vector of length 32 that describes the user $j$ with features $x_u^{(j)}$
* $v_m^{(i)}$ is a vector of length 32 that describes the user $j$ with features $x_m^{(i)}$ 

To find movies similar to movie $i$:
$|| V_m^{(k)} - V_m^{(i)}||^2$ has to be small so using this approach we can find movies who's vectors are similar for each user. 

Note: This can be pre-computed ahead of time. Meaning we can compute the distance before the user want to access it. 
If we are to implement this in practice, we have to spend time to engineer good features for this. One limitation of this algorithm is that it's computationally expensive.  

---
**Recommending From a Large Catalogue** 
We will sometimes need to pick a handful of items to recommend from a catalog of millions of items or more. But how do we do this efficiently? Since we have a large amount of users and a large amount of items we want to recommend, running the neural network thousands of time is very computationally expensive. 

**Large Scale Recommender Systems are Implemented In Two Steps:** 

**Retrieval & Ranking** 

**Retrieval:**
* Generate large list of plausible item candidates 
* Will cover a lot of possible things we might want to recommend to the user and maybe items the user might not like. 
	Example:
1. For each of the last 10 movies watched by the user, find 10 most similar movies 
2. For most viewed 3 genres, find the top 10 movies 
3. Top 20 movies in the country 
* Combine retrieved items into list, removing duplicates and items already watched / purchased. 

*This will include some good options and some options that the user might not like at all but that's okay because the goal of the retrieval step is to ensure broad coverage*

**Ranking:** 
* Take list retrieved and rank using learned model, meaning we would feed the user vector and each of the movie vectors and make a prediction like we explained above. 
* Display ranked items to user. 

One additional optimization is that if you've computed the $V_m$ 's for all the movies in advance, then all you need to do is to do inference on the user network of the model a single time to compute the $V_u$ and then take the inner product of the $V_u$ and $V_m$ to calculate the potential rating. 

One of the decisions we need to make is to choose how many items we choose to retrieve, to get an accurate ranking step. 
* Retrieving more items results in better performance but slower recommendations. 
* To analyze / optimize the trade-off, carry out offline experiments to see if retrieving additional items results in more relevant recommendations 

---
**TensorFlow Implementation**

We can use a sequential model, for both the user and the item. 
```
# For the user Nueral Network 
user_NN = tf.keras.models.Sequential([
	tf.keras.layers.Dense(256, activation = 'relu')
	tf.keras.layers.Dense(128, activation = 'relu')	
	tf.keras.layers.Dense(32)
]) 
```

```
# For the item Neural Network 
item_NN = tf.keras.models.Sequential([
	tf.keras.layers.Dense(256, activation = 'relu')
	tf.keras.layers.Dense(128, activation = 'relu')
	tf.keras.layers.Dense(32)
])
```

```
# create the user input and point to the base network 
input_user = tf.keras.layers.Input(shape =(num_user_features))
vu = user_NN(input_user)
vu = tf.linalg.l2_normalize(vu, axis = 1) 
# The line above normalizes the vu to have length one
```

```
# create the item input and point to the base network 
input_item = tf.keras.layers.input(shape = (num_item_features))
vm = item_NN(input_item)
vm = tf.linalg.l_normalize(vm, axis = 1)
# normalizes the length to be equal to one
```

```
# taking the dot product of the two vectors 
# and uses a speical keras layer to take the dot product of two vectors 
output = tf.keras.layers.Dot(axes=1)([vu,vm])
```

```
#specify the inputs and the outputs of the model 
model = Model([input_user, input_item], output)
```

```
# Specify the cost function 
cost_fn = tf.keras.loses.MeanSquaredError() 
```

# 🧠 Principal Component Analysis (PCA)

Principal Component Analysis (PCA) is a **dimensionality reduction** technique used to simplify complex datasets by transforming them into a lower-dimensional space — while preserving as much **variance** as possible.

---

## 🔍 Why Use PCA?

- Reduce dimensionality of large datasets  
- Remove noise and redundancy  
- Visualize high-dimensional data  
- Improve performance of ML algorithms  

---

## 📊 Core Idea

PCA finds **new axes (principal components)** that:

1. Are **orthogonal** (i.e., uncorrelated)  
2. Capture the **maximum variance** in the data  

The **first principal component** captures the most variance, the second captures the next most (orthogonal to the first), and so on.

---

# 🧠 Principal Component Analysis (PCA)

Principal Component Analysis (PCA) is a **dimensionality reduction** technique used to simplify complex datasets by transforming them into a lower-dimensional space — while preserving as much **variance** as possible.

---

## 🔍 Why Use PCA?

- Reduce dimensionality of large datasets  
- Remove noise and redundancy  
- Visualize high-dimensional data  
- Improve performance of ML algorithms  

---

## 📊 Core Idea

PCA finds **new axes (principal components)** that:

1. Are **orthogonal** (i.e., uncorrelated)  
2. Capture the **maximum variance** in the data  

The **first principal component** captures the most variance, the second captures the next most (orthogonal to the first), and so on.

---

## 🧮 PCA Step-by-Step

1. **Standardize the data**  
   Zero mean and unit variance for each feature.

   ```python
   from sklearn.preprocessing import StandardScaler
   X_std = StandardScaler().fit_transform(X)