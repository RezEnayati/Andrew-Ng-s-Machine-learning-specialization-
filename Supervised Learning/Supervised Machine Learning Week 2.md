**Multivariate Linear Regression:** 
Linear Regression with more than just one one feature. (Multiple Variable Regression)

Last week we learned that with one variable the model was:
$$
f_{w,b}(x) = wx + b
$$
No we have multiple features. Going back to the house price example, instead of just using the house square footage we use, parameters like umber of bedrooms, number of floors and etc. To denote these parameters we can use these variable names: 
$$
x_1, \ x_2, \ x_3, \ ...
$$
Notation:

$$x_j = j^{th} feature $$
$$n = number \ of \ features $$
$$\vec{x}^{(i)} = features\ of \ i^{th} \ training \ example$$
$$ x_j^{(i)} = value \ of \ feature \ j \ in \ the \ i^{th} \ training \ example$$
$$\vec{x} \ is \ vector$$

New Model For multiple Variable Linear Regression:

$$
f_{w,b}(x) =w_1x_1 + w_2x_2 + ... + w_nx_n + b
$$
In the house price example $b$ can for example represent the base price of a house, not because it's "b" because it can be any starting value for the model. 
We can rewrite this equation:
$$
\vec{w} = [w_1,w_2,w_3...w_n]
$$
$$
b \ is \ a\ number
$$
$$
\vec{x} = [x_1,x_2,x_3...x_n]
$$
The new model is:
$$
f_{\vec{w},b}(\vec{x}) = \vec{w} \ \cdot \ \vec{x} \ + \ b
$$
This represents the dot product of the vector w and the vector x, which is the same as multiplying the corresponding values in the same position and adding them. 
This model is represents **Multiple Linear Regression.**

**Vectorization:** Using vectorization will make our code much shorter and much more efficient. 
We can use *NumPy* dot vectorization to calculate the dot product. Behind the scenes NumPy uses parallel hardware, this makes it much more efficient and practical, making the code much cleaner and faster:

							` f = np.dot(w,b) + b `

**What is the computer doing behind the scenes to make vectorization faster?**
* ***Without vectorization:** A for loop performs operations one after the other, finds the values and adds them.
* ***With Vectorization:** The computer gets all the values corresponding to the vector all at the same time and performs the multiplication at the same time. The addition is carried out at the same time with specialized hardware. 

This also applies to gradient descent, since the derivatives of the values also form a NumPy array. This can be used to calculate the new values of the descent after every iteration. Basically we take the old $\vec{w}$, take the derivative of every element in the cost function and multiply it by the learning rate alpha and replace the values back in w in one step using vectorization. 

**Gradient Descent for multiple linear Regression:**
Parameters:
$$\vec{w}$$
$$b$$
Model: 
$$
f_{\vec{w},b}(\vec{x}) = \vec{w} \ \cdot \ \vec{x} \ + \ b
$$
Cost Function:
$$J(\vec{w},b)$$
Gradient Descent:
repeat {
$$
w_j = w_j - \alpha \frac{d}{dw_j}J(\vec{w},b)
$$
$$
b = b - \alpha \frac{d}{db}J(\vec{w},b)
$$

}

If we take the derivative:
repeat{
$$
w_1 = w_1 - \alpha \frac{1}{m} \sum_{i=1}^m(f_{\vec{w},b}(\vec{x^{(i)}}) \ - \ y^{(i)})x_1^{(i)}
$$
For $w_1$ to $w_n$
$$
b = b - \alpha \frac{1}{m} \sum_{i=1}^m(f_{\vec{w},b}(\vec{x^{(i)}}) \ - \ y^{(i)})
$$
simultaneous for $b$ and  $w_i$

**An Alternative to Gradient Descent:**
- Normal equation:
	* Only For linear Regression
	* Solve for w, b without iterations. 

Disadvantage:
* Doesn't generalize to other learning algorithm
* Slow when number of features is large (> 10,000)
* Normal equation method may be used in machine learning libraries that implement linear regression.
* **Gradient descent is recommended.** 

**Feature Scaling:** 
Enables Gradient Descent to run much faster. 
$x_1$: 300-2,000 Square Feet
$x_2$: 0-5 Number of bedrooms. 
When the possible range of features is large, like the size is in square feet a good model will choose a small parameter value. When the possible values of the feature is small the reasonable value for the parameter is predicted to be large. 
In this case gradient descent will have a hard time finding the global minimum because it will have to bounce back and fourth a lot. 
In situations like this it's better to perform some kind of transformation on our training data so the features, both range from the same scale. The key point is to make them take comparable values against each other, so the gradient descent can find the global minimum faster.
How can we implement this? 
1. **Basic Normalization:** We can divide the features by the maximum value in the range of the feature. 
2. **Mean normalization**: First find the average for one feature, and subtract the mean by the feature and divide by the difference of the max and the min. 
		$$ \mu_x = \frac{1}{m} \sum^{m}_{i=1} x^{(i)}$$
		$$
		 R_x = \underset{1 \ \le \ i \ \le \ m}{max}  x^{(i)} \ - \ \underset{1 \ \le \ i \ \le \ m}{min}  x^{(i)}
		$$
		$$
		\tilde{x}^{(i)} = \frac{x^{(i)} - \mu_x}{R_x} \ \ i=1,...m
		$$
3. **Z-Score normalization**: (Gaussian distribution). first calculate the mean and the standard deviation for each feature and to normalize, we take the feature, subtract the mean from it and divide by the standard deviation. 
		$$ \mu_x = \frac{1}{m} \sum^{m}_{i=1} x^{(i)}$$
		$$\sigma_x = \sqrt{ \frac{1}{m} \sum_{i=1}^{m} \left( x^{(i)} - \mu_x \right)^2 }
		$$
		$$x^{(i)} = \frac{x^{(i)} -\mu_x}{\sigma_x} \ \ i = 1,...,m$$
**Rule of thumb:** 
We want to get the features from -1 to 1 for each feature, the point of this procedure is to make gradient descent faster and more efficient. There is almost never harm for feature scaling and should be s standard practice. 


**How to recognize if Gradient is Converging?** 
repeat {
$$
w_j = w_j - \alpha \frac{d}{dw_j}J(\vec{w},b)
$$
$$
b = b - \alpha \frac{d}{db}J(\vec{w},b)
$$

}

One of the key choices is choosing the learning rate alpha correctly.

**Recall:** Objective of gradient descent is to minimize the cost function J.
$$
min_{\vec{w},b}J(\vec{w},b)
$$
**Plotting:** if you plot the function J, at each iteration, the horizontal axis is the number of iterations and the y axis is the value of the cost. This is called the learning curve. Looking at the graph helps to see how the cost changes, if the cost is decreasing after each iteration this means that it's working. After a while the graph should more or less stay the same or *converge*, but the number of iterations is application dependent. Creating a learning curve will help understand if the we are converging. 

**Automatic Convergence Test:** 

$$
Let \ \epsilon \  be \ 10^{-3}
$$
If the cost J decreases by less than epsilon, in one iteration declare *convergence*. Finding the epsilon is hard,  but looking at the graph is easy. So plotting a learning curve is the best method. 

**How to choose an appropriate learning rate:** 

The algorithm will run fast with a good alpha.
If alpha is small -> Gradient Descent is slow.
If alpha is big -> Gradient Descent might never converge. 

If you plot the cost of the number of iterations and the cost function value, and notice that the cost goes up and down. **This is a sign that gradient descent is not working properly.** Means there are bugs, or learning rate might be too big.

If the plot increases with the number of iterations this can also mean the learning rate might be too large, and could be addressed by choosing a small learning rate.

**Debug Tip:** With a small enough learning rate, J should decrease on every iteration. So one thing to do is to set alpha to a small number, but if J still does not decrease there could be a bug. 

Range of values to try:

	0.001.   0.01     0.1     1
	
After choosing each alpha, plot a hand full of the iterations and see if the learning rate is decreasing rapidly and consistently. What you should do is try a range of values starting from small to large. And see what works best. After that, pick the largest possible learning rate.

**Feature Engineering:** 
Choosing the features is a critical step towards making the algorithm work well. 
Using your intuition to design new features by transforming or combining original features. Usually by transforming or converting. 

**Polynomial Regression:** fit non-linear lines through our data like curves, this is a feature of feature engineering, like a quadratic function for the data. 
Example model:
$$
f_{\vec{w},b}(x) = w_1x \ + \ w_2x^2 \ + \ w_3x^3 \ + \ b 
$$
In this case feature scaling becomes a lot more important because the ranges will vary a lot. This is covered Next Week!