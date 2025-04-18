# Machine Learning

Field of study that gives computers the ability to learn without being explicitly programmed.

---

## Two Main Machine Learning Algorithms

1. **Supervised Learning:** A type of machine learning where the model is trained on a labeled dataset, meaning each input has a known, correct output. (Used most in real life applications)
2. **Unsupervised Learning:** A type of machine learning where the model is given data without labels, and it tries to find patterns or structure in the data on its own.

## Supervised Learning

Algorithms that learn *input* to *output* mappings, the key is that you give your algorithm the correct answer or *labels* and they learn the input to output mappings.

1. **Regression:** Learns to predict numbers out of infinitely possible numbers. For example predicting house price, height, temperature or scores.
2. **Classification:** Learns to predict categories or label classes for the input data. The difference between **regression** and **classification** is that, classification predicts categories from a small number of possible outputs. For example, detecting breast cancer, labeling news articles or classifying pictures of dogs and cats.

## Unsupervised Learning

We feed *unlabeled* data to our model and the model's job is to find a structure or a pattern that emerges from the dataset.

1. **Clustering:** Unsupervised learning technique where the goal is to group similar data points together given no labels.
2. **Anomaly Detection:** Finding unusual data points in a dataset. Like credit card fraud and manufacturing defects.
3. **Dimensionality Reduction:** Compressing data using fewer numbers. For example image compression and data visualization.

## Recap

**Supervised Learning:**
* Regression: Predicting output from infinite possible outputs.
* Classification: Predicting output from limited possible classes.

**Unsupervised Learning:**
* Clustering: Grouping related data.
* Anomaly Detection: Finding unusual data points.
* Dimensionality Reduction: Compress Data using fewer numbers.

---

## Deep Dive

### Linear Regression

Fitting a *straight* line through the data points, and use the function you get to predict outputs for unseen data. Linear regression can be visualized in two ways, graph or data table.

#### Terminology

* **Training Set:** Data used to train the model, including both the feature and the target.
* **Input Variable, Feature:** *x* (eg. size of house in training data)
* **Output variable, Target:** *y* (eg. price of house in training data)
* **Total number of training example:** *m*
* **Single Training Example:** (*x, y*)
* **$i\ th$ Training Example:** referring to a specific example: $(x^i, y^i)$

**Flow Of Regression:**  
Training Set -> Learning Algorithm -> *f*

The role of *f* is to take input *x* and output a prediction called $\hat{y}$, note that this is the prediction. The real value is denoted by just $y$.

**Representation:**  
We represent $f$ as $f_{w,b}(x) = wx+b$ which means that this is a *Linear Regression* model with one variable ($w$) or commonly referred to as a *univariate Linear Regression model.*

**Cost Function:**  
A function that measures how good our model is doing at predicting the output, we can use the cost function to improve the accuracy of our model.

$$f_{w,b}(x) = wx+b$$

The function above is our model, the $w$ and $b$ are called the parameters of the model. We adjust these variables to improve the model. These variables are also referred to as coefficients and weights sometimes. In linear regression, we have to pick values of $w$ and $b$ so that the line fits our data well. Or in other words our line passes through or is close to our training examples. But how do we pick these parameters so that the value of $\hat{y}$ is true for many or all of our training examples in $(x^i,y^i)$? This is where the cost function comes in handy.

For many linear regression models, we use the **Squared error cost function:**

$$J_{(w,b)} = \frac{1}{2m} \sum_{i=1}^m (\hat y^{(i)} - y^{(i)})^2$$

The difference of $\hat{y}$ and $y$ is called the error. $m$ is the number of training examples. We take the average of the cost and that's why we divide by $m$ and we add the 2 in to make our calculations neater for the future. Since $\hat{y}$ is the result of our model we can re-write the function as:

$$J_{(w,b)} = \frac{1}{2m} \sum_{i=1}^m (f_{w,b}(x^{(i)}) - y^{(i)})^2$$

**The goal:** Find $w$ and $b$ to get $\hat{y}$ as close to $y$ for all $(y^i, x^i)$. The cost function helps because it calculates the difference between the models prediction to the actual true value for the training example. Meaning the goal is:

$$\min_{w,b} \; J_{(w,b)}$$

If we graph the **Squared Error Cost Function** we will get a three dimensional graph that looks like soup bowl.

---

### Gradient Descent

We use gradient descent to minimize the cost function, the gradient descent works for cost functions with any number of parameters. We start with an initial guess for the parameters $w$ and $b$, you update them until they are at the minimum, note that there can be more than one minimum. The way gradient descent works is that you start at a random position in the graph, look around and take a step that gets you closer to the bottom of the bowl, and do this until we get to the lowest point in the bowl.

**Implementation of Gradient Descent:**  
At each iteration in **gradient descent** we update our parameters $w$ and $b$, by calculating the derivative of the cost function, multiplying it by a learning rate denoted by $\alpha$, the learning rate is called alpha and it decides the magnitude of the step we take which is usually a small positive number between 0 and 1. The direction of the step is decided by the derivative of the cost function. We control alpha, but we have to make sure we pick a suitable value (more on this later). The way we update both $w$ and $b$ is:

$$w = w - \alpha \frac{d}{dw} J(w,b)$$
$$b = b - \alpha \frac{d}{db}J(w,b)$$

The "=" sign here denotes assignment like in coding not equals as in both sides are equal in math. We repeat these steps until the algorithm converges meaning we reach a local minima where the parameters $w$ and $b$ no longer change much with each iteration. The important note is that we **must** update these values simultaneously so that we use the fresh set of variable in the next iteration, meaning that in code we do like the following:

$$temp\_w = w - \alpha \frac{d}{dw}J(w,b)$$
$$temp\_b = b - \alpha \frac{d}{db}J(w,b)$$
$$w = temp\_w$$
$$b = temp\_b$$

We will repeat these steps until the values of $w$ and $b$ converge, and by converge we mean that we will reach a **local minimum** where the parameters $w$ and $b$ no longer change much with each additional iteration. It is very important to update the values of these parameters at the same time. It's more natural to update these in code in this way. **Note:** it will work if we don't update them simultaneously but this convention keeps steps clear.

**Intuition of Gradient Descent:** What's going on? and why it makes sense?  
This algorithm updates these parameters using a **Learning rate** ($\alpha$) and the derivative of the cost function is used to simultaneously reach convergence.  
A way to think of the derivative at a point in line is to draw the tangent line, and the slope of the tangent line is the derivative. The important note about the derivative is that if the tangent line faces up it means that it's going towards a higher value in the y axis, so the value of the slope will be a positive number. Since we have a minus sign between the old value and the term, we decrease the parameter. On the other hand if the tangent line has a decreasing slope meaning it's a negative number and since there is a minus between the old parameter and our term, we increase the value of the parameter. (minus * minus will be plus).

**How do we choose Alpha?**  
This is a very important choice and has a huge impact on the implementation of **gradient descent**. If the alpha is too small, the algorithm will not be very efficient since the steps we take will be too small and the values of $w$ and $b$ will not change much after each iteration. Causing our algorithm to take more steps than needed for convergence. On the other hand if the value of the learning rate is too large, we over shoot the value of $w$ and $b$, thus ending up in a worse position that we started in and get further from the minimum.

**Recap for Choosing Learning Rate:**
* If $\alpha$ is too small: Gradient Descent will be too slow to reach convergence.
* If $\alpha$ is too large: Gradient Descent can over shoot and never actually reach the minimum and converge.

**What Happens When we Reach a Local Minimum?**  
When we reach a local minimum the slope of the tangent line will be zero or close to zero, so the values of $w$ and $b$ will not change.  
**Note:** As we get closer to the local minimum, gradient descent will automatically take smaller steps because the derivative of the loss function will gradually get smaller. This means that gradient descent will reach a local minimum with a fixed learning rate. This is what makes Gradient Descent so powerful and can be used for any cost function.

## Recap

**Linear regression model:**
$$ f_{w,b}(x) = wx+b$$

**Cost function:**
$$J_{(w,b)} = \frac{1}{2m} \sum_{i=1}^m (f_{w,b}(x^{(i)}) - y^{(i)})^2$$

**Gradient Descent:**  
Repeat until convergence {

$temp\_w = w - \alpha \frac{d}{dw}J(w,b)$

$temp\_b = b - \alpha \frac{d}{db}J(w,b)$

$w = temp\_w$

$b = temp\_b$
}

Now If We Take The Derivatives:

$w = w - \alpha \frac{1}{m} \sum_{i=1}^m (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)}$

$b = b - \alpha \frac{1}{m} \sum_{i=1}^m (f_{w,b}(x^{(i)}) - y^{(i)})$

We carry out these functions until convergence (simultaneously)

**Issue:**  
The issue is that Gradient Descent can reach a local minimum instead of a global minimum. Except that in this case since our **Square Cost Function Difference** creates a bowl shape, it only has one minimum. The technical term for this kind of bowl shape is a *convex* function.

Convex function -> Bowl shape -> Only one local minimum

**Batch Gradient Descent:**  
This kind of gradient descent means that at every step of the gradient descent, the algorithm uses all the training examples instead of a small example of the data. The term "batch" is somewhat off-putting, but it's important that it means all of the data.
