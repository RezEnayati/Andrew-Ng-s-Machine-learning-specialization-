# Classification

In classification our output variable y can only take one of a small handful of possible outputs instead of any number in an infinite range like Linear Regression. Since Linear Regression is not a powerful algorithm for classification we will use **Logistic Regression**, which is one of the most popular and widely used algorithms today.

Examples:

| Question                   | Answer "y" |
| -------------------------- | ---------- |
| Is this email spam?        | no / yes   |
| Is this transaction fraud? | no / yes   |
| Is the tumor malignant?    | no / yes   |

**y can only have two values.**

This is a type of **binary classification**, where binary refers to there being only two possible categories. In this case we are going to use *class* and *category* interchangeably. By common convention we can refer to these two classes of "yes" and "no" in a few ways, "true" and "false" or most commonly 0 and 1. A common terminology is to refer to the "yes" class as the "positive class" and the "no" class as the "negative class".

Example: An email that is **not** spam will be referred to as the "negative class" because the output to the question is if the email **is** spam. So a spam email would be referred to as a positive training example.

## Logistic Regression: **Most used classification algorithm**

Used to solve binary classification problems where output label y is either 0 or 1.

### Sigmoid Function (logistic function)

Outputs values between zero and one, if we use $g$ to denote this function then the formula of $g$ is:

$$g(z) = \frac{1}{1 + e^{-z}} \quad 0 < g(z) < 1$$

When $z$ is large, $g(z)$ will be very close to 1 because the denominator will be close to one. And conversely if $z$ is a very large negative number, $g(z)$ will be close to 0 since the denominator will be large. So the sigmoid function will have an s shape where it will start at zero and move towards one. Also when the value of $z$ is zero, the denominator will be 1 + 1 so $g(z)$ will be equal to 0.5 and it will pass the vertical axis at 0.5.

### Logistic Regression Algorithm: Two Steps

**First Step:** Take the linear Regression Model:

$$f_{\vec{w},b} = \vec{w} \cdot \vec{x} + b$$

Set it equal to $z$:

$$z = \vec{w} \cdot \vec{x} + b$$

**Second Step:** Pass it to the Sigmoid Function:

$$g(z) = \frac{1}{1 + e^{-z}}$$

This will give us the **logistic regression** model:

$$f_{w,b}(\vec{x}) = g({\vec{w} \cdot \vec{x} + b}) = \frac{1}{1+ e^{-({\vec{w} \cdot \vec{x} + b})}}$$

### How to Interpret The Sigmoid Function

Think about it like the "probability" that class is 1 given a certain input $x$.

$x$: tumor size
$y$: 0 (not malignant) 1 (malignant)

$f_{\vec{w},b} = 0.7$

This means that the model is giving the tumor a 70% chance that $y$ is 1 or 70% chance that the tumor is malignant. This is denoted by:

$$P(y = 1 | x;\vec{w},b)$$

## Decision Boundary

A decision boundary is the surface or line that separates different classes based on the model's predictions. It's the point or the region where the model is *equally unsure* between the two classes.

The decision Boundary is where $z = 0$

How Logistic Regression is computing its predictions:

When does the model predict 1?

$\hat{y} = 1$

$f_{\vec{w},b}(\vec{x}) \geq 0.5$

$g(z) \geq 0.5$

$z \geq 0$

$\vec{w} \cdot \vec{x} + b \geq 0$

$\hat{y} = 1$

This conversely applies to $\hat{y} = 0$

## Cost Function for Logistic Regression

The problem with using the **square error cost** function for logistic regression is that when we use the sigmoid function with $J$ we get a non-convex function unlike the convex function resulted for *linear regression*. The problem with a non-convex function is that it's bumpy and has many local minimums making it hard for gradient descent to find the global minimum. Therefore we use a different cost function for *logistic regression* to make the cost function convex.

Squared Error Cost Function:

$$J_{(w,b)} = \frac{1}{2m} \sum_{i=1}^m (\hat y^{(i)} - y^{(i)})^2$$

for linear regression where:

$$f_{\vec{w},b} = \vec{w} \cdot \vec{x} + b$$

If we re-write the cost function as the following (for simplicity of later calculations):

$$J_{(w,b)} = \frac{1}{m} \sum_{i=1}^m \frac{1}{2}(\hat y^{(i)} - y^{(i)})^2$$

The loss for a single training example will be (denoted by L):

$$L(f_{\vec{w,b}}(\vec{x}^{(i)}),y^{(i)})$$

so:

$$L(f_{\vec{w,b}}(\vec{x}^{(i)}),y^{(i)}) = \frac{1}{2}(f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)})^2$$

And L, the Logistic ***loss*** function (not the cost, the cost function measures accuracy for the entire dataset) is:

$$L(f_{\vec{w,b}}(\vec{x}^{(i)}),y^{(i)}) = 
\begin{cases} 
-\log(f_{\vec{w},b}(\vec{x}^{(i)})) & \text{if } y^{(i)} = 1 \\
-\log(1 - f_{\vec{w},b}(\vec{x}^{(i)})) & \text{if } y^{(i)} = 0 
\end{cases}$$

If the true value of y is 1, the closer the algorithm predicts to 1 the loss function will be closer to zero, on the other hand if it predicts closer to zero when the real value is 1 then the loss will be a really big number (approaches infinity).

On the other hand if the true value is 0, the closer the algorithm predicts to zero the loss function is closer to zero, and if it predicts closer to one then the value of the loss function will be a really big number (approaches infinity).

**General Idea:** The further away the prediction is from the actual value, the higher the **loss**. This makes the overall cost function convex and thus we can use gradient descent to compute to take us to the global minimum and minimize the cost function.

### Simplified Cost Function

We can re-write the cost loss function in a simpler way to use later:

$$L(f_{\vec{w,b}}(\vec{x}^{(i)}),y^{(i)}) = -y^{(i)}\log(f_{\vec{w},b}(x^{(i)})) -(1-y^{(i)})\log(1-f_{\vec{w},b}(\vec{x}^{(i)}))$$

This makes sense since $y^{(i)}$ can only take values of 0 and 1. If it's zero the first term cancels out and if it's 1 the second term cancels out, leaving us with the same set of equations in the unsimplified version up top. This cost function comes from the *maximum likelihood* which is an idea from statistics.

If we take this loss function and plug it into the **cost** function we get this, which is pretty much the standard cost function used for most logistic regression today:

$$J(\vec{w},b) = \frac{1}{m} \sum_{i=1}^{m}[L(f_{\vec{w,b}}(\vec{x}^{(i)}),y^{(i)})]$$

If we plug it in and take the minus signs outside:

$$J(\vec{w},b) = -\frac{1}{m} \sum_{i=1}^{m}[y^{(i)}\log(f_{\vec{w},b}(x^{(i)})) +(1-y^{(i)})\log(1-f_{\vec{w},b}(\vec{x}^{(i)}))]$$

Thus, this is our final **cost** function in logistic regression.

## Gradient Descent

Algorithm used to minimize the **cost** function.

Usual gradient descent:

repeat {

$$w_j = w_j - \alpha \frac{d}{dw_j} J(w,b)$$

$$b = b - \alpha \frac{d}{db}J(w,b)$$

}

$j = 1, \dots, n \quad \text{where } j \text{ is the number of features}$

The derivatives:

$$\frac{d}{dw_j} J(w,b) = \frac{1}{m} \sum_{i=1}^{m} (f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})x_j^{(i)}$$

$$\frac{d}{db}J(w,b) = \frac{1}{m} \sum_{i=1}^{m} (f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})$$

where the values of $w$ and $b$ have to be updated simultaneously for the function to be correctly implemented.

If we plug in these values:

$$w_j = w_j - \alpha\left[\frac{1}{m} \sum_{i=1}^{m} (f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})x_j^{(i)}\right]$$

$$b = b - \alpha \left[\frac{1}{m} \sum_{i=1}^{m} (f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})\right]$$

Which looks like the gradient descent algorithm of the **linear regression** model but the difference is that the equation of the function $f$ has changed.

For **linear regression** $f$ is:

$$f_{\vec{w},b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$$

but for **logistic regression** $f$ is the sigmoid function:

$$f_{\vec{w},b}(\vec{x}) = \frac{1}{1 + e^{-(\vec{w} \cdot \vec{x} + b)}}$$

therefore making the algorithms different since the definition of $f$ of $x$ is different.

**Where they are the same?**
* Monitor gradient descent (learning curve)
* Vectorized implementation
* Feature Scaling

---

## Overfitting and Underfitting

### For Linear Regression:

**Under fit (bad):** Does not fit the training set well, also referred to as **high bias**

**Generalization (good):** Fits the training set pretty well.

**Over fit (bad):** Fits the training example extremely well. Also referred to as **high variance**

### For Classification:

**Under fit:** High bias

**Generalized:** good

**Over fit:** High variance

## How To Address Overfitting:

1. Collect more training data (sometimes isn't an option)
2. Select features to include/exclude
3. **Regularization:** Reduce the size of parameters $w_j$. Regularization lets us keep all the features but prevents the features from having an overly large effect. It's common to regularize parameters $w_1$ to $w_n$ but not $b$.

### Cost Function With Regularization:

The goal of the cost function is to minimize the values of $w$ and $b$:

$$\underset{\vec{w},b}{\min} \frac{1}{2m} \sum_{i=1}^{m}(f_{\vec{w,b}}(\vec{x}^{(i)})-y^{(i)})^{2} + 100w_{j}^{2} + 1000w_{k}^{2}$$

This way we are penalizing the parameters $w_{j}$ and $w_{k}$ so that when we run gradient descent we try to minimize those two features as well as minimizing the value of $\vec{w}$ and $b$. But most of the time we don't know which features will have a high impact on the model, therefore we try to minimize the effects of all of the features on the model. By convention we usually don't regularize $b$ because it does not have an effect on the model. So we have to find a way to regularize all the features, we do that by adding a *regularization term* to the cost function.

The idea is that if we have smaller values for $\vec{w}$ we will have a simpler model, with fewer features which is therefore less prone to overfitting that is smoother and less wiggly.

$$J(\vec{w},b) = \frac{1}{2m} \sum_{i=1}^{m}(f_{\vec{w,b}}(\vec{x}^{(i)})-y^{(i)})^{2} + \frac{\lambda}{2m}\sum_{j=1}^{n}w_{j}^{2}$$

where $n$ is the number of the features and $\lambda$ is called the lambda parameter or commonly referred to as a regularization parameter. We scale lambda by $\frac{1}{2m}$ so that both terms are scaled the same amount and it becomes easier to pick a good $\lambda$. This will make sure if we add more features we can reuse the same $\lambda$ we picked earlier.

**To Recap:** The MSE will make the model fit the data well by minimizing the squared difference of the predictions and the **L2 Regularization** will try to keep the parameters of $\vec{w}$ as small as possible which will reduce the overfitting.

### Choosing Lambda $\lambda$:
* If lambda is zero, the model will **overfit** the data
* If lambda is enormous like $10^{10}$ the model will **underfit**

### How to get gradient descent to work with regularized linear regression:

In our old Gradient Descent Implementation we repeatedly took our old values of $w_1$ through $w_j$ and the old value of $b$, multiplied the learning rate $\alpha$ by the derivative of the cost function $J$ to update the values of $\vec{w}$ and $b$. This implementation will pretty much stay the same with only one difference that the derivative term of the cost function $J$ will be different since we added the new **L2 regularization** term. So the new derivative terms will look like the following. One important note is that since we don't regularize $b$ the new derivative term will look the same as before.

$$w_j = w_j - \alpha \frac{d}{dw_j}J(\vec{w},b)$$

$$\frac{d}{dw_j}J(\vec{w},b) = \frac{1}{m} \sum_{i=1}^{m}(f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)})x_j^{(i)} + \frac{\lambda}{m}w_j$$

and for $b$:

$$b = b - \alpha \frac{d}{db}J(\vec{w},b)$$

$$\frac{d}{db}J(\vec{w},b) = \frac{1}{m} \sum_{i=1}^{m}(f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)})$$

**Simultaneous updates for these parameters as always.**

The Gradient Descent is the same for both but the definition of $f$ is now different.
