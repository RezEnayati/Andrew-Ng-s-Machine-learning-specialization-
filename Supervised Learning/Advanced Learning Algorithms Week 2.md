### **Training  a Neural Network using TensorFlow:**
Let's go with the classic example of recognizing hand written digits of 1 and 0. The network architecture we are using is a network with 3 layers. First layer has 25 units, second layer 15 and the last layer with 1 unit. Let's say we have $(x,y)$ examples. 
Training with TensorFlow has three steps, and we will get into the details of how the training is done later in this explanation. 

**First:** The first step is to create our layers using the model sequentially using the `Dense`. 

**Second:** The second step is to *compile* the model using a loss function, The loss function here is *Binary Cross Entropy* 

**Third:** The third step is to use gradient descent, we do that by calling `fit` on the model and the number of `epochs` is basically the number of steps in our gradient descent. 

```
import tensorflow as tf
from tensorflow.keras import Sequentail
from tensorflow.keras.layers import Dense 

#First Step 
model = Sequential([
	Dense(25, activation = 'sigmoid),
	Dense(15, activation = 'sigmpid'),
	Dense(1, activation = 'sigmoid')
])

#Second Step 
from tensorflow.keras.losses import BinaryCrossentropy

model.compile(loss = BinaryCrossentropy())

#Third Step 
model.fit(X,Y,epochs = 100)
```

**Training *Logistic Regression***: Recall that training a *Logistic Regression* model had 3 steps which are more or less the same as training a neural network. The three steps were as follows:

**Step 1:**
Specify how to compute output given input x and parameters w and b (define the model):
$f_{\vec{x},b}(\vec{x}) = ?$
```
z = np.dot(x, w) + b 
f_x = 1 / (1 + np.exp(z))
```

**Step 2:** 
Specify the loss and the cost function $J$. Recall that the loss function was the loss of a single training example and the cost was the average of those loss function:
$L(f_{\vec{x},b}(\vec{x}),y)$ : 1 example 
$J(\vec{w},b) = \frac{1}{m} \sum_{i=1}^m L(f_{\vec{x},b}(\vec{x}),y)$ 
and the logistic loss was:
```
loss = -y * np.log(f_x) - (1-y) * np.log(1-f_x)
```

**Step 3:**
Train on the data to minimize the cost function $J(\vec{w},b)$ using gradient descent:
```
w = w - alpha * dj_dw
b = b - alpha * dj_db
```

**These steps map to the neural Network in TensorFlow as follows:**
**Step 1:**
Defining the model. This is the architecture of the whole model and everything TensorFlow needs to know to define the model.  
```
model = Sequential([
	Dense(...)
	Dense(...)
	Dense(...)
])
```

**Step 2:**
Defining the Loss function in this case Binary Cross Entropy and calculating the average.
Here we define the loss function, the most common used is the Binary Cross Entropy which is actually the same loss function as the logistic regression model:
$L(f(\vec{x}),y) = -y \ log(f(\vec{x})) - (1 -y) \ log(1-f({\vec{x}}))$ 
TensorFlow will calculate the cost function and take the average itself. 
```
model.compile (
	loss = BinaryCrossentropy()
)
```

If we wanted to do a *Linear Regression* Model and not do binary classification our loss function would be different for example we would use the Mean Squared Error and optimize that:
```
model.complie(loss = MeanSquaredError())
```

**Step 3:**
Using Gradient Descent to minimize the cost function by using `fit`. Which is calculating the partial derivative of the cost function w.r.t w and b, multiplying by the learning rate $\alpha$ and updating the parameters lie so:
repeat {
	$w_{j}^{[l]} = w_{j}^{[l]} - \alpha \frac{d}{d w_j}J(\vec{w},b)$  
	$b_{j}^{[l]} = b_{j}^{[l]} - \alpha \frac{d}{d w_j}J(\vec{w},b)$
}
TensorFlow computes these derivatives for gradient descent using *back propagation*. And the number of steps is specified by `epochs`. 
```
model.fit(X, y, epochs = 100)
```


### Alternatives to the Sigmoid Function: 
Recall the example of predicting if a shirt will be a top seller item or not from last week. We had three activation values which were, awareness, perceived quality and price. In that example we considered awareness to be binary but that's not always the case. A product can be viral or averagely known or loss known or not known at all. To work with these kind of values we can use a different activation function, and here are two examples which are pretty self explanatory: 
recall: 
$a_{2}^{[1]} = g(\vec{w}_{2}^{[1]} \cdot \vec{x} + b_{2}^{[1]})$ 

**ReLU:** Rectified Linear Unit
$g(z) = max(0,g)$
![[Screenshot 2025-04-26 at 7.10.48 AM.png|350]]

**Sigmoid:**
$g(z) = \frac{1}{1 + e^{-z}}$  
![[Screenshot 2025-04-26 at 7.12.17 AM.png|350]]

**Linear Activation Function:** Sometimes referred to as "no activation function"
$g(z) = z$
![[Screenshot 2025-04-26 at 7.13.32 AM.png|350]]
These three function are the most commonly used function is ML along with the softmax function we will talk about later in the week. 

**How to choose between Activation Functions:**
We can choose different activation functions for different layers in the network and considering the activation function for the output layer there is often one natural choice depending on the label $y$. 
**Output Layer:**
**Binary Classification:** Sigmoid Function will be the natural choice, because then the network 
will learn the probability of the output being equal to one. 

**Regression Problem:** If $y$ can be positive or negative using the linear activation function will be the natural choice. 

**Regression Problem:**  If $y$ can only be a positive number like the house price prediction problem then **ReLU** can be the natural choice for the problem since it only outputs positive and 0 values. 

**Hidden Layers:** 
It turns out that the for the hidden layers the **ReLU** function is by far the most commonly used activation function used by machine learning engineers today. At the start of Neural Networks, sigmoid function was the most widely used for neural nets but the field has evolved to using the **ReLU** activation function for most of the hidden layers, except naturally for binary classification problems. But why? 
1. ReLU is faster. 
2. ReLU function goes flat in only one part which is the negative part whereas sigmoid goes flat both in the positive and negative parts of the graph. And if we are using gradient descent for a function that is flat in a lot of places then **gradient descent** will be slow. 

**Summary:**
**Output Layer**
* Binary Classification: `activation = 'sigmoid'`
* Regression (y can be positive and negative): `acivation = 'linear'`
* Regression (y can only be positive or zero): `activation = 'relu'`
**Hidden Layer**
* Use the ReLU by default. 

**Syntax in TensorFlow:** If we are using **ReLU** for the **hidden layers** and **sigmoid** for the output layer:

```
from tf.keras.layers import Dense

model = Sequential([
	Dense(units = 25, activation = 'relu'),
	Dense(units = 15, activation = 'relu'),
	Dense(units = 1, activation = 'sigmoid'),
]) 
```


**Multi-class classification problems:**
Problems where we can have more than two possible output labels. 

**Softmax:**
A generalization of logistic regression which is a binary classification algorithm to the multi-class classification contexts. 
If we take a look at the logistic regression model, it only had 2 possible output values and gives us the probability of the output being one and being zero for example:
$$a_1 = g(z) = \frac{1}{1+e^{-z}} = P(y =1 \mid\vec{x}) $$
$$a_2 = 1 - a_1 = P(y = 0\mid x)$$
So $a_1$ gives us the probability of the output being 1 and $a_2$ gives us the probability of the output being 0. On the other hand the Softmax Regression can give us probability of a numerous amount of possibilities for example if we have 4 choices:

$z_1 = \vec{w_1} \cdot {\vec{x}} + b_1$  $a_1 = \frac{e^{z_1}}{e^{z_1} + e^{z_2} e^{z_3} e^{z_4}}$
$z_2 = \vec{w_2} \cdot {\vec{x}} + b_2$ $a_2 = \frac{e^{z_2}}{e^{z_1} + e^{z_2} e^{z_3} e^{z_4}}$
$z_3 = \vec{w_1} \cdot {\vec{x}} + b_3$ $a_3 = \frac{e^{z_3}}{e^{z_1} + e^{z_2} e^{z_3} e^{z_4}}$
$z_4 = \vec{w_4} \cdot {\vec{x}} + b_4$ $a_4 = \frac{e^{z_4}}{e^{z_1} + e^{z_2} e^{z_3} e^{z_4}}$

This would give us the probability of each of the options, from $a_1$ to $a_4$. The general format  of Softmax Regression:

$$z_j = \vec{w}_j \cdot \vec{x} + b_j \ \ j = 1,...,N$$
$$a_j = \frac{a_j}{\sum_{k=1}^N e^{z_k}} = P(y=j\mid\vec{x})$$
where $\sum_{k=1}^N e^{z_k} = 1$
**Note:** It turns out that if you use Softmax Regression when $y$ can have only two output values the results you get will be the same as our original logistic regression. Which is why the softmax is a generalization of logistic regression. 

**Cost Function of The Softmax Function:**
Recall that for **Logistic Regression:**
$z = \vec{x} * \vec{x} + b$
$a_1 = g(z) = \frac{1}{1+e^{-z}} = P(y=1\mid\vec{x})$
$a_2 = 1 - a_1$
$loss = -y \ log \ a_1 - (1-y) \ log(1-a_1)$ since $a_2 = 1 - a_1$ we can rewrite the loss as:
$loss = -y \ log \ a_1 - (1-y) \ log(a_2)$
$J(\vec{x},b) = average \ loss$

Therefore we can do the same thing for the softmax regression loss:

$$
\begin{aligned}
a_1 &= \frac{e^{z_1}}{e^{z_1} + e^{z_2} + \cdots + e^{z_N}} = P(y=1\mid \vec{x}) \\
\vdots \\
a_N &= \frac{e^{z_N}}{e^{z_1} + e^{z_2} + \cdots + e^{z_N}} = P(y=N\mid \vec{x})
\end{aligned}
$$ 
**Cross Entropy Loss**
$$
\text{loss}(a_1, \ldots, a_N, y) = 
\begin{cases}
-\log a_1 & \text{if } y = 1 \\
-\log a_2 & \text{if } y = 2 \\
\vdots \\
-\log a_N & \text{if } y = N
\end{cases}
$$
Which means the loss for a single activation value is:
$$loss = -log \ a_j \ \text{if} \ y =j$$
Which means that this incentivized the algorithm to make $a_j$ as close to one as possible and punished the algorithm if $a_j$ is not close 0. In this algorithm if for example $j =2$ we would only calculate: $-log \ a_2$. And we take the average for the Gradient Descent. 

**Neural Network with Softmax output**: 
Before when we were doing handwritten digit recognition for 0 and 1 our output unit only had one unit. If we want to recognize digits from 1 to 9 we would have 10 output units. The way we would do forward prop for the softmax output is that if the architecture of our Network has 3 layers we would  for example have 25 units in layer 1, 15 units in layer 2 and 10 units in our output layer where the output of the output layer is a vector with 10 values each corresponding to the probability of the input to be a digit from 0 to 9. The softmax activation is different from the other activations because in order to calculate $a_1$ to $a_{10}$ in this example we would need the values from $z_1$ to $z_{10}$ because each $a$ depends on all the other values of $z$ meaning we need to first calculate all the values of $z$ to obtain all the values in $\vec{a}$ for the last layer, we can implement Softmax output in TensorFlow like the following: 
**Note:** The loss function we defined before for the Softmax Regression is called `SparseCategoricalCrossentropy`

```
import tensorflow as tf
from tensorflow.keras import Sequential 
from tensorflow.keras import Dense


model = Sequential([
	Dense(25, activation = 'relu')
	Dense(15, activation = 'relu')
	Dense(10, activation = 'softmax')
])

from tensorflow.keras import SparseCategroricalCrossentropy 

model.compile(loss = SparseCategoricalCrossentropy())

model.fit(X, y, epochs = 100) 
```
**Note:** Although this code works for Softmax Regression, TensorFlow offers a better way of implementing this which we will cover next. The code in the next section will make TensorFlow give more accurate results. 

**What's wrong with this implementation**
When calculating with intermediate values computers make numerical roundoff errors, For example in Logistic regression we have the model:
$$a = g(z) = \frac{1}{1+e^{-z}}$$
Original Loss:
$$loss = -y \ log(a) - (1-y) \ log(1-a)$$
If we get rid of the intermediate value we get a more accurate loss:
$$loss = - y \ log(\frac{1}{1+e^{-z}}) - (1-y) \ log(1 - \frac{1}{1+e^{-z}})$$
In TensorFlow:
```
model = Sequential([
	Dense(units = 25, activation = 'relu'),
	Dense(units = 15, activation = 'relu'),
	Dense(units = 1, activation = 'linear'), #instead of the sigmoid
])
#We write this 
model.compile(loss = BinaryCrossentropy(from_logits = True))

#Instead of 
# model.compile(loss = BinaryCrossentropy())
```
This code works okay for logistic regression but in terms of the softmax regression the roundoff gets more inaccurate. 
If we doe the same of getting rid of the intermediate values, we can use the definition of a in the loss function without having to calculate the intermediate values of the activations. This makes TensorFlow rearrange the terms and get a more accurate results. So we calculate the values of z using a linear activation but still use the loss function for softmax regression which is the `SparseCategoricalCrossEntropy`. The code is as follows:
```
model = Sequential([
	Dense(25, activation = 'relu')
	Dense(15, activation = 'relu')
	Dense(10, activation = 'linear') #instead of sofmax to get z 
])

model.compile(loss = SparseCategoricalCrossEntropy(from_logits = True))
```
So the model does not output $\vec{a}$ it outputs $\vec{x}$ so to predict and do forward prop we have to do this:
```
#after model.compile 
logits = model(X) 
f_x = tf.nn.softmax(logits) #this gives the mapping 
```

**Multi-Label Classification:** Classification with multiple outputs. 
Meaning in the example of classifying images there can be multiple labels associated with each image. For example if we are designing a car that detects object on the road like cars, buses and pedestrians one approach would be to design three different machine learning for each classification problem. Another approach would be to train one neural network with three outputs. The architecture of the model would be for example a network that has multiple layers and multiple units in each layer, but the output layer would have three units to predict the existence of bus, car or pedestrian. Therefore since this is 3 binary classification problems at once we can use sigmoid activation for the output layers. And each output would correspond to the existence of the object on the road.  
It's important to note that **multi class** and **multi label** classification are different, multi class will try to predict what something is out of different known classes but multi-label will determine the existence of  objects in a single image for example. 

---
**Advanced Optimization:**
Gradient Descent is an optimization algorithm, that is one of the foundations of ML and was used in **Linear Regression**, **Logistic Regression** and early implementation of neural networks. But over time there are newer and more efficient ways of training neural networks which we will cover. 
Recall that gradient descent will help minimize the cost function $J$ and so if we plot the contour plot of the ellipsis formed by the cost function $J$, we need to start somewhere in the graph and keep taking uniform steps towards the center where the cost is at a minimum. But if we know which direction to take, taking bigger steps will make us reach the center faster. So by taking larger and larger steps with each iteration we will reach the center much faster than having a fixed learning rate. There is an algorithm called the "Adam Algorithm" that if it sees that the learning rate is too small and we are taking small steps in a similar direction over and over we should just make the learning rate bigger, It can also make the learning rate smaller in cases where we are oscillating and are having a hard time finding the minimum.  

**Adam Algorithm**
This algorithm can adjust the learning rate automatically. 
**Adam:** Adaptive Moment Estimation. The interesting thing about the Adam Algorithm is that it does not just use one learning rate $\alpha$ it uses a different learning rate for every single parameter of your model. So for example if we have parameters $w_1$ through $w_{10}$ and $b$ we will have 11 different values of $\alpha$ 
The intuition behind the Adam Algorithm is that if we see a parameter $w_j$ (or $b$) keeps moving in the same direction, increase $a_j$. Conversely if the parameter $w_j$ (or $b$)  keep oscillating we can reduce $a_j$. 
**Adam Algorithm in Code:**
```
#model 
model = Sequential([
	tf.keras.layers.Dense(units = 25, activation = 'sigmoid'),
	tf.keras.layers.Dense(units = 15, activation = 'sigmoid'),
	tf.keras.layers.Dense(units = 10, activation = 'linear')
])

#compile 
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3),
     loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
)

#fit
model.fit(X,Y, epochs = 100)
```
Everything is the same as we have done before except that now we have one extra argument which is the Adam optimizer. It does need an initial learning rate and we've set it equal to $10^{-3}$ 

**Additional Layer Types:**
All the layers that we saw so far have been the `Dense` layer type. In the Dense layer all the neurons in the layer get all the activations from the pervious layer. 
**Convolutional Layer:** Each neuron only looks at part of the previous layer's outputs. One of the benefits is that it speeds up computation. Second a network that can use a convolutional layer need less training data meaning it's less prone to overfitting. If you have multiple layers of convolutional layer the neural network might be called a Convolutional Network or CovNet. 