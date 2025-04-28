We will start learning about neural networks also called deep learning algorithms as well as decision trees + advice on how to build machine learning systems. 
This weeks focus is on Neural networks and how to carry out inference or prediction. 

# Neural Networks
### How The Human Brain Works:
The original motivation for building neural networks was to write software that could mimic how the human brain thinks and **learns.**  
* Origins: Algorithms that try to mimic the brain. 
A neuron in the human brain has many inputs where it receives many electric impulses from other neurons. The neuron then carries out some computations and will then send it's outputs to other neurons by electrical impulses. 
A biological neuron has a cell body or formally a nucleus along with input wires to the nucleus called dendrites. After computation it sends out an electrical input to other neurons via an axon. An artificial neural network uses a very simplified mathematical model of what a biological neuron does. What it does is that it takes one or more inputs which are just numbers and does some computation, finally outputs another number which can be used as an input to another artificial neuron. 

### Why is Deep Learning so Important all of a Sudden? 
In the past decade specially after the rise of the internet, the amount of data humans have  been able to obtain has increased drastically. The traditional AI and machine learning algorithms like **logistic regression** and **linear regression**, were not able to scale well with the rise of data at hand meaning their performance did not increase at the amount of data increased. But, with the rise of data we are able to scale deep learning algorithms to perform much better that traditional AI, with a small neural network we are able to produce relatively well performed algorithms and we are able to scale to much better performance if we increase the size of the neural network. One thing that helps us reach this level is having much faster computer processors and better GPUs that allow us to scale neural networks and utilize the amount of data we have nowadays. 

### Details of How Neural Networks work. 
**Demand Prediction:** When we look at a product and try to predict of the product will be a top seller or not. 
In this example we would like to know if a T-shirt we be predicted as a top seller or not. 
If we take a look at it through a **logistic regression** lens with one feature $x$ which is the price we can use the sigmoid function to create a logistic regression model the output will look like the following: 
$$ f(x) = \frac{1}{1 + e^{-(wx+b)}}$$
In logistic regression we referred to this as the $f$ of $x$. For neural networks we will denote this output by the letter $a$. The term $a$ stands for activation, which originates to neuroscience and refers to how much a neuron is sending a high output to other neurons that downstream form it. 
It turns out that this logistic regression algorithm can be though of as a very simplified model of a single neuron in the brain. This neuron takes the input, "price" $x$, computes the sigmoid function and outputs the number $a$, which is essentially the probability of the T-shirt being a top seller. From this idea we can start building neural networks which is taking a number of these neurons and wiring them together. 
Let's look at a more complex example now, the new features we're going to look at is:

* price
* shipping cost 
* marketing 
* material 

1. We can create a one neuron to predict if the t shirt is perceived as an affordable shirt, and this neuron is mainly a function of the price and the shipping cost. 
2. Second we can create another artificial neuron to estimate high awareness of this shirt which is a function of the marketing done for the shirt. 
3. Finally we can create another neuron to predict if the shirt is perceived to be high quality which is mainly a function of the material and the price of shirt. Price is a factor here because usually people correlate high price with a high quality. 

### Example: T shirt sales prediction 
So here we will have three artificial neurons, affordability, awareness and quality. We take these three neurons and feed the output of these neurons to another logistic regression unit which will then output the probability of the shirt being a top seller.
![[Screenshot 2025-04-23 at 10.16.25 AM.png]]

In the terminology of neural networks, we will group the three blue middle neurons together into what's called a layer. And a layer is a grouping of neurons. A layer can have multiple neurons or it can also have a single neuron like the output layer to the right. The last layer on the right is also called an **output** layer and the layer in the middle is called a **hidden layer**. In neural network terminology, we will call affordability, awareness and perceived quality as **activations**. 
So in the neural network above we have 4 inputs that output 3 outputs and these numbers go as inputs to the output layer to computer 1 number. The list of 4 numbers is called the input layer and the 3 numbers are called activation values. 

One simplification we can do for this neural network is that in the above example we went through the input layer and chose what inputs it would take for the next layer but in reality we can't go through the neurons and pick what features we're going to select for the neurons so we input all values into the neurons. And every layer will have access to every feature from the layer before. To further simplify the neural network we will take the input features and call them vector x denoted as $\vec{x}$.
So the values in the vector $x$ are fed into the hidden layer which then they compute 3 activation values and these 3 values become another vector $\vec{a}$ and are fed into the output layer that finally outputs a number predicting the probability the shirt being a top layer. 

One way of thinking about neural network is that they are just **logistic regression**, but a version of logistic regression where it can learn it's own features that makes it easier to make more accurate prediction. So instead of manually feature engineering the neural networks do it for us and picks the values of the hidden layer by itself. 
The choice of the number of hidden layers and how many neurons we want each hidden layer to have. This question is a question of the architecture of the neural network, which effect the performance of the algorithm. 

### Neural Network Model
How to build a layer and a layer works. High level view of a neural network. For the hidden layer:
![[Screenshot 2025-04-23 at 11.19.29 AM.png]]

And for the output layer: 
![[Screenshot 2025-04-23 at 11.22.32 AM.png]]

If we want a binary prediction, we can use thresholding just like we did with logistic regression: 
![[Screenshot 2025-04-23 at 11.24.07 AM.png]]



### Inference: making predictions (forward propagation)
**Handwritten digit recognition** a binary classification problem, where we predict if the handwritten digit is 0 or 1. We can use and 8px by 8px image where, in the matrix 255 denotes a bright white pixel and a 0 denotes a black pixel. 
We will use a three layered network where:
* layer 1: 25 units
* layer 2: 15 units 
* layer 3: 1 unit 
So we start with a vector of out input features $\vec{x}$ or more commonly $\vec{a}^{[0]}$, we use this to input into the first hidden layer and so after computing the values through the sigmoid function we get a vector with 25 values which is our second activation vector called $\vec{a}^{[1]}$. We feed this vector into the second layer with 15 units and it will compute a vector with 15 values called $\vec{a}^{[2]}$.  These values will then be fed to layer 3 which is our output layer $\vec{a}^{[3]}$. Since this layer is the output layer and only has one neuron or one unit this will be scalar, giving us the probability of the digit being a 0 or 1. We then can set a threshold to see get a binary classification. Because this computation goes from left to right, we go from $\vec{a}^{[0]}$ to $\vec{a}^{[3]}$ it's called a **forward propagation**. Meaning we are propagating these values from left to right which is in contrast to another algorithm called back propagating. One note on this neural network is that the number of the units in the layers decrease as we go forward which is a typical architecture choice for this type of neural networks. 

### TensorFlow Implementation: (Inference)
**TensorFlow**: Deep learning framework that let us build and train neural networks. 
The wonderful thing about neural networks is that the same algorithm can be applied to different applications. 
Let's take an example of roasting coffee beans. Let's say our features are the temperature and the minutes we roast the coffee for. By inference we mean that come to a prediction to see if the coffee will taste good or not. This is done after we find our parameters. In TensorFlow the code will go as follows. Note that the term `Dense`just means a layer of neurons:

```
x = np.array([[200.0, 17.0]])
layer_1 = Dense(units = '3', activation = 'sigmoid')
a1 = layer_1(x)

layer_2 = Dense(units = '1', activation = 'sigmoid')
at = layer_2(a1)
```

These are the key steps for forward propagation. Note that  `layer_2` here represents the output layer of the network. 

One thing to note is that for **linear regression** and **logistic regression** we used regular 1-D arrays. But for TensorFlow we have to use 2-D arrays or matrices because that's what TensorFlow uses under the hood. 

### Forward Prop by hand: Non-Vectorized and Vectorized
It's important to understand that all a layer is doing is multiplying the input values by the parameters and adding b to them to create a new vector to pass into the next layer. For example if we have a vector of 400 features coming into  a layer with 25 units, the first layer will have 401 parameters for the first unit, that's 400 $w$ and 1 $b$,  then it will multiply these values and add $b$ to them to create a new vector of size 25, then let's say if the second layer has 15 activation units. The second layer will have 390 total parameters. 375 for the $w$ and 15 $b$. So it will do the same thing as the first layer, multiply these parameters by the parameters $w$ and add $b$ to them. This will output a vector of size 15, If the final layers or layer 3 has 1 unit it will have 16 parameters 15 for $w$ and 1 for $b$. 
We can implement these layers in two ways:
* **Non Vectorized:** Meaning we use a for loop to go through the values, multiply and add (Inefficient)
* **Vectorized:** Meaning we use matrix multiplication to multiply and add, and apply the sigmoid function element wise and create a new vector for output (This is the preferred method of doing it)

**Non Vectorized:**
```
def dense(a_in, W, b_in, g)
	"""
	Arguments:
	a_in: (n,) narray: The input values for the forward prop 
	W: (n,m) Matrix: The input matrix of n features m units
	b_in: The biasas for the first layer
	g: sigmoid function as our activation function 
	Return Value:
	a_out: (m,): the output vector of the activation values
	"""
	units = W.shape[1]
	a_out = np.zeros(units)
	for j in range(units):
		w = W[:,j] #Get each unit parameters 
		z = np.dot(w, a_in) + b[j]
		a_out[j] = g(z)
		
	return a_out 
```
**Vectorized:**
```
#Note that here the input featues have to be a Matrix 
def dense_v(A_in, W, b_in, g):
	"""
	Arguments:
	a_in: (n,1) Matrix: The input values for the forward prop 
	W: (n,m) Matrix: The input matrix of n features m units
	b_in: The biasas for the first layer
	g: sigmoid function as our activation function 
	Return Value:
	a_out: (m,): the output vector of the activation values
	"""
	z = np.matmul(A_in, W) + b
	A_out = g(z)
```

Both of these are considered as subroutines for the sequential implementation of the dense layers, we can implement as follows:
**Non Vectorized:**
```
def sequential(x, W1, b1, W2, b2, W3, b3):
	a1 = dense(x, W1, b1)
	a2 = dense(a1, W2, b2)
	a3 = dense(a2, W3, b3)
	return a3
```
**Vectorized:**
```
def sequential_v(X, W1, b1, W2, b2, W3, b3):
	A1 = dense_v(X, W1, b1)
	A2 = dense_v(A2, W2, B2)
	A3 = dense_v(A2, W3, B3)
	return A3
```

**Note:** The vectorized implementation runs much more faster than loop because of NumPy's optimization under the hood, so if implemented by hand the vectorized implementation is preferred. 
