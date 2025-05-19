# ML Diagnostics 
Let's say you have implemented regularized linear regression for the housing prices problem your cost function would look something like this:

$$
J({\vec{w},b}) = \frac{1}{2m} \sum_{i=1}^m (f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})^2 + \frac{\lambda}{2m}\sum_{j=1}^n w_j^2
$$

But it makes unacceptably large errors in predictions. What would we do next? In this chapter we go over techniques to debug machine learning algorithms. Below is a list of steps that you can maybe try to use to improve the model: 
* Get more training examples 
* Try smaller sets of features 
* Try getting additional features 
* Try adding polynomial features 
* Try decreasing $\lambda$ 
* Try increasing $\lambda$ 

With any algorithm these choices can tend to be fruitful, but it's important to understand where to invest your time to get optimal results. So this week we will learn machine learning diagnostics and by **diagnostics** we mean:
A test that you run to gain insight into what is/isn't working with a learning algorithm, to gain guidance into improving its performance. The note is that diagnostics can take time to implement but it's important to invest time to get optimal results. 

---
**How to Evaluate a Model's Performance?**
Let's take the example of predicting house prices again, we have a model that fits the training data really well but fails to generalize to new examples not in the training set. 
Our model is a fourth order polynomial so it in fact does fit our training data very well. Below is a what out model and it's features look like:
* $x_1 = \ size \ in \ feet^2$ 
* $x_2 = \ number \ of \ bedrooms$
*  $x_3 = \ number \ of \ floors$
* $x_4 = \ age \ of \ homes \ in \ years$
So therefore the model looks like this: 

$f_{\vec{w},b}(\vec{x}) = w_1 x + w_2x^2+ w_3x^3 + w_4x^4 +b$  

Note that since the model is a fourth order polynomial, it's difficult to plot what the model looks like because how do we even plot a model in four dimensions? So we need to find a more systematic way of evaluating the model's performance. One technique we can use is splitting our training set into two subsets, one subset for example 70% can go into the training set and 30% can go into the tests set. So we train our model parameters on the 70% of the data and test it on the other 30%. 
We start of by fitting the parameters by minimizing the cost function $J(\vec{w},b)$ with the regularization term over the training data, and to compute how well the model is doing we would $J_{test}(\vec{w},b)$ which is the regular mean squared error cost function but in the test cost function we don't include the regularization term. This will give us a test error, one thing that is also worth doing is computing the training cost error, note that this also does not include the regularization term. 
This procedure would be the same for logistic regression problems, but in this case our cost functions would be different so instead of mean squared error we would be using the **log loss** or **Binary cross-entropy**. But one thing that is more common for logistic regression is finding out the fraction of the test set and the training set that the algorithm has misclassified,  so we would get the label of the prediction for example if $f_{\vec{w},b}$ is greater that 0.5 we would label it 1 and if it's lower we would label it 0. Then we would count the number of mistakes in both the training set and the test set. So this would make our cost function the fraction of the test or training set that has been misclassified. 
#### Model selection using cross validation
Let's say we want to find the best model in a linear regression problem, one thing we might do is have models with different order polynomials, for example a linear model up to a 10th order polynomial, one thing we could try which turns out not to be the best action, is train the different models on the training set, find the cost function for the test set and pick the one with the lowest cost function on the test example. Let's say from these models, you pick the one with a fifth order polynomial and report the cost function of the test set. This procedure is flawed because the fifth order polynomial is likely to be an optimistic estimate of the generalization error because an extra parameter d (degree of polynomial) was chosen during the test set. Instead we can modify the model selection procedure by splitting our data into three different subsets:
1. Training Set 
2. Cross validation Set (validation set), (dev set)
3. Test Set 

The way we utilize, the cross validation is by computing the training error, cross validation error and the test error and as usual none of these include the regularization term. This procedure is carried out by, first training the different models, that we pick on the training set, then calculating the model's cross validation cost, for each model and picking the model with the lowest cross validation cost. After picking the appropriate model, we can then find the cost function on the test data set and this will give us an un biased generalization error. This let's us automatically pick the best model, and is applicable to different ML models, for example the best architecture for a neural network architecture. 

---
#### Bias And Variance
Most of machine learning applications fails to work well the first time we create a new model so looking at bias and variance gives you a way to see what to try next. In this section let's take a look on what this means. In machine learning algorithms, the bias and variance are a fancy way to say that the model is under-fitting or over-fitting.  High Bias happens is when the model has under-fit the data, meaning that our J-train will be high also our J-cv will be high too, for linear regression this can happen when we for example try to fit a linear function to a noisy data, the model will fail to fit the parameters accordingly. On the other hand when we say High variance, we mean that the model had over-fit the data, this will result in the J train to be low and the J cv to be much higher than the J train. For linear regression this can happen when we are trying to fit a high order polynomial to our data. In some cases, specially in neural networks we can have instances where the Bias and the variance are both very high, this is usually not the case for linear regression.  In summary: 
* **High Bias: Underfit**
	* $J_{train} \ will \ be \ high$ 
	* $J_{train} \approx J_{cv}$
* **High Variance: (overfit)**
	* $J_{train} \ will \ be \ low$ 
	* $J_{train} \gg J_{cv}$
* **High variance & High Bias**
	* $J_{train} \ will \ be \ high$ 
	* $J_{train} \gg J_{cv}$

**How Does Regularization Effects Bias and Variance:**
The choice of the regularization parameter $\lambda$ plays an important role on the accuracy and the final generalization error of the ML algorithm. The choice of the $\lambda$  can't be too high because that will cause the model to have a high bias meaning it will underfit, it also cannot be too low because that makes the model have a High variance meaning it will overfit, or in simple terms applying regularization will not have an effect. In particular, a low lambda will cause our training set cost function to be low and the cross validation cost function to be high, on the other hand if the lambda is too high it will cause us to have both a high cost function for cross validation and a high cost function for the training set meaning that the model is not learning. We have to pick $\lambda$ by testing on different amount of lambda and seeing which model gives us the lowest cost. We have to test with a range of values which are not too high and not too low for our specific algorithm. 

**How do we actually know a cost function is low or high?** 
We have to establish a baseline level of performance and judge if the cost functions are high or low. To better understand the problem we will take a look at the task of speech recognition with ML. If we were to train a speech recognition and test figure out the training error, and it's 10.8%, also the cross validation error is 14.8%. But let's add another performance metric to this problem, and say that the **human** level performance for speech recognition is 10.6%. This happens because the audio clips that we can have often have a lot of noise so it's even hard for humans to transcribe some of the audio clips. So in this case the number that results from our training cost function is not that different from the one gotten from humans, so in fact the model is performing great on the training data but the number we got from the cross validation set is much higher than both so at the end model does have some variance issues. 
So if the training error is high, it's sometimes best to establish a base line for performance by seeing how well a human can do the task that we want our models to do for us. This is because humans are generally great at things like audio, images or text. 

#### Learning Curve
Learning curve is a way of finding out how our model is doing as a function of how much experience it has, by experience we mean how much training examples we have. If we plot the cost function of the training set and the cost function of the cross validation set over the number of training examples, we see that as the number of the training data increases, we see some interesting behavior for both the cv cost and the training cost. One thing that is important is that it is a lot easier to fit a parameters to say 1 or 3 training examples so our cost for the training data will start at zero or close to zero and as the number of the training examples increase the cost for the training examples will also increase until it stabilizes. On the other hand as the training data increases our cv cost decreases until it stabilizes, this is how the plot looks like: 
![[Screenshot 2025-05-18 at 2.53.40 PM.png]]

**For High Bias (Underfit)**, if we were to plot the training cost and the cv cost we get something like this. This plot will look more or less the same as our model but with a difference the level where the cost of the training flatten out is gonna be much higher than the level of our baseline performance, the plot will look like this. 
![[Screenshot 2025-05-18 at 4.53.22 PM.png]]
One interesting thing about this plot is that even if we increase the amount of training data, we  still wont be able to improve the cost of the training data and the cost of the cross validation data, they will just continue to stay flat as we go and will not dip down. So with an algorithm with high bias more data is not effective. 

**High Variance (Overfit)**, in this case the plot will look a little different, the distance between the cost of the training and the distance between the cost of the cv will be high. If we establish a baseline performance, it's probable that the training cost will be lower since it has over fit to the data in the training examples. So this is how the plot will look like: 
![[Screenshot 2025-05-18 at 5.01.25 PM.png]]In this case of high variance, if we increase the training size it could be fruitful. Increasing the training examples will do two things, one it will increase the cost function of the training set and it will also decrease the cost function of the cross validation set. This will cause the two plots to get close to each other so we therefore have a better understanding of the model's performance. 

**Summary:**
* In cases of High Bias: Increasing training size **does not help** 
* In case of High Variance: Increasing training size **does help**

**Deciding What to try next:**
Going back to the initial example of predicting housing prices, by a linear regression model with regularization. These were the six ideas we had:
* Get more training examples 
* Try smaller sets of features 
* Try getting additional features 
* Try adding polynomial features 
* Try decreasing $\lambda$ 
* Try increasing $\lambda$ 
But how do we choose what to do next? 
Each of these ideas can help with either high bias or high variance issues, specifically If the learning algorithm has high bias, 3 of these ideas are helpful and if the algorithm has high variance the other 3 is helpful. 

**For High Bias: Underfit**
* Getting additional features 
* Adding polynomial features
* Try decreasing $\lambda$ 

**For High Variance: Overfit** 
* Trying smaller set of features 
* Get more training examples
* Try increasing $\lambda$ 

---
### How Bias and Variance Apply to Neural Networks
Large neural networks are low bias machines. It turns out when you have a large enough neural network with small to moderate data by following the recipe bellow you can get very accurate results. 

1. Start with your training data and see if the model does well on the training set meaning look at the $J_{train}(\vec{w},b)$. 
2. If the model is not performing well on the training set, then increase the size of the network and repeat step 1. 
3. If the model does perform well on the training set then check to see if it does well on the cross validation set $J_{cv}(\vec{w},b)$ 
4. If the model is not doing well on the training set, increase the data and start from step 1.
5. If the model is performing well on the cross validation set then you are done. 

This algorithm can be helpful for creating good models but it can also get very computationally expensive, that's why hardware has been getting so much better with GPUs. It can also can be very hard to gather data. 

An important note is that a large neural network will usually do as well or better than a smaller one as long as regularization is chosen appropriately. Another way to say this, is that it is always a better idea to go for a bigger network, but one downside is that it does get more computationally expensive to have a bigger network. 

**To summarize:** 
1. It is always a better idea to have a large neural network. 
2. A large neural network is often a low bias machine meaning that it fits very complicated functions very well 

---
**Iterative Loop of Machine Learning Model:**

Chose Architecture -> Train Model -> Diagnostics

Think of this a cycle where diagnostics points back to choosing the architecture.   

**Error Analysis:**
 Let's say that we have 500 examples in a cross validation set and our algorithm misclassifies 100 of them. Error analysis means to examine the 100 examples and categorize them based on common traits to gain insights to where the algorithm is being wrong. If the size of the cross validation set is large and the number of examples where the model made a mistake is a lot, it can be time consuming to examine all the mistakes so a better approach is to randomly sample 100 examples and see where the model had the most error. Error analysis is beneficial to tasks where humans do a good job in, for example detecting phishing emails but not very fruitful on tasks where humans are generally not good at like thinking where to place ads or recommender systems. 

---
### The Problem with Adding Data 
Trying to get more data can be time consuming and expensive instead an alternative to adding data can be to add more data of the types where error analysis has indicated it might help. For example in the problem of recognizing spam emails, if our error analysis indicates that we have issues with emails regarding pharma spams it can be better to find more examples of emails that are pharma spams and feed that into the training set to help the model recognize these types of data better. 

**Data Augmentation**
Beyond getting brand new training examples another technique mostly applicable to image and audio data is **data augmentation** which means taking the old training examples and modifying them to create new training example. For example in the problem of recognizing hand written digits, we can use data augmentation to maybe shrink, rotate, mirror or zoom in the data to create new examples from the data. In the case of speech recognition we can add noisy background, in a crowded back ground or car noise to make the model see newer, richer data. 
But data augmentation has to be done carefully, meaning that it has to be representative in the test set, think about it like modifying the data to get them more similar to the data in the training sample. 

**Data Synthesis**
Using artificial data input to create a new training example. An example of synthetic data can be done in the problem of OCR, in images. Let's say we want a model to recognize letters in images, these letters can have multiple different fonts, and the model will probably fail to generalize on only the training set. One thing we can do is, open a text editor and capture screen shots of letter using the numerous different fonts and different colors, we have access to and feed them into the training data so we have a richer data set. Synthetic data has been mostly applicable to image recognition. 

Conventional ML has been mostly focused on models and this has caused us to come up with really good algorithms such as **Linear Regression**, **Logistic Regression** and **Neural Networks**. This is called a model-centric approach to ML. It turns out that using these algorithms, when we focus on data we get much better results that trying to come up with a new model, hence this is called a data-centric approach to ML and AI 
AI = Code + Data 

---
### Transfer Learning
This technique can help us create models where data is scares or expensive to access. Here is how it works. 
Let's say we want to classify handwritten digits from 0-9 but we don't have a lot of training examples, on the other hand we have a million images of 1000 different objects like cats, dogs, cars and humans. This model would for example be 5 layers with the output layer having 1,000 outputs for each class. What we can do is take this model's layers not including the output layer and add an output layer for the 10 classes of data that we are trying to recognize, meaning that we "freeze" the first four layers' parameters and add a new output layer with fresh parameters. This gives us two options:
1. Only train output layers parameters 
2. Train all parameters
We can choose from these two options by looking at how much data we have at hand, If we don't have access to a lot of data the first option is a better choice, if do have a reasonable amount of data we can go for option 2 and train all the parameters but start with the parameters we got from the first model. This first task of training the model is often called **Supervised Pre-training** and the second task of training the output or all the parameters is called **Fine tuning**. The good thing about fine tuning is that we have accessed to pre-trained models on the internet making it faster and cheaper to do new tasks with models other people have already trained. 

**How Does Transfer Learning Work?**
It turns out for similar tasks, models learn patterns rather than specific features of the data, for example in the task of image recognition a model learns edges, corners curves and basic shapes which is applicable to other recognition tasks. 

**Summary:**
1. Download a neural network parameters pre-trained on a large dataset with same input type (eg, images audio, text) as you application. (Or train your own)
2. Further train (fine tune) the network on you own data. 

---
**Full cycle of Machine Learning Project**

1. Scope Project: Define Project 
2. Collect Data: Define and collect data 
3. Train Model: Training, error analysis & iterative improvements 
4. Deploy in production: Deploy, monitor and maintain system

**Deployment:**
A common way to deploy a model is to place the model in an inference server, so the application can call an API in the server and get back an inference for the application to use. 
Software engineering maybe needed for:
* Ensuring reliable and efficient predictions 
* Scaling 
* Logging 
* System monitoring
* Model Update 
MLOps: Machine Learning operations, combining machine learning engineering and software engineering. 

---
**Fairness, bias and ethics**
* Deepfakes
* Spreading toxic/incendiary speech through optimizing for engagement.
* Generating fake content for commercial or political purposes.
* Using ML to build harmful products, commit fraud etc.
* Spam vs anti-spam: fraud vs fraud.

**Guidelines:**
* Get a diverse team to brainstorm things that might go wrong, with emphasis on possible harm to vulnerable groups.
* Carry out literature search on standards/guidelines for industry.
* Audit systems against possible harm prior to deployment.
* Check bias against possible hard prior to deployment.
* Develop mitigation plan (if applicable), and after deployment, monitor possible harms. 

---
**Skewed Data Sets**
If the positive to negative ratio is ver skewed, meaning that it is far from 50-50 it turns out that the usual metrics for accuracy do not work that well. Let's say we want to classify a rare disease and do this with a binary classification algorithm. After training we find that we've got 1% error on the test set and 99% correct diagnoses, this is actually not very accurate if only 0.5% of the population have this rare disease. So we usually have a different error metric rather than just classification error. A common pair of error metrics we use in this case is called the precision/recall. In particular it's useful to construct what's called a confusion matrix which is a table that helps us calculate the recall and the precision. 

The rows columns depict the actual class and the rows depict the predicted class

|     | 1                     | 0                    |
| --- | --------------------- | -------------------- |
| 1   | True Positive **15**  | False positive 5     |
| 0   | False negative **10** | True Negative **70** |
**Precision:** Of all patients where we predicted y=1 what fraction actually have the rare disease?

$\frac{True \ Positive}{Predicted \ positive} = \frac{True \ Positive}{True\ Positive\ +\ False \ Positive} = \frac{15}{15+5} = \frac{15}{20} = 0.75$

**Recall:** Of all patients that actually have the rare disease what fraction did we correctly detect has having it?

$\frac{True \ Positive}{Actual \ positive} = \frac{True \ Positive}{True\ Positive\ +\ False \ Negative} = \frac{15}{15+10} = \frac{15}{25} = 0.6$

This learning algorithm would have 0.75 precision and 0.6 recall. 

When we have a rare class, looking at precision and recall can give us a better understanding of the accuracy of the learning algorithm. 

**Trading off Precision and Recall:**
**High Precision:** If a diagnosis of patients have that rare disease, probably the patient does have it and it's and accurate diagnosis.
**High Recall:** If there is a patient with that rare disease, probably the algorithm will correctly identify that they do have that disease. 
But in practice there is often a tradeoff between recall and precision. 
Say we are using logistic regression to predict the rare disease and want to predict the rare disease only if we are very confidant in our prediction, so instead of the threshold we regularly use for logistic regression which is (0.5) we use a higher threshold like 0.7, this means that only predict the disease if we are very confidant in the patient having a disease, this will cause the **Precision** to go up, but **Recall** will go down. 
On the flip side if we want to avoid missing too many cases of rare disease we can lower the threshold, this will cause **Precision** to go down and the **Recall** to go up. 
Mostly in medical cases recall is often prioritized. 

**Combined Metric: F1 Score**

$$F_1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$
Used to balance both, but pays more attention to which is lower. (Also called the Harmonic Mean which gives more value to the smaller number)

|             | Precision (P) | Recall (R) | Average | $F_1$ Score |
| ----------- | ------------- | ---------- | ------- | ----------- |
| Algorithm 1 | 0.5           | 0.4        | 0.45    | 0.444       |
| Algorithm 2 | 0.7           | 0.1        | 0.4     | 0.175       |
| Algorithm 3 | 0.02          | 1.0        | 0.501   | 0.0392      |

Based on this table, we can see that just taking the average is not very beneficial, but taking the $F_1$ score gives us more insight as to how the algorithm is doing. 
