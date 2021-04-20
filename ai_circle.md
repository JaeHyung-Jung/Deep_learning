---
description: This page is edited as a guide for deep_learning in KINGS' AI_Circle
---

# AI\_Circle

## Installation \(First Week\)

This week\('21.4.12 ~ '21. 4. 16\), we are going to classify some datasets\(MNIST\) with tensorflow.

Let's proceed assuming that you have installed **python**, **Anaconda** and **tensorflow** after looking at the paper handed out in the first OT.

If you didn't install [tensorflow](https://www.tensorflow.org/install), click the link and type command below in python.

\(If you also didn't install python, there is a download link below. You should install python before install tensorflow or anaconda\)

```text
$ pip install --upgrade pip
```

```text
$ pip install tensorflow
```

```text
$ pip install tf-nightly
```

Then, tensorflow install is finished.



The installations of [**python**](https://www.python.org/downloads/) and [Anaconda](https://www.anaconda.com/products/individual#Downloads) are very easy, So i have skipped it.

 But if you didn't install it, click the link.

## Jupyter\_notebook

I recommend you to use Colab or Jupyter\_notebook. It enable python to be run in web.

I prefer Jupyter\_notebook, so I will introduce with using it.

After you install Anaconda, run '**Anaconda Navigator**' 

![](.gitbook/assets/image%20%2815%29.png)

Then, you can see screen like this.

Click 'launch'button below Notebook.

![](.gitbook/assets/image%20%286%29.png)

Then, click the 'New' button and then click 'Python 3'. You can ignore many folders above.

![](.gitbook/assets/image%20%282%29.png)

Then you can see screen like this.

Finally, I will introduce the basic shortcut keys before go into practice.

**'Ctrl + Enter'** : Run cell

**'Alt + Enter'** : Create cell below after run cell

**'Ctrl + Z'** : undo

**'Ctrl + Y'** : redo

Now you're ready to use notebook to classify many MNIST data sets.

## Tensorflow : Quickstart for beginners

Now, let's practice. You can type below codes in Jupyter\_notebook\(recommended\).

I have introduced how to use notebook above, so check it if you don't know how to use it.



We need tensorflow, so import it.

```text
import tensorflow as tf
```

We will use MNIST dataset. Below code means loading the data and convert it from integers to floating-point numbers.

```text
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
```

Please enter the code below. The code below loads the keras model and sets the optimizer and loss function.

```text
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
```

Please enter the code below.

```text
predictions = model(x_train[:1]).numpy()
predictions
```

You can see this output after run above code  \(**This is an output, do not enter code below**\)

```text
array([[ 0.67086774, -0.25231966,  0.01695401, -0.20872438, -0.5840499 ,
         0.20415965, -0.07967779,  0.01230302,  0.2564202 ,  0.19890268]],
      dtype=float32)
```

Please enter the code below. \(Softmax is an activation function. The feature is that when all the output values ​​are added up, sum is always '1'\)

```text
tf.nn.softmax(predictions).numpy()
```

You can see this output after run above code  \(**This is an output, do not enter code below**\)

```text
array([[0.18120685, 0.07198457, 0.09422877, 0.07519217, 0.05166196,
        0.11362814, 0.08554938, 0.09379152, 0.11972431, 0.11303235]],
      dtype=float32)
```

Please enter the code below. \(Crossentropy is a loss function. It is hard to describe, so if you want to know more about that, go [here](https://en.wikipedia.org/wiki/Cross_entropy).\)

```text
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

Please enter the code below.

```text
loss_fn(y_train[:1], predictions).numpy()
```

You can see this output after run above code  \(**This is an output, do not enter code below**\)

```text
2.1748242
```

Please enter the code below.

```text
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
```

Please enter the code below. \(The Model.fit method adjusts the model parameters to minimize the loss:\)

```text
model.fit(x_train, y_train, epochs=5)
```

output of above code : 

```text
Epoch 1/5
1875/1875 [==============================] - 3s 2ms/step - loss: 0.4813 - accuracy: 0.8565
Epoch 2/5
1875/1875 [==============================] - 3s 2ms/step - loss: 0.1533 - accuracy: 0.9553
Epoch 3/5
1875/1875 [==============================] - 3s 2ms/step - loss: 0.1057 - accuracy: 0.9686
Epoch 4/5
1875/1875 [==============================] - 3s 2ms/step - loss: 0.0908 - accuracy: 0.9721
Epoch 5/5
1875/1875 [==============================] - 3s 2ms/step - loss: 0.0700 - accuracy: 0.9788
<tensorflow.python.keras.callbacks.History at 0x7f7c15df5358>
```

Please enter the code below. \(The model.evaluate method checks the models performance\)

```text
model.evaluate(x_test,  y_test, verbose=2)
```

output of above code : 

```text
313/313 - 0s - loss: 0.0748 - accuracy: 0.9758
[0.07476752996444702, 0.9757999777793884]
```

Please enter the code below.

```text
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
```

```text
probability_model(x_test[:5])
```

output of above code :

```text
<tf.Tensor: shape=(5, 10), dtype=float32, numpy=
array([[7.78855878e-08, 6.61358468e-11, 6.59998250e-07, 1.59961201e-05,
        2.46321262e-11, 1.29930243e-07, 2.94833365e-14, 9.99982715e-01,
        4.22193658e-08, 4.47160573e-07],
       [4.33228813e-08, 1.27517624e-05, 9.99970555e-01, 1.60829786e-05,
        2.47642311e-16, 7.49611928e-09, 1.37607294e-08, 4.11349470e-12,
        6.27970280e-07, 9.24811917e-14],
       [2.03916397e-06, 9.99185383e-01, 1.84247561e-04, 1.05477593e-05,
        2.75765397e-05, 5.58228692e-07, 1.01305332e-05, 4.32787347e-04,
        1.45807702e-04, 8.87280294e-07],
       [9.99742925e-01, 5.94857603e-08, 8.63709865e-05, 1.54006088e-08,
        1.39324254e-06, 4.43872267e-07, 1.60094583e-04, 5.25048790e-07,
        8.63345750e-09, 8.21989215e-06],
       [5.87329941e-06, 3.34152190e-07, 3.92818802e-05, 3.36201609e-08,
        9.96013522e-01, 5.50971926e-08, 4.14997248e-06, 1.14215931e-04,
        2.20527431e-06, 3.82039533e-03]], dtype=float32)>
```

Great job. If you want to know more about this tutorial or other tutorials, go [here](https://www.tensorflow.org/tutorials)

**Goal : raise accuracy up to 99%**

\*\*\*\*

## Raising up Accuracy \(Second Week\)

Last week\(4. 12 ~ 4. 16\), We have installed **python**, **anaconda**, **tensorflow**.

And we have run the code of tutorial\(tensorflow : quickstart for beginners\)



This week\('21. 4. 19 ~ '21. 4. 23\) , We will make the accuracy up to 98% or more higher.

So how can we do that? Let's go practicicing.

{% hint style="info" %}
Hint : the more many epochs, the more higher accuracy
{% endhint %}



Below picture is an example of increased accuracy 

![](.gitbook/assets/image%20%2814%29.png)

There are many ways to increase accuracy.



Adjusting the value of \(**epoch**, **dropout**\) can increases the accuracy.

Also use different Activation & Loss function such as

\(**Sigmoid**, **Softmax**\) in Activation function or \(**Cross-entropy**, **Binary-entropy**\) to increase accuracy.

The MNIST datasets used in the tutorial are relatively easy to classify, so even a simple method can increase the accuracy up to 99%.

If you are curious about the principle of the above methods, not only by simply changing the value, but also increasing the accuracy, please refer to the following



### Epoch

Epoch refers to one cycle through the full training dataset. Usually, training a neural network takes more than a few epochs. In other words, if we feed a neural network the training data for more than one epoch in different patterns, we hope for a better generalization when given a new "unseen" input . 

An epoch is often mixed up with an iteration. Iterations is the number of batches or steps through partitioned packets of the training data, needed to complete one epoch.  Heuristically, one motivation is that \(especially for large but finite training sets\) it gives the network a chance to see the previous data to readjust the model parameters so that the model is not biased towards the last few data points during training.  

Simply put, epoch means the number of times all data has been learned once.

\*\*\*\*

### Dropout

Dropout is a regularization technique for reducing overfitting in artificial neural networks by preventing complex co-adaptations on training data.

The term dropout refers to randomly "dropping out", or omitting, units \(both hidden and visible\) during the training process of a neural network.

![Srivastava, Nitish, et al. &#x201D;Dropout: a simple way to prevent neural networks from overfitting&#x201D;, JMLR 2014](.gitbook/assets/image%20%288%29.png)

With above picture, you can easily understand the principle of 'Drop out'

### Activation Function

What is Activation Function?

There are many Activation Functions.

![source : https://docs.paperspace.com/machine-learning/wiki/activation-function](.gitbook/assets/image%20%283%29.png)

But commonly, we almost use **ReLU**, **Sigmoid**, **Softmax** functions.

Simply, the function of activation function is like this.

 input data =&gt; {activation function} =&gt; output data



**ReLU :**

![source : https://en.wikipedia.org/wiki/Rectifier\_\(neural\_networks\)](.gitbook/assets/image%20%2811%29.png)

If the input value is less than 0, it outputs as 0, and if it is greater than 0, it outputs the input value as it is.

\*\*\*\*

**Sigmoid** :

![Source : https://en.wikipedia.org/wiki/Sigmoid\_function](.gitbook/assets/image%20%284%29.png)

A function that adjusts the value from the output node in the range of 0 to 1.



**Softmax** :

A function that has the condition that all values ​​in the output node fall within the range of 0 to 1, and all values ​​are added together to become 1.

You can refer [here](https://en.wikipedia.org/wiki/Softmax_function)



### Loss Function

What is Loss Function?

Loss function defines the difference of value between outpude node & right answer



#### MSE

MSE function usually used in the purpose of regression using deep\_learning models.



![source : https://en.wikipedia.org/wiki/Mean\_squared\_error](https://wikimedia.org/api/rest_v1/media/math/render/svg/e258221518869aa1c6561bb75b99476c4734108e)

The result of MSE function is the squared of difference between the output value and the correct answer.



#### **Categorical Cross-entropy**

Categorical cross entropy is used for multiclass classification, i.e. when there are more than 3 classes to be classified.

It is used when the label is provided in one-hot format such as \[0,0,1,0,0\], \[1,0,0,0,0\], \[0,0,0,1,0\].



Additionally, we used sparse-categorical cross-entropy in tutorial.

It is used when labels come in the form of integers, such as 0, 1, 2, 3, 4.









