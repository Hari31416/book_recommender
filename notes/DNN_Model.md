# Deep Neural Network Models

Another way to create a recommendation system is to use a deep neural network model. The idea is to use a neural network to learn the user and item embeddings. The good thing about this is that we can add some extra information related to user or item to the model which can give better performance.

## Softmax Model

One possible DNN model is softmax, which treats the problem as a multiclass prediction problem in which:

- The input is the user query.
- The output is a probability vector with size equal to the number of items in the corpus, representing the probability to interact with each item.

### What are the inputs?

The input to a DNN can include:

- dense features (for example, watch time and time since last watch)
- sparse features (for example, watch history and country)

This is where DNN based models shines. Unlike matrix factorization, we can add any feature we want to the model. For example, we can add the user's watch history to the model. This is not possible in matrix factorization. Furthermore, in this case, we are free to choose the complexity of the model. Using more layers and non-linearity may help the model learn more complex patterns.

### The Output

We will denote the output of the last layer of the model as $\psi (x) \in \mathbb R^d$. We add another layer, a softmax layer, to the model to convert the output to a probability vector. The probability distribution is defined as:

$$
p(i | x) = h(\psi(x)V^T)
$$

Here:

- $h$ is the softmax function.
- $V \in \mathbb R^{d \times n}$ is the item embedding matrix. This is the matrix weight of the softmax layer.

### The Loss Function

We can use the cross entropy loss function to train the model since this is a multiclass classification problem. The loss function is defined as:

$$
L = - \sum_{i=1}^n y_i \log p(i | x)
$$

### The Embeddings

Note that the probability distribution is defined as:

$$
\hat p_j = \frac{\exp(\langle \psi(x), V_j\rangle)}{Z}\\
\log(\hat p_j) = \langle \psi(x), V_j\rangle - log(Z)
$$

Here $Z$ is a normalization constant. We can see that the log probability of an item $j$ is (up to an additive constant) the dot product of two $d$-dimensional vectors, which can be interpreted as query and item embeddings:

- $\psi(x) \in \mathbb R^d$ is the query embedding of the query $x$.
- $V_j \in \mathbb R^d$ is the item embedding of item $j$.

We see that we do get the embeddings of the query and item from the model. This is similar to matrix factorization. There is a difference though: Instead of learning one embedding $U_i$ per query $i$ the system learns a mapping from the query feature $x$ to the embedding $\psi(x) \in \mathbb R^d$. Therefore, you can think of this DNN model as a generalization of matrix factorization, in which you replace the query side by a nonlinear function $\psi(.)$.

## Using Item Features

Till now, we are using extra features only on the user side. We can, however, use the item features as well. For example, we can add the item category to the model. For this to work, we can use a neural network that consist of two parts:

### The User Network

The user network is a neural network that takes the query features $x_{\text{query}}$ and maps it the query embedding $\psi(x_{\text{query}}) \in \mathbb R^d$.

### The Item Network

The item network is a neural network that takes the item features $x_{\text{item}}$ and maps it the item embedding $\phi(x_{\text{item}}) \in \mathbb R^d$.

### The Output of the Model

The output of such a model can be determined by using the dot product of the query and item embeddings:

$$
p(i | x) = h(\langle \psi(x_{\text{query}}), \phi(x_{\text{item}})\rangle)
$$

Note that this is a real number, meaning that the model predicts the similarity between the query and the item. This is a major difference between this model and the softmax model. In the softmax model, the output is a probability vector where each element represents the probability of interacting with the corresponding item. In this model, the output is a real number representing the similarity between the query and the item.