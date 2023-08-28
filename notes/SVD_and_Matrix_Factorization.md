# Imports


```python
import pandas as pd
import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
```

# Data


```python
DATA_DIR = os.path.join("..", "data", "final_dataset")
```


```python
df = pd.read_parquet(os.path.join(DATA_DIR, 'ratings.parquet'))
books= pd.read_parquet(os.path.join(DATA_DIR, 'books_all.parquet'))
df = df[df["isbn"].isin(books["isbn"])]
df = df.query("provided_rating!=0")
df.reset_index(drop=True, inplace=True)
print(f"Number of ratings: {len(df)}")
print(f"Number of unique users: {df['user_id'].nunique()}")
print(f"Number of books: {df['isbn'].nunique()}")
df.head()
```

    Number of ratings: 104756
    Number of unique users: 31940
    Number of books: 22020
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>isbn</th>
      <th>provided_rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17</td>
      <td>0891075275</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17</td>
      <td>0553264990</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26</td>
      <td>0449005615</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>39</td>
      <td>0671888587</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>69</td>
      <td>1853260053</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



## Smaller Dataset


```python
num_ratings = df.groupby('isbn')['provided_rating'].count().sort_values(ascending=False)
most_rated_books = num_ratings.index[:10]
num_ratings.head()
```




    isbn
    0316666343    707
    0060928336    320
    0671027360    269
    067976402X    256
    0786868716    242
    Name: provided_rating, dtype: int64




```python
df.groupby('user_id')['provided_rating'].count().sort_values(ascending=False)
```




    user_id
    11676     1593
    98391      595
    189835     371
    76499      333
    153662     322
              ... 
    59675        1
    157184       1
    59685        1
    59697        1
    278854       1
    Name: provided_rating, Length: 31940, dtype: int64




```python
ratings = pd.DataFrame(df.groupby('isbn')['provided_rating'].mean())
ratings['num_ratings'] = pd.DataFrame(df.groupby('isbn')['provided_rating'].count())
ratings.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>provided_rating</th>
      <th>num_ratings</th>
    </tr>
    <tr>
      <th>isbn</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0002163578</th>
      <td>5.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>0002190915</th>
      <td>9.5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>0002210479</th>
      <td>6.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>0002222469</th>
      <td>8.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>0002241358</th>
      <td>8.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
min_ratings = 5
books_ = ratings.query(f"num_ratings > {min_ratings}").index
print(f"Number of books_ with more than {min_ratings} ratings: {len(books_)}")
print(f"Original number of books_: {df['isbn'].nunique()}")
print(f"Number of rows in the original dataset: {df.shape[0]}")
df_small = df[df['isbn'].isin(books_)]
print(f"Number of rows in the new dataset: {df_small.shape[0]}")
```

    Number of books_ with more than 5 ratings: 3823
    Original number of books_: 22020
    Number of rows in the original dataset: 104756
    Number of rows in the new dataset: 72190
    


```python
min_ratings = 10
books_ = ratings.query(f"num_ratings > {min_ratings}").index
print(f"Number of books_ with more than {min_ratings} ratings: {len(books_)}")
print(f"Original number of books_: {df['isbn'].nunique()}")
print(f"Number of rows in the original dataset: {df.shape[0]}")
df_small = df[df['isbn'].isin(books_)]
unique_users = df_small['user_id'].nunique()
print(f"Number of rows in the new dataset: {df_small.shape[0]}")
print(f"Number of unique users in the new dataset: {unique_users}")
```

    Number of books_ with more than 10 ratings: 1963
    Original number of books_: 22020
    Number of rows in the original dataset: 104756
    Number of rows in the new dataset: 58166
    Number of unique users in the new dataset: 22560
    

## Preprocessing


```python
n_users = df_small.user_id.nunique()
n_items = df_small.isbn.nunique()

print('Num. of Users: '+ str(n_users))
print('Num of Movies: '+str(n_items))
```

    Num. of Users: 22560
    Num of Movies: 1963
    


```python
user_id_map = dict(zip(df_small.user_id.unique(), list(range(n_users))))
book_id_map = dict(zip(df_small.isbn.unique(), list(range(n_items))))
user_id_map_df  = pd.DataFrame(
    {
        "user_id":user_id_map.keys(),
        "user_id_new": user_id_map.values(),
    }
)
book_id_map_df  = pd.DataFrame(
    {
        "isbn":book_id_map.keys(),
        "isbn_new": book_id_map.values(),
    }
)
df_small["user_id"] = df_small["user_id"].map(user_id_map)
df_small["isbn"] = df_small["isbn"].map(book_id_map)
```

# SVD

## Introduction

In linear algebra, the singular value decomposition (SVD) is a factorization of a real or complex matrix. The SVD of an $n\times m$ complex matrix $\mathbf{M}$ is a factorization of the form

$$
\mathbf{M} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^*
$$

where:

$\mathbf{U}$ is an $n\times m$ unitary matrix over the field $\mathbb{C}$, called the left singular vectors of $\mathbf{M}$,

$\mathbf{\Sigma}$ is an $m\times m$ diagonal matrix with non-negative real numbers on the diagonal, called the singular values of $\mathbf{M}$,

and $\mathbf{V}^*$ denotes the conjugate transpose of an $n\times n$ unitary matrix $\mathbf{V}$ over $\mathbb{C}$, called the right singular vectors of $\mathbf{M}$.

If the matrix $\mathbf{M}$ is real, $\mathbf{V}^*$ denotes the transpose of $\mathbf{V}$ and we can denote the SVD of a real matrix with

$$
\mathbf{M} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T
$$

Furthermore, for this case, $\mathbf{U}$ and $\mathbf{V}$ form two sets of orthonormal bases.

The singular values are ordered such that $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_p \geq 0$.

> Imagine we collected some book reviews such that books are columns and people are rows, and the entries are the ratings that a person gave to a book. In that case, $\mathbf{M}\cdot \mathbf{M}^T$ would be a table of person-to-person which the entries would mean the sum of the ratings one person gave match with another one. Similarly $\mathbf{M}^T\cdot \mathbf{M}$ would be a table of book-to-book which entries are the sum of the ratings received match with that received by another book. What can be the hidden connection between people and books? That could be the genre, or the author, or something of similar nature.

## Low rank approximation with the SVD

The rank of a matrix, $\mathbf{A}$, is determined by the dimension of the vector space spanned by its columns. The SVD can be used to approximate a matrix with a lower rank, which ultimately decreases the dimensionality of data required to store the information represented by the matrix. The rank-r approximation of $\mathbf{A}$ in terms of the SVD is defined by the formula:

$$
{\mathrm{A_r} } = {\mathrm{U_r} } \Sigma_r {\mathrm{V_r} }^T
$$

where

$\mathbf{U_r}$ is the $m\times r$ matrix consisting of the first $r$ columns of $\mathbf{U}$,

$\mathbf{\Sigma_r}$ is the $r\times r$ diagonal matrix consisting of the first $r$ singular values of $\mathbf{A}$,

and $\mathbf{V_r}$ is the $n\times r$ matrix consisting of the first $r$ columns of $\mathbf{V}$.

## Using SVD for Recommendation

We have a matrix of user-to-item ratings, and we can use SVD decomposition to create two new matrices that can be used to predict unknown ratings.


```python
M = df_small.pivot_table(index='user_id', columns='isbn', values='provided_rating')
M = M.fillna(0)
M = M.values
M.shape
```




    (22560, 1963)



We have 22560 users and 1963 books. In notation denoted above, we have $n = 22560$ and $m = 1963$. We can use `numpy.svd` to decompose the matrix into three matrices.


```python
U, sigma, Vt = svd(M, full_matrices=False)
U.shape, sigma.shape, Vt.shape
```




    ((22560, 1963), (1963,), (1963, 1963))



Let us see if it approximates the original matrix well.


```python
def calculate_rmse(M, M_hat, threhsold = 0.5):
    M_copy = M.copy()
    M_copy = M_copy[M_hat > threhsold]
    M_hat_copy = M_hat.copy()
    M_hat_copy = M_hat_copy[M_hat > threhsold]
    print(f"Number of ratings: {len(M)}")
    return np.sqrt(np.mean((M - M_hat)**2))
```


```python
M_ = np.dot(U, np.dot(np.diag(sigma), Vt))
rmse = calculate_rmse(M, M_, threhsold=0.5)
print(f"RMSE: {rmse}")
```

    Number of ratings: 22560
    RMSE: 1.2170026062959825e-15
    

It does! The problem is, the matrices still have very high dimensions. We can use the low rank approximation to reduce the dimensions.


```python
r = 100
s_r, U_r, V_r = sigma[..., :r], U[..., :, :r], Vt[..., :, :r].T
print(f"U_r: {U_r.shape}")
print(f"s_r: {s_r.shape}")
print(f"V_r: {V_r.shape}")
M_r = np.dot(U_r, np.dot(np.diag(s_r), V_r))
```

    U_r: (22560, 100)
    s_r: (100,)
    V_r: (100, 1963)
    


```python
def rank_r_approx(s, U, V, r, verbose=False):
  s_r, U_r, V_r = s[..., :r], U[..., :, :r], V[..., :, :r].T #need to transpose V
  # Compute the low-rank approximation and its size
  M_r = np.dot(U_r, np.dot(np.diag(s_r), V_r))
  M_r_size = np.size(U_r) + np.size(s_r) + np.size(V_r)
  og_size = np.size(U) + np.size(s) + np.size(V)
  if verbose:
    print(f"Approximation Size: {M_r_size}")
    print(f"Original Size: {og_size}")
    print(f"Compression Ratio: {og_size/M_r_size}")
  return M_r, M_r_size

M_r, M_r_size = rank_r_approx(sigma, U, Vt, 20, verbose=True)
```

    Approximation Size: 490480
    Original Size: 48140612
    Compression Ratio: 98.15
    

Using rank of 20 gives a compression ratio of about 100. Let us check the RMSE:


```python
rmse = calculate_rmse(M, M_r, threhsold=0.5)
print(f"RMSE: {rmse}")
```

    Number of ratings: 22560
    RMSE: 0.30479232583716404
    

We can see that the RMSE has increased. This was expected, since we are using a lower rank approximation. However, the RMSE is still very low, and we have reduced the dimensions of the matrices by a factor of 20. This is a huge improvement.

# Matrix Factorization

SVD, discussed above is a special case of matrix factorization. Matrix factorization is a class of collaborative filtering algorithms used in recommender systems. Matrix factorization algorithms work by decomposing the user-item interaction matrix into the product of two lower dimensionality rectangular matrices. The matrix factorization algorithms are usually more scalable than the neighborhood methods and can deal better with sparsity in the user-item matrix.

## Basics

In matrix factorizations, the user-item interactions are modeled as inner products in the latent factor space with dimensionality $f$. For each item, we have a vector $q_i \in \mathbb{R}^f$ which represents the item in the latent factor space. For each user, we have a vector $p_u \in \mathbb{R}^f$ which represents the user in the same latent factor space. The rating $r_{ui}$ is the inner product of the corresponding vectors in the latent factor space:

$$
r_{ui} \approx q_i^Tp_u
$$

## Biases

We usually need to add some biases to the model. For example, some users may tend to give higher ratings than others, regardless of the items. Thus ratings alone cannot determine what movies should be recommended and that is when biases are added to the equation. The first order approximation of this bias is:

$$
\hat{r}_{ui} = \mu + b_u + b_i + q_i^Tp_u
$$

where $\mu$ is the average rating over all items, $b_u$ is the bias of user $u$, and $b_i$ is the bias of item $i$.

## Other Inputs

In addition to the ratings and biases discussed prior, we must take into account many of the other forms of inputs that affect the outcome of the recommendation. These inputs are generally made up of the implicit data. For example, the time of day, the day of the week, the location of the user, the device being used, the weather, etc. are all examples of implicit data that can be used to improve the recommendation.

To incorporate this, we can define another vector $x_i \in \mathbb{R}^d$ for each item $i$. Furthermore, we can also use some attributes of users such as income, gender, age, etc. to define a vector $y_a \in \mathbb{R}^d$ for each user $a$. The rating can then be modeled as:

$$
\hat{r}_{ui} = \mu + b_u + b_i + q_i^T(p_u + x_i + y_u)
$$

> We can make these biases as a function of time to incorporate the idea that people's tastes change over time.

## Learning the Parameters

These parameters can be learned by minimizing the regularized squared error on the set of known ratings. We can define the loss function as:

$$
\mathcal{L} = \sum_{(u,i) \in \mathcal{K}} (r_{ui} - \hat{r}_{ui})^2 + \lambda \left( \sum_u ||p_u||^2 + \sum_i ||q_i||^2 + \sum_i ||x_i||^2 + \sum_u ||y_u||^2 \right)
$$

where $\mathcal{K}$ is the set of known ratings and $\lambda$ is the regularization parameter.

In shorthand, we have:

$$
e_{ui} = r_{ui} - \mathbf{q}_i^T \mathbf{p}_u
$$

then we modify the vectors as:

$$
\mathbf{p}_u \leftarrow \mathbf{p}_u + \gamma (e_{ui} \mathbf{q}_i - \lambda \mathbf{p}_u) \\
\mathbf{q}_i \leftarrow \mathbf{q}_i + \gamma (e_{ui} \mathbf{p}_u - \lambda \mathbf{q}_i)
$$

where $\gamma$ is the learning rate.

There are two methods to learn the parameters:

1. Stochastic Gradient Descent (SGD)
2. Alternating Least Squares (ALS)

SGD is a generic method that can be used to learn any model. What makes this method less than ideal in this situation is that the error function above is not convex for both $p_u$ and $q_i$ simultaneously. This means that the error function is not bowl shaped, and there are many local minima. This means that SGD can get stuck in a local minimum and not find the global minimum.

This is where ALS comes in. ALS is a method that can be used to learn the parameters of a matrix factorization model. ALS works by holding one set of parameters constant and optimizing the other. Then the roles are reversed and the other set of parameters are optimized while the other set is held constant. This process is repeated until convergence.

## SVD and User-Item Vectors

We already discussed that the SVD of the user-item matrix $\mathbf{M}$ is given by:

$$
\mathbf{M} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T
$$

Using this, we can approximate:

$$
\mathbf{p}_u \approx \mathbf{U}_k\sqrt{\mathbf{\Sigma}}^T \\
\mathbf{q}_i \approx \sqrt{\mathbf{\Sigma}}\mathbf{V}_k^T
$$
