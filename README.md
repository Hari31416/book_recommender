# Book Recommendation

A small project to create a book recommendation system using collaborative filtering as well as content based filtering. I will be using various datasets from Kaggle.

## Dataset Creation

### Data Sources 

The final dataset created is made of multiple sources. The sources used are:

- [Book Depository Dataset](https://www.kaggle.com/datasets/sp1thas/book-depository-dataset) This is denotes as [1]. (It has over 1M books with genre and description. Does not have user ratings. More than one author per book.) [1]

- [Goodreads Book Datasets With User Rating 2M](https://www.kaggle.com/datasets/bahramjannesarr/goodreads-book-datasets-10m) This is denotes as [2]. (It has about 5M books scraped from GoodReads. Also has about 11000 user ratings. Does not has the genre though. On author per book.) [2]

- [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) This is denotes as [2]. (Has about 279k users with some demographic info, 271k books. No description and no genre) [3]

The reasons behind using three different sources were these:

- [1] has better information like book genre and multiple authors but does not have user ratings. Also, it does not have a lot of book metadata like ratings, number of ratings, etc.
- [2] has a number of book metadata that [1] did not have. It also has user ratings. However, it does not have the genre and has only one author per book. Also, the user ratings are very few and not enough to create a good recommendation system. Also, there is no information about user, like age or location.
- [3] has better user information. It has more interactions between users and books and also has some information about user. However, it does not have book metadata.

### Plan of Action

The idea was to use the genre and author information from [1], other book info from [2] and the user and interaction from [3] create a final dataset. There were some complications regrading the book matching. I mean the books in the three sources does not have some unique identifier. Of course, we have ISBN but they need not match as the same book with different publication date and publisher will have different ISBN. So, I decided to use the book title and author name to match the books.

Initially, I thought of using a fuzzy match, however, considering the huge number of books, it was taking a very long time. So, I decided to use a simple match. So, I tried using embeddings. Embeddings of the title + author combination were created using the `sentence-transformers` model from `transformers` library. Then, I used `faiss` to create an index of the embeddings. Finally, a threshold was defined to get the closest match. I considered only a single match with the lowest distance. If this lowest distance was below the threshold, then the book was considered a match otherwise no match of considered. The threshold was decided empirically. I experimented with some threshold value and used what I thought was the best. This match was done for first [1] to [2] and then to [2] to [3]. The final dataset was created by merging the three datasets.

### Final Dataset

The final dataset has four files:

- `books.parquet`: This file contains the final book that are matched in all the three data sources. It turned out that very few books are matching (60812 books). But I think I have to work with what I get.
- `users.parquet`: This file contains the final users that are matched in all the three data sources. This was copied from [3] as is.
- `ratings.parquet`: This file contains the final ratings that are matched in all the three data sources. This was copied from [3] as is.
- `books_all.parquet`: This has all the books that were matching in [1] and [2]. This was created so that we will have a larger version of book dataset (215397 books), even though not all the books have user interactions.

## Dataset using Goodreads Book Graph Datasets

[Goodreads Book Graph Datasets](https://mengtingwan.github.io/data/goodreads.html) is a comprehensive dataset that was scraped in 2017 from Goodreads. From the website:
> We collected three groups of datasets: (1) meta-data of the books, (2) user-book interactions (users' public shelves) and (3) users' detailed book reviews. These datasets can be merged together by joining on book/user/review ids.
>
>Basic Statistics of the Complete Book Graph:
2,360,655 books (1,521,962 works, 400,390 book series, 829,529 authors)
876,145 users; 228,648,342 user-book interactions in users' shelves (include 112,131,203 reads and 104,551,549 ratings)

I might work on this dataset later. But for now, I am using the dataset created above.