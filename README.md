# item-based collaborative filtering implementation in python
This repository is the python implementation of muffaddal's codes.


He has explained the code in the documentation Fully and clearly, you can read it [here](https://towardsdatascience.com/comprehensive-guide-on-item-based-recommendation-systems-d67e40e2b75d) 

Also, you can fine the link to the original R implementation [here](https://github.com/muffaddal52/Item-Based-Collaborative-Recommendation-System/tree/master)


item-based (item to item) collaborative filtering is a strategy in recommendation sytems that finds most similar items using formulas like cosine similarity, Pearson similarity, ....

**Steps invovled in this recommendation system are as follow:**
* Prepare dataset (your dataset should have index of user_ids or user_name or ...)
* Calculate similarity between all items (Here Adjusted Cosine Similarity is used)
* based on user's ratings to items, find best matching items from the similarity matrix (considering the ratings to that particular item)
* Finally, set n your prefered number to get n results (here, n recommended movies to each user)     
