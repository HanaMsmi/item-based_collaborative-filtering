import numpy as np
import pandas as pd


def calculate_adjusted_cosine_similarity(normalized_df, sourse_df):
    similarity_matrix = pd.DataFrame(index=sourse_df.columns, columns=sourse_df.columns)

    for i in sourse_df.columns:
        for j in sourse_df.columns:
            # Caculate numerator 
            A = np.sum(normalized_df[i]*normalized_df[j])

            # Calculate denominator
            B = np.sqrt(np.sum(sourse_df[i]**2)) * np.sqrt(np.sum(sourse_df[j]**2))

            similarity_matrix.loc[i, j] = A / B

    return similarity_matrix


def calculate_score(history, similarities, avgRating):
    return np.nansum((history - avgRating) * similarities) / np.sum(similarities)


def get_recommendation_scores(user_means, source_df, similarity_matrix):

    recommendaton_scores = pd.DataFrame(index=source_df.index, columns=source_df.columns)

    for i in source_df.index:
        meanAvg = user_means[i]

        for j in source_df.columns:

            # Check if user has rated the movie 'j'
            if not pd.isnull(source_df.loc[i, j]):
                recommendaton_scores.loc[i, j] = -1
                continue
            else:
                topN = similarity_matrix[j].sort_values(ascending=False).head(11)

                # Dropping first movie as it will be the same movie
                topN_names = topN.index[1:]

                topN_similarities = topN.values[1:]

                # We then get the user's rating history for those 10 movies.
                topN_history = source_df.loc[source_df.index==i, topN_names].values[0]

                # Get average ratings for similar movies
                item_rating_avg = source_df[topN_names].mean(axis=0).values

                # Calculate score for the given movie and the user
                recommendaton_scores.loc[i, j] = meanAvg + calculate_score(similarities=topN_similarities, 
                                                                           history=topN_history, 
                                                                           avgRating=item_rating_avg)     
    return recommendaton_scores
                

def restructure_score_matrix(recommendaton_scores, n):
    recommendaton_scores_holder = pd.DataFrame(columns=['User'] + list(range(1, n)), index=recommendaton_scores.index)

    for i in recommendaton_scores.index:
        sorted_scores = recommendaton_scores.loc[i].sort_values(ascending=False)
        recommendaton_scores_holder.loc[i] = sorted_scores.index[:n]

    return recommendaton_scores_holder


if __name__ == "__main__":
    # n: the number of items to be recommended 
    n = 10

    # Prepare dataset
    df = pd.read_csv('IBCF - Movie Ratings.xlsx - Ratings.csv').set_index('User')
    df = df.replace(0, np.nan)

    # Normalize matrix
    user_means = df.mean(axis=1)

    df_normalized = df.copy()

    # Where df is NaN, store zero
    df_normalized[df.isna()] = 0

    # Where df is not NaN, subtract user_means
    df_normalized[df.notna()] = df[df.notna()].sub(user_means, axis=0)

    # Get similarity between all items using adjusted cosine similarity formula
    similarity_matrix = calculate_adjusted_cosine_similarity(df_normalized, df)

    # Caculate how each item is good enough to be recommended to each user
    recommendaton_scores = get_recommendation_scores(
        user_means=user_means.to_dict(),
        similarity_matrix=similarity_matrix,
        source_df=df
    )

    # Get the final result of recommendation system
    recommendaton_scores_holder = restructure_score_matrix(recommendaton_scores, n)
