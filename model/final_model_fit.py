import pandas as pd
import numpy as np
import graphlab as gl


if __name__ == "__main__":
    sample_sub_fname = "data/sample_submission.csv"
    ratings_data_fname = "data/ratings.dat"
    side_features_fname = "side_features_75.csv"
    output_fname = "data/our_test_ratings.csv"

    ratings = gl.SFrame(ratings_data_fname, format='tsv')
    sf = pd.read_csv(side_features_fname)
    side_features = gl.SFrame(sf)
    sample_sub = pd.read_csv(sample_sub_fname)
    for_prediction = gl.SFrame(sample_sub)


    rec_eng = gl.ranking_factorization_recommender.create(observation_data=ratings,
                                                     user_id="user_id",
                                                     item_id="joke_id",
                                                     target='rating',
                                                     item_data=side_features,
                                                     solver='auto',
                                                     ranking_regularization = 0,
                                                     num_factors=16)
    sample_sub.rating = rec_eng.predict(for_prediction) #update with ratings
    sample_sub.to_csv(output_fname, index=False)
