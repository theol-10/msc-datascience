import torch
import numpy as np
import pandas as pd
import sys
import os
from cross_validation import random_train_test_split
from explicit import ExplicitFactorizationModel
from interactions import Interactions

def load_data(movies_path, ratings_path, tags_path):
    """
    Load and preprocess the MovieLens dataset from CSV files,
    automatically removing movies without ratings
    
    Parameters:
    -----------
    movies_path : str
        Path to the movies.csv file
    ratings_path : str
        Path to the ratings.csv file
    tags_path : str
        Path to the tags.csv file
    
    Returns:
    --------
    tuple
        (movies DataFrame, ratings DataFrame, tags DataFrame)
    """
    try:
        # Load raw data
        movies = pd.read_csv(movies_path)
        ratings = pd.read_csv(ratings_path)
        tags = pd.read_csv(tags_path)
        
        # Preprocess ratings DataFrame
        ratings = ratings[['userId', 'movieId', 'rating', 'timestamp']]
        
        # Convert IDs to integers
        ratings['userId'] = ratings['userId'].astype(int)
        ratings['movieId'] = ratings['movieId'].astype(int)
        movies['movieId'] = movies['movieId'].astype(int)
        
        # Ensure rating is float
        ratings['rating'] = ratings['rating'].astype(float)
        
        # Get the set of movies that have ratings
        rated_movies = set(ratings['movieId'].unique())
        
        # Count movies before filtering
        total_movies_before = len(movies)
        
        # Filter movies to keep only those with ratings
        movies = movies[movies['movieId'].isin(rated_movies)].copy()
        
        # Count movies after filtering
        total_movies_after = len(movies)
        movies_removed = total_movies_before - total_movies_after
        
        # Print cleaning report
        if movies_removed > 0:
            print(f"\nData Cleaning Report:")
            print(f"Removed {movies_removed} movies without ratings")
            print(f"Remaining movies: {total_movies_after}")
        
        # Sort ratings by userId and timestamp
        ratings = ratings.sort_values(['userId', 'timestamp'])
        
        # Process tags (if any exist for the remaining movies)
        tags['userId'] = tags['userId'].astype(int)
        tags['movieId'] = tags['movieId'].astype(int)
        tags = tags[tags['movieId'].isin(rated_movies)].copy()
        
        return movies, ratings, tags
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find one or more data files: {str(e)}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def prepare_dataset(ratings_df):
    """
    Convert ratings DataFrame to Interactions object while preserving original user IDs
    
    Parameters:
    -----------
    ratings_df : pandas.DataFrame
        DataFrame containing columns: userId, movieId, rating, timestamp
    
    Returns:
    --------
    spotlight.interactions.Interactions
        Interactions object for the Spotlight model
    """
    # Verify ratings DataFrame has required columns
    required_columns = ['userId', 'movieId', 'rating']
    if not all(col in ratings_df.columns for col in required_columns):
        raise ValueError(f"ratings_df must contain columns: {required_columns}")
    
    # Extract arrays for Interactions
    user_ids = ratings_df['userId'].values
    movie_ids = ratings_df['movieId'].values
    ratings = ratings_df['rating'].values
    
    # Create and return Interactions object
    interactions = Interactions(
        user_ids=user_ids,
        item_ids=movie_ids,
        ratings=ratings,
        num_users=ratings_df['userId'].max() + 1,
        num_items=ratings_df['movieId'].max() + 1
    )
    
    return interactions

def train_model(train_data, embedding_dim=128, n_iter=10, batch_size=1024):
    """
    Train the recommendation model
    
    Parameters:
    -----------
    train_data : spotlight.interactions.Interactions
        Training data
    embedding_dim : int
        Dimension of the latent factors
    n_iter : int
        Number of training iterations
    batch_size : int
        Training batch size
    
    Returns:
    --------
    ExplicitFactorizationModel
        Trained model
    """
    model = ExplicitFactorizationModel(
        loss='regression',
        embedding_dim=embedding_dim,
        n_iter=n_iter,
        batch_size=batch_size,
        l2=1e-9,
        learning_rate=1e-3,
        use_cuda=torch.cuda.is_available()
    )
    
    print("Training model...")
    model.fit(train_data, verbose=True)
    return model

def get_user_predictions(user_id, num_predictions=10, model=None, movies_df=None, 
                        ratings_df=None, tags_df=None):
    """
    Get movie recommendations for a specific user
    
    Parameters:
    -----------
    user_id : int
        User ID to get recommendations for
    num_predictions : int
        Number of recommendations to return
    model : ExplicitFactorizationModel
        Trained recommendation model
    movies_df : pandas.DataFrame
        DataFrame containing movie information
    ratings_df : pandas.DataFrame
        DataFrame containing ratings information
    tags_df : pandas.DataFrame
        DataFrame containing tags information
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing recommended movies with their details
    """
    if not all([model, movies_df is not None, ratings_df is not None, tags_df is not None]):
        raise ValueError("All required parameters must be provided")

    # Verify user exists in the dataset
    if user_id not in ratings_df['userId'].unique():
        raise ValueError(f"User ID {user_id} not found in ratings data")
    
    # Get all possible movie IDs from the movies DataFrame
    all_movie_ids = movies_df['movieId'].values
    
    # Get predictions for all movies
    predictions = model.predict(user_id, all_movie_ids)
    
    # Get top K movie indices
    top_k_indices = np.argsort(predictions)[-num_predictions:][::-1]
    recommended_movie_ids = all_movie_ids[top_k_indices]
    
    # Get movie information
    recommendations = movies_df[movies_df['movieId'].isin(recommended_movie_ids)].copy()
    
    # Add average rating for each movie
    avg_ratings = ratings_df.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
    recommendations = recommendations.merge(avg_ratings, on='movieId', how='left')
    recommendations = recommendations.rename(columns={'mean': 'avg_rating', 'count': 'num_ratings'})
    
    # Add tags (concatenated)
    movie_tags = tags_df.groupby('movieId')['tag'].agg(lambda x: ', '.join(x.unique())).reset_index()
    recommendations = recommendations.merge(movie_tags, on='movieId', how='left')
    
    # Add user's rating if available
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    user_ratings = user_ratings[['movieId', 'rating']].rename(columns={'rating': 'user_rating'})
    recommendations = recommendations.merge(user_ratings, on='movieId', how='left')
    
    # Add predicted ratings
    movie_id_to_pred_idx = {movie_id: idx for idx, movie_id in enumerate(all_movie_ids)}
    recommendations['predicted_rating'] = [
        predictions[movie_id_to_pred_idx[movie_id]] 
        for movie_id in recommendations['movieId']
    ]
    
    # Sort by predicted rating
    recommendations = recommendations.sort_values('predicted_rating', ascending=False)
    
    # Reorder columns and round numeric columns
    columns = ['movieId', 'title', 'genres', 'tag', 'predicted_rating', 
               'avg_rating', 'user_rating', 'num_ratings']
    recommendations = recommendations[columns]
    recommendations = recommendations.round({'predicted_rating': 2, 'avg_rating': 2})
    
    return recommendations

def get_recommendations_for_user(user_id, num_recommendations=10, data_folder="./"):
    """
    End-to-end function to get recommendations for a user
    
    Parameters:
    -----------
    user_id : int
        User ID to get recommendations for
    num_recommendations : int
        Number of recommendations to return
    data_folder : str
        Path to the folder containing the CSV files
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing recommended movies with their details
    """
    try:
        # Construct file paths
        movies_path = f"{data_folder}/movies.csv"
        ratings_path = f"{data_folder}/ratings.csv"
        tags_path = f"{data_folder}/tags.csv"
        
        # Load and preprocess data
        print("Loading and preprocessing data...")
        movies, ratings, tags = load_data(movies_path, ratings_path, tags_path)
        
        # Print some information about the data
        print(f"\nDataset Statistics:")
        print(f"Number of users: {ratings['userId'].nunique()}")
        print(f"Number of movies: {movies['movieId'].nunique()}")
        print(f"Number of ratings: {len(ratings)}")
        print(f"Average ratings per user: {len(ratings) / ratings['userId'].nunique():.2f}")
        print(f"Average ratings per movie: {len(ratings) / movies['movieId'].nunique():.2f}")
        
        # Prepare dataset
        print("\nPreparing dataset...")
        dataset = prepare_dataset(ratings)
        
        # Split data
        print("Splitting data...")
        train, test = random_train_test_split(dataset, random_state=np.random.RandomState(42))
        
        # Train model
        model = train_model(train)
        
        # Get recommendations
        print(f"\nGenerating recommendations for user {user_id}...")
        recommendations = get_user_predictions(
            user_id=user_id,
            num_predictions=num_recommendations,
            model=model,
            movies_df=movies,
            ratings_df=ratings,
            tags_df=tags
        )
        
        return recommendations
        
    except Exception as e:
        print(f"Error getting recommendations: {str(e)}")
        raise

# Example usage showing the expected format of the CSV files:
"""
Expected CSV file formats:

ratings.csv:
userId,movieId,rating,timestamp
1,1,4.0,964982703
1,3,4.0,964981247
...

movies.csv:
movieId,title,genres
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
2,Jumanji (1995),Adventure|Children|Fantasy
...

tags.csv:
userId,movieId,tag,timestamp
1,1,pixar,1445714994
1,1,fun,1445714994
...
"""

if __name__ == "__main__":
    # Set these parameters according to your needs
    USER_ID = 1  # The user ID you want recommendations for (1-620 within current dataset)
    NUM_RECOMMENDATIONS = 10  # Number of recommendations to generate (sorted in descending rating)
    DATA_FOLDER = "/data"  # Path to the folder containing your CSV files
    
    try:
        # Get recommendations
        recommendations = get_recommendations_for_user(
            user_id=USER_ID,
            num_recommendations=NUM_RECOMMENDATIONS,
            data_folder=DATA_FOLDER
        )
        
        # Display recommendations
        print("\nTop Recommendations:")
        print("===================")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(recommendations)
        
    except Exception as e:
        print(f"Error: {str(e)}")