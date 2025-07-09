from flask import Flask, render_template, request, session, redirect, url_for, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import pandas as pd
import sqlite3
import os
import random
import csv

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Replace with a secure key in production

# Define the path to the database
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(BASE_DIR, 'users.db')

# Database initialization
def init_db():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT UNIQUE,
            quiz_completed INTEGER DEFAULT 0
        )
    """)
    
    # Create movies table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS movies (
            movieId INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            genres TEXT
        )
    """)
    
    # Create ratings table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ratings (
            user_id INTEGER NOT NULL,
            movie_id INTEGER NOT NULL,
            rating INTEGER NOT NULL,
            PRIMARY KEY (user_id, movie_id),
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (movie_id) REFERENCES movies (movieId)
        )
    """)
    
    # Create quiz_responses table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quiz_responses (
            user_id INTEGER NOT NULL,
            question_id INTEGER NOT NULL,
            answer TEXT NOT NULL,
            PRIMARY KEY (user_id, question_id),
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    
    # Create recommendations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie TEXT NOT NULL,
            genres TEXT NOT NULL,
            predicted_rating REAL NOT NULL,
            user_rating REAL,
            recommended_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, movie),
            FOREIGN KEY (user_id) REFERENCES users (id)
        )           
    """)
    
    conn.commit()
    conn.close()
    
    # Load movies from CSV if movies table is empty
    load_movies()
    
    # Run migrations to ensure all columns are present
    migrate_db()

def load_movies():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM movies")
    count = cursor.fetchone()[0]
    conn.close()
    
    if count == 0:
        try:
            with open('movies.csv', 'r', encoding='utf-8') as f:
                reader = pd.read_csv(f)
                movies = reader[['movieId', 'title', 'genres']].values.tolist()
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.executemany("INSERT INTO movies (movieId, title, genres) VALUES (?, ?, ?)", movies)
                conn.commit()
                conn.close()
                print("Movies loaded into the database.")
        except FileNotFoundError:
            print("movies.csv not found. Please ensure the file exists in the project directory.")
        except Exception as e:
            print(f"Error loading movies: {e}")

def migrate_db():
    print("Running database migrations...")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Check if 'answer' column exists in 'quiz_responses'
        cursor.execute("PRAGMA table_info(quiz_responses)")
        columns = [info[1] for info in cursor.fetchall()]
        print(f"Existing columns in 'quiz_responses': {columns}")
        if 'answer' not in columns:
            cursor.execute("ALTER TABLE quiz_responses ADD COLUMN answer TEXT NOT NULL DEFAULT ''")
            conn.commit()
            print("Added 'answer' column to 'quiz_responses' table.")
        else:
            print("'quiz_responses' table already has 'answer' column.")
    except Exception as e:
        print(f"Error during migration: {e}")
    finally:
        conn.close()

def get_random_movies(n=20):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT movieId, title FROM movies")
    all_movies = cursor.fetchall()
    conn.close()
    if len(all_movies) <= n:
        return all_movies
    return random.sample(all_movies, n)

# Placeholder for the actual recommendation logic
def get_recommendations_for_user(user_id, num_recommendations, data_folder):
    # Replace this with your actual recommendation logic
    # For demonstration, we'll return random movies not already recommended
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get movies already recommended
    cursor.execute("SELECT movie FROM recommendations WHERE user_id = ?", (user_id,))
    recommended_movies = set([row[0] for row in cursor.fetchall()])
    
    # Fetch all movies not yet recommended
    cursor.execute("""
        SELECT title, genres FROM movies 
        WHERE title NOT IN (SELECT movie FROM recommendations WHERE user_id = ?)
    """, (user_id,))
    available_movies = cursor.fetchall()
    conn.close()
    
    if not available_movies:
        return pd.DataFrame(columns=['title', 'genres', 'predicted_rating'])
    
    # Randomly select movies as dummy recommendations
    selected_movies = random.sample(available_movies, min(num_recommendations, len(available_movies)))
    data = {
        'title': [movie[0] for movie in selected_movies],
        'genres': [movie[1] for movie in selected_movies],
        'predicted_rating': [round(random.uniform(3.0, 5.0), 2) for _ in selected_movies]
    }
    recommendations = pd.DataFrame(data)
    return recommendations

def save_recommendations(user_id, recommendations):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        for i in range(len(recommendations)):
            title = recommendations.iloc[i]['title']
            genres = recommendations.iloc[i]['genres']
            predict = recommendations.iloc[i]['predicted_rating']
            cursor.execute("""
                INSERT OR IGNORE INTO recommendations (user_id, movie, genres, predicted_rating) 
                VALUES (?, ?, ?, ?)
            """, (user_id, title, genres, predict))
        conn.commit()
        conn.close()
        print(f"Recommendations saved for user_id {user_id}.")
    except Exception as e:
        print(f"Error saving recommendations: {e}")

# Route Definitions

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        
        # Collect all ratings with names starting with 'rating-'
        ratings = {}
        for key, value in request.form.items():
            if key.startswith('rating-'):
                try:
                    movie_id = int(key.split('-')[1])
                    rating = int(value)
                    if 1 <= rating <= 5:
                        ratings[movie_id] = rating
                except (ValueError, IndexError):
                    continue  # Ignore invalid inputs
        
        # Debug: Print received ratings
        print(f"Signup Attempt - User: {username}, Ratings: {ratings}")
        
        # Validation
        if not username or not email or not password:
            error = "All fields are required."
            movies = get_random_movies()
            return render_template('signup.html', error=error, movies=movies)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if username or email already exists
        cursor.execute("SELECT 1 FROM users WHERE username = ? OR email = ?", (username, email))
        if cursor.fetchone():
            error = "User with this username or email already exists."
            conn.close()
            movies = get_random_movies()
            return render_template('signup.html', error=error, movies=movies)
        
        if len(ratings) < 1:
            error = f"Please rate at least 5 movies. You have rated {len(ratings)} movies."
            conn.close()
            movies = get_random_movies()
            return render_template('signup.html', error=error, movies=movies)
        
        # Assign a unique user_id and store user in the users table
        try:
            hashed_password = generate_password_hash(password)  # Correct usage without specifying method
            cursor.execute("""
                INSERT INTO users (username, email, password) VALUES (?, ?, ?)
            """, (username, email, hashed_password))
            conn.commit()
        except sqlite3.IntegrityError:
            error = "User with this username or email already exists."
            conn.close()
            movies = get_random_movies()
            return render_template('signup.html', error=error, movies=movies)
        
        # Get the user_id of the newly created user
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        user_id = cursor.fetchone()[0]
        
        # Store ratings in the ratings table
        valid_ratings = {}
        for movie_id, rating in ratings.items():
            # Ensure movie exists
            cursor.execute("SELECT 1 FROM movies WHERE movieId = ?", (movie_id,))
            if cursor.fetchone():
                valid_ratings[movie_id] = rating
        
        if len(valid_ratings) < 5:
            error = f"Please rate at least 5 valid movies. You have rated {len(valid_ratings)} valid movies."
            conn.close()
            movies = get_random_movies()
            return render_template('signup.html', error=error, movies=movies)
        
        # Insert or update ratings
        for movie_id, rating in valid_ratings.items():
            cursor.execute("""
                INSERT INTO ratings (user_id, movie_id, rating) VALUES (?, ?, ?)
                ON CONFLICT(user_id, movie_id) DO UPDATE SET rating=excluded.rating
            """, (user_id, movie_id, rating))
        
        conn.commit()
        conn.close()
        
        print(f"User '{username}' registered successfully with user_id {user_id}.")
        print(f"User '{username}' rated {len(valid_ratings)} movies.")
        
        flash("Signup successful! Please log in.")
        return redirect(url_for('login'))
    
    # GET request
    movies = get_random_movies()
    return render_template('signup.html', movies=movies)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        if not username or not password:
            error = "Please enter both username and password."
            return render_template('login.html', error=error)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, password, quiz_completed FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            user_id_db, hashed_password, quiz_completed = result
            if check_password_hash(hashed_password, password):
                # Successful login
                session['user_id'] = user_id_db
                session['username'] = username
                
                if quiz_completed:
                    flash("Welcome back! Here are your recommendations.")
                    return redirect(url_for('recommendations'))
                else:
                    flash("Welcome! Please complete the questionnaire.")
                    return redirect(url_for('questionnaire', q=1))
            else:
                error = "Invalid username or password."
        else:
            error = "Invalid username or password."
    
    return render_template('login.html', error=error)

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    error = None
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        if not email:
            error = "Please enter your email."
            return render_template('forgot_password.html', error=error)
        
        # Check if email exists
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT username FROM users WHERE email = ?", (email,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            username = result[0]
            session['reset_user'] = username
            flash("Password reset link has been sent to your email (simulated).")
            # Here, you'd normally send an actual email with a reset link.
            return redirect(url_for('reset_password'))
        else:
            error = "No account found for this email."
            return render_template('forgot_password.html', error=error)
    
    return render_template('forgot_password.html')

@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    error = None
    if 'reset_user' not in session:
        flash("No password reset request found.")
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        new_password = request.form.get('new_password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()
        
        if not new_password or not confirm_password:
            error = "Please fill out both password fields."
            return render_template('reset_password.html', error=error)
        
        if new_password != confirm_password:
            error = "Passwords do not match."
            return render_template('reset_password.html', error=error)
        
        # Hash the new password
        hashed_password = generate_password_hash(new_password)
        
        # Update the password in the database
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET password = ? WHERE username = ?", (hashed_password, session['reset_user']))
            conn.commit()
            conn.close()
            flash("Password reset successful! Please log in.")
            session.pop('reset_user')
            return redirect(url_for('login'))
        except Exception as e:
            error = "An error occurred while resetting your password. Please try again."
            print(f"Error resetting password: {e}")
            return render_template('reset_password.html', error=error)
    
    return render_template('reset_password.html')

@app.route('/questionnaire', methods=['GET', 'POST'])
def questionnaire():
    if 'username' not in session:
        flash("You need to log in to access the questionnaire.")
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    
    # Define questionnaire questions
    questions = [
        {
            "number": 1,
            "text": "How much time do you have to watch something?",
            "type": "radio",
            "options": [
                "Less than 30 minutes",
                "30 minutes to 1 hour",
                "1 to 2 hours",
                "More than 2 hours"
            ]
        },
        {
            "number": 2,
            "text": "What genre are you in the mood for?",
            "type": "radio",
            "options": [
                "Action/Adventure",
                "Comedy",
                "Drama",
                "Horror/Thriller",
                "Sci-Fi/Fantasy",
                "Documentary"
            ]
        },
        {
            "number": 3,
            "text": "What release timeframe are you interested in?",
            "type": "radio",
            "options": [
                "New Releases (past 1-2 years)",
                "Modern Movies (last 10 years)",
                "Classics (older than 10 years)"
            ]
        },
        {
            "number": 4,
            "text": "What language do you want the movie to be in?",
            "type": "dropdown",
            "options": [
                "No Preference",
                "English",
                "Spanish",
                "French",
                "German",
                "Chinese",
                "Japanese",
                "Hindi",
                "Other"
            ]
        },
        {
            "number": 5,
            "text": "Are there specific themes you’d like to explore?",
            "type": "radio",
            "options": [
                "Friendship",
                "Survival",
                "Love",
                "Mystery",
                "No Preference"
            ]
        },
        {
            "number": 6,
            "text": "What kind of vibe are you looking for?",
            "type": "radio",
            "options": [
                "Relaxing and chill",
                "Intense and thrilling",
                "Romantic and heartwarming",
                "Fun and lighthearted",
                "Deep and thought-provoking"
            ]
        },
        {
            "number": 7,
            "text": "Do you want recommendations similar to movies you’ve already enjoyed?",
            "type": "radio",
            "options": [
                "Yes",
                "No"
            ]
        }
    ]
    
    total_questions = len(questions)
    q = request.args.get('q', 1, type=int)  # Current question number
    
    if q < 1 or q > total_questions:
        flash("Invalid question number.")
        return redirect(url_for('questionnaire', q=1))
    
    current_question = questions[q-1]
    
    error = None
    
    if request.method == 'POST':
        response = request.form.get('response')
        if not response:
            error = "Please select an option to proceed to the next question."
        else:
            # Check if the user has already answered this question
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT answer FROM quiz_responses WHERE user_id = ? AND question_id = ?
            """, (user_id, current_question['number']))
            existing_response = cursor.fetchone()
            
            if existing_response:
                # Update existing answer
                try:
                    cursor.execute("""
                        UPDATE quiz_responses SET answer = ? WHERE user_id = ? AND question_id = ?
                    """, (response, user_id, current_question['number']))
                    conn.commit()
                    print(f"User ID {user_id} updated Answer for Question {current_question['number']}: {response}")
                except Exception as e:
                    error = "An error occurred while updating your response. Please try again."
                    print(f"Error updating response: {e}")
            else:
                # Insert new answer
                try:
                    cursor.execute("""
                        INSERT INTO quiz_responses (user_id, question_id, answer) VALUES (?, ?, ?)
                    """, (user_id, current_question['number'], response))
                    conn.commit()
                    print(f"User ID {user_id} answered Question {current_question['number']}: {response}")
                except Exception as e:
                    error = "An unexpected error occurred. Please try again."
                    print(f"Error saving response: {e}")
            conn.close()
        
        if not error:
            if q < total_questions:
                return redirect(url_for('questionnaire', q=q+1))
            else:
                # Mark quiz as completed
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE users SET quiz_completed = 1 WHERE id = ?
                    """, (user_id,))
                    conn.commit()
                    conn.close()
                    flash("Questionnaire completed successfully!")
                    return redirect(url_for('recommendations'))
                except Exception as e:
                    error = "An error occurred while completing the questionnaire."
                    print(f"Error marking quiz as completed: {e}")
    
    # Fetch existing answer if any
    existing_answer = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT answer FROM quiz_responses WHERE user_id = ? AND question_id = ?
        """, (user_id, current_question['number']))
        result = cursor.fetchone()
        conn.close()
        if result:
            existing_answer = result[0]
    except Exception as e:
        print(f"Error fetching existing answer: {e}")
    
    return render_template(
        'questionnaire.html',
        question_number=q,
        total_questions=total_questions,
        question=current_question['text'],
        question_type=current_question['type'],
        options=current_question['options'],
        error=error,
        existing_answer=existing_answer  # Pass existing answer to pre-select option
    )

@app.route('/recommendations')
def recommendations():
    user_id = session.get("user_id")
    
    
    if not user_id:
        flash("You need to log in to view recommendations.")
        return redirect(url_for('login'))
    
    try:
        # Get recommendations (Assuming the function is correctly implemented)
        recommendations = get_recommendations_for_user(
            user_id=user_id,
            num_recommendations=5,
            data_folder="/path/to/your/frontend/"  # Update this path accordingly
        )
        
        # Debug: Print recommendations
        print("\nTop Recommendations:")
        print("===================")
        print(recommendations)
        
        # Store recommendations in the database
        save_recommendations(user_id, recommendations)
    except Exception as e:
        print('Error generating recommendations:')
        print(f"Error: {str(e)}")
        flash("An error occurred while generating recommendations. Please try again.")
        return redirect(url_for('login'))
    
    # Format recommendations for display
    display_recommendations = []
    for i in range(len(recommendations)):
        title = recommendations.iloc[i]['title']
        genres = recommendations.iloc[i]['genres']
        predict = recommendations.iloc[i]['predicted_rating']
        predict = round(predict, 2)
        display = {
            "title": title,
            "image": "https://m.media-amazon.com/images/M/MV5BYjU3MzRjNzktN2IxMi00OWZkLWJmODEtZDc3MTc2NmQzNmYwXkEyXkFqcGc@._V1_.jpg",  # Placeholder image
            "description": f"We think you will give this movie a {predict:.2f} out of 5!"
        }
        display_recommendations.append(display)
    
    return render_template('recommendations.html', recommendations=display_recommendations)

@app.route('/ratings', methods=['GET', 'POST'])
def ratings():
    if 'username' not in session:
        flash("You need to log in to access the ratings page.")
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    error = None
    
    if request.method == 'POST':
        # Collect all ratings with names starting with 'rating-'
        ratings = {}
        for key, value in request.form.items():
            if key.startswith('rating-'):
                try:
                    movie_id = int(key.split('-')[1])
                    rating = int(value)
                    if 1 <= rating <= 5:
                        ratings[movie_id] = rating
                except (ValueError, IndexError):
                    continue  # Ignore invalid inputs
        
        print(f"Ratings Update Attempt - User ID: {user_id}, Ratings: {ratings}")
        
        if len(ratings) < 1:
            error = f"Please rate at least 5 movies. You have rated {len(ratings)} movies."
            movies = get_random_movies()
            return render_template('ratings.html', error=error, movies=movies, ratings=get_user_ratings(user_id))
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Validate that all movie_ids exist
        valid_ratings = {}
        for movie_id, rating in ratings.items():
            cursor.execute("SELECT 1 FROM movies WHERE movieId = ?", (movie_id,))
            if cursor.fetchone():
                valid_ratings[movie_id] = rating
        
        if len(valid_ratings) < 1:
            error = f"Please rate at least 5 valid movies. You have rated {len(valid_ratings)} valid movies."
            conn.close()
            movies = get_random_movies()
            return render_template('ratings.html', error=error, movies=movies, ratings=get_user_ratings(user_id))
        
        # Insert or update ratings
        csv_data = []
        for movie_id, rating in valid_ratings.items():
            cursor.execute("""
                INSERT INTO ratings (user_id, movie_id, rating) VALUES (?, ?, ?)
                ON CONFLICT(user_id, movie_id) DO UPDATE SET rating=excluded.rating
            """, (user_id, movie_id, rating))
            
            #code to update the csv
            csv_data.append([user_id, movie_id, rating])
        
        with open("newRatings.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(csv_data)
        
        conn.commit()
        conn.close()
        
        flash("Ratings updated successfully!")
        return redirect(url_for('recommendations'))  # Redirect to Recommendations Page
    
    # GET request
    movies = get_random_movies()
    user_ratings = get_user_ratings(user_id)
    return render_template('ratings.html', movies=movies, ratings=user_ratings)


# @app.route('/settings')
def settings2():
    if 'user_id' not in session:  # Check if the user is logged in
        return redirect('/login')  # Redirect to login if not logged in
    
    user_id = session['user_id']  # Get the logged-in user’s ID
    
    # Connect to the database and fetch rated movies
    conn = sqlite3.connect(db_path)

    cursor = conn.cursor()
    
    # Query the database for movies rated by the user
    cursor.execute("""
        SELECT m.title, r.rating
        FROM ratings r
        JOIN movies m ON r.movieId = m.id
        WHERE r.id = ?
    """, (user_id,))
    
    rated_movies = cursor.fetchall()  # Get the list of rated movies
    
    conn.close()  # Close the database connection

    # Render the settings page with the rated movies
    return render_template('settings.html', rated_movies=rated_movies)



@app.route('/settings')
def settings():
    if 'user_id' not in session:  # Check if the user is logged in
        return redirect('/login')  # Redirect to login if not logged in
    
    user_id = session['user_id']  # Get the logged-in user’s ID
    
    # Connect to the database and fetch rated movies
    conn = sqlite3.connect(db_path)  # Ensure the correct path to your db
    cursor = conn.cursor()
    
    # Query the database for movies rated by the user
    cursor.execute("""
        SELECT m.title, r.rating
        FROM ratings r
        JOIN movies m ON r.movie_id = m.movieId 
        WHERE r.user_id = ?
    """, (user_id,))
    
    rated_movies = cursor.fetchall()  # Get the list of rated movies
    
    conn.close()  # Close the database connection

    # Render the settings page with the rated movies
    return render_template('settings.html', rated_movies=rated_movies)


@app.route('/search_movies')
def search_movies():
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify([])
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    like_query = f"%{query}%"
    cursor.execute("SELECT movieId, title FROM movies WHERE title LIKE ? LIMIT 20", (like_query,))
    results = cursor.fetchall()
    conn.close()
    
    # Return list of dictionaries
    movies = [{'movieId': movie[0], 'title': movie[1]} for movie in results]
    return jsonify(movies)

@app.route('/reload_movies')
def reload_movies():
    # Fetch a new set of random movies
    movies = get_random_movies()
    # Fetch existing ratings
    user_id = session.get('user_id')
    user_ratings = get_user_ratings(user_id) if user_id else {}
    # Prepare movie data
    movie_data = []
    for movie in movies:
        movie_id, title = movie
        rating = user_ratings.get(movie_id, 0)
        movie_data.append({'movieId': movie_id, 'title': title, 'rating': rating})
    return jsonify(movie_data)

@app.route('/update-profile', methods=['GET', 'POST'])
def update_profile():
    if 'username' not in session:
        flash("You need to log in to update your profile.")
        return redirect(url_for('login'))
    
    username = session['username']
    user_id = session['user_id']
    error = None
    
    if request.method == 'POST':
        current_password = request.form.get('current_password', '').strip()
        new_username = request.form.get('username', '').strip()
        new_password = request.form.get('new_password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()
        email = request.form.get('email', '').strip()
        
        # Fetch current user data
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE id = ?", (user_id,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            error = "User not found."
            return render_template('update_profile.html', user={'username': username, 'email': email}, error=error)
        
        hashed_password = result[0]
        
        # Verify current password
        if not check_password_hash(hashed_password, current_password):
            error = "Current password is incorrect."
            return render_template('update_profile.html', user={'username': username, 'email': email}, error=error)
        
        # Validate new passwords
        if new_password:
            if new_password != confirm_password:
                error = "New passwords do not match."
                return render_template('update_profile.html', user={'username': username, 'email': email}, error=error)
            new_hashed_password = generate_password_hash(new_password)
        else:
            new_hashed_password = hashed_password  # No change
        
        # Update username if changed
        if new_username and new_username != username:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM users WHERE username = ?", (new_username,))
            if cursor.fetchone():
                error = "Username already exists."
                conn.close()
                return render_template('update_profile.html', user={'username': username, 'email': email}, error=error)
            else:
                try:
                    cursor.execute("UPDATE users SET username = ? WHERE id = ?", (new_username, user_id))
                    conn.commit()
                    conn.close()
                    
                    # Update session and username variable
                    session['username'] = new_username
                    username = new_username
                except Exception as e:
                    error = "An error occurred while updating your username."
                    print(f"Error updating username: {e}")
                    conn.close()
                    return render_template('update_profile.html', user={'username': username, 'email': email}, error=error)
        
        # Update email if changed
        if email:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM users WHERE email = ? AND id != ?", (email, user_id))
            if cursor.fetchone():
                error = "Email already in use."
                conn.close()
                return render_template('update_profile.html', user={'username': username, 'email': email}, error=error)
            else:
                try:
                    cursor.execute("UPDATE users SET email = ? WHERE id = ?", (email, user_id))
                    conn.commit()
                    conn.close()
                except Exception as e:
                    error = "An error occurred while updating your email."
                    print(f"Error updating email: {e}")
                    conn.close()
                    return render_template('update_profile.html', user={'username': username, 'email': email}, error=error)
        
        # Update password if changed
        if new_password:
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("UPDATE users SET password = ? WHERE id = ?", (new_hashed_password, user_id))
                conn.commit()
                conn.close()
            except Exception as e:
                error = "An error occurred while updating your password."
                print(f"Error updating password: {e}")
                return render_template('update_profile.html', user={'username': username, 'email': email}, error=error)
        
        flash("Profile updated successfully!")
        return redirect(url_for('recommendations'))
    
    # Fetch user data
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT email FROM users WHERE id = ?", (user_id,))
    result = cursor.fetchone()
    conn.close()
    
    user_email = result[0] if result else ''
    
    return render_template('update_profile.html', user={'username': username, 'email': user_email}, error=error)

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.")
    return redirect(url_for('welcome'))

# Run the application
if __name__ == '__main__':
    init_db()
    app.run(debug=True, use_reloader=False)
