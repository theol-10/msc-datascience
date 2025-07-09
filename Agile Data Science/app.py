from flask import Flask, render_template, request, session, redirect, url_for, flash
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['PREFERRED_URL_SCHEME'] = 'http'

# In-memory user database for demonstration purposes
users = {}
user_preferences = {}

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in users and users[username]['password'] == password:
            session['username'] = username
            if username in user_preferences:
                return redirect(url_for('recommendations'))
            else:
                return redirect(url_for('questionnaire'))
        else:
            error = "Invalid username or password. Please try again."
    return render_template('login.html', error=error)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        if username in users:
            error = "User already exists."
        else:
            users[username] = {'email': email, 'password': password}
            return redirect(url_for('login'))
    return render_template('signup.html', error=error)

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')  # Get the email from the form
        for username, info in users.items():
            if info['email'] == email:  # Check if the email exists in the users database
                session['reset_user'] = username  # Store the username in the session
                return redirect(url_for('reset_password'))  # Redirect to reset password page
        # If email is not found, display an error message
        return render_template('forgot_password.html', error="No account found for this email.")
    return render_template('forgot_password.html')  # Render the forgot-password page by default
@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    error = None
    if 'reset_user' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        if new_password == confirm_password:
            users[session['reset_user']]['password'] = new_password
            session.pop('reset_user')
            return redirect(url_for('login'))
        else:
            error = "Passwords do not match. Please try again."
    return render_template('reset_password.html', error=error)

@app.route('/questionnaire', methods=['GET', 'POST'])
def questionnaire():
    if 'username' not in session:
        return redirect(url_for('login'))

    # Questionnaire questions and options
    questions = [
        "What is your preferred movie duration?",
        "What genre do you prefer?",
        "Which actor would you like to see?",
        "What vibe are you looking for?"
    ]
    options = [
        ["90 mins", "120 mins", "150 mins"],
        ["Action", "Romance", "Comedy", "Thriller", "Horror"],
        ["Leonardo DiCaprio", "Scarlett Johansson", "Tom Cruise", "Meryl Streep"],
        ["Exciting", "Romantic", "Chill", "Mysterious", "Scary"]
    ]

    current_question = int(request.args.get('q', 0))
    error = None

    if request.method == 'POST':
        # Collect ratings for each option
        ratings = []
        for i in range(len(options[current_question])):
            rating = request.form.get(f'rating-{i}')
            if rating:
                ratings.append(rating)
            else:
                error = "You need to rate all options to proceed to the next question."
                break

        if not error:
            if 'preferences' not in session:
                session['preferences'] = []
            session['preferences'].append(ratings)

            # Check if it's the last question
            if current_question + 1 < len(questions):
                return redirect(url_for('questionnaire', q=current_question + 1))
            else:
                user_preferences[session['username']] = session['preferences']
                session.pop('preferences')
                return redirect(url_for('recommendations'))

    return render_template(
        'questionnaire.html',
        question=questions[current_question],
        options=options[current_question],
        current_question=current_question + 1,
        total_questions=len(questions),
        error=error
    )

@app.route('/recommendations')
def recommendations():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    sample_recommendations = [
        {
            "title": "Inception",
            "image": "https://m.media-amazon.com/images/M/MV5BMjExMjkwNTQ0Nl5BMl5BanBnXkFtZTcwNTY0OTk1Mw@@._V1_.jpg",
            "description": "A mind-bending thriller by Christopher Nolan."
        },
        {
            "title": "The Grand Budapest Hotel",
            "image": "https://i.ebayimg.com/images/g/r~IAAOSwuwRYLgpm/s-l400.jpg",
            "description": "A whimsical comedy set in a grand European hotel."
        },
        {
            "title": "Interstellar",
            "image": "https://upload.wikimedia.org/wikipedia/en/b/bc/Interstellar_film_poster.jpg",
            "description": "A journey through space and time to save humanity."
        }
    ]
    
    preferences = user_preferences.get(session['username'], [])
    return render_template('recommendations.html', recommendations=sample_recommendations)


def settings():
    if 'username' not in session:
        return redirect(url_for('login'))
    error = None
    if request.method == 'POST':
        username = session['username']
        new_email = request.form.get('email')
        new_username = request.form.get('username')
        if new_username and new_username != username:
            if new_username in users:
                error = "Username already exists."
            else:
                users[new_username] = users.pop(username)
                session['username'] = new_username
        if new_email:
            users[session['username']]['email'] = new_email
        if not error:
            return redirect(url_for('settings'))
    return render_template('settings.html', user=users[session['username']], error=error)

@app.route('/update-profile', methods=['GET', 'POST'])
def update_profile():
    if 'username' not in session:
        return redirect(url_for('login'))

    username = session['username']
    user = users.get(username, {})
    error = None

    if request.method == 'POST':
        # Retrieve form data
        current_password = request.form.get('current_password')
        new_username = request.form.get('username')  # Note: Field renamed to align with requirements
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        name = request.form.get('name', user.get('name', ''))
        middle_name = request.form.get('middle_name', user.get('middle_name', ''))
        last_name = request.form.get('last_name', user.get('last_name', ''))
        birthday = request.form.get('birthday', user.get('birthday', ''))
        nationality = request.form.get('nationality', user.get('nationality', ''))

        # Validate the current password
        if users[username]['password'] != current_password:
            error = "Current password is incorrect."
        elif new_password and new_password != confirm_password:
            error = "New passwords do not match."
        else:
            # Update username if it has changed
            if new_username and new_username != username:
                if new_username in users:
                    error = "Username already exists."
                else:
                    users[new_username] = users.pop(username)
                    session['username'] = new_username
            
            # Update password
            if new_password:
                users[session['username']]['password'] = new_password
            
            # Update other fields
            users[session['username']]['name'] = name
            users[session['username']]['middle_name'] = middle_name
            users[session['username']]['last_name'] = last_name
            users[session['username']]['birthday'] = birthday
            users[session['username']]['nationality'] = nationality

            if not error:
                # Password validation pop-up
                return redirect(url_for('recommendations'))

    return render_template(
        'update_profile.html',
        user=user,
        error=error,
        current_year=datetime.now().year
    )


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('welcome'))

if __name__ == '__main__':
    app.run(debug=True)