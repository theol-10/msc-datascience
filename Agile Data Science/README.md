# MOVIEMAGIC

The general idea is to have a web application that gives movie recommendations to signed in users. For this we want to use Streamlit to program everything in Python and use the Spotlight recommender model.

The main pages and logic of the website is as follows:

# 1. Log in page
Users will be prompted for a username and password. These are stored in an Azure(?) database that also contains other information for training our recommender model to give personalised and relevant recommendations to each user.

This page also has sign up option for new users, which we describe further below:

# 2. Sign up page
New users will input some general data that is then stored on the Azure database. The important part here is obtaining initial training data from the user, which basically consists of a list of movies that the user has watched with the ratings he would give to these. 

To make this sign up process a bit more interactive and entertaining, the process of choice and rating of the movies will be performed in a knock-out style tournament, as follows:
1. An initial list of 50-100 movies is displayed to the user in a board format, showing the poster of the movie to make it easy to recognise. The user needs to select 16 movies from here he has already seen: 8 that they like and 8 that they dislike.
2. Then these will be randomly matched against each other on a 1vs1 game format. The user has to rate both out of 5, to decide on the winner.
3. Iterate step two and go past all the phases: quarter finals, semifinals and finals
4. At the end of the tournament all 8 movies are shown in descending order of starts. If there are any draws are ammendmends the user wants to make, they have the option to change their rating at this stage.
5. These ratings and movies are then saved along the user information into the Azure database

# 3. MovieMagic page: get a recommendation
This is the main page of the website, where the customer has to answer 7-8 direct questions about what kind of movie he is interested in finding to then receive a list of recommendations. 

Based on these answers the database of possible outputs the model can show is restricted, to fit only movies within these filters.

Below are the questions that the user will be taken through:

## 1. Time Availability

**How much time do you have to watch something?**
- A. Less than 30 minutes  
- B. 30 minutes to 1 hour  
- C. 1 to 2 hours  
- D. More than 2 hours  

---

## 2. Genre Preferences

**What genre are you in the mood for?**
- A. Action/Adventure  
- B. Comedy  
- C. Drama  
- D. Horror/Thriller  
- E. Sci-Fi/Fantasy  
- F. Documentary  

---

## 3. Release Timeframe

**What release timeframe are you interested in?**
- A. New Releases (past 1-2 years)  
- B. Modern Movies (last 10 years)  
- C. Classics (older than 10 years)  

---

## 4. Language Preferences

**What language do you want the movie to be in?**
- [Dropdown List of Languages]  
- A. No Preference  

---

## 5. Themes

**Are there specific themes you’d like to explore?**
- A. Friendship  
- B. Survival  
- C. Love  
- D. Mystery  
- E. No Preference  

---

## 6. Mood and Vibe

**What kind of vibe are you looking for?**
- A. Relaxing and chill  
- B. Intense and thrilling  
- C. Romantic and heartwarming  
- D. Fun and lighthearted  
- E. Deep and thought-provoking  

---

## 7. Similar Recommendations

**Do you want recommendations similar to movies you’ve already enjoyed?**
- Yes  
- No  

---

# 4. Personal page
Contains all the users ratings of different movies and other analytical data of the users activity in the platform.

Also includes different set of preferences to be modified.