import altair as alt
import pandas as pd
import streamlit as st

# Show the page title and description.
st.set_page_config(page_title="Movies dataset", page_icon="🎬")
st.title("🎬 Moviemagic")
st.write(
    """
    Here are some movie recommendations based on the answers you provided!  """
)


# Load the data from a CSV. We're caching this so it doesn't reload every time the app
# reruns (e.g. if the user interacts with the widgets).
@st.cache_data
def load_data():
    df = pd.read_csv("data/movie_metadata.csv")
    return df


df = load_data()

# Show a multiselect widget with the genres using `st.multiselect`.
#genres = st.multiselect(
 #   "Genres",
  #  df.genres.unique(),
   # ["Action", "Adventure","Animation", "Biography", "Comedy", "Crime","Drama", "Family","Fantasy","History","Horror","Music","Mystery","Romance","Sci-Fi","Thriller","War"],
#)

# Show a slider widget with the years using `st.slider`.
years = st.slider("Years", 1986, 2006, (2000, 2016))

# Filter the dataframe based on the widget input and reshape it.
# df_filtered = df[(df["genres"].isin(genres)) & (df["year"].between(years[0], years[1]))]
df_filtered = (df["title_year"].between(years[0], years[1]))

#df_reshaped = df_filtered.pivot_table(
 #   index="year", columns="genre", values="movie_title"
#)
#df_reshaped = df_reshaped.sort_values(by="year", ascending=False)


# Display the data as a table using `st.dataframe`.
st.dataframe(
    df_filtered,
    use_container_width=True,
    column_config={"movie_title": st.column_config.TextColumn("movie title")},
)


