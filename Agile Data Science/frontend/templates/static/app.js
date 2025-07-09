document.getElementById("recommendBtn").addEventListener("click", async () => {
    const movieInput = document.getElementById("movieInput").value;

    const response = await fetch("/recommend", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ favoriteMovie: movieInput }),
    });

    const recommendations = await response.json();
    const recDiv = document.getElementById("recommendations");
    recDiv.innerHTML = recommendations
        .map(
            (movie) => `
            <div class="movie">
                <img src="${movie.poster}" alt="${movie.title}">
                <p>${movie.title} (${movie.genre})</p>
            </div>`
        )
        .join("");
});
