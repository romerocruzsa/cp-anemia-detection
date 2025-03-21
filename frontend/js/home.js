function logout() {
    localStorage.removeItem("isLoggedIn");
    localStorage.removeItem("username");
    localStorage.removeItem("userId");
    localStorage.removeItem("isAdmin"); // Ensure admin login is also cleared
    alert("You have been logged out.");
    window.location.href = "login.html";
}

window.onload = function () {
    if (localStorage.getItem("isLoggedIn") !== "true" && localStorage.getItem("isAdmin") !== "true") {
        localStorage.removeItem("isLoggedIn");
        localStorage.removeItem("username");
        localStorage.removeItem("userId");
        localStorage.removeItem("isAdmin"); // Extra safety for admin session
        window.location.href = "login.html";
    } else {
        const userName = localStorage.getItem("username") || "Guest";
        const userId = localStorage.getItem("userId") || "####";
        
        if (document.getElementById("user-name")) {
            document.getElementById("user-name").textContent = userName;
        }
        if (document.getElementById("user-id")) {
            document.getElementById("user-id").textContent = "ID: " + userId;
        }
    }
}