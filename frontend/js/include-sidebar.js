document.addEventListener("DOMContentLoaded", () => {
  const sidebarContainer = document.getElementById("sidebar-container");
  if (sidebarContainer) {
    fetch("/frontend/html/sidebar.html")
      .then(response => response.text())
      .then(data => {
        sidebarContainer.innerHTML = data;

        const script = document.createElement("script");
        script.src = "/frontend/js/sidebar-utils.js"; 
        script.onload = () => {
          if (typeof loadSidebarDetails === "function") {
            loadSidebarDetails();
          } else {
            console.error("loadSidebarDetails function is not defined.");
          }
        };
        document.body.appendChild(script);
      })
      .catch(error => console.error("Error loading sidebar:", error));
  }
});