document.addEventListener("DOMContentLoaded", () => {
  const sidebarContainer = document.getElementById("sidebar-container");
  if (sidebarContainer) {
    fetch("/frontend/html/sidebar.html")
      .then(response => response.text())
      .then(data => {
        sidebarContainer.innerHTML = data;
      })
      .catch(error => console.error("Error loading sidebar:", error));
  }
});
