document.addEventListener("DOMContentLoaded", () => {
  const moreInfoBtn = document.getElementById("more-info-btn");
  const patientInfo = document.getElementById("patient-info");

  if (moreInfoBtn && patientInfo) {
    moreInfoBtn.addEventListener("click", () => {
      patientInfo.classList.toggle("hidden");
      moreInfoBtn.textContent = patientInfo.classList.contains("hidden")
        ? "More patient info"
        : "Hide patient info";
    });
  }
});

async function loadSidebarDetails() {
  console.log("Loading sidebar details..."); // Debug log
  const patientID = localStorage.getItem("patientID");
  if (!patientID) {
    alert("No user is logged in!");
    return;
  }

  console.log(`Fetching details for patientID: ${patientID}`); // Debug log

  try {
    const res = await fetch(`http://localhost:8000/get_patient/${patientID}`);
    console.log(`Response status: ${res.status}`); // Debug log

    if (res.ok) {
      const user = await res.json();
      console.log("User data:", user); // Debug log

      // Populate the sidebar with user details
      document.getElementById("sidebar-name").textContent = `${user.FirstName} ${user.LastName}`;
      document.getElementById("sidebar-patient-id").textContent = `Patient ID: ${user.PatientID}`;
    } else {
      const err = await res.json();
      console.error("Error response:", err); // Debug log
      alert(`Failed to load user details: ${err.detail || err.error}`);
    }
  } catch (error) {
    console.error("Network error:", error); // Debug log
    alert("Network error – is the API running on port 8000?");
  }
}