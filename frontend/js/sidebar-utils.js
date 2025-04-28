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
