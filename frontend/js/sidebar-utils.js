async function loadSidebarDetails() {
  const userID = localStorage.getItem("userID");
  const role = localStorage.getItem("role");
  const patientID = localStorage.getItem("patientID");
  const clinicianID = localStorage.getItem("clinicianID");
  const adminID = localStorage.getItem("adminID");

  console.log("Local Storage values:", {
    userID,
    role,
    patientID,
    clinicianID,
    adminID
  });

  if (!userID || !role) {
    alert("No user is logged in!");
    return;
  }

  console.log(`Fetching details for ${role} with ID: ${userID}`); // Debug log

  try {
    let endpoint;
    let id;
    if (role === 'patient') {
      endpoint = `/get_patient/${patientID}`;
      id = patientID;
    } else if (role === 'clinician') {
      endpoint = `/get_clinician/${clinicianID}`;
      id = clinicianID;
    } else if (role === 'administrator') {
      endpoint = `/get_administrator/${adminID}`;
      id = adminID;
    } else {
      console.error("Invalid role:", role);
      return;
    }

    console.log("Making request to:", `http://localhost:8000${endpoint}`);
    const res = await fetch(`http://localhost:8000${endpoint}`);
    console.log(`Response status: ${res.status}`); // Debug log

    if (res.ok) {
      const user = await res.json();
      console.log("Raw user data:", user); // Debug log
      console.log("User data keys:", Object.keys(user)); // Debug log

      // Populate the sidebar with user details
      const nameElement = document.getElementById("sidebar-name");
      const idElement = document.getElementById("sidebar-id");
      const profilePicture = document.getElementById("profile-picture");
      const patientInfo = document.getElementById("patient-info");

      console.log("Found elements:", {
        nameElement: !!nameElement,
        idElement: !!idElement,
        profilePicture: !!profilePicture,
        patientInfo: !!patientInfo
      });

      if (nameElement) {
        nameElement.textContent = `${user.FirstName} ${user.LastName}`;
        console.log("Set name to:", `${user.FirstName} ${user.LastName}`);
      }
      
      if (idElement) {
        if (role === 'patient') {
          idElement.textContent = `Patient ID: ${user.PatientID}`;
        } else if (role === 'clinician') {
          idElement.textContent = `Clinician ID: ${user.ClinicianID}`;
        } else if (role === 'administrator') {
          idElement.textContent = `Admin ID: ${user.AdminID}`;
        }
        console.log("Set ID to:", idElement.textContent);
      }

      if (profilePicture) {
        if (user.Gender === "Male") {
          profilePicture.src = "/frontend/static/images/4.jpg";
        } else if (user.Gender === "Female") {
          profilePicture.src = "/frontend/static/images/9.jpg";
        } else {
          profilePicture.src = "/frontend/static/images/1.jpg"; // Default image
        }
        console.log("Set profile picture to:", profilePicture.src);
      }

      if (patientInfo) {
        if (role === 'patient') {
          patientInfo.innerHTML = `
            <p><strong>Age:</strong> ${user.Age || 'N/A'}</p>
            <p><strong>Gender:</strong> ${user.Gender || 'N/A'}</p>
            <p><strong>Blood Type:</strong> ${user.BloodType || 'N/A'}</p>
            <p><strong>Condition:</strong> ${user.Condition || 'N/A'}</p>
          `;
        } else if (role === 'clinician') {
          patientInfo.innerHTML = `
            <p><strong>License:</strong> ${user.LicenseNumber || 'N/A'}</p>
            <p><strong>Specialization:</strong> ${user.Specialization || 'N/A'}</p>
            <p><strong>Email:</strong> ${user.Email || 'N/A'}</p>
          `;
        } else if (role === 'administrator') {
          patientInfo.innerHTML = `
            <p><strong>Email:</strong> ${user.Email || 'N/A'}</p>
          `;
        }
        console.log("Set patient info to:", patientInfo.innerHTML);
      }
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