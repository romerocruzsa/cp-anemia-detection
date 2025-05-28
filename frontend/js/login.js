async function handleLogin(event) {
  event.preventDefault();
  
  const email = document.getElementById('email').value;
  const password = document.getElementById('password').value;

  try {
    const response = await fetch('http://localhost:8000/login', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ email, password }),
    });

    const data = await response.json();
    console.log("Login response:", data); // Debug log

    if (response.ok) {
      // Store user information in localStorage
      localStorage.setItem('userID', data.UserID);
      localStorage.setItem('role', data.Role);
      
      // Store role-specific ID
      if (data.Role === 'patient') {
        localStorage.setItem('patientID', data.PatientID);
      } else if (data.Role === 'clinician') {
        localStorage.setItem('clinicianID', data.ClinicianID);
      } else if (data.Role === 'administrator') {
        localStorage.setItem('adminID', data.AdminID);
      }

      console.log("Stored in localStorage:", {
        userID: localStorage.getItem('userID'),
        role: localStorage.getItem('role'),
        patientID: localStorage.getItem('patientID'),
        clinicianID: localStorage.getItem('clinicianID'),
        adminID: localStorage.getItem('adminID')
      });

      // Redirect based on role
      if (data.Role === 'patient') {
        window.location.href = '/frontend/html/capture.html';
      } else {
        // Both clinicians and administrators go to the main dashboard
        window.location.href = '/frontend/html/dashboard.html';
      }
    } else {
      alert(data.detail || 'Login failed');
    }
  } catch (error) {
    console.error('Error:', error);
    alert('An error occurred during login');
  }
} 