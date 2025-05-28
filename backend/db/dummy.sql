-- Insert dummy users
INSERT INTO Users (Email, PasswordHash, Role, IsActive, LastLogin) VALUES
('john.doe@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBAQ.3J5J5qK8i', 'patient', true, CURRENT_TIMESTAMP),
('jane.smith@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBAQ.3J5J5qK8i', 'patient', true, CURRENT_TIMESTAMP),
('dr.wilson@clinic.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBAQ.3J5J5qK8i', 'clinician', true, CURRENT_TIMESTAMP),
('dr.brown@clinic.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBAQ.3J5J5qK8i', 'clinician', true, CURRENT_TIMESTAMP),
('admin@clinic.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBAQ.3J5J5qK8i', 'administrator', true, CURRENT_TIMESTAMP),
('super.admin@clinic.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBAQ.3J5J5qK8i', 'administrator', true, CURRENT_TIMESTAMP);

-- Insert dummy patients
INSERT INTO Patients (UserID, FirstName, LastName, DateOfBirth, Gender) VALUES
(1, 'John', 'Doe', '1985-06-15', 'Male'),
(2, 'Jane', 'Smith', '1990-03-22', 'Female');

-- Insert dummy clinicians
INSERT INTO Clinicians (UserID, FirstName, LastName, LicenseNumber, Specialization) VALUES
(3, 'Robert', 'Wilson', 'MD123456', 'General Medicine'),
(4, 'Sarah', 'Brown', 'MD789012', 'Hematology');

-- Insert dummy administrators
INSERT INTO Administrators (UserID, FirstName, LastName) VALUES
(5, 'Michael', 'Johnson'),
(6, 'Emily', 'Davis');

-- Insert dummy image uploads
INSERT INTO ImageUploads (PatientID, ImagePath, Status) VALUES
(1, '/uploads/patient1_image1.jpg', 'Processed'),
(1, '/uploads/patient1_image2.jpg', 'Processed'),
(2, '/uploads/patient2_image1.jpg', 'Processed'),
(2, '/uploads/patient2_image2.jpg', 'Pending');

-- Insert dummy anemia analysis
INSERT INTO AnemiaAnalysis (ImageID, AnemiaStatus, ConfidenceScore) VALUES
(1, 'Positive', 0.95),
(2, 'Negative', 0.88),
(3, 'Positive', 0.92),
(4, 'Indeterminate', 0.65);

-- Insert dummy medical notes
INSERT INTO MedicalNotes (AnalysisID, ClinicianID, NoteContent) VALUES
(1, 1, 'Patient shows clear signs of anemia. Recommended iron supplements and follow-up in 2 weeks.'),
(2, 1, 'No signs of anemia detected. Patient advised to maintain current diet.'),
(3, 2, 'Moderate anemia detected. Prescribed iron supplements and scheduled follow-up in 1 month.');

-- Insert dummy audit records
INSERT INTO AuditRecords (UserID, Action, TableName, RecordID, IPAddress) VALUES
(1, 'login', 'Users', 1, '192.168.1.100'),
(3, 'create', 'MedicalNotes', 1, '192.168.1.101'),
(4, 'create', 'MedicalNotes', 2, '192.168.1.102'),
(5, 'view', 'Patients', 1, '192.168.1.103');

-- Note: All passwords in this dummy data are hashed versions of 'password123'
-- The hash shown is just an example and should be replaced with actual bcrypt hashes in production
