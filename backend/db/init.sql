-- Create custom types
CREATE TYPE upload_status AS ENUM ('Pending', 'Processed', 'Error');
CREATE TYPE anemia_status AS ENUM ('Positive', 'Negative', 'Indeterminate');
CREATE TYPE user_role AS ENUM ('patient', 'clinician', 'administrator');
CREATE TYPE audit_action AS ENUM ('view', 'create', 'update', 'delete', 'login', 'logout');

-- Create Users table (base table for authentication)
CREATE TABLE IF NOT EXISTS Users (
    UserID SERIAL PRIMARY KEY,
    Email VARCHAR(100) UNIQUE NOT NULL,
    PasswordHash VARCHAR(255) NOT NULL,
    Role user_role NOT NULL,
    IsActive BOOLEAN DEFAULT true,
    LastLogin TIMESTAMP,
    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UpdatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create Patients table
CREATE TABLE IF NOT EXISTS Patients (
    PatientID SERIAL PRIMARY KEY,
    UserID INTEGER UNIQUE REFERENCES Users(UserID) ON DELETE CASCADE,
    FirstName VARCHAR(50) NOT NULL,
    LastName VARCHAR(50) NOT NULL,
    DateOfBirth DATE NOT NULL,
    Gender VARCHAR(10) NOT NULL,
    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UpdatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create Clinicians table
CREATE TABLE IF NOT EXISTS Clinicians (
    ClinicianID SERIAL PRIMARY KEY,
    UserID INTEGER UNIQUE REFERENCES Users(UserID) ON DELETE CASCADE,
    FirstName VARCHAR(50) NOT NULL,
    LastName VARCHAR(50) NOT NULL,
    LicenseNumber VARCHAR(50) UNIQUE NOT NULL,
    Specialization VARCHAR(100),
    IsActive BOOLEAN DEFAULT true,
    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UpdatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create ImageUploads table
CREATE TABLE IF NOT EXISTS ImageUploads (
    ImageID SERIAL PRIMARY KEY,
    PatientID INTEGER NOT NULL REFERENCES Patients(PatientID) ON DELETE CASCADE,
    ImagePath VARCHAR(255) NOT NULL,
    UploadDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    Status upload_status DEFAULT 'Pending',
    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UpdatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create AnemiaAnalysis table
CREATE TABLE IF NOT EXISTS AnemiaAnalysis (
    AnalysisID SERIAL PRIMARY KEY,
    ImageID INTEGER NOT NULL REFERENCES ImageUploads(ImageID) ON DELETE CASCADE,
    AnemiaStatus anemia_status NOT NULL,
    ConfidenceScore DECIMAL(5,4) NOT NULL,
    AnalysisDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UpdatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_confidence_score CHECK (ConfidenceScore >= 0 AND ConfidenceScore <= 1)
);

-- Create MedicalNotes table
CREATE TABLE IF NOT EXISTS MedicalNotes (
    NoteID SERIAL PRIMARY KEY,
    AnalysisID INTEGER NOT NULL REFERENCES AnemiaAnalysis(AnalysisID) ON DELETE CASCADE,
    ClinicianID INTEGER NOT NULL REFERENCES Clinicians(ClinicianID) ON DELETE CASCADE,
    NoteContent TEXT NOT NULL,
    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UpdatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create AuditRecords table
CREATE TABLE IF NOT EXISTS AuditRecords (
    AuditID SERIAL PRIMARY KEY,
    UserID INTEGER NOT NULL REFERENCES Users(UserID) ON DELETE CASCADE,
    Action audit_action NOT NULL,
    TableName VARCHAR(50) NOT NULL,
    RecordID INTEGER NOT NULL,
    OldValue JSONB,
    NewValue JSONB,
    IPAddress VARCHAR(45),
    Timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_patients_userid ON Patients(UserID);
CREATE INDEX IF NOT EXISTS idx_clinicians_userid ON Clinicians(UserID);
CREATE INDEX IF NOT EXISTS idx_imageuploads_patientid ON ImageUploads(PatientID);
CREATE INDEX IF NOT EXISTS idx_anemiaanalysis_imageid ON AnemiaAnalysis(ImageID);
CREATE INDEX IF NOT EXISTS idx_medicalnotes_analysisid ON MedicalNotes(AnalysisID);
CREATE INDEX IF NOT EXISTS idx_medicalnotes_clinicianid ON MedicalNotes(ClinicianID);
CREATE INDEX IF NOT EXISTS idx_auditrecords_userid ON AuditRecords(UserID);
CREATE INDEX IF NOT EXISTS idx_auditrecords_timestamp ON AuditRecords(Timestamp);

-- Create function to update UpdatedAt timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.UpdatedAt = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for UpdatedAt
CREATE TRIGGER update_patients_updated_at
    BEFORE UPDATE ON Patients
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_clinicians_updated_at
    BEFORE UPDATE ON Clinicians
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_imageuploads_updated_at
    BEFORE UPDATE ON ImageUploads
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_anemiaanalysis_updated_at
    BEFORE UPDATE ON AnemiaAnalysis
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_medicalnotes_updated_at
    BEFORE UPDATE ON MedicalNotes
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();