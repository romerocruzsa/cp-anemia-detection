-- Create tables if they don't exist
CREATE TABLE IF NOT EXISTS Patients (
    PatientID SERIAL PRIMARY KEY,
    FirstName VARCHAR(50) NOT NULL,
    LastName VARCHAR(50) NOT NULL,
    DateOfBirth DATE NOT NULL,
    Gender VARCHAR(10) NOT NULL,
    Email VARCHAR(100) UNIQUE NOT NULL,
    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TYPE upload_status AS ENUM ('Pending', 'Processed', 'Error');
CREATE TYPE anemia_status AS ENUM ('Positive', 'Negative', 'Indeterminate');

CREATE TABLE IF NOT EXISTS ImageUploads (
    ImageID SERIAL PRIMARY KEY,
    PatientID INTEGER NOT NULL,
    ImagePath VARCHAR(255) NOT NULL,
    UploadDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    Status upload_status DEFAULT 'Pending',
    FOREIGN KEY (PatientID) REFERENCES Patients(PatientID) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS AnemiaAnalysis (
    AnalysisID SERIAL PRIMARY KEY,
    ImageID INTEGER NOT NULL,
    AnemiaStatus anemia_status NOT NULL,
    ConfidenceScore DECIMAL(5,4) NOT NULL,
    AnalysisDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (ImageID) REFERENCES ImageUploads(ImageID) ON DELETE CASCADE,
    CONSTRAINT valid_confidence_score CHECK (ConfidenceScore >= 0 AND ConfidenceScore <= 1)
);