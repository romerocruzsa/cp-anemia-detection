document.getElementById('submitButton').addEventListener('click', () => {
    const imageUpload = document.getElementById('imageUpload');
    const predictionStatus = document.getElementById('predictionStatus');
    const hemoglobinLevel = document.getElementById('hemoglobinLevel');

    if (imageUpload.files.length === 0) {
        predictionStatus.textContent = 'Please upload an image.';
        return;
    }

    // Mock Prediction Logic
    predictionStatus.textContent = 'Analyzing...';
    hemoglobinLevel.textContent = '';

    setTimeout(() => {
        // Mock prediction results
        const mockPrediction = {
            status: 'Moderate Anemia',
            hemoglobin: '10.2 g/dL'
        };

        predictionStatus.textContent = `Anemia Status: ${mockPrediction.status}`;
        hemoglobinLevel.textContent = `Predicted Hemoglobin Level: ${mockPrediction.hemoglobin}`;
    }, 2000);  // Simulating a delay for prediction
});
