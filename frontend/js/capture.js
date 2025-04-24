document.addEventListener("DOMContentLoaded", () => {
    const previewContainer = document.getElementById('sample-preview');
    const screeningResults = document.getElementById('screening-results');
    const fileInput = document.getElementById('sample-upload');
    const startCameraBtn = document.getElementById('start-camera');
    const captureBtn = document.getElementById('capture-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const clearBtn = document.getElementById('clear-sample');
    const video = document.getElementById('camera-stream');
    const canvas = document.getElementById('snapshot-canvas');

    const contactBtn = document.getElementById('contact-pmp-btn');
    const contactModal = document.getElementById('contact-pmp-modal');
    const scheduleModal = document.getElementById('schedule-modal');
    const cancelSchedule = document.getElementById('cancel-schedule');
    const confirmSchedule = document.getElementById('confirm-schedule');
    const logoutBtn = document.getElementById('logout-btn');

    function resetCamera() {
        if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        }
        video.classList.add('hidden');
        document.getElementById('camera-overlay').classList.add('hidden');
    }

    function clearAll() {
        previewContainer.innerHTML = '';
        screeningResults.innerHTML = '';
        fileInput.value = '';
        resetCamera();
        captureBtn.classList.add('hidden');
        analyzeBtn.classList.add('hidden');
        clearBtn.classList.add('hidden');
    }

    function createResultCard(imageSrc, prediction) {
        const card = document.createElement('div');
        card.className = 'bg-white p-4 rounded-lg shadow';
        card.innerHTML = `
        <h3 class="text-lg font-semibold mb-2">Screening Result</h3>
        <div class="flex items-start gap-4">
            <img src="${imageSrc}" class="w-24 h-24 object-cover rounded border" />
            <div class="text-sm space-y-1">
            <p>Estimated Hemoglobin: <strong>${prediction.hgb}</strong> g/dL</p>
            <p>Anemia Classification: <strong class="text-yellow-600">${prediction.class}</strong></p>
            <p>Probability: <strong>${prediction.proba}</strong>%</p>
            <p class="text-xs text-gray-500 italic mt-2">Preliminary result. Confirm with a clinician.</p>
            </div>
        </div>
        <div class="flex gap-4 mt-4">
            <button class="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700">Send to PCP</button>
            <button class="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700" onclick="document.getElementById('schedule-modal').classList.remove('hidden')">Schedule Appointment</button>
        </div>
        `;
        screeningResults.appendChild(card);
    }

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file && file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = function (event) {
            clearAll();
            const img = document.createElement('img');
            img.src = event.target.result;
            img.className = 'w-full h-40 object-cover rounded border shadow';

            previewContainer.innerHTML = '';
            previewContainer.appendChild(img);
            analyzeBtn.classList.remove('hidden');
            clearBtn.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
        fileInput.value = '';
        } else {
        alert('Please upload a valid image file.');
        }
    });

    startCameraBtn.addEventListener('click', () => {
        clearAll();
        navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
        video.srcObject = stream;
        video.classList.remove('hidden');
        captureBtn.classList.remove('hidden');
        clearBtn.classList.remove('hidden');
        document.getElementById('camera-overlay').classList.remove('hidden');
        });
    });

    captureBtn.addEventListener('click', () => {
        previewContainer.innerHTML = '';
        const ctx = canvas.getContext('2d');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);
        const dataUrl = canvas.toDataURL('image/png');
        const img = document.createElement('img');
        img.src = dataUrl;
        img.className = 'w-full h-40 object-cover rounded border shadow';
        previewContainer.appendChild(img);
        analyzeBtn.classList.remove('hidden');
        clearBtn.classList.remove('hidden');
        document.getElementById('camera-overlay').classList.add('hidden');
    });

    analyzeBtn.addEventListener('click', () => {
        const img = previewContainer.querySelector('img');
        if (!img) {
        alert('No image to analyze.');
        return;
        }

        previewContainer.innerHTML = '';
        createResultCard(img.src, {
        hgb: '12.7',
        class: 'Moderate',
        proba: '87'
        });

        analyzeBtn.classList.add('hidden');
    });

    clearBtn.addEventListener('click', clearAll);
    logoutBtn.addEventListener('click', () => window.location.href = 'login.html');
    contactBtn.addEventListener('click', () => contactModal.classList.remove('hidden'));
    cancelSchedule.addEventListener('click', () => scheduleModal.classList.add('hidden'));
    confirmSchedule.addEventListener('click', () => {
        const datetime = document.getElementById('appt-datetime').value;
        if (!datetime) return alert('Please choose date & time.');
        alert('Appointment scheduled for ' + new Date(datetime).toLocaleString());
        scheduleModal.classList.add('hidden');
    });
});