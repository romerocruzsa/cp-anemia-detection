document.addEventListener("DOMContentLoaded", () => {

    document.getElementById('more-info-btn').addEventListener('click', () => {
      const info = document.getElementById('patient-info');
      info.classList.toggle('hidden');
    });
    
    const logoutBtn = document.getElementById('logout-btn');
    logoutBtn.addEventListener('click', () => {
      window.location.href = 'login.html';
    });
    
    const contactModal = document.getElementById('contact-pmp-modal');
    const contactModalContent = document.getElementById('contact-modal-content');
    const contactPcpBtn = document.getElementById('contact-pmp-btn');

    function openContactModal() {
      contactModal.classList.remove('hidden');
      setTimeout(() => {
        contactModal.classList.add('opacity-100');
        contactModalContent.classList.remove('scale-95', 'opacity-0');
        contactModalContent.classList.add('scale-100', 'opacity-100');
      }, 10);
    }

    function closeContactModal() {
      contactModalContent.classList.remove('scale-100', 'opacity-100');
      contactModalContent.classList.add('scale-95', 'opacity-0');
      contactModal.classList.remove('opacity-100');
      setTimeout(() => {
        contactModal.classList.add('hidden');
      }, 300);
    }

    contactPcpBtn.addEventListener('click', openContactModal);
    
    const results = [
      {
        patientId: '789087', gender: 'Female', age: 32,
        date: '2024-04-05', time: '09:15 AM', hb: '13.1', anemia: 'No',
        doctor: 'Dr. Santos', hospital: 'San Juan Medical Center',
        note: 'Normal range. Continue regular checkups.', steps: 'No further action required.', sample: '/frontend/static/images/sample_img2.png'
      },
      {
        patientId: '789087', gender: 'Female', age: 32,
        date: '2024-04-01', time: '01:30 PM', hb: '7.2', anemia: 'Severe',
        doctor: 'Dr. Vega', hospital: 'Bayamón General Hospital',
        note: 'Critical level detected.', steps: 'Refer to hematologist immediately.', sample: '/frontend/static/images/sample_img1.png'
      }
    ];

    const resultsTable = document.getElementById('results-table');
    const searchInput = document.getElementById('search-input');
    const anemiaFilter = document.getElementById('anemia-filter');
    const dateFilter = document.getElementById('date-filter');
    const sortBy = document.getElementById('sort-by');
    const applyFiltersBtn = document.getElementById('apply-filters');
    const clearFiltersBtn = document.getElementById('clear-filters');
    const resultCount = document.getElementById('result-count');

    applyFiltersBtn.addEventListener('click', renderFilteredResults);
    clearFiltersBtn.addEventListener('click', () => {
      searchInput.value = '';
      anemiaFilter.value = '';
      dateFilter.value = '';
      sortBy.value = 'date-asc';
      renderFilteredResults();
    });

    function renderFilteredResults() {
      const query = searchInput.value.toLowerCase();
      const anemia = anemiaFilter.value;
      const date = dateFilter.value;
      const sort = sortBy.value;

      let filtered = results.filter(r => {
        const matchesQuery = Object.values(r).some(val => String(val).toLowerCase().includes(query));
        const matchesAnemia = anemia ? r.anemia === anemia : true;
        const matchesDate = date ? r.date === date : true;
        return matchesQuery && matchesAnemia && matchesDate;
      });

      filtered.sort((a, b) => {
        if (sort === 'date-asc') return new Date(a.date) - new Date(b.date);
        if (sort === 'date-desc') return new Date(b.date) - new Date(a.date);
        if (sort === 'hb-asc') return parseFloat(a.hb) - parseFloat(b.hb);
        if (sort === 'hb-desc') return parseFloat(b.hb) - parseFloat(a.hb);
        return 0;
      });

      resultsTable.innerHTML = '';

      filtered.forEach(r => {
        const row = document.createElement('tr');
        row.className = 'hover:bg-gray-50 cursor-pointer';
        row.innerHTML = `
          <td class="py-2 px-4"><img src="${r.sample}" class="w-10 h-10 rounded object-cover" onerror="this.style.display='none'" /></td>
          <td class="py-2 px-4">${r.patientId}</td>
          <td class="py-2 px-4">${r.date}</td>
          <td class="py-2 px-4">${r.time}</td>
          <td class="py-2 px-4">${r.hb}</td>
          <td class="py-2 px-4">${r.anemia}</td>
          <td class="py-2 px-4">${r.doctor}</td>
          <td class="py-2 px-4">${r.hospital}</td>`;

        const expand = document.createElement('tr');
        expand.className = 'expand-details hidden';
        expand.innerHTML = `
          <td colspan="8" class="p-4 bg-gray-50">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div class="flex flex-col items-center">
                <img src="${r.sample}" class="rounded shadow-md max-w-full" onerror="this.style.display='none'" />
                <p class="text-xs text-gray-400 mt-2">Sample evaluated</p>
              </div>
              <div class="text-sm text-gray-800 space-y-2">
                <p><strong>Notes:</strong> ${r.note}</p>
                <p><strong>Recommendation:</strong> ${r.steps}</p>
                <p class="text-xs text-gray-500 italic">Reference Range: Normal hemoglobin level is 12–16 g/dL for females, 14–18 g/dL for males.</p>
              </div>
            </div>
          </td>`;

        row.addEventListener('click', () => expand.classList.toggle('hidden'));
        resultsTable.appendChild(row);
        resultsTable.appendChild(expand);
      });

      resultCount.textContent = `${filtered.length} result${filtered.length !== 1 ? 's' : ''} shown`;
    }

    renderFilteredResults();
});