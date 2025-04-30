document.addEventListener("DOMContentLoaded", () => {
  const sidebarContainer = document.getElementById("sidebar-container");

  if (sidebarContainer) {
    fetch("/frontend/html/sidebar.html")
      .then(response => response.text())
      .then(data => {
        sidebarContainer.innerHTML = data;

        // Now define the results array globally
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

        const elements = {
          moreInfoBtn: document.getElementById('more-info-btn'),
          logoutBtn: document.getElementById('logout-btn'),
          contactModal: document.getElementById('contact-pmp-modal'),
          contactModalContent: document.getElementById('contact-modal-content'),
          contactPcpBtn: document.getElementById('contact-pmp-btn'),
          resultsTable: document.getElementById('results-table'),
          searchInput: document.getElementById('search-input'),
          anemiaFilter: document.getElementById('anemia-filter'),
          dateFilter: document.getElementById('date-filter'),
          sortBy: document.getElementById('sort-by'),
          applyFiltersBtn: document.getElementById('apply-filters'),
          clearFiltersBtn: document.getElementById('clear-filters'),
          resultCount: document.getElementById('result-count')
        };

        // --- Event Listeners --- //
        if (elements.moreInfoBtn) {
          elements.moreInfoBtn.addEventListener('click', togglePatientInfo);
        }

        if (elements.logoutBtn) {
          elements.logoutBtn.addEventListener('click', () => redirectTo('login.html'));
        }

        if (elements.contactPcpBtn) {
          elements.contactPcpBtn.addEventListener('click', openContactModal);
        }

        if (elements.applyFiltersBtn) {
          elements.applyFiltersBtn.addEventListener('click', renderFilteredResults);
        }

        if (elements.clearFiltersBtn) {
          elements.clearFiltersBtn.addEventListener('click', clearFilters);
        }

        // --- Functions --- //

        function togglePatientInfo() {
          document.getElementById('patient-info').classList.toggle('hidden');
        }

        function redirectTo(url) {
          window.location.href = url;
        }

        function openContactModal() {
          elements.contactModal.classList.remove('hidden');
          setTimeout(() => {
            elements.contactModal.classList.add('opacity-100');
            elements.contactModalContent.classList.replace('scale-95', 'scale-100');
            elements.contactModalContent.classList.replace('opacity-0', 'opacity-100');
          }, 10);
        }

        function closeContactModal() {
          elements.contactModalContent.classList.replace('scale-100', 'scale-95');
          elements.contactModalContent.classList.replace('opacity-100', 'opacity-0');
          elements.contactModal.classList.remove('opacity-100');
          setTimeout(() => {
            elements.contactModal.classList.add('hidden');
          }, 300);
        }

        function clearFilters() {
          elements.searchInput.value = '';
          elements.anemiaFilter.value = '';
          elements.dateFilter.value = '';
          elements.sortBy.value = 'date-asc';
          renderFilteredResults();
        }

        function renderFilteredResults() {
          const query = elements.searchInput.value.toLowerCase();
          const anemia = elements.anemiaFilter.value;
          const date = elements.dateFilter.value;
          const sort = elements.sortBy.value;

          let filtered = results.filter(r => {
            const matchesQuery = Object.values(r).some(val => String(val).toLowerCase().includes(query));
            const matchesAnemia = anemia ? r.anemia === anemia : true;
            const matchesDate = date ? r.date === date : true;
            return matchesQuery && matchesAnemia && matchesDate;
          });

          sortResults(filtered, sort);
          displayResults(filtered);
        }

        function sortResults(array, sortKey) {
          array.sort((a, b) => {
            if (sortKey === 'date-asc') return new Date(a.date) - new Date(b.date);
            if (sortKey === 'date-desc') return new Date(b.date) - new Date(a.date);
            if (sortKey === 'hb-asc') return parseFloat(a.hb) - parseFloat(b.hb);
            if (sortKey === 'hb-desc') return parseFloat(b.hb) - parseFloat(a.hb);
            return 0;
          });
        }

        function displayResults(filtered) {
          elements.resultsTable.innerHTML = '';

          filtered.forEach(r => {
            const row = createRow(r);
            const expand = createExpandRow(r);

            row.addEventListener('click', () => expand.classList.toggle('hidden'));

            elements.resultsTable.appendChild(row);
            elements.resultsTable.appendChild(expand);
          });

          elements.resultCount.textContent = `${filtered.length} result${filtered.length !== 1 ? 's' : ''} shown`;
        }

        function createRow(r) {
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
          return row;
        }

        function createExpandRow(r) {
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
          return expand;
        }

        // --- Initialize --- //
        renderFilteredResults();
      })
      .catch(error => {
        console.error("Error loading sidebar:", error);
      });
  } else {
    console.error("Error: Sidebar container not found");
  }
});
