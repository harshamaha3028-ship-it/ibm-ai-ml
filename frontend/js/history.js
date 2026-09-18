// Prediction History JavaScript Controller

let currentPage = 1;
const limit = 15;
let currentTotal = 0;
let totalPages = 1;
let currentRecords = [];

document.addEventListener('DOMContentLoaded', () => {
  setupFilterListeners();
  loadHistory();
});

function setupFilterListeners() {
  const searchInput = document.getElementById('filter-search');
  let debounceTimeout = null;

  searchInput.addEventListener('input', () => {
    clearTimeout(debounceTimeout);
    debounceTimeout = setTimeout(() => {
      currentPage = 1;
      loadHistory();
    }, 300);
  });

  document.getElementById('filter-efficiency').addEventListener('change', () => {
    currentPage = 1;
    loadHistory();
  });

  document.getElementById('filter-profile').addEventListener('change', () => {
    currentPage = 1;
    loadHistory();
  });

  document.getElementById('filter-sort').addEventListener('change', () => {
    currentPage = 1;
    loadHistory();
  });

  document.getElementById('filter-order').addEventListener('change', () => {
    currentPage = 1;
    loadHistory();
  });
}

async function loadHistory() {
  const tbody = document.getElementById('history-table-tbody');
  
  const search = document.getElementById('filter-search').value.trim();
  const efficiency = document.getElementById('filter-efficiency').value;
  const profile = document.getElementById('filter-profile').value;
  const sortBy = document.getElementById('filter-sort').value;
  const order = document.getElementById('filter-order').value;

  const params = new URLSearchParams({
    page: currentPage,
    limit: limit,
    sort_by: sortBy,
    order: order
  });

  if (search) params.append('search', search);
  if (efficiency) params.append('efficiency', efficiency);
  if (profile) params.append('profile', profile);

  try {
    const res = await fetch(`/predictions?${params.toString()}`);
    const data = await res.json();

    currentRecords = data.items || [];
    currentTotal = data.total || 0;
    totalPages = data.total_pages || 1;

    if (currentRecords.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" class="text-center" style="color: var(--text-muted); padding: 2rem;">No historical predictions found matching criteria.</td></tr>`;
    } else {
      tbody.innerHTML = currentRecords.map(r => {
        let badgeClass = 'badge-moderate';
        if (r.efficiency_category.includes('High')) badgeClass = 'badge-high';
        else if (r.efficiency_category.includes('Low')) badgeClass = 'badge-low';

        return `
          <tr>
            <td class="font-mono" style="color: var(--text-muted);">#${r.id}</td>
            <td style="font-size: 0.8rem; color: var(--text-secondary); white-space: nowrap;">${r.created_at || 'Just now'}</td>
            <td>
              <strong>${r.cylinders} Cyl</strong> • ${r.displacement} ci • ${r.horsepower} hp • ${r.weight} lbs
              <div style="font-size: 0.75rem; color: var(--text-muted);">0-60 in ${r.acceleration}s • Year '19${r.model_year}</div>
            </td>
            <td><span class="badge badge-secondary font-mono">${r.origin}</span></td>
            <td class="font-mono" style="font-size: 1.05rem; font-weight: 800; color: var(--text-primary);">
              ${r.predicted_mpg.toFixed(1)} <span style="font-size: 0.75rem; color: var(--text-muted);">MPG</span>
            </td>
            <td><span class="badge ${badgeClass}">${r.efficiency_category}</span></td>
            <td><span class="badge badge-cluster">${r.profile_name}</span></td>
          </tr>
        `;
      }).join('');
    }

    // Update Pagination UI
    const startIdx = currentTotal === 0 ? 0 : (currentPage - 1) * limit + 1;
    const endIdx = Math.min(currentPage * limit, currentTotal);
    document.getElementById('page-records-range').innerText = `${startIdx}-${endIdx}`;
    document.getElementById('page-total-records').innerText = `${currentTotal}`;
    document.getElementById('page-indicator').innerText = `Page ${currentPage} of ${totalPages}`;

    document.getElementById('btn-prev-page').disabled = (currentPage <= 1);
    document.getElementById('btn-next-page').disabled = (currentPage >= totalPages);
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="7" class="text-center" style="color: var(--accent-rose); padding: 2rem;">Error querying predictions history: ${err.message}</td></tr>`;
  }
}

function changePage(delta) {
  const target = currentPage + delta;
  if (target >= 1 && target <= totalPages) {
    currentPage = target;
    loadHistory();
  }
}

async function confirmClearHistory() {
  if (confirm("Are you sure you want to clear all stored predictions from the database? This action cannot be undone.")) {
    try {
      const res = await fetch('/predictions', { method: 'DELETE' });
      if (res.ok) {
        currentPage = 1;
        loadHistory();
      }
    } catch (e) {
      alert("Failed to clear history: " + e.message);
    }
  }
}

function exportData(format) {
  if (currentRecords.length === 0) {
    alert("No records to export.");
    return;
  }

  if (format === 'json') {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(currentRecords, null, 2));
    const dlAnchor = document.createElement('a');
    dlAnchor.setAttribute("href", dataStr);
    dlAnchor.setAttribute("download", `automobile_predictions_${Date.now()}.json`);
    dlAnchor.click();
  } else if (format === 'csv') {
    const headers = ['id', 'created_at', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'predicted_mpg', 'efficiency_category', 'cluster_id', 'profile_name'];
    const rows = currentRecords.map(r => headers.map(h => `"${r[h] !== undefined ? r[h] : ''}"`).join(','));
    const csvContent = "data:text/csv;charset=utf-8," + [headers.join(','), ...rows].join('\n');
    const encodedUri = encodeURI(csvContent);
    const dlAnchor = document.createElement('a');
    dlAnchor.setAttribute("href", encodedUri);
    dlAnchor.setAttribute("download", `automobile_predictions_${Date.now()}.csv`);
    dlAnchor.click();
  }
}
