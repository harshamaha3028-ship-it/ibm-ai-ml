// Dashboard JavaScript Controller

document.addEventListener('DOMContentLoaded', () => {
  initDashboard();
});

let clusterChart = null;
let leaderboardChart = null;

async function initDashboard() {
  await checkHealth();
  await loadStatistics();
  await loadClusterChart();
  await loadModelLeaderboard();
  await loadRecentPredictions();
  setupQuickPredictForm();
}

async function checkHealth() {
  try {
    const res = await fetch('/health');
    const data = await res.json();
    const dot = document.getElementById('system-status-dot');
    const text = document.getElementById('system-status-text');
    if (data.status === 'healthy') {
      dot.style.backgroundColor = 'var(--accent-emerald)';
      text.innerText = `Online (${data.database_backend})`;
    }
  } catch (e) {
    const dot = document.getElementById('system-status-dot');
    const text = document.getElementById('system-status-text');
    if (dot) dot.style.backgroundColor = 'var(--accent-rose)';
    if (text) text.innerText = 'API Offline';
  }
}

async function loadStatistics() {
  try {
    const res = await fetch('/statistics');
    const data = await res.json();

    if (data.dataset_summary) {
      document.getElementById('kpi-avg-mpg').innerText = `${data.dataset_summary.average_mpg}`;
    }
    if (data.model_summary) {
      document.getElementById('kpi-best-r2').innerText = `${data.model_summary.r2_score}`;
      document.getElementById('kpi-best-model').innerText = `${data.model_summary.best_model}`;
    }
    if (data.clustering_summary) {
      document.getElementById('kpi-clusters-count').innerText = `${data.clustering_summary.num_profiles}`;
    }
    if (data.application_stats) {
      document.getElementById('kpi-predictions-count').innerText = `${data.application_stats.total_predictions_made}`;
    }
  } catch (e) {
    console.error('Failed to load dashboard statistics:', e);
  }
}

async function loadClusterChart() {
  try {
    const res = await fetch('/clusters');
    const data = await res.json();

    const ctx = document.getElementById('clusterDistChart').getContext('2d');
    const labels = data.clusters.map(c => c.profile_name);
    const counts = data.clusters.map(c => c.characteristics.sample_count || 1);

    const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

    if (clusterChart) clusterChart.destroy();

    clusterChart = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: labels,
        datasets: [{
          data: counts,
          backgroundColor: colors.slice(0, labels.length),
          borderColor: '#111827',
          borderWidth: 2
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'bottom',
            labels: { color: '#94a3b8', boxWidth: 12, padding: 15 }
          }
        },
        cutout: '65%'
      }
    });
  } catch (e) {
    console.error('Error rendering cluster chart:', e);
  }
}

async function loadModelLeaderboard() {
  try {
    const res = await fetch('/model-info');
    const data = await res.json();

    const ctx = document.getElementById('modelLeaderboardChart').getContext('2d');
    const models = Object.keys(data.models_comparison);
    const r2_scores = models.map(m => data.models_comparison[m].test_r2);
    const rmse_scores = models.map(m => data.models_comparison[m].test_rmse);

    if (leaderboardChart) leaderboardChart.destroy();

    leaderboardChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: models.map(m => m.replace(' Regression', '')),
        datasets: [
          {
            label: 'Test R² Score',
            data: r2_scores,
            backgroundColor: '#3b82f6',
            borderRadius: 4
          },
          {
            label: 'Test RMSE (MPG)',
            data: rmse_scores,
            backgroundColor: '#f59e0b',
            borderRadius: 4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: '#94a3b8' }
          },
          y: {
            grid: { color: '#1e293b' },
            ticks: { color: '#94a3b8' }
          }
        },
        plugins: {
          legend: {
            labels: { color: '#94a3b8', boxWidth: 12 }
          }
        }
      }
    });
  } catch (e) {
    console.error('Error rendering leaderboard chart:', e);
  }
}

async function loadRecentPredictions() {
  const tbody = document.getElementById('recent-predictions-tbody');
  try {
    const res = await fetch('/predictions?limit=5');
    const data = await res.json();

    if (!data.items || data.items.length === 0) {
      tbody.innerHTML = `<tr><td colspan="4" class="text-center" style="color: var(--text-muted); padding: 1.5rem;">No predictions recorded yet. Run a vehicle analysis above!</td></tr>`;
      return;
    }

    tbody.innerHTML = data.items.map(item => {
      let badgeClass = 'badge-moderate';
      if (item.efficiency_category.toLowerCase().includes('high')) badgeClass = 'badge-high';
      else if (item.efficiency_category.toLowerCase().includes('low')) badgeClass = 'badge-low';

      return `
        <tr>
          <td>
            <strong>${item.cylinders} Cyl</strong> • ${item.weight} lbs<br>
            <span style="font-size: 0.75rem; color: var(--text-muted);">${item.horsepower}hp, '19${item.model_year} (${item.origin})</span>
          </td>
          <td class="font-mono" style="font-weight: 700; color: var(--text-primary); font-size: 1rem;">
            ${item.predicted_mpg.toFixed(1)} <span style="font-size: 0.75rem; color: var(--text-muted);">MPG</span>
          </td>
          <td><span class="badge ${badgeClass}">${item.efficiency_category}</span></td>
          <td><span class="badge badge-cluster">${item.profile_name}</span></td>
        </tr>
      `;
    }).join('');
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="4" class="text-center" style="color: var(--text-muted);">Failed to load recent activity.</td></tr>`;
  }
}

function setupQuickPredictForm() {
  const form = document.getElementById('quick-predict-form');
  const btn = document.getElementById('quick-btn');
  const resBox = document.getElementById('quick-result');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    btn.disabled = true;
    btn.innerText = 'Calculating...';

    const payload = {
      cylinders: parseInt(document.getElementById('quick-cylinders').value),
      displacement: parseFloat(document.getElementById('quick-disp').value),
      horsepower: parseFloat(document.getElementById('quick-hp').value),
      weight: parseFloat(document.getElementById('quick-weight').value),
      acceleration: 15.5,
      model_year: parseInt(document.getElementById('quick-year').value),
      origin: document.getElementById('quick-origin').value
    };

    try {
      const res = await fetch('/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) throw new Error('Analysis failed');
      const data = await res.json();

      document.getElementById('quick-res-mpg').innerText = data.predicted_mpg.toFixed(1);
      const badge = document.getElementById('quick-res-badge');
      badge.innerText = data.efficiency_category;
      badge.className = 'badge ' + (
        data.efficiency_category.includes('High') ? 'badge-high' :
        data.efficiency_category.includes('Low') ? 'badge-low' : 'badge-moderate'
      );
      document.getElementById('quick-res-profile').innerText = data.vehicle_profile;

      resBox.style.display = 'block';

      // Refresh recent predictions and stats
      loadRecentPredictions();
      loadStatistics();
    } catch (err) {
      alert('Error during prediction: ' + err.message);
    } finally {
      btn.disabled = false;
      btn.innerText = '⚡ Analyze Vehicle Specifications';
    }
  });
}
