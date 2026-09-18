// Model Analytics JavaScript Controller

document.addEventListener('DOMContentLoaded', () => {
  initAnalyticsPage();
});

let impactChart = null;

async function initAnalyticsPage() {
  await loadModelDiagnostics();
}

async function loadModelDiagnostics() {
  try {
    const res = await fetch('/model-info');
    const data = await res.json();

    // 1. Populate Leaderboard Table
    const tbody = document.getElementById('models-table-tbody');
    const comp = data.models_comparison;
    const bestModel = data.best_model;

    tbody.innerHTML = Object.keys(comp).map(mName => {
      const m = comp[mName];
      const isBest = (mName === bestModel);
      const rowStyle = isBest ? 'background-color: rgba(59, 130, 246, 0.08); font-weight: 600;' : '';
      const bestBadge = isBest ? ' <span class="badge badge-high" style="font-size: 0.65rem;">Selected Best</span>' : '';

      let paramStr = 'None (Standard OLS)';
      if (m.params && Object.keys(m.params).length > 0) {
        paramStr = Object.entries(m.params).map(([k, v]) => `${k}=${Number(v).toFixed(4)}`).join(', ');
      }

      return `
        <tr style="${rowStyle}">
          <td><strong>${mName}</strong>${bestBadge}</td>
          <td class="font-mono">${m.train_r2.toFixed(4)}</td>
          <td class="font-mono">${m.cv_r2_mean.toFixed(4)} <span style="font-size: 0.75rem; color: var(--text-muted);">(±${m.cv_r2_std.toFixed(4)})</span></td>
          <td class="font-mono text-emerald">${m.test_r2.toFixed(4)}</td>
          <td class="font-mono">${m.test_rmse.toFixed(4)}</td>
          <td class="font-mono">${m.test_mae.toFixed(4)}</td>
          <td class="font-mono" style="font-size: 0.75rem; color: var(--accent-cyan);">${paramStr}</td>
        </tr>
      `;
    }).join('');

    // 2. Best Model Badge & Summary
    document.getElementById('badge-best-model-tag').innerText = bestModel;
    if (comp[bestModel]) {
      document.getElementById('best-test-r2').innerText = comp[bestModel].test_r2.toFixed(4);
      document.getElementById('best-test-rmse').innerText = `${comp[bestModel].test_rmse.toFixed(2)} MPG`;
    }

    // 3. Render Feature Impact Chart
    renderFeatureImpactChart(data.feature_importances);
  } catch (err) {
    console.error('Failed to load model diagnostics:', err);
  }
}

function renderFeatureImpactChart(impacts) {
  const ctx = document.getElementById('featureImpactChart').getContext('2d');
  
  // Sort features in reverse for top-to-bottom bar display
  const sorted = [...impacts].reverse();
  const labels = sorted.map(i => formatFeatName(i.feature));
  const coefs = sorted.map(i => i.coefficient);
  const colors = coefs.map(c => (c >= 0 ? '#10b981' : '#ef4444'));

  if (impactChart) impactChart.destroy();

  impactChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Standardized Coefficient Impact',
        data: coefs,
        backgroundColor: colors,
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: 'Standardized Coefficient Magnitude (MPG)', color: '#94a3b8' },
          grid: { color: '#1e293b' },
          ticks: { color: '#94a3b8' }
        },
        y: {
          grid: { display: false },
          ticks: { color: '#f8fafc', font: { size: 11, weight: '500' } }
        }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });
}

function formatFeatName(feat) {
  const map = {
    'cylinders': 'Cylinders',
    'displacement': 'Displacement',
    'horsepower': 'Horsepower',
    'weight': 'Curb Weight',
    'acceleration': 'Acceleration',
    'model_year': 'Model Year',
    'power_to_weight': 'Power / Weight',
    'displacement_per_cylinder': 'Displacement / Cyl',
    'weight_per_cylinder': 'Weight / Cyl',
    'vehicle_age': 'Vehicle Age',
    'origin_usa': 'Origin: USA',
    'origin_europe': 'Origin: Europe',
    'origin_japan': 'Origin: Japan'
  };
  return map[feat] || feat;
}
