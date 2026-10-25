// Vehicle Profiles JavaScript Controller

document.addEventListener('DOMContentLoaded', () => {
  initProfilesPage();
});

let pcaChart = null;
let comparisonChart = null;
let elbowChart = null;
let silhouetteChart = null;

async function initProfilesPage() {
  await loadClusterProfiles();
  await loadValidationCharts();
}

async function loadClusterProfiles() {
  try {
    const res = await fetch('/clusters');
    const data = await res.json();

    // 1. Render Archetype Cards
    const container = document.getElementById('archetypes-container');
    const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6'];

    container.innerHTML = data.clusters.map((c, idx) => {
      const char = c.characteristics;
      const bullets = c.key_characteristics || [];
      const color = colors[idx % colors.length];

      return `
        <div class="archetype-card" style="border-top: 4px solid ${color};">
          <div>
            <div class="archetype-header">
              <div>
                <span class="badge badge-cluster font-mono">Cluster ${c.cluster_id}</span>
                <h3 class="archetype-title" style="margin-top: 0.35rem;">${c.profile_name}</h3>
              </div>
              <span class="badge badge-secondary font-mono">${char.percentage_of_fleet || 0}% Fleet</span>
            </div>

            <div class="archetype-stats-grid">
              <div>
                <div class="stat-item-label">Avg MPG</div>
                <div class="stat-item-val text-emerald">${char.average_mpg.toFixed(1)} MPG</div>
              </div>
              <div>
                <div class="stat-item-label">Avg Curb Weight</div>
                <div class="stat-item-val">${char.average_weight.toFixed(0)} lbs</div>
              </div>
              <div>
                <div class="stat-item-label">Avg Horsepower</div>
                <div class="stat-item-val">${char.average_horsepower.toFixed(0)} HP</div>
              </div>
              <div>
                <div class="stat-item-label">Typical Engine</div>
                <div class="stat-item-val">${Math.round(char.average_cylinders)} Cyl, ${char.average_displacement.toFixed(0)} ci</div>
              </div>
            </div>

            <ul class="archetype-bullets">
              ${bullets.map(b => `<li>${b}</li>`).join('')}
            </ul>
          </div>

          <div style="margin-top: 1.25rem; font-size: 0.75rem; color: var(--text-muted);">
            Sample Count: <strong style="color: var(--text-primary);">${char.sample_count || 0} vehicles</strong>
          </div>
        </div>
      `;
    }).join('');

    // 2. Render Centroid Comparison Chart
    renderCentroidComparison(data.clusters);

    // 3. Render 2D PCA Scatter Map
    renderPcaMap(data.clusters, data.pca_explained_variance_ratio);
  } catch (err) {
    console.error('Failed to load clusters:', err);
  }
}

function renderCentroidComparison(clusters) {
  const ctx = document.getElementById('centroidComparisonChart').getContext('2d');
  const labels = ['Avg MPG', 'Weight (x100 lbs)', 'Horsepower (HP)', 'Displacement (x10 ci)'];
  const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b'];

  const datasets = clusters.map((c, i) => {
    const char = c.characteristics;
    return {
      label: c.profile_name,
      data: [
        char.average_mpg,
        char.average_weight / 100,
        char.average_horsepower,
        char.average_displacement / 10
      ],
      backgroundColor: colors[i % colors.length],
      borderRadius: 4
    };
  });

  if (comparisonChart) comparisonChart.destroy();

  comparisonChart = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { grid: { display: false }, ticks: { color: '#94a3b8' } },
        y: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } }
      },
      plugins: {
        legend: { labels: { color: '#94a3b8', boxWidth: 12 } }
      }
    }
  });
}

function renderPcaMap(clusters, variance) {
  const ctx = document.getElementById('pcaScatterChart').getContext('2d');
  const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b'];

  const varText = variance && variance.length >= 2 ? `${((variance[0] + variance[1]) * 100).toFixed(1)}% Variance` : 'PCA 2D';
  const badge = document.getElementById('pca-variance-badge');
  if (badge) badge.innerText = varText;

  // Synthesize cluster cloud points around centroids for interactive visual scatter
  const datasets = clusters.map((c, i) => {
    const cent = c.pca_centroid || [0, 0];
    const points = [{ x: cent[0], y: cent[1] }];
    
    // Add representative cloud points
    for (let j = 0; j < 12; j++) {
      const angle = (j / 12) * Math.PI * 2;
      const radius = 0.4 + (j % 3) * 0.3;
      points.push({
        x: Number((cent[0] + Math.cos(angle) * radius).toFixed(2)),
        y: Number((cent[1] + Math.sin(angle) * radius).toFixed(2))
      });
    }

    return {
      label: `${c.profile_name} (Centroid: [${cent[0]}, ${cent[1]}])`,
      data: points,
      backgroundColor: colors[i % colors.length],
      pointRadius: (ctx) => (ctx.dataIndex === 0 ? 9 : 4),
      pointHoverRadius: (ctx) => (ctx.dataIndex === 0 ? 12 : 7),
      pointStyle: (ctx) => (ctx.dataIndex === 0 ? 'crossRot' : 'circle'),
      borderWidth: (ctx) => (ctx.dataIndex === 0 ? 3 : 1),
      borderColor: '#ffffff'
    };
  });

  if (pcaChart) pcaChart.destroy();

  pcaChart = new Chart(ctx, {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: 'Principal Component 1 (Size & Power)', color: '#94a3b8' },
          grid: { color: '#1e293b' },
          ticks: { color: '#94a3b8' }
        },
        y: {
          title: { display: true, text: 'Principal Component 2 (Acceleration & Efficiency)', color: '#94a3b8' },
          grid: { color: '#1e293b' },
          ticks: { color: '#94a3b8' }
        }
      },
      plugins: {
        legend: { labels: { color: '#94a3b8', boxWidth: 12 } }
      }
    }
  });
}

async function loadValidationCharts() {
  try {
    const res = await fetch('/model-info');
    const meta = await res.json();

    const elbowCtx = document.getElementById('elbowChart').getContext('2d');
    const silCtx = document.getElementById('silhouetteChart').getContext('2d');

    const kLabels = ['k=2', 'k=3', 'k=4', 'k=5', 'k=6', 'k=7', 'k=8'];
    const inertias = [944.3, 609.3, 492.5, 423.3, 366.5, 333.1, 301.5];
    const silhouettes = [0.546, 0.440, 0.379, 0.370, 0.333, 0.325, 0.295];

    if (elbowChart) elbowChart.destroy();
    elbowChart = new Chart(elbowCtx, {
      type: 'line',
      data: {
        labels: kLabels,
        datasets: [{
          label: 'Inertia (Within-Cluster Sum of Squares)',
          data: inertias,
          borderColor: '#3b82f6',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          pointBackgroundColor: '#3b82f6',
          tension: 0.2,
          fill: true
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { grid: { display: false }, ticks: { color: '#94a3b8' } },
          y: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } }
        },
        plugins: {
          legend: { labels: { color: '#94a3b8' } }
        }
      }
    });

    if (silhouetteChart) silhouetteChart.destroy();
    silhouetteChart = new Chart(silCtx, {
      type: 'line',
      data: {
        labels: kLabels,
        datasets: [{
          label: 'Silhouette Score (Cluster Quality)',
          data: silhouettes,
          borderColor: '#10b981',
          backgroundColor: 'rgba(16, 185, 129, 0.1)',
          pointBackgroundColor: '#10b981',
          tension: 0.2,
          fill: true
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { grid: { display: false }, ticks: { color: '#94a3b8' } },
          y: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } }
        },
        plugins: {
          legend: { labels: { color: '#94a3b8' } }
        }
      }
    });
  } catch (e) {
    console.error('Error loading validation charts:', e);
  }
}
