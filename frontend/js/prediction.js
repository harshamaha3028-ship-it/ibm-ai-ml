// Prediction Studio JavaScript Controller

const presets = {
  compact: {
    name: 'Honda Civic / Toyota Corolla',
    cylinders: 4,
    displacement: 98,
    horsepower: 68,
    weight: 2050,
    acceleration: 17.0,
    model_year: 80,
    origin: '3'
  },
  sedan: {
    name: 'Chevrolet Malibu / Ford Torino',
    cylinders: 6,
    displacement: 231,
    horsepower: 110,
    weight: 3300,
    acceleration: 15.2,
    model_year: 76,
    origin: '1'
  },
  v8: {
    name: 'Pontiac Grand Prix / Dodge Charger',
    cylinders: 8,
    displacement: 350,
    horsepower: 180,
    weight: 4250,
    acceleration: 12.0,
    model_year: 73,
    origin: '1'
  },
  euro: {
    name: 'Volkswagen Golf / BMW 320i',
    cylinders: 4,
    displacement: 121,
    horsepower: 90,
    weight: 2450,
    acceleration: 14.5,
    model_year: 79,
    origin: '2'
  }
};

document.addEventListener('DOMContentLoaded', () => {
  setupStudioForm();
  // Run initial test prediction on page load
  runAnalysis();
});

function loadPreset(key) {
  const p = presets[key];
  if (!p) return;

  document.getElementById('input-car-name').value = p.name;
  document.getElementById('input-cylinders').value = p.cylinders;
  document.getElementById('input-displacement').value = p.displacement;
  document.getElementById('input-horsepower').value = p.horsepower;
  document.getElementById('input-weight').value = p.weight;
  document.getElementById('input-acceleration').value = p.acceleration;
  document.getElementById('input-model-year').value = p.model_year;
  document.getElementById('input-origin').value = p.origin;

  runAnalysis();
}

function resetStudioForm() {
  document.getElementById('prediction-studio-form').reset();
  runAnalysis();
}

function setupStudioForm() {
  const form = document.getElementById('prediction-studio-form');
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    runAnalysis();
  });
}

async function runAnalysis() {
  const btn = document.getElementById('studio-submit-btn');
  btn.disabled = true;
  btn.innerText = 'Analyzing Specifications...';

  const payload = {
    car_name: document.getElementById('input-car-name').value || 'Custom Vehicle',
    cylinders: parseInt(document.getElementById('input-cylinders').value),
    displacement: parseFloat(document.getElementById('input-displacement').value),
    horsepower: parseFloat(document.getElementById('input-horsepower').value),
    weight: parseFloat(document.getElementById('input-weight').value),
    acceleration: parseFloat(document.getElementById('input-acceleration').value),
    model_year: parseInt(document.getElementById('input-model-year').value),
    origin: parseInt(document.getElementById('input-origin').value)
  };

  try {
    const res = await fetch('/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.message || 'Analysis failed');
    }

    const data = await res.json();
    renderAnalysisResults(data, payload);
  } catch (err) {
    alert('Prediction Error: ' + err.message);
  } finally {
    btn.disabled = false;
    btn.innerText = '⚡ Run AI Regression & Profiling';
  }
}

function renderAnalysisResults(data, inputSpecs) {
  // 1. Dial Update
  const mpgVal = data.predicted_mpg;
  document.getElementById('res-mpg-value').innerText = mpgVal.toFixed(1);

  const circle = document.getElementById('mpg-dial-circle');
  const maxMpg = 50;
  const progressRatio = Math.min(Math.max(mpgVal / maxMpg, 0), 1);
  const strokeOffset = 440 * (1 - progressRatio);
  circle.style.strokeDashoffset = strokeOffset;

  // Efficiency color
  const effBadge = document.getElementById('res-efficiency-badge');
  effBadge.innerText = data.efficiency_category;
  if (data.efficiency_category.includes('High')) {
    effBadge.className = 'badge badge-high';
    circle.style.stroke = '#10b981';
  } else if (data.efficiency_category.includes('Low')) {
    effBadge.className = 'badge badge-low';
    circle.style.stroke = '#ef4444';
  } else {
    effBadge.className = 'badge badge-moderate';
    circle.style.stroke = '#f59e0b';
  }

  // 2. Profile Archetype
  document.getElementById('res-profile-badge').innerText = data.vehicle_profile;
  document.getElementById('res-archetype-name').innerText = data.vehicle_profile;
  document.getElementById('res-cluster-id-tag').innerText = `Cluster ${data.cluster_id}`;

  const bulletsContainer = document.getElementById('res-archetype-bullets');
  if (data.key_characteristics && data.key_characteristics.length > 0) {
    bulletsContainer.innerHTML = data.key_characteristics.map(c => `<li>${c}</li>`).join('');
  }

  // 3. Cluster Comparison Deltas
  const comp = data.cluster_comparison || {};
  const deltaWt = document.getElementById('delta-weight');
  const deltaMpg = document.getElementById('delta-mpg');

  if (comp.weight_diff !== undefined) {
    const isLighter = comp.weight_diff < 0;
    deltaWt.innerText = `${comp.weight_diff > 0 ? '+' : ''}${comp.weight_diff} lbs`;
    deltaWt.className = 'font-mono ' + (isLighter ? 'text-emerald' : 'text-rose');
  }

  if (comp.mpg_diff !== undefined) {
    const isHigher = comp.mpg_diff > 0;
    deltaMpg.innerText = `${comp.mpg_diff > 0 ? '+' : ''}${comp.mpg_diff} MPG`;
    deltaMpg.className = 'font-mono ' + (isHigher ? 'text-emerald' : 'text-rose');
  }

  // 4. Feature Contributions / Explainability
  const attrContainer = document.getElementById('attribution-container');
  if (data.feature_contributions) {
    const topContribs = data.feature_contributions.slice(0, 6);
    attrContainer.innerHTML = topContribs.map(c => {
      const isPositive = c.impact_direction.includes('Increases');
      const tagClass = isPositive ? 'positive' : 'negative';
      const tagText = isPositive ? `+ ${Math.abs(c.coefficient).toFixed(2)} (Boosts MPG)` : `- ${Math.abs(c.coefficient).toFixed(2)} (Reduces MPG)`;

      return `
        <div class="attribution-item">
          <div class="attr-info">
            <span class="attr-name">${formatFeatureName(c.feature)}</span>
            <span class="attr-desc">${c.description}</span>
          </div>
          <span class="attr-tag ${tagClass}">${tagText}</span>
        </div>
      `;
    }).join('');
  }
}

function formatFeatureName(feat) {
  const map = {
    'cylinders': 'Engine Cylinders',
    'displacement': 'Engine Displacement',
    'horsepower': 'Engine Horsepower',
    'weight': 'Vehicle Curb Weight',
    'acceleration': 'Acceleration Performance',
    'model_year': 'Engineering Era / Model Year',
    'power_to_weight': 'Power-to-Weight Ratio',
    'displacement_per_cylinder': 'Displacement per Cylinder',
    'weight_per_cylinder': 'Weight per Cylinder',
    'vehicle_age': 'Vehicle Age Index',
    'origin_usa': 'USA Manufacturing',
    'origin_europe': 'European Engineering',
    'origin_japan': 'Japanese Engineering'
  };
  return map[feat] || feat;
}
