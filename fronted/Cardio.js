  const API_URL = 'https://cardioscan-cgwj.onrender.com';
  let apiOnline = false;
  let wakeInterval = null;
  let countdownInterval = null;
  let seconds = 0;

  async function checkAPI() {
    const dot  = document.getElementById('statusDot');
    const text = document.getElementById('statusText');
    const banner = document.getElementById('wakeBanner');

    try {
      const r = await fetch(`${API_URL}/health`, {
        signal: AbortSignal.timeout(5000)
      });
      if (r.ok) {
        dot.className   = 'status-dot online';
        text.textContent = 'API connectée ✓';
        apiOnline = true;
        banner.classList.remove('show');
        stopWakeCountdown();
        return true;
      }
    } catch {}

    dot.className   = 'status-dot offline';
    text.textContent = 'Démarrage serveur...';
    apiOnline = false;

    if (!banner.classList.contains('show')) {
      banner.classList.add('show');
      startWakeCountdown();
    }
    return false;
  }

  function startWakeCountdown() {
    seconds = 0;
    const fill = document.getElementById('wakeFill');
    const cd   = document.getElementById('countdown');
    cd.textContent = '60';
    fill.style.width = '0%';

    countdownInterval = setInterval(() => {
      seconds++;
      const remaining = Math.max(0, 60 - seconds);
      cd.textContent = remaining;
      fill.style.width = (seconds / 60 * 100) + '%';
      if (seconds >= 60) {
        cd.textContent = '...';
      }
    }, 1000);
  }

  function stopWakeCountdown() {
    if (countdownInterval) {
      clearInterval(countdownInterval);
      countdownInterval = null;
    }
  }

  // Vérifier toutes les 5 secondes
  checkAPI();
  setInterval(checkAPI, 5000);

  async function predict() {
    const fields = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'];
    const data = {};

    for (const f of fields) {
      const el = document.getElementById(f);
      if (!el || el.value === '') {
        el.style.borderColor = '#dc2626';
        el.focus();
        setTimeout(() => el.style.borderColor = '', 2000);
        return;
      }
      data[f] = parseFloat(el.value);
    }

    if (!apiOnline) {
      alert('⏳ Le serveur démarre encore. Attendez que l\'indicateur passe au vert puis réessayez.');
      return;
    }

    const btn = document.getElementById('predictBtn');
    btn.disabled = true;
    btn.classList.add('loading');

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
        signal: AbortSignal.timeout(30000)
      });

      if (!response.ok) throw new Error(await response.text());
      const result = await response.json();
      showResult(result);

    } catch (e) {
      alert('❌ Erreur de connexion. Vérifiez que l\'indicateur est vert et réessayez.');
    } finally {
      btn.disabled = false;
      btn.classList.remove('loading');
    }
  }

  function showResult(result) {
    const card    = document.getElementById('resultCard');
    const idle    = document.getElementById('resultIdle');
    const content = document.getElementById('resultContent');
    const label   = document.getElementById('resultLabel');
    const icon    = document.getElementById('resultIcon');
    const prob    = document.getElementById('resultProb');
    const fill    = document.getElementById('gaugeFill');
    const ecgPath = document.getElementById('ecgPath');

    const isHigh = result.risk === 'HIGH';
    card.className = 'result-card ' + (isHigh ? 'high' : 'low');
    idle.style.display = 'none';
    icon.textContent = isHigh ? '⚠️' : '✅';
    label.textContent = isHigh ? 'Risque Élevé' : 'Risque Faible';
    label.className = 'result-label ' + (isHigh ? 'high' : 'low');
    prob.textContent = result.probability + '%';
    fill.className = 'gauge-fill ' + (isHigh ? 'high' : 'low');
    ecgPath.style.stroke = isHigh ? '#dc2626' : '#16a34a';

    content.style.display = 'flex';
    content.classList.remove('show');
    void content.offsetWidth;
    content.classList.add('show');
    setTimeout(() => { fill.style.width = result.probability + '%'; }, 200);
    ecgPath.classList.remove('animate');
    void ecgPath.offsetWidth;
    ecgPath.classList.add('animate');

    const factorsCard = document.getElementById('factorsCard');
    const factorList  = document.getElementById('factorList');
    factorsCard.style.display = 'block';
    factorList.innerHTML = '';
    result.factors.forEach((f, i) => {
      const item = document.createElement('div');
      item.className = 'factor-item';
      item.innerHTML = `<div class="factor-dot ${f.status}"></div><span>${f.text}</span>`;
      factorList.appendChild(item);
      setTimeout(() => item.classList.add('visible'), i * 120 + 400);
    });
  }