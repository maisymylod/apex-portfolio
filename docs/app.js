// fetches state files and renders the dashboard
const REPO = 'maisymylod/apex-portfolio';

const fmtUsd = n => `$${n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
const fmtPct = n => `${n >= 0 ? '+' : ''}${n.toFixed(2)}%`;
const upDown = n => n >= 0 ? 'up' : 'down';

async function loadAll() {
  const [portfolio, historyText] = await Promise.all([
    fetch('data/portfolio.json').then(r => r.json()),
    fetch('data/history.csv').then(r => r.text()),
  ]);
  let learning = { biases: {} };
  try { learning = await fetch('data/learning.json').then(r => r.json()); } catch (e) {}
  const history = parseHistory(historyText);
  renderStats(portfolio, history);
  renderChart(history);
  renderPositions(portfolio, history);
  renderRisk(portfolio);
  renderTimeline(history);
}

function parseHistory(csv) {
  const rows = csv.trim().split('\n').slice(1); // drop header
  return rows.map(r => {
    const [date, total, cash, holdings, pnl, pnlPct] = r.split(',');
    return {
      date,
      total: parseFloat(total),
      cash: parseFloat(cash),
      holdings: parseFloat(holdings),
      pnl: parseFloat(pnl),
      pnlPct: parseFloat(pnlPct),
    };
  });
}

function renderStats(p, history) {
  const last = history[history.length - 1];
  if (!last) return;
  const totalEl = document.getElementById('stat-total');
  const pnlEl = document.getElementById('stat-pnl');
  const pctEl = document.getElementById('stat-pnl-pct');
  const dayEl = document.getElementById('stat-day');
  totalEl.textContent = fmtUsd(last.total);
  pnlEl.textContent = (last.pnl >= 0 ? '+' : '') + fmtUsd(last.pnl);
  pnlEl.classList.add(upDown(last.pnl));
  pctEl.textContent = fmtPct(last.pnlPct);
  pctEl.classList.add(upDown(last.pnlPct));
  dayEl.textContent = `Day ${history.length}`;
  document.getElementById('last-run').textContent = `last run ${last.date}`;
}

function renderChart(history) {
  const svg = document.getElementById('equity-chart');
  const W = 800, H = 280, PAD_L = 50, PAD_R = 20, PAD_T = 20, PAD_B = 35;
  svg.innerHTML = '';
  if (!history.length) return;

  const values = history.map(h => h.total).concat([1000]);
  const min = Math.min(...values) * 0.995;
  const max = Math.max(...values) * 1.005;
  const xStep = history.length > 1 ? (W - PAD_L - PAD_R) / (history.length - 1) : 0;
  const y = v => PAD_T + (H - PAD_T - PAD_B) * (1 - (v - min) / (max - min));
  const x = i => PAD_L + i * xStep;

  // gridlines
  const ticks = 4;
  for (let i = 0; i <= ticks; i++) {
    const v = min + (max - min) * i / ticks;
    const yy = y(v);
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', PAD_L); line.setAttribute('x2', W - PAD_R);
    line.setAttribute('y1', yy); line.setAttribute('y2', yy);
    line.setAttribute('stroke', 'rgba(255,255,255,0.06)');
    svg.appendChild(line);
    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    label.setAttribute('x', PAD_L - 8); label.setAttribute('y', yy + 4);
    label.setAttribute('fill', '#a8b2cf'); label.setAttribute('font-size', '10');
    label.setAttribute('text-anchor', 'end'); label.setAttribute('font-family', 'Inter');
    label.textContent = `$${v.toFixed(0)}`;
    svg.appendChild(label);
  }

  // baseline at 1000
  if (1000 >= min && 1000 <= max) {
    const bl = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    bl.setAttribute('x1', PAD_L); bl.setAttribute('x2', W - PAD_R);
    bl.setAttribute('y1', y(1000)); bl.setAttribute('y2', y(1000));
    bl.setAttribute('stroke', '#a8b2cf'); bl.setAttribute('stroke-dasharray', '4 4');
    bl.setAttribute('stroke-width', '1');
    svg.appendChild(bl);
  }

  // area under line
  if (history.length > 1) {
    const areaPath = history.map((h, i) => `${i === 0 ? 'M' : 'L'} ${x(i)} ${y(h.total)}`).join(' ')
      + ` L ${x(history.length - 1)} ${y(min)} L ${x(0)} ${y(min)} Z`;
    const area = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    area.setAttribute('d', areaPath);
    area.setAttribute('fill', 'rgba(212, 245, 61, 0.12)');
    svg.appendChild(area);
  }

  // main line
  const path = history.map((h, i) => `${i === 0 ? 'M' : 'L'} ${x(i)} ${y(h.total)}`).join(' ');
  const line = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  line.setAttribute('d', path);
  line.setAttribute('fill', 'none');
  line.setAttribute('stroke', '#d4f53d');
  line.setAttribute('stroke-width', '2.5');
  line.setAttribute('stroke-linejoin', 'round');
  svg.appendChild(line);

  // points
  history.forEach((h, i) => {
    const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    c.setAttribute('cx', x(i)); c.setAttribute('cy', y(h.total));
    c.setAttribute('r', 4);
    c.setAttribute('fill', '#0d1738');
    c.setAttribute('stroke', '#d4f53d');
    c.setAttribute('stroke-width', '2');
    svg.appendChild(c);
  });

  // x-axis date labels (first, mid, last)
  const labelIdx = history.length <= 4 ? history.map((_, i) => i) : [0, Math.floor(history.length / 2), history.length - 1];
  labelIdx.forEach(i => {
    const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    t.setAttribute('x', x(i)); t.setAttribute('y', H - 12);
    t.setAttribute('fill', '#a8b2cf'); t.setAttribute('font-size', '10');
    t.setAttribute('text-anchor', 'middle'); t.setAttribute('font-family', 'Inter');
    t.textContent = history[i].date.slice(5); // MM-DD
    svg.appendChild(t);
  });
}

function renderPositions(p, history) {
  const grid = document.getElementById('positions-grid');
  grid.innerHTML = '';
  const lastDate = history.length ? history[history.length - 1].date : null;
  const ranked = Object.entries(p.positions).map(([t, pos]) => ({ ticker: t, ...pos }));
  // pull latest price from state by re-marking inline (cost_basis is the cost; we don't have current price here)
  // For pnl & price we'd want a richer state file. For v1, use the journal markdown? Skip: derive weight from cost.
  ranked.sort((a, b) => b.shares * b.cost_basis - a.shares * a.cost_basis);
  ranked.forEach(r => {
    const card = document.createElement('div');
    card.className = `position-card ${r.conviction.toLowerCase()}`;
    card.innerHTML = `
      <div class="pos-header">
        <div class="pos-ticker">${r.ticker}</div>
        <div class="pos-pnl ${''}">$${(r.shares * r.cost_basis).toFixed(2)}</div>
      </div>
      <div class="pos-thesis">${r.thesis}</div>
      <div class="pos-meta">
        <span>${r.conviction}</span>
        <span>${r.shares.toFixed(4)} sh @ $${r.cost_basis.toFixed(2)}</span>
      </div>
    `;
    grid.appendChild(card);
  });
}

function renderRisk(p) {
  const sectorMap = {
    BE: 'power', CEG: 'power', VST: 'power', GEV: 'power',
    MSFT: 'hyperscaler', GOOG: 'hyperscaler', META: 'hyperscaler',
    NVDA: 'semi', TSM: 'semi', AVGO: 'semi',
    CLSK: 'miners', RIOT: 'miners', BITF: 'miners',
  };
  const buckets = {};
  let totalCost = 0;
  Object.entries(p.positions).forEach(([t, pos]) => {
    const sec = sectorMap[t] || 'other';
    const cost = pos.shares * pos.cost_basis;
    buckets[sec] = (buckets[sec] || 0) + cost;
    totalCost += cost;
  });
  const flags = [];
  let status = 'GREEN';
  Object.entries(buckets).forEach(([s, v]) => {
    if (v / totalCost * 100 > 50) {
      flags.push(`Sector concentration: ${s} ${(v/totalCost*100).toFixed(1)}% > 50%`);
      status = 'AMBER';
    }
  });

  const statusCard = document.querySelector('.risk-status-card');
  document.getElementById('risk-status').textContent = status;
  if (status !== 'GREEN') statusCard.classList.add('red');

  const sectorList = document.getElementById('sector-list');
  sectorList.innerHTML = '<h4>Sector exposure (by cost)</h4>';
  Object.entries(buckets).sort((a, b) => b[1] - a[1]).forEach(([s, v]) => {
    const pct = v / totalCost * 100;
    const row = document.createElement('div');
    row.className = 'sector-row';
    row.innerHTML = `
      <span>${s}</span>
      <div class="sector-bar-wrap"><div class="sector-bar" style="width: ${Math.min(100, pct)}%"></div></div>
      <span>${pct.toFixed(1)}%</span>
    `;
    sectorList.appendChild(row);
  });

  const flagList = document.getElementById('flag-list');
  flagList.innerHTML = '<h4>Flags</h4>';
  if (flags.length === 0) {
    const d = document.createElement('div');
    d.className = 'flag muted';
    d.textContent = 'No veto conditions active.';
    flagList.appendChild(d);
  } else {
    flags.forEach(f => {
      const d = document.createElement('div');
      d.className = 'flag';
      d.textContent = f;
      flagList.appendChild(d);
    });
  }
}

function renderTimeline(history) {
  const tl = document.getElementById('timeline');
  tl.innerHTML = '';
  history.slice().reverse().forEach(h => {
    const row = document.createElement('div');
    row.className = 'timeline-row';
    const cls = h.pnl >= 0 ? 'pnl-up' : 'pnl-down';
    const sign = h.pnl >= 0 ? '+' : '';
    row.innerHTML = `
      <div class="timeline-date">${h.date}</div>
      <div class="timeline-summary">
        Total <strong>${fmtUsd(h.total)}</strong>,
        P&amp;L <span class="${cls}">${sign}${fmtUsd(h.pnl)} (${fmtPct(h.pnlPct)})</span>,
        cash ${fmtUsd(h.cash)}
      </div>
      <a class="timeline-link" href="https://github.com/${REPO}/blob/main/journal/${h.date}.md" target="_blank">view journal</a>
    `;
    tl.appendChild(row);
  });
}

loadAll().catch(e => {
  console.error(e);
  document.getElementById('stat-total').textContent = 'load error';
});
