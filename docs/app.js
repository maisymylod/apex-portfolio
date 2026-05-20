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
  renderQuant(portfolio, history);
}

function parseHistory(csv) {
  const rows = csv.trim().split('\n').slice(1); // drop header
  return rows.map(r => {
    const [date, total, cash, holdings, pnl, pnlPct, var95, cvar95, sharpe, pRuin, blRun] = r.split(',');
    return {
      date,
      total: parseFloat(total),
      cash: parseFloat(cash),
      holdings: parseFloat(holdings),
      pnl: parseFloat(pnl),
      pnlPct: parseFloat(pnlPct),
      var95: var95 ? parseFloat(var95) : null,
      cvar95: cvar95 ? parseFloat(cvar95) : null,
      sharpe: sharpe ? parseFloat(sharpe) : null,
      pRuin: pRuin ? parseFloat(pRuin) : null,
      blRun: blRun === '1',
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

function renderQuant(p, history) {
  // BL weights vs current
  const blRows = document.getElementById('bl-rows');
  const blMeta = document.getElementById('bl-meta');
  blRows.innerHTML = '';
  if (!p.bl_weights) {
    blMeta.textContent = 'not yet computed';
    blRows.innerHTML = '<div class="quant-empty">BL output will appear after the next cron run.</div>';
  } else {
    blMeta.textContent = `run ${(p.bl_run_ts || '').slice(0, 16).replace('T', ' ')} UTC`;
    const currentByTicker = {};
    Object.entries(p.positions).forEach(([t, pos]) => {
      currentByTicker[t] = pos.shares * pos.cost_basis;
    });
    const totalCost = Object.values(currentByTicker).reduce((a, b) => a + b, 0) || 1;
    const sorted = Object.entries(p.bl_weights).sort((a, b) => b[1] - a[1]);
    sorted.forEach(([t, target]) => {
      const cur = (currentByTicker[t] || 0) / totalCost;
      const delta = target - cur;
      const row = document.createElement('div');
      row.className = 'bl-row';
      const maxW = Math.max(0.25, target, cur);
      row.innerHTML = `
        <span class="bl-tk">${t}</span>
        <div class="bl-bar-wrap">
          <div class="bl-bar cur" style="width: ${(cur / maxW * 100).toFixed(1)}%"></div>
          <div class="bl-bar tgt" style="width: ${(target / maxW * 100).toFixed(1)}%"></div>
        </div>
        <span class="bl-delta ${delta >= 0 ? 'pos' : 'neg'}">${delta >= 0 ? '+' : ''}${(delta * 100).toFixed(1)}pp</span>
      `;
      blRows.appendChild(row);
    });
  }

  // MC metrics
  const mc = p.mc_report;
  const mcGrid = document.getElementById('mc-grid');
  const mcMeta = document.getElementById('mc-meta');
  const mcVeto = document.getElementById('mc-veto');
  mcGrid.innerHTML = '';
  mcVeto.innerHTML = '';
  if (!mc) {
    mcMeta.textContent = 'not yet computed';
    mcGrid.innerHTML = '<div class="quant-empty">Risk metrics will appear after the next cron run.</div>';
  } else {
    mcMeta.textContent = `${mc.n_paths.toLocaleString()} paths × ${mc.horizon_days}d`;
    const cells = [
      { label: 'VaR 95% (1d)', value: `$${mc.var_1d_usd.toFixed(2)}`, flag: mc.var_1d_usd > 80 },
      { label: 'CVaR 95% (1d)', value: `$${mc.cvar_1d_usd.toFixed(2)}`, flag: mc.cvar_1d_usd > 120 },
      { label: 'Median MaxDD', value: `${(mc.median_max_drawdown * 100).toFixed(1)}%`, flag: mc.median_max_drawdown < -0.25 },
      { label: 'Sim Sharpe', value: mc.sim_sharpe.toFixed(2), flag: false },
      { label: 'Sim return (ann)', value: `${(mc.sim_return_ann * 100 >= 0 ? '+' : '')}${(mc.sim_return_ann * 100).toFixed(1)}%`, flag: false },
      { label: `P(ruin < $${mc.ruin_threshold_usd.toFixed(0)})`, value: `${(mc.p_ruin * 100).toFixed(2)}%`, flag: mc.p_ruin > 0.15 },
    ];
    cells.forEach(c => {
      const d = document.createElement('div');
      d.className = `mc-cell ${c.flag ? 'veto' : ''}`;
      d.innerHTML = `<div class="mc-val">${c.value}</div><div class="mc-lab">${c.label}</div>`;
      mcGrid.appendChild(d);
    });
    if (mc.veto) {
      const v = document.createElement('div');
      v.className = 'mc-veto-banner';
      v.innerHTML = `<strong>VETO</strong> — ${(mc.veto_flags || []).join(' · ')}`;
      mcVeto.appendChild(v);
    }
  }

  // Options overlay
  const opt = p.options_overlay;
  const optStatus = document.getElementById('opt-status');
  const optBody = document.getElementById('opt-body');
  optBody.innerHTML = '';
  if (!opt) {
    optStatus.textContent = 'no signal';
    optBody.innerHTML = '<div class="quant-empty">BSM overlay will appear after the next cron run.</div>';
  } else {
    optStatus.textContent = opt.status;
    optStatus.classList.add(opt.status === 'ACTIVE' ? 'status-active' : 'status-monitoring');
    optBody.innerHTML = `
      <div class="opt-summary">
        ${opt.ticker} protective put · K=$${opt.strike.toFixed(2)} · T=${opt.tenor_days}d · σ=${(opt.sigma * 100).toFixed(1)}%
      </div>
      <div class="opt-greeks">
        <span><b>BSM</b> $${opt.put_value_per_share.toFixed(2)}/sh</span>
        <span><b>Δ</b> ${opt.delta.toFixed(3)}</span>
        <span><b>Γ</b> ${opt.gamma.toFixed(4)}</span>
        <span><b>Θ</b> $${opt.theta_per_day.toFixed(3)}/d</span>
        <span><b>Vega</b> $${opt.vega.toFixed(3)}</span>
      </div>
      <div class="opt-triggers">
        ${Object.entries(opt.triggers || {}).map(([k, v]) =>
          `<span class="opt-trig ${v ? 'pass' : 'fail'}">${v ? '●' : '○'} ${k.replace(/_/g, ' ')}</span>`
        ).join('')}
      </div>
    `;
  }

  renderQuantChart(history);
}

function renderQuantChart(history) {
  const svg = document.getElementById('quant-chart');
  if (!svg) return;
  svg.innerHTML = '';
  const withMC = history.filter(h => h.var95 != null);
  if (withMC.length === 0) {
    const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    t.setAttribute('x', 400); t.setAttribute('y', 120);
    t.setAttribute('fill', '#a8b2cf'); t.setAttribute('text-anchor', 'middle');
    t.setAttribute('font-family', 'Inter'); t.setAttribute('font-size', '12');
    t.textContent = 'Awaiting first MC-enabled run...';
    svg.appendChild(t);
    return;
  }
  const W = 800, H = 240, PAD_L = 50, PAD_R = 50, PAD_T = 20, PAD_B = 35;
  const varVals = withMC.map(h => h.var95);
  const sharpeVals = withMC.map(h => h.sharpe);
  const varMin = Math.min(0, ...varVals), varMax = Math.max(80, ...varVals) * 1.1;
  const shMin = Math.min(0, ...sharpeVals), shMax = Math.max(1.0, ...sharpeVals) * 1.1;
  const xStep = withMC.length > 1 ? (W - PAD_L - PAD_R) / (withMC.length - 1) : 0;
  const yVar = v => PAD_T + (H - PAD_T - PAD_B) * (1 - (v - varMin) / (varMax - varMin || 1));
  const ySh = v => PAD_T + (H - PAD_T - PAD_B) * (1 - (v - shMin) / (shMax - shMin || 1));
  const x = i => PAD_L + i * xStep;

  // VaR limit line at $80
  if (80 <= varMax) {
    const lim = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    lim.setAttribute('x1', PAD_L); lim.setAttribute('x2', W - PAD_R);
    lim.setAttribute('y1', yVar(80)); lim.setAttribute('y2', yVar(80));
    lim.setAttribute('stroke', '#ff6b6b'); lim.setAttribute('stroke-dasharray', '4 4');
    lim.setAttribute('stroke-width', '1');
    svg.appendChild(lim);
  }

  const varPath = withMC.map((h, i) => `${i === 0 ? 'M' : 'L'} ${x(i)} ${yVar(h.var95)}`).join(' ');
  const shPath = withMC.map((h, i) => `${i === 0 ? 'M' : 'L'} ${x(i)} ${ySh(h.sharpe)}`).join(' ');
  [['#d4f53d', varPath], ['#3030ff', shPath]].forEach(([stroke, d]) => {
    const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    p.setAttribute('d', d);
    p.setAttribute('fill', 'none');
    p.setAttribute('stroke', stroke);
    p.setAttribute('stroke-width', '2.5');
    svg.appendChild(p);
  });

  // y-axis labels: left = VaR, right = Sharpe
  for (let i = 0; i <= 3; i++) {
    const v = varMin + (varMax - varMin) * i / 3;
    const yy = yVar(v);
    const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    t.setAttribute('x', PAD_L - 8); t.setAttribute('y', yy + 4);
    t.setAttribute('fill', '#d4f53d'); t.setAttribute('font-size', '10');
    t.setAttribute('text-anchor', 'end'); t.setAttribute('font-family', 'Inter');
    t.textContent = `$${v.toFixed(0)}`;
    svg.appendChild(t);
    const s = shMin + (shMax - shMin) * i / 3;
    const yys = ySh(s);
    const t2 = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    t2.setAttribute('x', W - PAD_R + 8); t2.setAttribute('y', yys + 4);
    t2.setAttribute('fill', '#a0a0ff'); t2.setAttribute('font-size', '10');
    t2.setAttribute('text-anchor', 'start'); t2.setAttribute('font-family', 'Inter');
    t2.textContent = s.toFixed(2);
    svg.appendChild(t2);
  }

  const labelIdx = withMC.length <= 4 ? withMC.map((_, i) => i) : [0, Math.floor(withMC.length / 2), withMC.length - 1];
  labelIdx.forEach(i => {
    const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    t.setAttribute('x', x(i)); t.setAttribute('y', H - 12);
    t.setAttribute('fill', '#a8b2cf'); t.setAttribute('font-size', '10');
    t.setAttribute('text-anchor', 'middle'); t.setAttribute('font-family', 'Inter');
    t.textContent = withMC[i].date.slice(5);
    svg.appendChild(t);
  });
}

loadAll().catch(e => {
  console.error(e);
  document.getElementById('stat-total').textContent = 'load error';
});
