/* EarthPulse — frontend controller */
'use strict';

const $ = (s, r = document) => r.querySelector(s);
const el = (t, c, h) => { const n = document.createElement(t); if (c) n.className = c; if (h !== undefined) n.innerHTML = h; return n; };
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, m => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[m]));
const nf = (n, d = 0) => Number(n ?? 0).toLocaleString('en-IN', { minimumFractionDigits: d, maximumFractionDigits: d });

const ICONS = {
  building: '<path d="M3 21h18M5 21V7l7-4 7 4v14M9 9h2m-2 4h2m-2 4h2m4-8h2m-2 4h2m-2 4h2"/>',
  tree: '<path d="M12 22v-7M9 15H5l3-5H6l3-5h2l3 5h-2l3 5h-4"/>',
  droplet: '<path d="M12 2.7 6.6 8.1a7.6 7.6 0 1 0 10.8 0z"/>',
  waves: '<path d="M2 6c2.5 0 2.5 2 5 2s2.5-2 5-2 2.5 2 5 2 2.5-2 5-2M2 12c2.5 0 2.5 2 5 2s2.5-2 5-2 2.5 2 5 2 2.5-2 5-2M2 18c2.5 0 2.5 2 5 2s2.5-2 5-2 2.5 2 5 2 2.5-2 5-2"/>',
  flame: '<path d="M12 22a7 7 0 0 0 7-7c0-5-4-6-4-10-3 2-5 5-5 8 0 1-1 2-2 1s-1-2-1-3c-2 2-2 4-2 6a7 7 0 0 0 7 5z"/>',
  sprout: '<path d="M7 20h10M12 20V9M12 9C12 5 9 3 5 3c0 4 3 6 7 6zM12 12c0-3 2-5 6-5 0 3-3 5-6 5z"/>',
  sparkles: '<path d="m12 3 1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z"/>',
  check: '<path d="M20 6 9 17l-5-5"/>',
  alert: '<path d="M12 9v4M12 17h.01M10.3 3.9 2 18a2 2 0 0 0 1.7 3h16.6A2 2 0 0 0 22 18L13.7 3.9a2 2 0 0 0-3.4 0z"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  layers: '<path d="m12 2 9 5-9 5-9-5z"/><path d="m3 12 9 5 9-5M3 17l9 5 9-5"/>',
  file: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/>',
  target: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
  compass: '<circle cx="12" cy="12" r="9"/><path d="m15.5 8.5-2 5-5 2 2-5z"/>',
  brain: '<path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44A2.5 2.5 0 0 1 4 17.5a2.5 2.5 0 0 1-.5-4.9A2.5 2.5 0 0 1 4 7.5 2.5 2.5 0 0 1 6.5 5 2.5 2.5 0 0 1 9.5 2zM14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44A2.5 2.5 0 0 0 20 17.5a2.5 2.5 0 0 0 .5-4.9A2.5 2.5 0 0 0 20 7.5 2.5 2.5 0 0 0 17.5 5 2.5 2.5 0 0 0 14.5 2z"/>',
  shield: '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
};
const svg = (k, sz = 16) =>
  `<svg viewBox="0 0 24 24" width="${sz}" height="${sz}" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${ICONS[k] || ICONS.check}</svg>`;

const STEPS = [
  ['interpret', 'Understand the question', 'Parsing location, time and phenomenon'],
  ['locate', 'Locate on Earth', 'Resolving the area of interest'],
  ['search', 'Search satellite archive', 'Finding season-matched, low-cloud scenes'],
  ['load', 'Stream imagery', 'Reading Sentinel-2 bands onto a common grid'],
  ['quality', 'Check data quality', 'Clouds, shadows and image alignment'],
  ['detect', 'Detect change', 'Comparing spectral indices across dates'],
  ['filter', 'Filter false change', 'Removing cloud, season and noise artefacts'],
  ['render', 'Render views', 'Before / after / change layers'],
  ['explain', 'Explain the evidence', 'Writing the report'],
];

let CUR = null;
let BUSY = false;

/* ---------------- pipeline UI ---------------- */
function resetPipe() {
  const p = $('#pipe'); p.innerHTML = '';
  STEPS.forEach(([k, t, s]) => {
    const n = el('div', 'step', `
      <div class="step-ic" data-ic>${STEPS.findIndex(x => x[0] === k) + 1}</div>
      <div class="step-tx"><b>${esc(t)}</b><span data-msg>${esc(s)}</span></div>`);
    n.dataset.step = k;
    p.appendChild(n);
  });
  $('#ptime').textContent = '';
}
function setStep(k, status, msg) {
  const n = $(`.step[data-step="${k}"]`);
  if (!n) return;
  n.className = 'step ' + (status === 'running' ? 'active' : status === 'error' ? 'err' : 'done');
  if (msg) $('[data-msg]', n).textContent = msg;
  const ic = $('[data-ic]', n);
  if (status === 'done') ic.innerHTML = svg('check', 11);
  if (status === 'error') ic.innerHTML = '!';
  if (status === 'running') ic.innerHTML = '';
}

/* ---------------- helpers ---------------- */
const bandOf = (c) => c >= 0.9 ? 'high' : c >= 0.7 ? 'moderate' : 'low';
const confColor = (c) => c >= 0.9 ? 'var(--ok)' : c >= 0.7 ? 'var(--warn)' : 'var(--hot)';
const statusColor = (s) => s === 'changed' ? 'var(--hot)' : s === 'emerging' ? 'var(--warn)' : 'var(--ok)';

function fmtArea(ha) {
  if (!ha) return '0 ha';
  if (ha >= 100) return nf(ha / 100, 2) + ' km²';
  return nf(ha, 1) + ' ha';
}

/* ---------------- results ---------------- */
function render(d) {
  const R = $('#results'); R.innerHTML = '';
  const plan = d.plan || {}, st = d.stats || {}, nar = d.narrative || {};
  const rep = nar.report || {};
  const imgs = d.images || {};
  const cl = d.clusters || [], tl = d.timeline || [], sim = d.similar || [];

  const wrap = el('div', 'grid');
  const left = el('div', 'stack');
  const right = el('div', 'stack');

  /* headline */
  const hl = el('div', 'card');
  hl.appendChild(el('div', 'headline', `
    <h3>${esc(nar.headline || 'Analysis complete')}</h3>
    <div class="meta">
      <span class="badge ${bandOf(d.confidence || 0)}">${svg('shield', 11)} ${Math.round((d.confidence || 0) * 100)}% ${esc(d.confidence_band || '')} confidence</span>
      <span class="badge n">${esc(st.index || '')}</span>
      <span class="badge n">${st.epoch_count || 0} acquisitions</span>
      <span class="badge n">~${nf(st.pixel_size_m, 0)} m/px</span>
      <span class="badge n">${esc(d.place?.label || '')}</span>
    </div>`));
  left.appendChild(hl);

  /* before / after / change */
  const vis = el('div', 'card');
  vis.appendChild(el('div', 'card-h', `${svg('layers')} Before · After · Change <span class="sp">${esc(st.baseline_date || '')} → ${esc(st.latest_date || '')}</span>`));

  const cmpWrap = el('div', 'compare-wrap');
  const cmp = el('div', 'compare');
  cmp.innerHTML = `
    <img src="${imgs.before}" alt="Before" />
    <img class="after-l" id="aimg" src="${imgs.overlay}" alt="After" />
    <div class="handle"></div>
    <span class="cl l">◀ ${esc(st.baseline_year || 'BEFORE')}</span>
    <span class="cl r">${esc(st.latest_year || 'AFTER')} ▶</span>`;
  cmpWrap.appendChild(cmp);
  vis.appendChild(cmpWrap);

  const tabs = el('div', 'viewtabs');
  const views = [
    ['overlay', 'Change overlay'], ['after', 'After (plain)'],
    ['heatmap', 'Change heatmap'], ['mask', 'Mask only'], ['scl_after', 'Cloud & land classes'],
  ];
  views.forEach(([k, lbl], i) => {
    const b = el('button', 'vt' + (i === 0 ? ' on' : ''), lbl);
    b.onclick = () => {
      $$all('.vt', tabs).forEach(x => x.classList.remove('on'));
      b.classList.add('on');
      $('#aimg').src = imgs[k] || imgs.after;
    };
    tabs.appendChild(b);
  });
  vis.appendChild(tabs);
  vis.appendChild(el('div', 'legend', `
    <span><i style="background:${plan.direction === 'increase' ? 'var(--hot)' : 'var(--warn)'}"></i>Detected change</span>
    <span><i style="background:#fff"></i>Cluster outline</span>
    <span style="margin-left:auto">Drag the divider to compare</span>`));
  left.appendChild(vis);

  /* swipe interaction */
  (function () {
    let drag = false;
    const set = (e) => {
      const r = cmp.getBoundingClientRect();
      const x = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
      cmp.style.setProperty('--pos', Math.max(0, Math.min(100, (x / r.width) * 100)) + '%');
    };
    cmp.addEventListener('mousedown', e => { drag = true; set(e); });
    window.addEventListener('mousemove', e => drag && set(e));
    window.addEventListener('mouseup', () => drag = false);
    cmp.addEventListener('touchstart', e => { drag = true; set(e); }, { passive: true });
    cmp.addEventListener('touchmove', e => drag && set(e), { passive: true });
    cmp.addEventListener('touchend', () => drag = false);
    cmp.style.setProperty('--pos', '50%');
  })();

  /* stats */
  const sc = el('div', 'card');
  sc.appendChild(el('div', 'card-h', `${svg('target')} Measurements`));
  const sg = el('div', 'stats');
  const cells = [
    ['Affected area', fmtArea(st.changed_area_ha), `${nf(st.changed_percent, 2)}% of area studied`],
    ['Clusters', nf(st.cluster_count), `largest ${fmtArea(st.largest_cluster_ha)}`],
    ['Area studied', fmtArea(st.analysed_area_ha), `${st.epoch_count || 0} dates compared`],
    ['Persistence', Math.round((st.mean_persistence || 0) * 100) + '<small>%</small>', 'holds across time'],
    ['Confidence', Math.round((d.confidence || 0) * 100) + '<small>%</small>', esc(d.confidence_band || '')],
  ];
  cells.forEach(([k, v, s]) => sg.appendChild(el('div', 'stat', `<div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div>`)));
  sc.appendChild(sg);
  left.appendChild(sc);

  /* timeline */
  if (tl.length) {
    const tc = el('div', 'card');
    tc.appendChild(el('div', 'card-h', `${svg('clock')} Change timeline <span class="sp">when did it happen?</span>`));
    const box = el('div', 'tl');
    const track = el('div', 'tl-track');
    track.appendChild(el('div', 'tl-line'));
    const fill = el('div', 'tl-fill'); fill.style.width = '100%'; track.appendChild(fill);
    tl.forEach((p, i) => {
      const pos = tl.length > 1 ? (i / (tl.length - 1)) * 100 : 50;
      const pt = el('div', 'tl-pt');
      pt.style.left = `calc(${pos}% )`;
      pt.innerHTML = `<div class="tl-node" style="background:${statusColor(p.status)}"></div><div class="tl-lb">${p.year}</div>`;
      pt.title = `${p.date}\n${p.status.toUpperCase()}\nCloud ${nf(p.cloud_cover, 1)}% · usable ${Math.round(p.valid_fraction * 100)}%\nChanged by this date: ${nf(p.changed_fraction * 100, 1)}%`;
      pt.onclick = () => {
        $$all('.tl-pt', track).forEach(x => x.classList.remove('sel'));
        pt.classList.add('sel');
        const th = (imgs.thumbs || []).find(t => t.year === p.year);
        if (th) { $('#aimg').src = th.src; $$all('.vt', tabs).forEach(x => x.classList.remove('on')); }
      };
      track.appendChild(pt);
    });
    box.appendChild(track);
    tc.appendChild(box);

    if (imgs.thumbs?.length) {
      const th = el('div', 'thumbs');
      imgs.thumbs.forEach(t => {
        const b = el('div', 'thumb', `<img src="${t.src}" alt="${t.year}" /><div class="ty">${t.year}</div>`);
        b.onclick = () => { $$all('.thumb', th).forEach(x => x.classList.remove('on')); b.classList.add('on'); $('#aimg').src = t.src; };
        th.appendChild(b);
      });
      tc.appendChild(th);
    }
    left.appendChild(tc);
  }

  /* explanation */
  const ex = el('div', 'card');
  ex.appendChild(el('div', 'card-h', `${svg('brain')} What EarthPulse found`));
  const eb = el('div', 'card-b');
  eb.appendChild(el('p', 'prose', esc(nar.summary || '')));
  if (nar.confidence_explanation) eb.appendChild(el('p', 'prose', `<strong>Confidence.</strong> ${esc(nar.confidence_explanation)}`));
  if (nar.so_what) { const s = el('div', 'sowhat'); s.innerHTML = esc(nar.so_what); s.style.marginTop = '13px'; eb.appendChild(s); }
  if (nar.recommendations?.length) {
    const rw = el('div'); rw.style.marginTop = '13px';
    rw.appendChild(el('div', '', '<div style="font-size:11px;text-transform:uppercase;letter-spacing:.6px;color:var(--txt-3);font-weight:600;margin-bottom:6px">Recommended next steps</div>'));
    nar.recommendations.forEach((r, i) => rw.appendChild(el('div', 'rec', `<div class="n">${i + 1}</div><div>${esc(r)}</div>`)));
    eb.appendChild(rw);
  }
  ex.appendChild(eb);
  left.appendChild(ex);

  /* ---------- right column ---------- */

  /* interpretation */
  const ic = el('div', 'card');
  ic.appendChild(el('div', 'card-h', `${svg('compass')} How I read your question`));
  const ib = el('div', 'card-b');
  [['Location', esc(d.place?.label || plan.place || '—')],
   ['Target', `<em>${esc(plan.mode_label || '')}</em>`],
   ['Period', `${plan.start_year}–${plan.end_year}`],
   ['Indicator', `${esc(plan.index)} · ${esc(plan.direction)}`],
   ['Analysis', 'Multi-temporal change detection'],
  ].forEach(([k, v]) => ib.appendChild(el('div', 'plan-row', `<div class="pk">${k}</div><div class="pv">${v}</div>`)));

  if (plan.reasoning?.length) {
    const det = el('details', 'adv', `<summary>Why I read it this way (${plan.reasoning.length})</summary>`);
    plan.reasoning.forEach(r => det.appendChild(el('div', 'ev plus', `<span class="ic">${svg('check', 15)}</span><span>${esc(r)}</span>`)));
    (plan.assumptions || []).forEach(r => det.appendChild(el('div', 'ev minus', `<span class="ic">${svg('alert', 15)}</span><span>${esc(r)}</span>`)));
    ib.appendChild(det);
  }
  ic.appendChild(ib);
  right.appendChild(ic);

  /* evidence */
  const evc = el('div', 'card');
  evc.appendChild(el('div', 'card-h', `${svg('shield')} Evidence <span class="sp">${(rep.evidence || []).length} checks</span>`));
  const evb = el('div', 'card-b');
  (rep.evidence || []).forEach(e => evb.appendChild(el('div', 'ev plus', `<span class="ic">${svg('check', 15)}</span><span>${esc(e)}</span>`)));
  (rep.limitations || []).forEach(e => evb.appendChild(el('div', 'ev minus', `<span class="ic">${svg('alert', 15)}</span><span>${esc(e)}</span>`)));
  evc.appendChild(evb);
  right.appendChild(evc);

  /* clusters */
  if (cl.length) {
    const cc = el('div', 'card');
    cc.appendChild(el('div', 'card-h', `${svg('target')} Detected clusters <span class="sp">top ${Math.min(cl.length, 8)} of ${cl.length}</span>`));
    const cb = el('div', 'card-b');
    cl.slice(0, 8).forEach(c => {
      const n = el('div', 'cluster', `
        <div class="rank">${c.id}</div>
        <div class="cinfo">
          <b>${fmtArea(c.area_ha)}</b>
          <span>${esc(c.label)}${c.onset_year ? ' · from ' + c.onset_year : ''} · ${c.centroid.lat.toFixed(4)}, ${c.centroid.lon.toFixed(4)}</span>
        </div>
        <div class="cconf" style="color:${confColor(c.confidence)}">${Math.round(c.confidence * 100)}%</div>`);
      n.title = [...(c.evidence || []), ...(c.caveats || [])].join('\n');
      n.onclick = () => window.open(`https://www.google.com/maps/@${c.centroid.lat},${c.centroid.lon},15z/data=!3m1!1e3`, '_blank');
      cb.appendChild(n);
    });
    cb.appendChild(el('div', '', `<div style="font-size:11px;color:var(--txt-3);margin-top:8px">Click a cluster to open it in satellite view for verification.</div>`));
    cc.appendChild(cb);
    right.appendChild(cc);
  }

  /* similar */
  if (sim.length) {
    const smc = el('div', 'card');
    smc.appendChild(el('div', 'card-h', `${svg('layers')} Similar patterns`));
    const smb = el('div', 'card-b');
    sim.forEach(s => smb.appendChild(el('div', 'simcard', `
      <div class="top"><b>${esc(s.label)}</b><span style="font-size:11.5px;font-weight:700;color:var(--accent)">${s.similarity_percent}%</span></div>
      <div class="bar"><i style="width:${Math.max(4, s.similarity_percent)}%"></i></div>
      <p>${esc(s.headline || s.why)}</p>`)));
    smc.appendChild(smb);
    right.appendChild(smc);
  }

  /* imagery + report */
  const dc = el('div', 'card');
  dc.appendChild(el('div', 'card-h', `${svg('file')} Imagery & report`));
  const db = el('div', 'card-b');
  (d.scenes || []).forEach(s => db.appendChild(el('div', 'kv',
    `<span class="k">${esc(s.date)}</span><span class="v mono">${nf(s.cloud_cover, 1)}% cloud</span>`)));
  const q = d.quality || {};
  db.appendChild(el('div', 'kv', `<span class="k">Alignment offset</span><span class="v">${nf(q.registration_shift_px, 1)} px</span>`));
  db.appendChild(el('div', 'kv', `<span class="k">Usable area</span><span class="v">${Math.round((q.analysis_valid_fraction || 0) * 100)}%</span>`));
  if (q.noise_rejection_rate != null) db.appendChild(el('div', 'kv', `<span class="k">Noise rejected</span><span class="v">${Math.round(q.noise_rejection_rate * 100)}%</span>`));
  if (q.spectral_gate_rejected_px != null) db.appendChild(el('div', 'kv', `<span class="k">Implausible rejected</span><span class="v">${nf(q.spectral_gate_rejected_px)} px</span>`));

  const dl = el('button', 'btn ghost', `${svg('file', 14)} Download full report (.md)`);
  dl.style.marginTop = '12px'; dl.style.width = '100%'; dl.style.justifyContent = 'center';
  dl.onclick = async () => {
    const r = await fetch('/api/report.md'); const t = await r.text();
    const b = new Blob([t], { type: 'text/markdown' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(b);
    a.download = `earthpulse-${(d.place?.label || 'report').replace(/[^a-z0-9]+/gi, '-').toLowerCase()}.md`;
    a.click();
  };
  db.appendChild(dl);
  dc.appendChild(db);
  right.appendChild(dc);

  wrap.appendChild(left); wrap.appendChild(right);
  R.appendChild(wrap);
}

const $$all = (s, r = document) => Array.from(r.querySelectorAll(s));

/* ---------------- run ---------------- */
async function run(q) {
  if (BUSY) return;
  const query = (q ?? $('#q').value).trim();
  if (!query) { $('#q').focus(); return; }
  BUSY = true;
  $('#go').disabled = true;
  $('#stage').classList.add('on');
  $('#results').innerHTML = '';
  resetPipe();
  $('#stage').scrollIntoView({ behavior: 'smooth', block: 'start' });

  const acc = {};
  const t0 = performance.now();
  const timer = setInterval(() => { $('#ptime').textContent = ((performance.now() - t0) / 1000).toFixed(1) + 's'; }, 100);

  try {
    const res = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        grid: parseInt($('#gridsel').value, 10),
        max_cloud: parseFloat($('#cloud').value) || 20,
      }),
    });
    if (!res.ok || !res.body) throw new Error('Server error ' + res.status);

    const rd = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await rd.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        let m; try { m = JSON.parse(line); } catch { continue; }
        Object.assign(acc, m);
        if (m.step === 'error') {
          setStep('interpret', 'error');
          $('#results').innerHTML = `<div class="err-box" style="margin-top:16px">${svg('alert', 15)} ${esc(m.message)}</div>`;
          break;
        }
        setStep(m.step, m.status, m.message);
        if (m.step === 'explain' && m.status === 'done') { CUR = acc; render(acc); }
      }
    }
  } catch (e) {
    $('#results').innerHTML = `<div class="err-box" style="margin-top:16px">${svg('alert', 15)} ${esc(e.message || e)}</div>`;
  } finally {
    clearInterval(timer);
    $('#ptime').textContent = ((performance.now() - t0) / 1000).toFixed(1) + 's total';
    BUSY = false; $('#go').disabled = false;
  }
}

/* ---------------- boot ---------------- */
$('#go').onclick = () => run();
$('#q').addEventListener('keydown', e => { if (e.key === 'Enter') run(); });

fetch('/api/health').then(r => r.json()).then(h => {
  $('#hdot').className = 'dot';
  $('#hstat').textContent = 'live · ' + (h.version || '');
}).catch(() => {
  $('#hdot').className = 'dot off';
  $('#hstat').textContent = 'offline';
});

fetch('/api/examples').then(r => r.json()).then(d => {
  const c = $('#chips');
  (d.examples || []).forEach(x => {
    const b = el('button', 'chip', `${svg(x.icon, 13)} ${esc(x.title)}`);
    b.title = x.query + (x.note ? ' — ' + x.note : '');
    b.onclick = () => { $('#q').value = x.query; run(x.query); };
    c.appendChild(b);
  });
}).catch(() => {});
