// script.js — Agentic Code Review Leaderboard (new)
//
// Data flow:
//   data/output_filelist.json  → list of data_<benchmark>.json filenames
//   data/data_<benchmark>.json → one entry per agent (output of leaderboar.py)
//   data/benchmark_meta.json   → display config: column_groups, primary_metric, display_name
//   data/metric_display_names.json → human-readable names for every column key
//   data/statistics.json       → global counts shown in hero

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------
let leaderboardData = [];          // [{filename, data:[...], meta:{...}}]
let metricDisplayNames = {};
let benchmarkMeta = {};            // keyed by benchmark name
let currentSort = { column: 'overall_weighted_score', direction: 'desc' };
let collapsedGroups = {};          // { benchmarkName: { groupName: bool } }

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
async function fetchJSON(url) {
    try {
        const r = await fetch(url);
        return r.ok ? await r.json() : null;
    } catch (e) {
        console.error('Fetch error:', url, e);
        return null;
    }
}

// displayName: check per-benchmark overrides first, then global map, then fallback
function displayName(col, tabName) {
    const bmOverrides = tabName && benchmarkMeta[tabName] ? (benchmarkMeta[tabName].display_names || {}) : {};
    return bmOverrides[col] || metricDisplayNames[col] || col.split('/').pop().replace(/_/g, ' ');
}

// aggregationMode: look up from the current benchmark's metric_aggregation config
function aggregationMode(col, tabName) {
    const agg = tabName && benchmarkMeta[tabName] ? (benchmarkMeta[tabName].metric_aggregation || {}) : {};
    if (col in agg) return agg[col];
    if (col.includes('_score')) return 'mean';
    if (col.startsWith('metric/')) return 'precision';
    return null;
}

function formatValue(value, col, tabName) {
    if (value === null || value === undefined) return '-';
    // Trajectory / cost / token columns — always raw numbers
    if (col.includes('tokens')) return Math.round(value).toLocaleString();
    if (col.includes('cost') || col.includes('costs')) return '$' + parseFloat(value).toFixed(4);
    if (col === 'steps') return parseFloat(value).toFixed(1);
    // Identity columns
    if (col === 'benchmark_goal') {
        const labels = {
            'human-alignment': 'Human Alignment',
            'bug-capacity': 'Bug Detection',
        };
        const label = labels[value] || value;
        const colors = {
            'human-alignment': '#0969da',
            'bug-capacity': '#cf222e',
        };
        const bg = colors[value] || '#57606a';
        return `<span style="background:${bg};color:#fff;padding:2px 8px;border-radius:12px;font-size:.72rem;white-space:nowrap">${label}</span>`;
    }
    if (['dataset_total_diffs', 'accomplished_diffs', 'total_diffs', 'total_comments'].includes(col)) return Math.round(value).toLocaleString();
    if (col === 'task_accomplishment_rate') return (parseFloat(value) * 100).toFixed(1) + '%';
    if (col === 'timestamp') return String(value).replace(/_/g, ' ');
    // Metric columns: use aggregation mode to decide formatting
    if (typeof value === 'number' && col.startsWith('metric/')) {
        const mode = aggregationMode(col, tabName);
        if (mode === 'mean') return parseFloat(value).toFixed(3);   // raw score, e.g. 0.443
        return (value * 100).toFixed(1) + '%';                      // precision / recall → %
    }
    // Group summary columns: format based on group_summary method in benchmark_meta
    if (typeof value === 'number' && col.startsWith('group_score/')) {
        const groupName = col.replace('group_score/', '');
        const meta = tabName ? (benchmarkMeta[tabName] || {}) : {};
        const method = meta.group_summary?.[groupName]?.method;
        if (method === 'pick') {
            // pick = trajectory/total_costs → show as cost
            const pickedCol = meta.group_summary[groupName].column || '';
            if (pickedCol.includes('cost')) return '$' + parseFloat(value).toFixed(4);
            if (pickedCol.includes('tokens')) return Math.round(value).toLocaleString();
            return parseFloat(value).toFixed(3);
        }
        // mean of metric columns: check if underlying cols are precision/recall (%) or mean (raw)
        const summaryCols = meta.group_summary?.[groupName]?.columns || [];
        const allMean = summaryCols.every(c => (meta.metric_aggregation?.[c] || 'precision') === 'mean');
        if (allMean) return parseFloat(value).toFixed(3);   // e.g. Text Similarity → 0.119
        return (value * 100).toFixed(1) + '%';              // e.g. Code Review Capability → 27.4%
    }
    if (typeof value === 'number') return parseFloat(value).toFixed(3);
    return String(value);
}

function getRankClass(rank) {
    return ['rank-1','rank-2','rank-3'][rank - 1] || 'rank-other';
}

function getTabName(filename) {
    return filename.replace('data_', '').replace('.json', '');
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
async function init() {
    const [stats, fileList, displayNames, bmMeta] = await Promise.all([
        fetchJSON('data/statistics.json'),
        fetchJSON('data/output_filelist.json'),
        fetchJSON('data/metric_display_names.json'),
        fetchJSON('data/benchmark_meta.json'),
    ]);

    metricDisplayNames = displayNames || {};
    benchmarkMeta = bmMeta || {};

    if (stats) {
        document.getElementById('total-agents').textContent   = (stats.total_agents || 0).toLocaleString();
        document.getElementById('total-models').textContent   = (stats.total_models || 0).toLocaleString();
        document.getElementById('total-reviews').textContent  = (stats.total_reviews || 0).toLocaleString();
        document.getElementById('total-comments').textContent = (stats.total_generated_comments || 0).toLocaleString();
    }

    if (fileList && fileList.length) {
        const results = await Promise.all(fileList.map(async filename => {
            const data = await fetchJSON(`data/${filename}`);
            const tabName = getTabName(filename);
            const meta = benchmarkMeta[tabName] || {};
            return data ? { filename, tabName, data, meta } : null;
        }));
        leaderboardData = results.filter(Boolean);
    }

    renderTabs();
    renderInfoPanels();
}

// ---------------------------------------------------------------------------
// Tab rendering
// ---------------------------------------------------------------------------
function renderTabs() {
    const tabsEl = document.getElementById('leaderboard-tabs');
    const contentEl = document.getElementById('leaderboard-tab-contents');
    tabsEl.innerHTML = '';
    contentEl.innerHTML = '';

    leaderboardData.forEach(({ filename, tabName, meta }, idx) => {
        const isActive = idx === 0;
        const label = meta.display_name || tabName.replace(/-/g, ' ').toUpperCase();

        // init collapsed state for this benchmark
        if (!collapsedGroups[tabName]) {
            collapsedGroups[tabName] = {};
            Object.keys(meta.column_groups || {}).forEach(g => {
                collapsedGroups[tabName][g] = true; // all groups collapsed by default
            });
        }

        tabsEl.innerHTML += `
            <li class="nav-item" role="presentation">
                <button class="nav-link ${isActive ? 'active' : ''}"
                    id="tab-btn-${idx}" data-bs-toggle="tab"
                    data-bs-target="#tab-pane-${idx}" type="button" role="tab">
                    ${label}
                </button>
            </li>`;

        contentEl.innerHTML += `
            <div class="tab-pane fade ${isActive ? 'show active' : ''}"
                 id="tab-pane-${idx}" role="tabpanel">
                <div id="tab-${idx}-container"></div>
            </div>`;
    });

    leaderboardData.forEach((item, idx) => renderTable(idx));
}

// ---------------------------------------------------------------------------
// Table rendering
// ---------------------------------------------------------------------------
function renderTable(idx) {
    const { tabName, data, meta } = leaderboardData[idx];
    const container = document.getElementById(`tab-${idx}-container`);
    if (!container) return;

    const sortedData = sortRows([...data]);
    assignRanks(sortedData, meta.primary_metric || 'overall_weighted_score');

    const groups = meta.column_groups || {};
    const groupNames = Object.keys(groups);

    // Build ordered column list
    // Fixed columns always visible regardless of group collapse state.
    // To add or remove a fixed column, edit this array and the formatValue / displayName maps.
    const fixedCols = [
        'agent',
        'model',
        'dataset_total_diffs',
        'task_accomplishment_rate',
        'total_comments',
        'overall_weighted_score',
    ];
    const groupCols = groupNames.flatMap(g => groups[g]);
    const allCols = [...new Set([...fixedCols, ...groupCols])];

    // Controls bar
    const controlsHtml = `
        <div class="leaderboard-controls">
            <span style="color:var(--muted);font-size:.8rem;margin-right:.5rem;">Groups:</span>
            <div class="btn-group-collapse">
                ${groupNames.map(g => `
                    <button onclick="toggleGroup('${tabName}','${g}',${idx})"
                            class="${collapsedGroups[tabName][g] ? '' : 'active'}"
                            id="grp-btn-${idx}-${g}">
                        ${g.charAt(0).toUpperCase() + g.slice(1)}
                    </button>`).join('')}
            </div>
        </div>`;

    // Header
    let headerHtml = '<thead>';

    // Group row
    headerHtml += '<tr>';
    headerHtml += `<th colspan="${fixedCols.length}" class="group-header" style="text-align:left">Agent</th>`;
    groupNames.forEach(g => {
        const cols = groups[g];
        const visible = !collapsedGroups[tabName][g];
        const count = visible ? cols.length : 1;
        const icon = visible ? '▾' : '▸';
        headerHtml += `<th colspan="${count}" class="group-header vertical-separator-left"
            onclick="toggleGroup('${tabName}','${g}',${idx})">${icon} ${g.toUpperCase()}</th>`;
    });
    headerHtml += '</tr>';

    // Column name row
    headerHtml += '<tr>';
    fixedCols.forEach((col, i) => {
        const sorted = currentSort.column === col;
        const arrow = sorted ? (currentSort.direction === 'asc' ? ' ▲' : ' ▼') : '';
        headerHtml += `<th class="sortable${i===0?' text-start':''}" data-col="${col}" onclick="sortBy('${col}',${idx})">${displayName(col, tabName)}${arrow}</th>`;
    });
    groupNames.forEach(g => {
        const cols = groups[g];
        const collapsed = collapsedGroups[tabName][g];
        if (collapsed) {
            // Show the group_score/{group} summary column when collapsed.
            // This key is computed by leaderboar.py from benchmark_info.json → group_summary.
            // To change the summary method or columns, edit benchmark_info.json — no JS changes needed.
            const summaryCol = `group_score/${g}`;
            const sorted = currentSort.column === summaryCol;
            const arrow = sorted ? (currentSort.direction === 'asc' ? ' ▲' : ' ▼') : '';
            const summaryLabel = meta.group_summary?.[g]?.method === 'pick'
                ? displayName(meta.group_summary[g].column, tabName)   // e.g. "Total Cost (USD)"
                : `Avg. ${g}`;                                          // e.g. "Avg. Code Review Capability"
            headerHtml += `<th class="sortable vertical-separator-left" data-col="${summaryCol}" onclick="sortBy('${summaryCol}',${idx})">${summaryLabel} ${arrow}<br><small style="opacity:.6">(${cols.length} cols)</small></th>`;
        } else {
            cols.forEach((col, i) => {
                const sorted = currentSort.column === col;
                const arrow = sorted ? (currentSort.direction === 'asc' ? ' ▲' : ' ▼') : '';
                const sep = i === 0 ? ' vertical-separator-left' : '';
                headerHtml += `<th class="sortable${sep}" data-col="${col}" onclick="sortBy('${col}',${idx})">${displayName(col, tabName)}${arrow}</th>`;
            });
        }
    });
    headerHtml += '</tr></thead>';

    // Body
    let bodyHtml = '<tbody>';
    sortedData.forEach(row => {
        bodyHtml += '<tr>';
        // fixed cols
        fixedCols.forEach((col, i) => {
            let cell = formatValue(row[col], col, tabName);
            if (col === 'agent') {
                cell = `<span class="rank-badge ${getRankClass(row._rank)}">${row._rank}</span>&nbsp;${row[col]}`;
            } else if (col === 'overall_weighted_score') {
                cell = `<span class="score-badge score-badge-primary">${formatValue(row[col], col, tabName)}</span>`;
            }
            bodyHtml += `<td${i===0?' style="text-align:left"':''}>${cell}</td>`;
        });
        // group cols
        groupNames.forEach(g => {
            const cols = groups[g];
            const collapsed = collapsedGroups[tabName][g];
            const isTrajectory = g.toLowerCase().includes('trajectory');
            const badgeClass = isTrajectory ? 'score-badge-traj' : 'score-badge-task';
            if (collapsed) {
                const summaryCol = `group_score/${g}`;
                bodyHtml += `<td class="vertical-separator-left"><span class="score-badge ${badgeClass}">${formatValue(row[summaryCol], summaryCol, tabName)}</span></td>`;
            } else {
                cols.forEach((col, i) => {
                    const sep = i === 0 ? ' class="vertical-separator-left"' : '';
                    bodyHtml += `<td${sep}><span class="score-badge ${badgeClass}">${formatValue(row[col], col, tabName)}</span></td>`;
                });
            }
        });
        bodyHtml += '</tr>';
    });
    bodyHtml += '</tbody>';

    container.innerHTML = controlsHtml + `<div class="table-responsive"><table>${headerHtml}${bodyHtml}</table></div>`;
}

// ---------------------------------------------------------------------------
// Sorting & ranking
// ---------------------------------------------------------------------------
function sortRows(data) {
    return data.sort((a, b) => {
        let va = a[currentSort.column] ?? -Infinity;
        let vb = b[currentSort.column] ?? -Infinity;
        if (typeof va === 'string') va = va.toLowerCase();
        if (typeof vb === 'string') vb = vb.toLowerCase();
        if (va < vb) return currentSort.direction === 'asc' ? -1 : 1;
        if (va > vb) return currentSort.direction === 'asc' ? 1 : -1;
        return 0;
    });
}

function sortBy(col, idx) {
    if (currentSort.column === col) {
        currentSort.direction = currentSort.direction === 'asc' ? 'desc' : 'asc';
    } else {
        currentSort.column = col;
        currentSort.direction = 'desc';
    }
    renderTable(idx);
}

function assignRanks(data, primaryMetric) {
    const sorted = [...data].sort((a, b) => (b[primaryMetric] ?? -1) - (a[primaryMetric] ?? -1));
    data.forEach(row => { row._rank = sorted.indexOf(row) + 1; });
}

// ---------------------------------------------------------------------------
// Group toggle
// ---------------------------------------------------------------------------
function toggleGroup(tabName, group, idx) {
    collapsedGroups[tabName][group] = !collapsedGroups[tabName][group];
    renderTable(idx);
}

// ---------------------------------------------------------------------------
// Theme toggle
// ---------------------------------------------------------------------------
function toggleTheme() {
    const html = document.documentElement;
    const current = html.getAttribute('data-theme') || 'light';
    const next = current === 'light' ? 'dark' : 'light';
    html.setAttribute('data-theme', next);
    localStorage.setItem('lb-theme', next);
    _updateToggleButton(next);
    renderParetoPlot(); // redraw canvas with new theme colors
}

function _updateToggleButton(theme) {
    const icon  = document.getElementById('theme-icon');
    const label = document.getElementById('theme-label');
    if (!icon || !label) return;
    if (theme === 'dark') {
        icon.innerHTML    = '<i class="far fa-sun"></i>';
        label.textContent = 'Light';
    } else {
        icon.innerHTML    = '<i class="far fa-moon"></i>';
        label.textContent = 'Dark';
    }
}

// ---------------------------------------------------------------------------
// Info panels — Benchmarks and Submission panels below the leaderboard table
// ---------------------------------------------------------------------------
function renderInfoPanels() {
    const GOAL_LABELS = {
        'human-alignment': 'Human Alignment',
        'bug-capacity':    'Bug Detection',
    };
    const GOAL_COLORS = {
        'human-alignment': '#0969da',
        'bug-capacity':    '#cf222e',
    };

    // ---- Benchmarks panel ------------------------------------------------
    // Benchmark page URLs — add new entries here when adding new benchmarks
    const BENCHMARK_PAGES = {
        'contextcrbench': 'benchmark-contextcrbench.html',
        'scrbench':       'benchmark-scrbench.html',
    };

    const benchEl = document.getElementById('benchmarks-panel-body');
    if (benchEl) {
        let html = '';
        for (const [bname, meta] of Object.entries(benchmarkMeta)) {
            const goal       = meta.benchmark_goal || bname;
            const label      = GOAL_LABELS[goal]   || goal;
            const color      = GOAL_COLORS[goal]   || '#57606a';
            const display    = meta.display_name   || bname;
            const totalDiffs = meta.dataset_total_diffs || '—';
            const nGroups    = Object.keys(meta.column_groups || {}).length;
            const primary    = meta.primary_metric
                ? (meta.display_names?.[meta.primary_metric] || meta.primary_metric.split('/').pop().replace(/_/g, ' '))
                : '—';
            const pageUrl    = BENCHMARK_PAGES[bname];
            const titleHtml  = pageUrl
                ? `<a href="${pageUrl}" style="color:inherit;text-decoration:none">${display} ↗</a>`
                : display;

            html += `
            <div class="benchmark-card">
                <div class="benchmark-card-title">
                    ${titleHtml}
                    <span class="benchmark-card-goal-badge" style="background:${color}">${label}</span>
                </div>
                <div class="benchmark-card-meta">
                    <span><i class="fas fa-code-pull-request me-1"></i>${totalDiffs} PRs</span>
                    <span><i class="fas fa-layer-group me-1"></i>${nGroups} metric groups</span>
                    <span><i class="fas fa-star me-1"></i>Primary: ${primary}</span>
                </div>
            </div>`;
        }
        
        // Placeholder for upcoming benchmarks
        html += `
            <div class="benchmark-card" style="opacity:0.5;border-style:dashed">
                <div class="benchmark-card-title">
                    <span style="color:var(--text-muted)">More Benchmarks Coming Soon</span>
                    <span class="benchmark-card-goal-badge" style="background:#d0d0d0;color:#666">Planned</span>
                </div>
                <div class="benchmark-card-meta">
                    <span><i class="fas fa-hourglass-end me-1"></i>Additional code review benchmarks in development</span>
                </div>
            </div>`;
        
        benchEl.innerHTML = html || '<p class="text-muted small mb-0">No benchmark metadata available.</p>';
    }

    // Submission panel is static HTML in index.html — nothing to render here.

    // Render the Pareto cost-vs-performance scatter plot
    renderParetoPlot();
}

// ---------------------------------------------------------------------------
// Pareto Plot — canvas-based scatter plot of cost vs. performance per benchmark
// ---------------------------------------------------------------------------
function renderParetoPlot() {
    const el = document.getElementById('pareto-panel-body');
    if (!el) return;

    // leaderboardData is an array of {tabName, data:[...rows], meta:{...}}
    if (!leaderboardData.length) {
        el.innerHTML = '<p class="text-muted small">No data available.</p>';
        return;
    }

    // Colour palette — one colour per benchmark
    const PALETTE = ['#0969da', '#cf222e', '#1a7f37', '#9a6700', '#8250df', '#0550ae'];

    // For each benchmark, collect {agent, cost, score} points (skip null cost or score)
    const datasets = leaderboardData.map(({ tabName, data, meta }, bi) => {
        const points = (data || [])
            .map(r => ({
                agent: r.agent,
                cost:  r['trajectory/trajectory_total_costs'],
                score: r.overall_weighted_score,
            }))
            .filter(p => p.cost != null && p.score != null);
        return { bname: tabName, color: PALETTE[bi % PALETTE.length], points, meta };
    });

    // Compute Pareto frontier for a dataset (lower cost, higher score = better)
    function paretoPts(points) {
        const sorted = [...points].sort((a, b) => a.cost - b.cost);
        const frontier = [];
        let bestScore = -Infinity;
        for (const p of sorted) {
            if (p.score > bestScore) { bestScore = p.score; frontier.push(p); }
        }
        return frontier;
    }

    // Build per-benchmark canvas + legend in a flex row
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const textColor  = isDark ? '#c9d1d9' : '#24292f';
    const gridColor  = isDark ? '#30363d' : '#d0d7de';
    const bgColor    = isDark ? '#161b22' : '#f6f8fa';
    const W = 480, H = 300, PAD = { top: 16, right: 20, bottom: 48, left: 52 };

    let html = '<div style="display:flex;flex-wrap:wrap;gap:1rem;align-items:flex-start">';

    datasets.forEach(({ bname, color, points, meta }) => {
        if (!points.length) return;
        const label  = (meta || {}).display_name || bname;
        const cid    = `pareto-canvas-${bname.replace(/[^a-z0-9]/gi, '-')}`;
        html += `
        <div style="flex:1;min-width:320px">
            <div style="font-size:.78rem;font-weight:600;color:${color};margin-bottom:.3rem">${label}</div>
            <canvas id="${cid}" width="${W}" height="${H}"
                    style="width:100%;max-width:${W}px;border-radius:6px;background:${bgColor}"></canvas>
        </div>`;
    });

    html += '</div>';

    // Legend row
    html += '<div style="display:flex;gap:1.5rem;flex-wrap:wrap;margin-top:.75rem;font-size:.75rem">';
    datasets.forEach(({ bname, color, points, meta }) => {
        if (!points.length) return;
        const label = (meta || {}).display_name || bname;
        html += `<span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${color};margin-right:4px"></span>${label}</span>`;
    });
    html += '<span style="color:var(--text-muted);margin-left:auto">— Pareto frontier &nbsp; &bull; Agent</span></div>';
    // Description appended after canvases
    html += `<p class="info-panel-text" style="margin-top:1rem;font-size:.8rem;color:var(--text-muted);border-top:1px solid var(--border);padding-top:.75rem">
        Each point is an agent — X axis: total cost per run ($), Y axis: overall benchmark score.
        The <strong style="color:var(--text)">dashed line</strong> connects the Pareto frontier: agents where no other agent achieves both lower cost <em>and</em> higher score simultaneously.
    </p>`;

    el.innerHTML = html;

    // Draw each canvas after DOM insertion
    requestAnimationFrame(() => {
        datasets.forEach(({ bname, color, points }) => {
            if (!points.length) return;
            const cid = `pareto-canvas-${bname.replace(/[^a-z0-9]/gi, '-')}`;
            const canvas = document.getElementById(cid);
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            const frontier = paretoPts(points);

            // Axis ranges
            const costs  = points.map(p => p.cost);
            const scores = points.map(p => p.score);
            const xMin = Math.max(0, Math.min(...costs)  * 0.85);
            const xMax = Math.max(...costs)  * 1.1;
            const yMin = Math.max(0, Math.min(...scores) * 0.85);
            const yMax = Math.min(1, Math.max(...scores) * 1.1);

            const toX = v => PAD.left + (v - xMin) / (xMax - xMin) * (W - PAD.left - PAD.right);
            const toY = v => H - PAD.bottom - (v - yMin) / (yMax - yMin) * (H - PAD.top - PAD.bottom);

            // Background
            ctx.fillStyle = bgColor;
            ctx.fillRect(0, 0, W, H);

            // Grid lines
            ctx.strokeStyle = gridColor;
            ctx.lineWidth = 1;
            for (let i = 0; i <= 4; i++) {
                const y = PAD.top + i * (H - PAD.top - PAD.bottom) / 4;
                ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(W - PAD.right, y); ctx.stroke();
            }
            for (let i = 0; i <= 4; i++) {
                const x = PAD.left + i * (W - PAD.left - PAD.right) / 4;
                ctx.beginPath(); ctx.moveTo(x, PAD.top); ctx.lineTo(x, H - PAD.bottom); ctx.stroke();
            }

            // Pareto frontier line
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.setLineDash([6, 3]);
            ctx.beginPath();
            frontier.forEach((p, i) => {
                const x = toX(p.cost), y = toY(p.score);
                i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            });
            ctx.stroke();
            ctx.setLineDash([]);

            // Dots + labels
            points.forEach(p => {
                const x = toX(p.cost), y = toY(p.score);
                const onFrontier = frontier.some(f => f.agent === p.agent);
                ctx.beginPath();
                ctx.arc(x, y, onFrontier ? 6 : 4, 0, Math.PI * 2);
                ctx.fillStyle = onFrontier ? color : color + '88';
                ctx.fill();
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 1.5;
                ctx.stroke();

                // Agent label for frontier points
                if (onFrontier) {
                    ctx.fillStyle = textColor;
                    ctx.font = '10px system-ui,sans-serif';
                    const name = p.agent.length > 18 ? p.agent.slice(0, 17) + '…' : p.agent;
                    ctx.fillText(name, x + 8, y + 4);
                }
            });

            // Axes labels
            ctx.fillStyle = textColor;
            ctx.font = '11px system-ui,sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Total Cost ($)', W / 2, H - 8);
            ctx.save();
            ctx.translate(14, H / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.fillText('Overall Score', 0, 0);
            ctx.restore();

            // Tick labels
            ctx.font = '10px system-ui,sans-serif';
            ctx.textAlign = 'center';
            for (let i = 0; i <= 4; i++) {
                const v = xMin + i * (xMax - xMin) / 4;
                ctx.fillText('$' + v.toFixed(2), PAD.left + i * (W - PAD.left - PAD.right) / 4, H - PAD.bottom + 14);
            }
            ctx.textAlign = 'right';
            for (let i = 0; i <= 4; i++) {
                const v = yMin + i * (yMax - yMin) / 4;
                const y = H - PAD.bottom - i * (H - PAD.top - PAD.bottom) / 4;
                ctx.fillText(v.toFixed(2), PAD.left - 6, y + 4);
            }
        });
    });
}


// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
    // Sync toggle button with saved theme
    const saved = localStorage.getItem('lb-theme') || 'light';
    _updateToggleButton(saved);
    init();
});
