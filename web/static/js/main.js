// NBA Momentum HMM Visualization - Frontend JavaScript

// Global variables
let games = [];
let currentGame = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing NBA Momentum Visualization...');
    loadGames();

    // Set up event listeners
    document.getElementById('analyze-btn').addEventListener('click', analyzeGame);
    document.getElementById('game-select').addEventListener('change', function() {
        const selected = this.value;
        document.getElementById('analyze-btn').disabled = !selected;
    });
});

// Load model statistics
async function loadModelStats() {
    try {
        const response = await fetch('/api/model-stats');
        const stats = await response.json();

        document.getElementById('stat-auc').textContent = stats.auc_final.toFixed(4);
        document.getElementById('stat-baseline').textContent = stats.auc_baseline.toFixed(4);
        document.getElementById('stat-improvement').textContent =
            `+${stats.improvement.toFixed(4)} (+${(stats.improvement / stats.auc_baseline * 100).toFixed(1)}%)`;
        document.getElementById('stat-pvalue').textContent = stats.anova_pvalue.toExponential(2);

        // Display transition matrix
        displayTransitionMatrix(stats.transition_matrix);

    } catch (error) {
        console.error('Error loading model stats:', error);
    }
}

// Load state statistics
async function loadStateStats() {
    try {
        const response = await fetch('/api/state-stats');
        const states = await response.json();

        const container = document.getElementById('states-container');
        container.innerHTML = '';

        const stateColors = {
            'COLD': '#3498db',
            'NEUTRAL': '#95a5a6',
            'HOT': '#e74c3c'
        };

        states.forEach(state => {
            const card = document.createElement('div');
            card.className = 'state-card';
            card.style.borderLeft = `5px solid ${stateColors[state.name]}`;

            card.innerHTML = `
                <h3 style="color: ${stateColors[state.name]}">${state.name}</h3>
                <div class="state-metric">
                    <span class="metric-label">Occurrences:</span>
                    <span class="metric-value">${state.count.toLocaleString()} (${state.pct.toFixed(1)}%)</span>
                </div>
                <div class="state-metric">
                    <span class="metric-label">Scoring Rate:</span>
                    <span class="metric-value">${state.scoring_rate.toFixed(1)}%</span>
                </div>
                <div class="state-metric">
                    <span class="metric-label">Avg Points:</span>
                    <span class="metric-value">${state.avg_points.toFixed(3)}</span>
                </div>
            `;

            container.appendChild(card);
        });

    } catch (error) {
        console.error('Error loading state stats:', error);
    }
}

function getGradientColor(val) {
    const green = [46, 204, 113];   // #2ecc71
    const yellow = [241, 196, 15];  // #f1c40f
    const orange = [230, 126, 34];  // #e67e22

    let start, end, ratio;

    if (val <= 0.5) {
        // Green → Yellow
        ratio = val / 0.5;
        start = green;
        end = yellow;
    } else {
        // Yellow → Orange
        ratio = (val - 0.5) / 0.5;
        start = yellow;
        end = orange;
    }

    const r = Math.round(start[0] + ratio * (end[0] - start[0]));
    const g = Math.round(start[1] + ratio * (end[1] - start[1]));
    const b = Math.round(start[2] + ratio * (end[2] - start[2]));

    return `rgb(${r}, ${g}, ${b})`;
}

// Display transition matrix
function displayTransitionMatrix(matrix) {
    const container = document.getElementById('transition-table');

    let html = '<table class="transition-table"><thead><tr><th>From \\ To</th>';
    const stateNames = ['COLD', 'NEUTRAL', 'HOT'];

    // Header
    stateNames.forEach(name => {
        html += `<th>${name}</th>`;
    });
    html += '</tr></thead><tbody>';

    // Rows
    matrix.forEach((row, i) => {
        html += `<tr><th>${stateNames[i]}</th>`;
        row.forEach(val => {
            const pct = (val * 100).toFixed(1);

            // Color gradient (green → yellow → orange)
            const color = getGradientColor(val);

            html += `<td style="background-color: ${color}; color: black;">${pct}%</td>`;
        });
        html += '</tr>';
    });

    html += '</tbody></table>';
    container.innerHTML = html;
}

// Load available games
async function loadGames() {
    try {
        const response = await fetch('/api/games');
        games = await response.json();

        const select = document.getElementById('game-select');
        select.innerHTML = '<option value="">-- Select a game --</option>';

        games.forEach(game => {
            const option = document.createElement('option');
            option.value = game.gameId;
            option.textContent = `${game.description} - ${game.season} (${game.n_possessions} poss)`;
            select.appendChild(option);
        });

    } catch (error) {
        console.error('Error loading games:', error);
        document.getElementById('game-select').innerHTML = '<option>Error loading games</option>';
    }
}

// Analyze selected game
async function analyzeGame() {
    const gameId = document.getElementById('game-select').value;
    if (!gameId) return;

    console.log('Analyzing game:', gameId);

    try {
        // Show loading
        const gameViz = document.getElementById('game-viz');
        gameViz.style.display = 'block';
        document.getElementById('game-title').textContent = 'Loading...';

        // Load game data
        const response = await fetch(`/api/game/${gameId}`);
        currentGame = await response.json();

        // Find game description from games list
        const gameInfo = games.find(g => g.gameId === gameId);
        const gameDesc = gameInfo ? gameInfo.description : `Game ${gameId}`;

        // Update header
        document.getElementById('game-title').textContent =
            `${gameDesc} - ${currentGame.season}`;
        document.getElementById('game-info').textContent =
            `Final Score: ${currentGame.finalScore} | ${currentGame.possessions.length} possessions`;

        // Load visualizations
        await Promise.all([
            loadMomentumTimeline(gameId),
            loadWinProbability(gameId)
        ]);

        // Display possession table
        displayPossessionTable(currentGame.possessions);

    } catch (error) {
        console.error('Error analyzing game:', error);
        alert('Error loading game data');
    }
}

// Load momentum timeline
async function loadMomentumTimeline(gameId) {
    try {
        const response = await fetch(`/api/momentum-timeline/${gameId}`);
        const figData = await response.json();

        Plotly.newPlot('momentum-timeline', figData.data, figData.layout, {
            responsive: true,
            displayModeBar: true
        });

    } catch (error) {
        console.error('Error loading momentum timeline:', error);
    }
}

// Load win probability curve
async function loadWinProbability(gameId) {
    try {
        const response = await fetch(`/api/win-probability/${gameId}`);
        const figData = await response.json();

        Plotly.newPlot('win-probability', figData.data, figData.layout, {
            responsive: true,
            displayModeBar: true
        });

    } catch (error) {
        console.error('Error loading win probability:', error);
    }
}

// Display possession-by-possession table
function displayPossessionTable(possessions) {
    const container = document.getElementById('possession-table');

    const stateNames = ['COLD', 'NEUTRAL', 'HOT'];
    const stateColors = {
        0: '#3498db',
        1: '#95a5a6',
        2: '#e74c3c'
    };

    let html = `
        <table class="possession-table">
            <thead>
                <tr>
                    <th>Poss #</th>
                    <th>Period</th>
                    <th>Clock</th>
                    <th>State</th>
                    <th>Score</th>
                    <th>Diff</th>
                    <th>Win %</th>
                    <th>Result</th>
                </tr>
            </thead>
            <tbody>
    `;

    possessions.forEach(poss => {
        const minutes = Math.floor(poss.clock / 60);
        const seconds = Math.floor(poss.clock % 60);
        const clock = `${minutes}:${seconds.toString().padStart(2, '0')}`;

        const stateName = stateNames[poss.state] || 'UNKNOWN';
        const stateColor = stateColors[poss.state] || '#95a5a6';

        html += `
            <tr>
                <td>${poss.possession}</td>
                <td>Q${poss.period}</td>
                <td>${clock}</td>
                <td style="background-color: ${stateColor}; color: white; font-weight: bold;">
                    ${stateName}
                </td>
                <td>${poss.teamScore}-${poss.oppScore}</td>
                <td style="color: ${poss.scoreDiff >= 0 ? '#27ae60' : '#e74c3c'}">
                    ${poss.scoreDiff >= 0 ? '+' : ''}${poss.scoreDiff}
                </td>
                <td>${(poss.winProb * 100).toFixed(1)}%</td>
                <td class="result-cell">
                    ${poss.pointsScored > 0 ? `<strong>+${poss.pointsScored}</strong>` : '--'}
                </td>
            </tr>
        `;
    });

    html += '</tbody></table>';
    container.innerHTML = html;
}
