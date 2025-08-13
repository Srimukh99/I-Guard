(function () {
  async function loadEvents() {
    try {
      const response = await fetch('/api/events', { cache: 'no-cache' });
      if (!response.ok) {
        console.error('Failed to fetch events', response.statusText);
        return;
      }
      const events = await response.json();
      const container = document.getElementById('events-container');
      if (!container) {
        return;
      }
      if (!events || events.length === 0) {
        container.textContent = 'No events detected.';
        return;
      }
      let html = '<table><thead><tr><th>Time</th><th>Camera</th><th>Flags</th><th>Score</th><th>Clip</th></tr></thead><tbody>';
      events.forEach(function (ev) {
        const date = new Date(ev.timestamp * 1000).toLocaleString();
        // Extract flag names where the value is truthy
        const flags = Object.keys(ev.flags || {}).filter(function (k) {
          return ev.flags[k];
        }).join(', ');
        const score = (ev.score !== null && ev.score !== undefined)
          ? Number(ev.score).toFixed(2)
          : '-';
        let clipCell;
        if (ev.clip_file) {
          const safeFilename = encodeURIComponent(ev.clip_file);
          clipCell =
            '<video width="160" controls muted>' +
            '<source src="/clips/' + safeFilename + '" type="video/mp4" />' +
            'Your browser does not support the video tag.' +
            '</video>';
        } else {
          clipCell = 'No clip available';
        }
        html +=
          '<tr>' +
          '<td>' + date + '</td>' +
          '<td>' + ev.camera_id + '</td>' +
          '<td>' + flags + '</td>' +
          '<td>' + score + '</td>' +
          '<td>' + clipCell + '</td>' +
          '</tr>';
      });
      html += '</tbody></table>';
      container.innerHTML = html;
    } catch (err) {
      console.error('Error loading events', err);
    }
  }
  window.addEventListener('DOMContentLoaded', function () {
    loadEvents();
    // Refresh events every 5 seconds
    setInterval(loadEvents, 5000);
  });
})();