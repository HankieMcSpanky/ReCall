const DEFAULT_URL = "http://localhost:7341";
let serverUrl = DEFAULT_URL;
let authToken = "";
let searchTimer = null;

async function loadSettings() {
  const s = await chrome.storage.local.get(["serverUrl", "authToken"]);
  serverUrl = s.serverUrl || DEFAULT_URL;
  authToken = s.authToken || "";
}

function headers() {
  const h = {"Content-Type": "application/json"};
  if (authToken) h["Authorization"] = "Bearer " + authToken;
  return h;
}

async function api(method, path, body) {
  const opts = {method, headers: headers()};
  if (body) opts.body = JSON.stringify(body);
  const r = await fetch(serverUrl + path, opts);
  if (!r.ok) throw new Error(r.status);
  return r.json();
}

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function toast(msg, type) {
  const el = document.createElement("div");
  el.className = "toast " + (type || "success");
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 2500);
}

async function checkHealth() {
  try {
    await api("GET", "/health");
    document.getElementById("status-dot").className = "dot ok";
    document.getElementById("status-text").textContent = "Connected";
  } catch {
    document.getElementById("status-dot").className = "dot";
    document.getElementById("status-text").textContent = "Offline";
  }
}

function renderResults(items, isSearch) {
  const list = document.getElementById("results-list");
  const title = document.getElementById("results-title");
  list.innerHTML = "";
  title.textContent = isSearch ? "Results" : "Recent";

  if (items.length === 0) {
    list.innerHTML = '<div class="empty">No memories found</div>';
    return;
  }

  items.forEach(item => {
    const div = document.createElement("div");
    div.className = "result";
    const l3 = esc(item.l3_abstract || item.content_preview || "");
    const tags = (item.tags || []).join(", ");
    let metaHtml = "";
    if (isSearch && item.score != null) {
      metaHtml += '<span class="score">' + item.score.toFixed(3) + '</span>';
    }
    if (tags) metaHtml += '<span class="tags">' + esc(tags) + '</span>';
    div.innerHTML = '<div class="l3">' + l3 + '</div><div class="meta">' + metaHtml + '</div>';
    list.appendChild(div);
  });
}

async function search(query) {
  if (query.length < 2) {
    await loadRecent();
    return;
  }
  try {
    const data = await api("POST", "/v1/recall", {query, limit: 10});
    renderResults(data.results, true);
  } catch {
    document.getElementById("results-list").innerHTML = '<div class="empty">Search failed</div>';
  }
}

async function loadRecent() {
  try {
    const data = await api("GET", "/v1/memories?limit=5");
    renderResults(data, false);
  } catch {
    document.getElementById("results-list").innerHTML = '<div class="empty">Cannot connect</div>';
  }
}

async function save() {
  const content = document.getElementById("save-content").value.trim();
  if (!content) return;
  const tagsStr = document.getElementById("save-tags").value.trim();
  const tags = tagsStr ? tagsStr.split(",").map(t => t.trim()).filter(Boolean) : [];
  try {
    await api("POST", "/v1/memories", {content, tags, source: "chrome-extension"});
    document.getElementById("save-content").value = "";
    document.getElementById("save-tags").value = "";
    toast("Saved!");
    await loadRecent();
  } catch (e) {
    toast("Save failed", "error");
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  await loadSettings();
  await checkHealth();
  await loadRecent();

  document.getElementById("search").addEventListener("input", e => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => search(e.target.value.trim()), 300);
  });

  document.getElementById("save-btn").addEventListener("click", save);

  document.getElementById("save-content").addEventListener("keydown", e => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) save();
  });
});
