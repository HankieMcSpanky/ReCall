const DEFAULT_URL = "http://localhost:7341";

async function loadSettings() {
  const s = await chrome.storage.local.get(["serverUrl", "authToken", "enabledSites"]);
  document.getElementById("server-url").value = s.serverUrl || DEFAULT_URL;
  document.getElementById("auth-token").value = s.authToken || "";

  const sites = s.enabledSites || { chatgpt: true, claude: true, gemini: true };
  document.getElementById("site-chatgpt").checked = sites.chatgpt !== false;
  document.getElementById("site-claude").checked = sites.claude !== false;
  document.getElementById("site-gemini").checked = sites.gemini !== false;
}

async function saveSettings() {
  const serverUrl = document.getElementById("server-url").value.trim() || DEFAULT_URL;
  const authToken = document.getElementById("auth-token").value.trim();
  const enabledSites = {
    chatgpt: document.getElementById("site-chatgpt").checked,
    claude: document.getElementById("site-claude").checked,
    gemini: document.getElementById("site-gemini").checked
  };

  await chrome.storage.local.set({ serverUrl, authToken, enabledSites });
  showStatus("Settings saved!", false);
}

async function testConnection() {
  const url = document.getElementById("server-url").value.trim() || DEFAULT_URL;
  const token = document.getElementById("auth-token").value.trim();

  const h = { "Content-Type": "application/json" };
  if (token) h["Authorization"] = "Bearer " + token;

  try {
    const r = await fetch(url + "/health");
    if (!r.ok) throw new Error("HTTP " + r.status);
    showStatus("Connected to NeuroPack!", false);
  } catch {
    showStatus("Cannot connect to " + url, true);
  }
}

function showStatus(msg, isError) {
  const el = document.getElementById("status-msg");
  el.textContent = msg;
  el.className = "status-msg " + (isError ? "err" : "ok");
  setTimeout(() => { el.className = "status-msg"; }, 3000);
}

document.addEventListener("DOMContentLoaded", () => {
  loadSettings();
  document.getElementById("save-btn").addEventListener("click", saveSettings);
  document.getElementById("test-btn").addEventListener("click", testConnection);
});
