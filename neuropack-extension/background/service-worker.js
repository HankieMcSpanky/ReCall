const DEFAULT_URL = "http://localhost:7341";

async function getSettings() {
  const s = await chrome.storage.local.get(["serverUrl", "authToken"]);
  return { url: s.serverUrl || DEFAULT_URL, token: s.authToken || "" };
}

function headers(token) {
  const h = { "Content-Type": "application/json" };
  if (token) h["Authorization"] = "Bearer " + token;
  return h;
}

// Context menu
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "save-to-neuropack",
    title: "Save to NeuroPack",
    contexts: ["selection"]
  });
});

chrome.contextMenus.onClicked.addListener(async (info) => {
  if (info.menuItemId !== "save-to-neuropack" || !info.selectionText) return;

  const { url, token } = await getSettings();
  try {
    const r = await fetch(url + "/v1/memories", {
      method: "POST",
      headers: headers(token),
      body: JSON.stringify({
        content: info.selectionText,
        tags: ["context-menu"],
        source: "chrome-extension"
      })
    });
    if (!r.ok) throw new Error(r.status);
    chrome.action.setBadgeText({ text: "✓", tabId: info.tab?.id });
    chrome.action.setBadgeBackgroundColor({ color: "#6c63ff" });
    setTimeout(() => chrome.action.setBadgeText({ text: "", tabId: info.tab?.id }), 2000);
  } catch {
    chrome.action.setBadgeText({ text: "!", tabId: info.tab?.id });
    chrome.action.setBadgeBackgroundColor({ color: "#ef4444" });
    setTimeout(() => chrome.action.setBadgeText({ text: "", tabId: info.tab?.id }), 2000);
  }
});

// Periodic health check + badge count
chrome.alarms.create("health-check", { periodInMinutes: 0.5 });

chrome.alarms.onAlarm.addListener(async (alarm) => {
  if (alarm.name !== "health-check") return;
  const { url, token } = await getSettings();
  try {
    const r = await fetch(url + "/v1/stats", { headers: headers(token) });
    if (!r.ok) return;
    const data = await r.json();
    const count = data.total_memories || 0;
    chrome.action.setBadgeText({ text: count > 0 ? String(count) : "" });
    chrome.action.setBadgeBackgroundColor({ color: "#6c63ff" });
  } catch {
    // Server offline, clear badge
    chrome.action.setBadgeText({ text: "" });
  }
});
