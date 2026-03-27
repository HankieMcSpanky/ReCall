(function() {
  "use strict";

  const SITE_CONFIG = {
    "chatgpt.com": {
      name: "chatgpt",
      selectors: [
        '[data-message-author-role] .markdown',
        '.text-message .markdown',
        'article .prose'
      ]
    },
    "chat.openai.com": {
      name: "chatgpt",
      selectors: [
        '[data-message-author-role] .markdown',
        '.text-message .markdown',
        'article .prose'
      ]
    },
    "claude.ai": {
      name: "claude",
      selectors: [
        '.font-claude-message',
        '[data-testid="chat-message-text"]',
        '.prose'
      ]
    },
    "gemini.google.com": {
      name: "gemini",
      selectors: [
        '.conversation-container',
        '.response-container',
        'message-content'
      ]
    }
  };

  const host = location.hostname.replace(/^www\./, "");
  const site = SITE_CONFIG[host];
  if (!site) return;

  const DEFAULT_URL = "http://localhost:7341";

  async function getSettings() {
    const s = await chrome.storage.local.get(["serverUrl", "authToken", "enabledSites"]);
    const enabled = s.enabledSites || { chatgpt: true, claude: true, gemini: true };
    return {
      url: s.serverUrl || DEFAULT_URL,
      token: s.authToken || "",
      enabled: enabled[site.name] !== false
    };
  }

  function extractConversation() {
    for (const sel of site.selectors) {
      const els = document.querySelectorAll(sel);
      if (els.length > 0) {
        return Array.from(els).map(el => el.innerText.trim()).filter(Boolean).join("\n\n---\n\n");
      }
    }
    return null;
  }

  function showToast(msg, isError) {
    const t = document.createElement("div");
    t.textContent = msg;
    Object.assign(t.style, {
      position: "fixed", bottom: "80px", right: "20px", zIndex: "1000000",
      padding: "10px 18px", borderRadius: "8px", fontSize: "13px",
      color: "#fff", fontFamily: "system-ui, sans-serif",
      background: isError ? "#ef4444" : "#6c63ff",
      boxShadow: "0 4px 12px rgba(0,0,0,0.3)",
      transition: "opacity 0.3s", opacity: "1"
    });
    document.body.appendChild(t);
    setTimeout(() => { t.style.opacity = "0"; setTimeout(() => t.remove(), 300); }, 2200);
  }

  async function saveConversation() {
    const settings = await getSettings();
    if (!settings.enabled) {
      showToast("NeuroPack disabled for " + site.name, true);
      return;
    }

    const text = extractConversation();
    if (!text || text.length < 20) {
      showToast("No conversation found", true);
      return;
    }

    const headers = { "Content-Type": "application/json" };
    if (settings.token) headers["Authorization"] = "Bearer " + settings.token;

    try {
      const r = await fetch(settings.url + "/v1/memories", {
        method: "POST",
        headers,
        body: JSON.stringify({
          content: text,
          tags: ["ai-conversation", site.name],
          source: site.name
        })
      });
      if (!r.ok) throw new Error(r.status);
      showToast("Saved to NeuroPack!");
    } catch {
      showToast("Save failed — is NeuroPack running?", true);
    }
  }

  // Floating button
  const btn = document.createElement("div");
  btn.id = "neuropack-fab";
  btn.title = "Save conversation to NeuroPack";
  btn.innerHTML = `<svg width="28" height="28" viewBox="0 0 28 28"><circle cx="14" cy="14" r="13" fill="#6c63ff"/><text x="14" y="18" text-anchor="middle" fill="#fff" font-size="10" font-weight="bold" font-family="system-ui">NP</text></svg>`;
  Object.assign(btn.style, {
    position: "fixed", bottom: "24px", right: "24px", zIndex: "999999",
    width: "44px", height: "44px", borderRadius: "50%", cursor: "pointer",
    display: "flex", alignItems: "center", justifyContent: "center",
    boxShadow: "0 4px 14px rgba(108,99,255,0.5)",
    transition: "transform 0.15s, box-shadow 0.15s",
    background: "transparent"
  });
  btn.addEventListener("mouseenter", () => {
    btn.style.transform = "scale(1.1)";
    btn.style.boxShadow = "0 6px 20px rgba(108,99,255,0.7)";
  });
  btn.addEventListener("mouseleave", () => {
    btn.style.transform = "scale(1)";
    btn.style.boxShadow = "0 4px 14px rgba(108,99,255,0.5)";
  });
  btn.addEventListener("click", saveConversation);
  document.body.appendChild(btn);
})();
