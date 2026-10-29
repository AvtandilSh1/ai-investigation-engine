async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  const text = await res.text();
  let data;
  try { data = JSON.parse(text); } catch { data = { detail: text }; }
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

async function refreshHealth() {
  try {
    const h = await api("/api/health");
    const el = document.getElementById("status");
    el.textContent = `${h.document_count} docs · ${h.chunk_count} chunks · llm ${h.llm_configured ? "on" : "off"}`;
  } catch (e) {
    document.getElementById("status").textContent = "backend offline";
  }
}

async function refreshDocs() {
  const docs = await api("/api/documents");
  const body = document.getElementById("docBody");
  body.innerHTML = "";
  for (const d of docs) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escapeHtml(d.title || d.filename)}<div style="color:#5c564c;font-size:0.8rem">${escapeHtml(d.filename)}</div></td>
      <td>${escapeHtml(d.format)}</td>
      <td>${d.chunk_count}</td>
      <td><button class="linkish" data-id="${d.id}">delete</button></td>`;
    tr.querySelector("button").onclick = async () => {
      if (!confirm("Delete this document?")) return;
      await api(`/api/documents/${d.id}`, { method: "DELETE" });
      await refreshDocs();
      await refreshHealth();
    };
    body.appendChild(tr);
  }
}

document.getElementById("uploadForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const msg = document.getElementById("uploadMsg");
  const file = document.getElementById("fileInput").files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append("file", file);
  const strategy = document.getElementById("strategy").value;
  msg.textContent = "indexing…";
  try {
    const rec = await api(`/api/documents/upload?strategy=${encodeURIComponent(strategy)}`, {
      method: "POST",
      body: fd,
    });
    msg.textContent = rec.deduplicated
      ? "duplicate content skipped"
      : `indexed ${rec.filename} (${rec.chunk_count} chunks)`;
    document.getElementById("fileInput").value = "";
    await refreshDocs();
    await refreshHealth();
  } catch (err) {
    msg.textContent = String(err.message || err);
  }
});

document.getElementById("askBtn").addEventListener("click", async () => {
  const question = document.getElementById("question").value.trim();
  const msg = document.getElementById("askMsg");
  const btn = document.getElementById("askBtn");
  if (!question) {
    msg.textContent = "enter a question";
    return;
  }
  btn.disabled = true;
  msg.textContent = "investigating… (first request after sleep can take up to a minute)";
  try {
    await refreshHealth();
    const top_k = Number(document.getElementById("topK").value || 8);
    const data = await api("/api/investigate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k }),
    });
    renderResult(data);
    msg.textContent = "";
  } catch (err) {
    const raw = String(err.message || err);
    if (raw === "Failed to fetch" || raw.includes("NetworkError") || raw.includes("fetch")) {
      msg.textContent =
        "Failed to fetch — free server may be waking up. Wait 30–60 seconds, refresh the page, then try again.";
    } else {
      msg.textContent = raw;
    }
  } finally {
    btn.disabled = false;
  }
});

function renderResult(data) {
  const box = document.getElementById("result");
  box.classList.remove("hidden");
  document.getElementById("abstain").textContent = data.abstained ? "abstained" : "answered";
  document.getElementById("confidence").textContent = `confidence ${Number(data.confidence).toFixed(2)}`;
  document.getElementById("retrieved").textContent = `retrieved ${data.retrieved_count}`;
  document.getElementById("answer").textContent = data.answer || "";

  const cites = document.getElementById("citations");
  cites.innerHTML = "";
  (data.citations || []).forEach((c) => {
    const li = document.createElement("li");
    li.innerHTML = `<strong>${escapeHtml(c.document_name)}</strong> <span class="cite">${escapeHtml(c.excerpt)}</span>`;
    cites.appendChild(li);
  });
  if (!data.citations?.length) cites.innerHTML = "<li>none</li>";

  const contras = document.getElementById("contras");
  contras.innerHTML = "";
  (data.contradictions || []).forEach((c) => {
    const li = document.createElement("li");
    li.textContent = `${c.topic}: ${c.detail} (${(c.sources || []).join(" vs ")})`;
    contras.appendChild(li);
  });
  if (!data.contradictions?.length) contras.innerHTML = "<li>none detected</li>";

  const steps = document.getElementById("steps");
  steps.innerHTML = "";
  (data.steps || []).forEach((s) => {
    const li = document.createElement("li");
    li.textContent = s;
    steps.appendChild(li);
  });
}

function escapeHtml(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

refreshHealth();
refreshDocs().catch(() => {});
