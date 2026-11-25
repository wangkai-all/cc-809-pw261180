// 共享工具: 取数、经校验的分类色板、SVG 构建、tooltip 与图例
const PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                 "#008300", "#6250d6", "#e34948"];
const DARK_PALETTE = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181",
                      "#008300", "#9085e9", "#e66767"];

function catColor(i) {
  const dark = window.matchMedia &&
    window.matchMedia("(prefers-color-scheme: dark)").matches &&
    document.documentElement.dataset.theme !== "light";
  return (dark ? DARK_PALETTE : PALETTE)[i % 8];
}

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try { msg = (await res.json()).error || msg; } catch (e) { /* ignore */ }
    throw new Error(msg);
  }
  return res.json();
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function")
      node.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) node.setAttribute(k, v);
  }
  for (const c of [].concat(children)) {
    if (c === null || c === undefined) continue;
    node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return node;
}

function fmt(x, d = 3) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return Number(x).toFixed(d);
}

// ---- 全局 tooltip ----
let tip;
function ensureTip() {
  if (!tip) {
    tip = el("div", { class: "svg-tip" });
    document.body.appendChild(tip);
  }
  return tip;
}
function showTip(html, evt) {
  const t = ensureTip();
  t.innerHTML = html;
  t.style.display = "block";
  moveTip(evt);
}
function moveTip(evt) {
  if (!tip) return;
  const pad = 14;
  let x = evt.clientX + pad, y = evt.clientY + pad;
  const r = tip.getBoundingClientRect();
  if (x + r.width > window.innerWidth - 8) x = evt.clientX - r.width - pad;
  if (y + r.height > window.innerHeight - 8) y = evt.clientY - r.height - pad;
  tip.style.left = x + "px";
  tip.style.top = y + "px";
}
function hideTip() { if (tip) tip.style.display = "none"; }

function legend(container, labels) {
  const wrap = el("div", { class: "legend" });
  labels.forEach((lab, i) => {
    wrap.appendChild(el("span", { class: "item" }, [
      el("span", { class: "swatch", style: `background:${catColor(i)}` }),
      el("span", {}, lab),
    ]));
  });
  container.appendChild(wrap);
  return wrap;
}

// 水平堆叠概率条 (带 2px 表面间隙, 段内直接标签 -> 满足对比度 relief 规则)
function probabilityBar(probs, labels) {
  const bar = el("div", { class: "prob-bar" });
  probs.forEach((p, i) => {
    const pct = Math.max(0, p) * 100;
    const seg = el("div", {
      class: "prob-seg",
      style: `width:${pct}%;background:${catColor(i)}`,
    });
    seg.addEventListener("mousemove", (e) => {
      showTip(`<b>${labels[i]}</b><br>概率 ${fmt(p, 4)} (${fmt(pct, 1)}%)`, e);
    });
    seg.addEventListener("mouseleave", hideTip);
    bar.appendChild(seg);
  });
  return bar;
}

function badge(kind, text) {
  return el("span", { class: `badge ${kind}` }, text);
}

function problemBadge(level) {
  return badge(level === "error" ? "bad" : "warn",
               level === "error" ? "问题" : "警告");
}

function setActiveNav() {
  const page = location.pathname.split("/").pop() || "index.html";
  document.querySelectorAll("nav a").forEach((a) => {
    if (a.getAttribute("href") === page) a.classList.add("active");
  });
}

function renderEmpty(container) {
  container.appendChild(el("div", { class: "card" }, [
    el("h2", {}, "暂无数据"),
    el("p", { class: "secondary" },
      "请先通过 CLI 导入 CSV 或生成虚构样例, 例如:"),
    el("code", {}, "python -m ordinal_app.cli synth --root <目录> " +
                   "--scenario base --batch b1"),
  ]));
}
