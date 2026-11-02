"use strict";
/* Real Chrome MV3 test. Requests are fulfilled locally; no site writes/network. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const os = require("node:os");
const { spawn } = require("node:child_process");
const EXT = path.resolve(__dirname, "..");
const LIVE = process.argv.includes("--live");
const RESEARCH = path.resolve(EXT, "../../experiments/ladder_scout_research");
const PROFILE = fs.mkdtempSync(path.join(os.tmpdir(), "botlab-scout-chrome-"));
const read = (name) => JSON.parse(fs.readFileSync(path.join(RESEARCH, name), "utf8"));
const ladder = read("ladder.json"), targetHistory = read("history_target.json"), mineHistory = read("history_me.json");
const targetId = targetHistory.player.id, myId = ladder.me.id;
const secondId = ladder.results.find((r) => r.id !== targetId && r.id !== myId).id;
const unauthId = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa";
const requests = [], failures = [], checks = [];
const chrome = spawn(process.env.CHROME_BINARY || "/opt/google/chrome/chrome", [
  "--headless=new", "--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu",
  "--remote-debugging-pipe", "--enable-unsafe-extension-debugging",
  "--disable-background-networking", "--disable-component-update", "--no-first-run",
  "--no-default-browser-check", "--disable-sync", "--user-data-dir=" + PROFILE, "about:blank"
], { stdio: ["ignore", "pipe", "pipe", "pipe", "pipe"] });
let stderr = "", pendingBuffer = Buffer.alloc(0), nextId = 1;
const pending = new Map(), listeners = new Map();
for (const pipe of [chrome.stdio[3], chrome.stdio[4]]) pipe.on("error", () => {});
chrome.stderr.on("data", (chunk) => { stderr = (stderr + chunk.toString()).slice(-20000); });
chrome.on("exit", () => { for (const item of pending.values()) item.reject(new Error("Chrome exited: " + stderr)); pending.clear(); });
chrome.stdio[4].on("data", (chunk) => {
  pendingBuffer = Buffer.concat([pendingBuffer, chunk]);
  let index;
  while ((index = pendingBuffer.indexOf(0)) >= 0) {
    const packet = JSON.parse(pendingBuffer.subarray(0, index).toString());
    pendingBuffer = pendingBuffer.subarray(index + 1);
    if (packet.id) {
      const item = pending.get(packet.id);
      if (!item) continue;
      pending.delete(packet.id); clearTimeout(item.timer);
      packet.error ? item.reject(new Error(JSON.stringify(packet.error))) : item.resolve(packet.result);
    } else for (const handler of listeners.get(packet.method) || []) {
      Promise.resolve(handler(packet.params, packet.sessionId)).catch((e) => failures.push(e.message));
    }
  }
});
function call(method, params = {}, sessionId) {
  return new Promise((resolve, reject) => {
    const id = nextId++;
    const timer = setTimeout(() => { pending.delete(id); reject(new Error("CDP timeout: " + method)); }, 20000);
    pending.set(id, { resolve, reject, timer });
    chrome.stdio[3].write(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }) + "\0");
  });
}
function on(method, handler) { listeners.set(method, [...listeners.get(method) || [], handler]); }
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
let session;
async function evaluate(expression, sid = session) {
  const value = await call("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true }, sid);
  if (value.exceptionDetails) throw new Error(JSON.stringify(value.exceptionDetails));
  return value.result.value;
}
const shadow = 'document.querySelector("botlab-ladder-scout")?.shadowRoot';
const text = `(${shadow})?.textContent || ""`;
async function until(expression, note, timeout = 16000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) {
    if (await evaluate(expression)) { checks.push(note); return; }
    await delay(100);
  }
  throw new Error("Timed out: " + note + "\nPanel: " + await evaluate(text));
}
function fixturePlayer(player) {
  if (!player) return player;
  const copy = structuredClone(player);
  copy.username = copy.id === myId ? "我的模型" : copy.id === targetId ? "示例对手" : "另一位用户";
  if (copy.model) copy.model.name = copy.id === targetId ? "精确搜索模型" : "示例模型";
  return copy;
}
function fixtureSeries(rows) {
  return rows.map((s) => ({ ...s, player_a: fixturePlayer(s.player_a), player_b: fixturePlayer(s.player_b) }));
}
on("Fetch.requestPaused", async (event, sid) => {
  const u = new URL(event.request.url);
  requests.push({ method: event.request.method, path: u.pathname + u.search });
  assert.equal(event.request.method, "GET", "Extension must never write to the site");
  if (LIVE) { await call("Fetch.continueRequest", { requestId: event.requestId }, sid); return; }
  let data, status = 200, type = "application/json";
  if (u.pathname === "/api/v1/ladder") {
    data = { ...ladder, me: fixturePlayer(ladder.me), results: ladder.results.map(fixturePlayer) };
  } else if (u.pathname === "/api/v1/ladder/series") {
    const id = u.searchParams.get("player"), offset = Number(u.searchParams.get("offset") || 0);
    if (id === unauthId) { status = 401; data = { detail: "Login required" }; }
    else if (id === targetId || id === myId) {
      const history = id === targetId ? targetHistory : mineHistory;
      data = { player: fixturePlayer(history.player), results: fixtureSeries(history.results.slice(offset, offset + 20)),
        next_offset: offset + 20 < history.results.length ? offset + 20 : null };
    } else {
      const player = fixturePlayer(ladder.results.find((r) => r.id === id));
      player.username = '<img src="https://example.invalid/x" onerror="alert(1)">';
      data = { player, results: [], next_offset: null };
    }
  } else {
    type = "text/html";
    data = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Botlab fixture</title><style>body{margin:0;padding:60px;background:#eef2f8;color:#33465f;font:16px system-ui}main{max-width:700px}small{color:#8b98aa}</style><main><small>本地浏览器回归 · 数据快照</small><h1>计分战绩</h1><p>插件会自动分析当前模型，区分开局必败与可避免失分。</p><p>此测试只回放本地接口快照，不会发起对战。</p></main></html>';
  }
  await call("Fetch.fulfillRequest", { requestId: event.requestId, responseCode: status,
    responseHeaders: [{ name: "Content-Type", value: type + "; charset=utf-8" }],
    body: Buffer.from(type === "application/json" ? JSON.stringify(data) : data).toString("base64") }, sid);
});
on("Runtime.exceptionThrown", (event) => { failures.push(event.exceptionDetails?.exception?.description || event.exceptionDetails?.text); });
async function main() {
  const version = await call("Browser.getVersion");
  const loaded = await call("Extensions.loadUnpacked", { path: EXT });
  const extensionId = loaded.id;
  assert.ok(extensionId, JSON.stringify(loaded));
  const target = await call("Target.createTarget", { url: "about:blank" });
  session = (await call("Target.attachToTarget", { targetId: target.targetId, flatten: true })).sessionId;
  await call("Runtime.enable", {}, session); await call("Page.enable", {}, session);
  await call("Network.enable", {}, session);
  if (LIVE) {
    const raw = process.env.BOTLAB_COOKIE;
    if (!raw) throw new Error("BOTLAB_COOKIE is required for live read-only smoke test");
    const cookies = raw.split(";").map((part) => part.trim()).filter((part) => /^(sessionid|csrftoken)=/.test(part))
      .map((part) => { const at = part.indexOf("="); return { name: part.slice(0, at), value: part.slice(at + 1),
        domain: "botlab.aiphys.cn", path: "/", secure: true, httpOnly: part.startsWith("sessionid=") }; });
    await call("Network.setCookies", { cookies }, session);
  }
  await call("Emulation.setDeviceMetricsOverride", { width: 1440, height: 1040, deviceScaleFactor: 1, mobile: false }, session);
  await call("Fetch.enable", { patterns: [{ urlPattern: "https://botlab.aiphys.cn/*" }] }, session);
  await call("Page.navigate", { url: "https://botlab.aiphys.cn/ladder/history?player=" + targetId }, session);
  if (LIVE) {
    await until(`(${text}).includes("正常系列样本") && !(${text}).includes("数据尚未读完") && !(${text}).includes("正在核对我方")`,
      "Extension works on authenticated live Botlab SPA", 30000);
    const rectangle = await evaluate(`{const r=(${shadow}).querySelector('.panel').getBoundingClientRect();({width:r.width,height:r.height,x:r.x,y:r.y})}`);
    assert.ok(rectangle.width > 300 && rectangle.width < 500 && rectangle.y >= 0);
    assert.equal(requests.filter((r) => r.method !== "GET").length, 0);
    assert.deepEqual(failures, []);
    const report = { passed: true, chrome: version.product, checks, panel: rectangle,
      apiReads: requests.filter((r) => r.path.startsWith("/api/")).length,
      siteWrites: 0, credentials: "Read only from environment, temporary browser profile deleted" };
    fs.writeFileSync(path.join(EXT, "docs/live-browser-test.json"), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report, null, 2)); return;
  }
  await until(`(${text}).includes("偏向跳过") && (${text}).includes("24 组") && !(${text}).includes("正在核对我方")`, "MV3 auto injection and full pagination");
  assert.ok((await evaluate(text)).includes("37 组旧模型")); checks.push("Current fingerprint filters 37 legacy-model series");
  assert.ok((await evaluate(text)).includes("今天已与该用户计分")); checks.push("Already-played status");
  assert.ok((await evaluate(text)).includes("-0.7 分/组")); checks.push("Rounded-per-series scenario matches independent calculation");
  const forbidden = requests.filter((r) => r.path.startsWith("/api/") && !/^\/api\/v1\/ladder(?:$|\/series\?)/.test(r.path));
  assert.deepEqual(forbidden, []); checks.push("Only allowed read-only ladder endpoints");
  const firstCount = requests.length;
  await evaluate(`{const r=${shadow};const i=r.querySelector('[name="myRating"]');i.value='1700';i.dispatchEvent(new Event('change',{bubbles:true}));}`);
  await delay(150);
  assert.equal(requests.length, firstCount); checks.push("Rating changes recalculate without requests");
  await evaluate(`{const r=${shadow};const i=r.querySelector('[name="myRating"]');i.value='1267';i.dispatchEvent(new Event('change',{bubbles:true}));}`);
  await evaluate(`{const r=${shadow};const i=r.querySelector('[name="assumeOptimal"]');i.checked=false;i.dispatchEvent(new Event('change',{bubbles:true}));}`);
  await until(`(${text}).includes("双方当前模型历史交手（1组）") && (${text}).includes("样本偏少")`, "Head-to-head uncertainty uses its actual sample count");
  await evaluate(`{const r=${shadow};const i=r.querySelector('[name="assumeOptimal"]');i.checked=true;i.dispatchEvent(new Event('change',{bubbles:true}));}`);
  const screenshot = await call("Page.captureScreenshot", { format: "png", captureBeyondViewport: false,
    clip: { x: 990, y: 0, width: 450, height: 1040, scale: 1 } }, session);
  fs.writeFileSync(path.join(EXT, "docs/preview.png"), Buffer.from(screenshot.data, "base64"));
  await evaluate(`(${shadow}).querySelector('[data-action="collapse"]').click()`);
  await until(`Boolean((${shadow})?.querySelector('.chip'))`, "Collapse control");
  await evaluate(`(${shadow}).querySelector('[data-action="expand"]').click()`);
  await until(`Boolean((${shadow})?.querySelector('.panel'))`, "Expand control");
  await evaluate(`history.pushState({},'', '/ladder');`);
  await until(`!document.querySelector('botlab-ladder-scout')`, "SPA departure removes panel");
  await evaluate(`history.pushState({},'', '/ladder/history?player=${secondId}');`);
  await until(`(${text}).includes("证据不足")`, "SPA target switch clears the prior recommendation");
  assert.equal(await evaluate(`(${shadow}).querySelectorAll('img').length`), 0); checks.push("Model/player text is HTML-escaped");
  await evaluate(`history.pushState({},'', '/ladder/history?player=${targetId}');`);
  await until(`(${text}).includes("偏向跳过") && (${text}).includes("本机缓存")`, "Cached revisit survives target/my concurrent cache writes");
  await evaluate(`history.pushState({},'', '/ladder/history?player=${unauthId}');`);
  await until(`(${text}).includes("请先在 Botlab 登录")`, "Authentication errors do not produce forecasts");
  await evaluate(`history.pushState({},'', '/ladder/history?player=${targetId}');`);
  await until(`(${text}).includes("偏向跳过")`, "Recovery after authentication error");
  const popup = await call("Target.createTarget", { url: "chrome-extension://" + extensionId + "/popup.html" });
  const popupSession = (await call("Target.attachToTarget", { targetId: popup.targetId, flatten: true })).sessionId;
  for (let i = 0; i < 40; i++) {
    if (await evaluate("Boolean(document.getElementById('enabled'))", popupSession)) break;
    await delay(50);
  }
  await evaluate("document.getElementById('enabled').click()", popupSession);
  await until(`!document.querySelector('botlab-ladder-scout')`, "Popup disables existing tabs");
  await evaluate("document.getElementById('enabled').click()", popupSession);
  await until(`(${text}).includes("偏向跳过")`, "Popup re-enables existing tabs");
  await call("Target.closeTarget", { targetId: popup.targetId });
  assert.deepEqual(failures, []);
  const report = { passed: true, chrome: version.product, extensionId, checks,
    writes: requests.filter((r) => r.method !== "GET").length, interceptedRequests: requests.length,
    network: "All Botlab requests fulfilled from local snapshots; no actual matches or site changes" };
  fs.writeFileSync(path.join(EXT, "docs/browser-test.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
}
main().catch((error) => { console.error(error.stack); console.error(JSON.stringify({ requests, failures })); console.error(stderr); process.exitCode = 1; })
  .finally(async () => { try { await call("Browser.close"); } catch (_) {} chrome.kill(); setTimeout(() => { fs.rmSync(PROFILE, { recursive: true, force: true }); }, 100); });
