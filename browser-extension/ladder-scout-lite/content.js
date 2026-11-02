(function () {
  "use strict";
  if (window.top !== window || globalThis.__botlabLiteInstalled) return;
  globalThis.__botlabLiteInstalled = true;
  const C = globalThis.BotlabScoutCore, BASE = "https://botlab.aiphys.cn";
  const cache = new Map();
  let route = "", host, root, controller, epoch = 0, data = {}, collapsed = false;
  const escape = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
  const sign = (n) => Number.isFinite(n) ? (n > 0 ? "+" : "") + n : "—";
  async function get(path, signal) {
    if (!/^\/api\/v1\/ladder(?:$|\/series\?)/.test(path)) throw new Error("不支持的请求");
    const response = await fetch(BASE + path, { method: "GET", credentials: "include",
      cache: "no-store", headers: { Accept: "application/json" }, signal });
    if ([401, 403].includes(response.status)) throw new Error("请先登录 Botlab，或检查战绩访问权限。");
    if (!response.ok) throw new Error("读取失败（HTTP " + response.status + "），点 ↻ 重试。");
    const result = await response.json();
    if (!Array.isArray(result.results)) throw new Error("接口格式已变化，暂不判断。");
    return result;
  }
  function mount() {
    if (host?.isConnected) return;
    host = document.createElement("botlab-ladder-scout-lite");
    host.id = "botlab-ladder-scout-lite";
    host.style.cssText = "position:relative;z-index:2147483000";
    root = host.attachShadow({ mode: "open" });
    root.addEventListener("click", (event) => {
      const action = event.target.closest("[data-action]")?.dataset.action;
      if (action === "refresh") load(true);
      if (action === "collapse" || action === "expand") { collapsed = action === "collapse"; render(); }
    });
    document.body.appendChild(host);
  }
  function render() {
    if (!root) return;
    if (collapsed) { root.innerHTML = styles() + '<button class="chip" data-action="expand">战绩简析</button>'; return; }
    let body = '<p class="muted">正在读取积分和战绩…</p>';
    if (data.error) body = '<p class="error" role="alert">' + escape(data.error) + '</p>';
    else if (data.history?.player && data.ladder) {
      const h = data.history, me = data.ladder.me;
      const live = data.ladder.results.find((p) => p.id === h.player.id);
      const player = live || h.player;
      const p = C.analyzeHistory(h.results, player.id, player.model, { limit: 400 });
      const e = C.elo(me?.rating, player.rating, data.ladder.k_factor || 32);
      const own = me?.id === player.id;
      let title, description, color = "neutral";
      if (data.loading) { title = "正在补齐分页"; description = "当前结论待全部已读战绩核对后再判断。"; }
      else if (!p.normalSeries) { title = "当前模型样本不足"; description = "旧模型、异常局或未结算记录不能代替当前模型的表现。"; }
      else if (p.avoidableMatches > 0) { title = "有可避免失分，值得核对"; description = "出现过本来可胜或可和却未守住的结果；能否利用仍需看具体对手和回放。"; color = "opportunity"; }
      else if (p.knownOpeningSeries < 8) { title = "暂未见破绽，样本较少"; description = "不能仅凭少量正常胜负判定模型强弱。"; }
      else { title = e?.draw.rounded < 0 ? "未见明显破绽，平手会掉分" : "暂未见可避免失分"; description = "预置开局已必败的后手负局不算破绽；不要仅凭败局多就挑战。"; color = "skip"; }
      const availability = own ? "这是你自己的战绩" : !player.enabled ? "对手当前未上榜" :
        live?.played_today ? "今天已与该用户计分" : data.ladder.active_series ? "你有计分对战尚未结束" : "";
      const counts = (v) => v.wins + "胜 " + v.draws + "和 " + v.losses + "负";
      const evidence = p.evidence.slice(0, 3).map((item) => '<a href="' + BASE + '/matches/' + encodeURIComponent(item.matchId) + '" target="_blank" rel="noopener noreferrer">' +
        (item.color === "black" ? "先手" : "后手") + '：' + (item.initial === 1 ? "胜势变" : "可和变") + (item.actual === 0 ? "和" : "负") + ' · 查看证据 ↗</a>').join("");
      const minWin = e ? ({ 0: "无", .25: "一和一负", .5: "两局打平", .75: "一胜一和", 1: "两战全胜" }[e.firstPositiveScore] || "未知") : "未知";
      body = '<div class="identity"><strong>' + escape(player.username) + '</strong><b>' + escape(player.rating ?? "—") + ' 分</b></div>' +
        '<p class="model">' + escape(player.model?.name || "当前模型未知") + ' · v' + escape(player.model?.number ?? "?") + '</p>' +
        (availability ? '<div class="availability">' + escape(availability) + '</div>' : '') +
        '<div class="verdict ' + color + '"><strong>' + title + '</strong><p>' + description + '</p></div>' +
        '<div class="stats"><div><b>' + p.normalSeries + '</b><span>当前模型正常组</span></div><div><b>' + p.avoidableMatches + '</b><span>可避免失分局</span></div><div><b>' + p.expectedLosses + '</b><span>必败后正常输</span></div></div>' +
        '<div class="colors"><span>先手</span>' + counts(p.colors.black) + '<br><span>后手</span>' + counts(p.colors.white) + '</div>' +
        (e ? '<div class="elo"><span>你 ' + e.myRating + ' → 对方 ' + e.theirRating + '</span><div><span>两局打平 <b class="' + (e.draw.rounded < 0 ? 'negative' : '') + '">' + sign(e.draw.rounded) + '</b></span><span>两战全胜 <b>' + sign(e.sweep.rounded) + '</b></span></div><p>单组要涨分，至少需 <b>' + minWin + '</b>（K=' + e.k + '）</p></div>' : '<p class="muted">无我方积分，暂不能计算涨分门槛。</p>') +
        (evidence ? '<div class="evidence">' + evidence + '</div>' : '') +
        '<p class="muted note">已读 ' + h.results.length + ' 组，剔除旧模型 ' + p.olderModelSeries + ' 组、技术异常 ' + p.technicalSeries + ' 组。' + (p.unknownOpeningSeries ? '另有 ' + p.unknownOpeningSeries + ' 组开局理论值未知。' : '') + (h.truncated ? '达到400组读取上限。' : '') + '</p>';
    }
    root.innerHTML = styles() + '<section class="panel" aria-label="Botlab 战绩简析"><header><h2>战绩简析 <small>简版</small></h2><div><button data-action="refresh" aria-label="刷新">↻</button><button data-action="collapse" aria-label="收起">−</button></div></header><div class="body">' + body + '</div><footer>只查开局胜负表 · 不读回放 · 不发起对战</footer></section>';
  }
  async function load(force = false) {
    controller?.abort(); controller = new AbortController();
    const signal = controller.signal, token = ++epoch;
    data = { loading: true }; render();
    try {
      const ladder = await get("/api/v1/ladder", signal);
      const id = new URL(location.href).searchParams.get("player") || ladder.me?.id;
      if (!id || !/^[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}$/i.test(id)) throw new Error("无有效的 player ID。");
      if (token !== epoch) return;
      data.ladder = ladder;
      const key = (ladder.me?.id || "anonymous") + ":" + id;
      const cached = cache.get(key), current = ladder.results.find((p) => p.id === id);
      if (!force && cached && Date.now() - cached.time < 120000 &&
          (!current?.model || C.sameModel(cached.player?.model, current.model))) {
        data.history = cached; data.loading = false; render(); return;
      }
      let offset = 0, history = { results: [], player: null, time: Date.now() };
      const offsets = new Set();
      for (let page = 0; page < 20; page++) {
        if (!Number.isInteger(offset) || offset < 0 || offsets.has(offset)) throw new Error("分页游标异常。");
        offsets.add(offset);
        const next = await get("/api/v1/ladder/series?offset=" + offset + "&player=" + id, signal);
        if (token !== epoch) return;
        if (next.player && next.player.id !== id) throw new Error("战绩用户与当前页面不一致。");
        history.player = next.player || history.player;
        history.results = history.results.concat(next.results);
        data.history = history; render();
        if (next.next_offset == null) {
          data.loading = false; history.time = Date.now(); cache.set(key, history);
          while (cache.size > 6) cache.delete(cache.keys().next().value);
          render(); return;
        }
        offset = next.next_offset;
      }
      history.truncated = true; data.loading = false; render();
    } catch (error) {
      if (error.name === "AbortError" || token !== epoch) return;
      data.loading = false; data.error = error.message || "读取失败"; render();
    }
  }
  function check() {
    const url = new URL(location.href);
    const next = url.origin === BASE && /^\/ladder\/history\/?$/.test(url.pathname) ? (url.searchParams.get("player") || "self") : "";
    if (!next) {
      if (host) { epoch++; controller?.abort(); host.remove(); host = root = null; }
      route = ""; return;
    }
    if (next === route && host?.isConnected) return;
    route = next; mount(); load();
  }
  function styles() {
    return `<style>
      :host{all:initial;color:#25354d;font:12px/1.65 system-ui,"Microsoft YaHei",sans-serif;color-scheme:light}*{box-sizing:border-box}button{font:inherit;cursor:pointer;background:#f5f7fb;color:#496080;border:1px solid #dce4ef;border-radius:7px;padding:2px 9px}button:hover{background:#e9eef7}button:focus-visible,a:focus-visible{outline:3px solid #7184d2;outline-offset:2px}.panel,.chip{position:fixed;right:18px;bottom:18px;z-index:2147483000}.panel{width:345px;max-width:calc(100vw - 24px);max-height:calc(100dvh - 36px);border:1px solid #d8e1ed;background:#fff;border-radius:14px;box-shadow:0 12px 60px #22385830;overflow:hidden;display:flex;flex-direction:column}.chip{background:#2b416d;color:white;border:0;padding:10px 16px;border-radius:24px;box-shadow:0 4px 20px #22385830}header{display:flex;align-items:center;justify-content:space-between;padding:12px 14px;background:#f4f7fc;border-bottom:1px solid #e5eaf2}h2{font-size:16px;margin:0}small{font-size:10px;font-weight:400;color:#8794a7;margin-left:5px}header>div{display:flex;gap:6px}.body{padding:13px 14px;overflow:auto}.identity{display:flex;justify-content:space-between;font-size:14px}.identity b{color:#456198}.model{margin:3px 0 10px;font-size:11px;color:#78869b;overflow-wrap:anywhere}.availability{border-radius:7px;padding:6px 8px;background:#edf4ff;color:#627ca8;font-size:10px;margin-bottom:9px}.verdict{padding:10px;background:#f2f5fa;border:1px solid #e0e6ef;border-radius:10px}.verdict strong{font-size:13px}.verdict p{margin:4px 0 0;font-size:11px;color:#6f7b90}.verdict.skip{background:#fff7ef;border-color:#f2dfc8}.verdict.opportunity{background:#eff8f2;border-color:#cde8d6}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin:11px 0}.stats>div{background:#f4f7fb;border-radius:8px;text-align:center;padding:7px 2px}.stats b{display:block;font-size:20px;color:#36517f}.stats span{font-size:9px;color:#7a879c}.colors{color:#5e6d86;font-size:11px}.colors>span{display:inline-block;width:37px;color:#93a0b2}.elo{border-top:1px solid #e6ebf3;margin-top:10px;padding-top:8px;font-size:11px}.elo>span{color:#8491a6;font-size:10px}.elo>div{display:flex;justify-content:space-between;margin-top:5px}.elo b{color:#385888;font-size:15px;margin-left:5px}.elo b.negative{color:#b8653f}.elo p{margin:4px 0;color:#8590a1;font-size:10px}.elo p b{font-size:11px;margin:0}.evidence{border-top:1px solid #e6ebf3;margin-top:9px;padding-top:7px}.evidence a{display:block;text-decoration:none;color:#4e69ad;font-size:11px;padding:3px 0}.muted{color:#8a96a7;font-size:10px}.note{margin:10px 0 0}.error{background:#fff2ed;color:#a15f4a;border-radius:8px;padding:10px}footer{background:#f7f9fc;padding:7px 12px;border-top:1px solid #e8edf4;color:#8b97aa;font-size:9px;text-align:center}@media(max-width:400px){.panel{right:8px;bottom:8px;width:calc(100vw - 16px);max-height:calc(100dvh - 16px)}}
    </style>`;
  }
  check(); setInterval(check, 900); addEventListener("popstate", check); addEventListener("pageshow", check);
  let timer;
  new MutationObserver(() => { clearTimeout(timer); timer = setTimeout(check, 80); })
    .observe(document.body, { subtree: true, childList: true });
})();
