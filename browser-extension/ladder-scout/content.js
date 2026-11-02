(function () {
  "use strict";
  if (window.top !== window || globalThis.__botlabScoutInstalled) return;
  globalThis.__botlabScoutInstalled = true;
  const Core = globalThis.BotlabScoutCore;
  const ORIGIN = "https://botlab.aiphys.cn";
  const STORE = "botlab_scout_settings_v1", CACHE = "botlab_scout_cache_v1";
  const DEFAULTS = { enabled: true, collapsed: false, assumeOptimal: true, limit: 50 };
  const TTL = 5 * 60 * 1000;
  let settings = { ...DEFAULTS }, host, shadow, routeKey = "", generation = 0;
  let controller, refreshTimer, renderTimer, cacheWrites = Promise.resolve();
  let state = {};
  const esc = (x) => String(x ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const percent = (v) => Number.isFinite(v) ? (100 * v).toFixed(1).replace(/\.0$/, "") + "%" : "—";
  const signed = (v, digits = 0) => Number.isFinite(v) ? (v > 0 ? "+" : "") + v.toFixed(digits) : "—";
  const labelScore = (v) => ({ 0: "两战皆负", 0.25: "一和一负", 0.5: "两局打平", 0.75: "一胜一和", 1: "两战全胜" }[v] || "—");
  const link = (kind, id) => ORIGIN + "/" + kind + "/" + encodeURIComponent(String(id));
  const date = (v) => { const d = new Date(v); return Number.isNaN(d.getTime()) ? "" : d.toLocaleDateString("zh-CN", { timeZone: "Asia/Shanghai", month: "2-digit", day: "2-digit" }); };
  async function readStorage(key) {
    try { return (await chrome.storage.local.get(key))[key]; } catch (_) { return null; }
  }
  async function saveSettings() {
    try { await chrome.storage.local.set({ [STORE]: settings }); } catch (_) { /* Panel remains usable. */ }
  }
  function renderSoon() {
    clearTimeout(renderTimer); renderTimer = setTimeout(render, 60);
  }
  async function api(path, signal) {
    if (!/^\/api\/v1\/ladder(?:\?|$|\/series\?)/.test(path)) throw new Error("不支持的读取路径");
    const response = await fetch(ORIGIN + path, { method: "GET", credentials: "include", cache: "no-store",
      headers: { Accept: "application/json" }, signal });
    if (response.status === 401 || response.status === 403) throw new Error("请先在 Botlab 登录，或确认有权查看这位用户的战绩。");
    if (!response.ok) throw new Error("战绩读取失败（HTTP " + response.status + "），可稍后刷新。");
    const data = await response.json();
    if (!data || !Array.isArray(data.results)) throw new Error("接口格式发生变化，暂不生成结论。");
    return data;
  }
  async function cachedHistory(id, ownerId, signal, force, onUpdate) {
    const allCache = await readStorage(CACHE) || {};
    const cached = allCache[ownerId + ":" + id];
    if (!force && cached && Date.now() - cached.time < TTL && cached.complete) {
      const data = { ...cached, cached: true };
      onUpdate(data); return data;
    }
    let offset = 0, player = null, rows = [], pages = 0;
    const offsets = new Set();
    while (offset !== null) {
      if (signal.aborted) throw new DOMException("Canceled", "AbortError");
      if (!Number.isInteger(offset) || offset < 0 || offsets.has(offset)) throw new Error("分页游标异常，已停止读取。");
      offsets.add(offset);
      const page = await api("/api/v1/ladder/series?offset=" + offset + "&player=" + encodeURIComponent(id), signal);
      if (page.player && page.player.id !== id) throw new Error("战绩用户与页面不一致，已停止读取。");
      player = page.player || player;
      rows = rows.concat(page.results); pages++;
      offset = page.next_offset == null ? null : page.next_offset;
      if (offset !== null && (!Number.isInteger(offset) || offset < 0)) throw new Error("不识别的下一页游标。");
      const data = { player, results: rows, pages, complete: offset === null, time: Date.now(), cached: false };
      onUpdate(data);
      if (pages >= 20) {
        data.truncated = offset !== null;
        if (data.truncated) return data;
      }
      if (offset === null) {
        // Keep a small, account-scoped local cache. Never cache cookies/tokens.
        cacheWrites = cacheWrites.catch(() => {}).then(async () => {
          const freshCache = await readStorage(CACHE) || {};
          freshCache[ownerId + ":" + id] = { ...data };
          const keys = Object.keys(freshCache).sort((a, b) => freshCache[b].time - freshCache[a].time);
          const pruned = {}; let bytes = 0;
          for (const key of keys.slice(0, 6)) {
            const size = JSON.stringify(freshCache[key]).length * 2;
            if (bytes + size <= 3_500_000) { pruned[key] = freshCache[key]; bytes += size; }
          }
          try { await chrome.storage.local.set({ [CACHE]: pruned }); } catch (_) { /* Caching is optional. */ }
        });
        await cacheWrites;
        return data;
      }
    }
    throw new Error("无法读取战绩。");
  }
  function ensureHost() {
    if (host && host.isConnected) return;
    host = document.createElement("botlab-ladder-scout");
    host.id = "botlab-ladder-scout";
    host.style.cssText = "position:relative;z-index:2147483000;";
    shadow = host.attachShadow({ mode: "open" });
    shadow.addEventListener("click", onClick);
    shadow.addEventListener("change", onChange);
    document.body.appendChild(host);
  }
  function metrics() {
    const data = state.target;
    if (!data || !data.player || !state.ladder) return null;
    const me = state.ladder.me;
    const targetLive = state.ladder.results.find((p) => p.id === data.player.id);
    const target = targetLive || data.player;
    const current = Core.analyzeHistory(data.results, target.id, target.model, { limit: settings.limit });
    const shown = settings.scope === "all" ? Core.analyzeHistory(data.results, target.id, target.model, { limit: settings.limit, scope: "all" }) : current;
    let myProfile = null;
    if (state.mine && me && me.model) myProfile = Core.analyzeHistory(state.mine.results, me.id, me.model, { limit: settings.limit });
    const myRating = state.myRatingOverride ?? me?.rating;
    const theirRating = state.theirRatingOverride ?? target.rating;
    const calculator = Core.elo(myRating, theirRating, state.ladder.k_factor || 32);
    const projection = Core.project(current, calculator, { assumeOptimal: settings.assumeOptimal,
      partial: state.loading, self: me?.id === target.id, meId: me?.id, myModel: me?.model,
      myVulnerable: Boolean(myProfile && myProfile.avoidableMatches) });
    return { me, target, targetLive, current, shown, myProfile, calculator, projection };
  }
  function render() {
    if (!host || !shadow) return;
    if (settings.collapsed) {
      shadow.innerHTML = style() + '<button class="chip" data-action="expand" aria-label="展开天梯侦察">◈ 天梯侦察</button>';
      return;
    }
    const m = metrics();
    const progress = state.target ? "已读 " + state.target.results.length + " 组 · " + state.target.pages + " 页" : "读取中…";
    let content;
    if (state.error) content = '<div class="notice error" role="alert">' + esc(state.error) + '</div><button data-action="refresh" class="primary">重新读取</button>';
    else if (!m) content = '<div class="loading"><span class="spinner"></span>正在读取积分和战绩…</div>';
    else content = report(m);
    shadow.innerHTML = style() + '<section class="panel" aria-label="Botlab 天梯战绩侦察">' +
      '<header><div><span class="eyebrow">BOTLAB SCOUT</span><h2>天梯侦察</h2></div><div class="header-actions"><button data-action="refresh" title="跳过缓存，重新读取" aria-label="刷新战绩">↻</button><button data-action="collapse" title="收起" aria-label="收起面板">−</button></div></header>' +
      '<div class="body">' + content + '</div><footer><span>' + esc(state.loading ? progress : state.target?.cached ? "本机缓存 · 5分钟内" : progress) + '</span><span>只读 · 不发起对战</span></footer></section>';
  }
  function report(m) {
    const { target, current: p, shown, projection: advice, calculator: e, myProfile, targetLive, me } = m;
    const k = e?.k || 32;
    let availability = "";
    if (me?.id !== target.id) {
      if (!target.enabled) availability = "该用户当前未参加日常榜。";
      else if (targetLive?.played_today) availability = "今天已与该用户计分，暂不能再次挑战。";
      else if (state.ladder.active_series) availability = "你有一组计分对战尚未完成。";
    }
    const band = advice.lowDelta == null ? "" : '<span class="sub">情景误差带 ' + signed(advice.lowDelta, 1) + ' ～ ' + signed(advice.highDelta, 1) + ' 分</span>';
    const evidence = p.evidence.slice(0, 8).map((r) =>
      '<a class="evidence" href="' + esc(link("matches", r.matchId)) + '" target="_blank" rel="noopener noreferrer"><span><b>' + (r.initial === 1 ? "胜势变" : "可和变") + (r.actual === 0 ? "和" : "负") + '</b> · ' + (r.color === "black" ? "先手" : "后手") + '<small>' + esc(date(r.createdAt)) + ' 对 ' + esc(r.opponent || "未知用户") + '</small></span><span>回放 ↗</span></a>').join("");
    const breakdown = [
      ["开场可胜", p.startWins], ["开场可和", p.startDraws], ["开场必败", p.startLosses],
      ["必败后正常输", p.expectedLosses], ["可避免失分局", p.avoidableMatches],
      ["技术异常组（排除）", p.technicalSeries], ["未知开局组", p.unknownOpeningSeries],
      ["未结算 / 作废", p.pending + " / " + p.voidSeries], ["异常数据组（排除）", p.malformedSeries],
    ].map(([label, value]) => '<div><span>' + label + '</span><b>' + esc(value) + '</b></div>').join("");
    const ownEvidence = !myProfile ? (state.mineError ? "我方战绩暂不可读；情景假设未验证。" : "正在核对我方当前模型战绩…") :
      "我方当前模型 " + myProfile.normalSeries + " 组正常记录，" + (myProfile.avoidableMatches ? "出现过 " + myProfile.avoidableMatches + " 局可避免失分。" : "未见这类失分。仍不等于逐手完美证明。");
    const color = (c) => c.wins + "胜 " + c.draws + "和 " + c.losses + "负";
    const kNote = p.formulaMismatches ? '<p class="notice">历史中有 ' + p.formulaMismatches + ' 组结算与当前标准公式不符，请谨慎使用估分。</p>' : "";
    return '<div class="identity"><div><strong>' + esc(target.username || "未知用户") + '</strong><span class="rating">' + esc(target.rating ?? "—") + ' 分</span></div><p>' + esc(target.model?.name || "当前模型未知") + ' · 平台 v' + esc(target.model?.number ?? "?") + '</p><span class="muted">当前模型已读 ' + p.selectedSeries + ' 组 · 另有 ' + p.olderModelSeries + ' 组旧模型记录</span></div>' +
      (availability ? '<div class="notice availability">' + esc(availability) + '</div>' : '') +
      '<div class="verdict ' + advice.kind + '"><span class="eyebrow">' + esc(advice.source || "战绩判断") + '</span><h3>' + esc(advice.title) + '</h3><p>' + esc(advice.explanation) + '</p><div class="scenario">' + (advice.expectedDelta == null ? "暂不预测收益" : '情景平均收益 <strong>' + signed(advice.expectedDelta, 1) + ' 分/组</strong>') + '</div>' + band + '</div>' +
      '<div class="stats"><div><b>' + p.normalSeries + '</b><span>正常系列样本</span></div><div><b>' + p.avoidableMatches + '</b><span>可避免失分局</span></div><div><b>' + (e ? signed(e.draw.rounded) : "—") + '</b><span>两局打平积分</span></div></div>' +
      '<div class="controls"><label>最近 <select name="limit"><option value="20"' + (settings.limit === 20 ? ' selected' : '') + '>20组</option><option value="50"' + (settings.limit === 50 ? ' selected' : '') + '>50组</option><option value="200"' + (settings.limit === 200 ? ' selected' : '') + '>200组</option><option value="400"' + (settings.limit === 400 ? ' selected' : '') + '>全部已读（上限400）</option></select></label><label><input type="checkbox" name="assumeOptimal"' + (settings.assumeOptimal ? ' checked' : '') + '>我方近似最优</label></div>' +
      '<p class="hint">' + esc(ownEvidence) + '</p>' +
      '<details open><summary>Elo 收益门槛 <span>K=' + k + '</span></summary><div class="rating-inputs"><label>我的分数<input name="myRating" type="number" min="0" max="10000" value="' + esc(e?.myRating ?? me?.rating ?? "") + '"></label><label>对方分数<input name="theirRating" type="number" min="0" max="10000" value="' + esc(e?.theirRating ?? target.rating ?? "") + '"></label></div>' +
      (e ? '<div class="payoffs">' + e.cases.map((c) => '<div><span>' + labelScore(c.score) + '</span><b class="' + (c.rounded > 0 ? 'positive' : c.rounded < 0 ? 'negative' : '') + '">' + signed(c.rounded) + '</b></div>').join('') + '</div><p class="hint">未取整公式的回本得分约 <b>' + percent(e.expected) + '</b>。单组要涨分，至少需<b>' + labelScore(e.firstPositiveScore) + '</b>。</p>' : '') + kNote + '</details>' +
      '<details><summary>先后手与开局诊断</summary><div class="breakdown">' + breakdown + '</div><p class="hint">双方各预置一手、同开局换色。第3/5列开局型为理论和棋，其余镜像首手型为先手胜。必败后手负局不作为可利用破绽。</p><p class="hint">先手：' + color(p.colors.black) + '<br>后手：' + color(p.colors.white) + '</p></details>' +
      '<details' + (p.evidence.length ? ' open' : '') + '><summary>当前模型的失分证据 <span>' + p.evidence.length + '局</span></summary>' + (evidence || '<p class="hint">样本中未见可避免的终局失分。这不证明它没有中途失误，也不证明模型完美。</p>') + '</details>' +
      '<details><summary>历史与版本</summary><label class="scope">统计范围 <select name="scope"><option value="current"' + (settings.scope !== 'all' ? ' selected' : '') + '>仅当前模型</option><option value="all"' + (settings.scope === 'all' ? ' selected' : '') + '>全部历史模型</option></select></label><p class="hint">' + (settings.scope === 'all' ? '下列包含旧模型，仅供回顾；收益判断仍只用当前模型。' : '模型按代码和棋谱 SHA 分组；相同权重的重复上传视为同一模型。') + '</p><p>系列 ' + shown.wins + '胜 ' + shown.draws + '平 ' + shown.losses + '负 · 得分率 ' + percent(shown.meanScore) + '</p><p class="hint">先手 ' + color(shown.colors.black) + '<br>后手 ' + color(shown.colors.white) + '</p><p class="hint">当前收益样本覆盖 ' + p.knownOpponentCount + ' 位对手。' + (state.target?.truncated ? '已达400组读取上限，仅统计已读数据。' : '') + '</p></details>' +
      '<details><summary>怎么算 / 适用范围</summary><p class="hint">Elo 按两局平均得分结算：K ×（实际得分 − 1/(1+10^((对方分−我方分)/400)))，显示平台整数取整近似值。</p><p class="hint">近似最优情景 = 50% + 对手当前模型历史可避免失分/每组两局。它假设我方不丢胜负值，也假设历史破绽能在本次对局复现；不是实际胜率保证。取消勾选后，仅参考双方当前模型的历史交手。</p><p class="hint">平均收益逐组取整后统计。误差带近似以每组整数积分为独立样本，按 Hoeffding 保守界计算，不包含对手分布变化和我方模型失误风险。异常局、旧模型及未知理论开局不混入此估计。</p></details>';
  }
  function onClick(event) {
    const action = event.target.closest("[data-action]")?.dataset.action;
    if (action === "collapse" || action === "expand") { settings.collapsed = action === "collapse"; saveSettings(); render(); }
    if (action === "refresh") load(true);
  }
  function onChange(event) {
    const el = event.target;
    if (el.name === "limit") settings.limit = Number(el.value);
    else if (el.name === "assumeOptimal") settings.assumeOptimal = el.checked;
    else if (el.name === "scope") settings.scope = el.value;
    else if (el.name === "myRating" || el.name === "theirRating") {
      const value = Number(el.value);
      state[el.name === "myRating" ? "myRatingOverride" : "theirRatingOverride"] = el.value !== "" && Number.isFinite(value) && value >= 0 && value <= 10000 ? value : undefined;
    } else return;
    saveSettings(); render();
  }
  async function load(force = false) {
    if (!host) return;
    controller?.abort(); controller = new AbortController();
    const signal = controller.signal, token = ++generation;
    state = { loading: true, route: location.href }; render();
    try {
      const requested = new URL(location.href).searchParams.get("player");
      const ladder = await api("/api/v1/ladder", signal);
      if (token !== generation) return;
      state.ladder = ladder;
      const id = requested || ladder.me?.id;
      if (!id || !/^[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}$/i.test(id)) throw new Error("战绩页缺少有效的 player ID。");
      const ownerId = ladder.me?.id || "anonymous";
      const targetWork = cachedHistory(id, ownerId, signal, force, (data) => {
        if (token === generation) { state.target = data; if (id === ownerId) state.mine = data; renderSoon(); }
      });
      const mineWork = id === ownerId ? Promise.resolve() : cachedHistory(ownerId, ownerId, signal, force, (data) => {
        if (token === generation) { state.mine = data; renderSoon(); }
      }).catch((error) => { if (token === generation && error.name !== "AbortError") state.mineError = true; });
      await Promise.all([targetWork, mineWork]);
      if (token !== generation) return;
      state.loading = false; render();
    } catch (error) {
      if (error.name === "AbortError" || token !== generation) return;
      state.loading = false; state.error = error.message || "读取失败"; render();
    }
  }
  function checkRoute() {
    const url = new URL(location.href);
    const active = settings.enabled && url.origin === ORIGIN && /^\/ladder\/history\/?$/.test(url.pathname);
    const key = active ? url.pathname + "?player=" + (url.searchParams.get("player") || "") : "";
    if (!active) {
      if (routeKey || host) { generation++; controller?.abort(); host?.remove(); host = shadow = null; }
      routeKey = ""; return;
    }
    if (key === routeKey && host?.isConnected) return;
    routeKey = key; ensureHost(); load();
  }
  function style() {
    return `<style>
      :host{all:initial;font-family:system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;color:#202c3d;font-size:13px;line-height:1.55;color-scheme:light}
      *{box-sizing:border-box}button,input,select{font:inherit}button,select{cursor:pointer}button{color:#42526a;background:#f4f6fa;border:1px solid #dce3ed;border-radius:8px;padding:5px 10px}button:hover{background:#e9edf6}button:focus-visible,a:focus-visible,input:focus-visible,select:focus-visible,summary:focus-visible{outline:3px solid #6c79f0;outline-offset:2px}a{color:#405ec5;text-decoration:none}
      .panel,.chip{position:fixed;right:18px;bottom:18px;z-index:2147483000}.panel{width:410px;max-width:calc(100vw - 24px);max-height:calc(100vh - 36px);background:#fff;border:1px solid #d9e0ea;border-radius:16px;box-shadow:0 16px 70px #1a284433;overflow:hidden;display:flex;flex-direction:column}.chip{background:#253763;color:#fff;padding:12px 17px;border:0;border-radius:24px;box-shadow:0 4px 24px #1a284440}
      header{display:flex;justify-content:space-between;align-items:center;padding:16px 18px 12px;background:#f5f7fc;border-bottom:1px solid #e5e9f1}h2{font-size:19px;margin:0;font-weight:750;letter-spacing:.04em}.eyebrow{font-size:10px;font-weight:700;letter-spacing:.1em;color:#66758e}.header-actions{display:flex;gap:7px}.header-actions button{font-size:20px;width:32px;height:32px;padding:0}.body{padding:16px 18px;overflow:auto;overscroll-behavior:contain}.identity strong{font-size:16px}.identity p{margin:4px 0;overflow-wrap:anywhere}.rating{float:right;font-weight:750;color:#344d8b}.muted,.sub,.hint{font-size:11px;color:#748095}.sub{display:block;margin-top:3px}.hint{margin:8px 0 5px;line-height:1.7}.identity{padding-bottom:13px}
      .verdict{border-radius:12px;background:#f1f4fa;border:1px solid #e1e6f0;padding:12px;margin:0 0 12px}.verdict h3{font-size:17px;margin:4px 0}.verdict p{margin:5px 0;font-size:12px}.verdict.skip{background:#fff7ee;border-color:#f5dfc4}.verdict.consider{background:#effaf4;border-color:#c3e9d1}.verdict.uncertain{background:#f6f5fc;border-color:#dfdbf4}.scenario{font-size:12px;margin-top:10px}.scenario strong{float:right;font-size:15px}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin-bottom:12px}.stats>div{background:#f5f7fb;border-radius:10px;padding:9px 5px;text-align:center}.stats b{font-size:21px;display:block;color:#30466e}.stats span{font-size:10px;color:#788398}.controls{display:flex;align-items:center;justify-content:space-between;gap:5px;font-size:11px}.controls input{vertical-align:middle;accent-color:#445cba}.controls select,.scope select{max-width:155px;border:1px solid #d9e0eb;background:#fff;border-radius:6px;padding:4px;color:#42526a}.controls label{white-space:nowrap}
      details{border-top:1px solid #e8ecf2;margin-top:12px;padding-top:9px}summary{cursor:pointer;font-weight:650;font-size:12px;padding:3px 0}summary>span{float:right;font-size:10px;color:#7a8599;font-weight:400}.rating-inputs{display:flex;gap:10px;margin:10px 0}.rating-inputs label{font-size:10px;color:#768399;flex:1}.rating-inputs input{display:block;min-width:0;width:100%;border:1px solid #d6dfec;border-radius:7px;padding:6px 9px;color:#263d6a;font-size:14px;margin-top:3px}.payoffs{display:grid;grid-template-columns:repeat(5,1fr);gap:4px}.payoffs div{text-align:center;background:#f5f7fb;border-radius:6px;padding:6px 0}.payoffs span{font-size:9px;display:block;color:#77849b}.payoffs b{font-size:16px}.positive{color:#25814f}.negative{color:#b15d39}.breakdown>div{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #f0f2f6;font-size:11px}.breakdown b{color:#485b78}.evidence{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #edf0f5;padding:9px 0;font-size:11px}.evidence small{display:block;color:#8a94a6;font-size:10px}.evidence>span:last-child{flex-shrink:0;color:#657bb9}.scope{display:block;margin-top:9px;font-size:11px}.notice{font-size:11px;border-radius:8px;padding:9px;margin:0 0 11px;background:#fff5e6;color:#91632f}.error{background:#fff1f0;color:#a54a45}.availability{background:#eef4fd;color:#4d6b9a}.primary{background:#425bb0;color:#fff;border:0}.loading{padding:28px 0;text-align:center;color:#7a879d}.spinner{display:inline-block;width:12px;height:12px;border-radius:50%;border:2px solid #d8deeb;border-top-color:#526ab8;animation:spin 1s linear infinite;margin-right:8px}@keyframes spin{to{transform:rotate(360deg)}}footer{padding:9px 16px;background:#f8f9fc;border-top:1px solid #e7ecf3;display:flex;justify-content:space-between;color:#8a94a6;font-size:10px}
      @media(max-width:480px){.panel{right:8px;bottom:8px;max-height:calc(100dvh - 16px);width:calc(100vw - 16px)}.body{padding:12px}.chip{right:10px;bottom:10px}}@media(prefers-reduced-motion:reduce){.spinner{animation:none}}
    </style>`;
  }
  async function boot() {
    const saved = await readStorage(STORE);
    if (saved) settings = { ...DEFAULTS, ...saved };
    if (![20, 50, 200, 400].includes(settings.limit)) settings.limit = 50;
    checkRoute();
    setInterval(checkRoute, 900);
    addEventListener("popstate", checkRoute);
    addEventListener("pageshow", checkRoute);
    const observer = new MutationObserver(() => {
      clearTimeout(refreshTimer); refreshTimer = setTimeout(checkRoute, 80);
    });
    observer.observe(document.body, { childList: true, subtree: true });
    try { chrome.storage.onChanged.addListener((changes, area) => {
      if (area === "local" && changes[STORE]) {
        settings = { ...DEFAULTS, ...changes[STORE].newValue };
        checkRoute(); if (host) renderSoon();
      }
    }); } catch (_) { /* Browser fixture can omit extension storage events. */ }
  }
  boot();
})();
