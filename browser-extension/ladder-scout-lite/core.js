/* Pure analysis shared by the content script and regression tests. */
(function (root, factory) {
  "use strict";
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.BotlabScoutCore = api;
})(globalThis, function () {
  "use strict";
  const SCORES = [0, 0.25, 0.5, 0.75, 1];
  const BLACK_WDL = Object.freeze([1, 1, 0, 1, 0, 1, 1]);
  const finite = (v) => typeof v === "number" && Number.isFinite(v);
  const clamp = (v, low, high) => Math.min(high, Math.max(low, v));
  function fingerprint(model) {
    if (!model) return null;
    if (model.code_sha256 && model.models_sha256) return "sha:" + model.code_sha256 + ":" + model.models_sha256;
    return model.version ? "version:" + model.version : null;
  }
  function sameModel(a, b) {
    if (!a || !b) return false;
    if (a.code_sha256 && a.models_sha256 && b.code_sha256 && b.models_sha256) {
      return a.code_sha256 === b.code_sha256 && a.models_sha256 === b.models_sha256;
    }
    return Boolean(a.version && b.version && a.version === b.version);
  }
  function openingValue(series) {
    const a = series && series.opening_actions;
    if (series.opening_pairs !== 1 || !Array.isArray(a) || a.length !== 2 ||
        !a.every((n) => Number.isInteger(n) && n >= 0 && n < 7) || a[0] + a[1] !== 6) return null;
    return BLACK_WDL[a[0]];
  }
  function slotFor(series, id) {
    if (series.player_a && series.player_a.id === id) return "a";
    if (series.player_b && series.player_b.id === id) return "b";
    return null;
  }
  function outcome(match, slot) {
    if (!match || !["a", "b"].includes(match.black_slot) || ![-1, 0, 1].includes(match.winner)) return null;
    const side = match.black_slot === slot ? 1 : -1;
    const wdl = match.winner === 0 ? 0 : match.winner === side ? 1 : -1;
    return { side, wdl, score: (wdl + 1) / 2 };
  }
  function roundEven(value) {
    const low = Math.floor(value), remainder = value - low;
    if (Math.abs(remainder - 0.5) <= 1e-12) return low % 2 === 0 ? low : low + 1;
    return Math.round(value) || 0;
  }
  function elo(myRating, theirRating, k = 32) {
    if (![myRating, theirRating, k].every(finite) || k <= 0 || k > 1000) return null;
    const expected = 1 / (1 + Math.pow(10, (theirRating - myRating) / 400));
    const delta = (score) => k * (score - expected);
    const cases = SCORES.map((score) => ({ score, raw: delta(score), rounded: roundEven(delta(score)) }));
    return { myRating, theirRating, k, expected, cases,
      draw: cases[2], sweep: cases[4], swept: cases[0],
      firstPositiveScore: (cases.find((c) => c.rounded > 0) || {}).score ?? null,
      firstNonnegativeScore: (cases.find((c) => c.rounded >= 0) || {}).score ?? null,
      sweepRateIfOnlyTieOrSweep: cases[4].rounded === cases[2].rounded ? null :
        clamp(-cases[2].rounded / (cases[4].rounded - cases[2].rounded), 0, 1) };
  }
  function emptyColor() { return { wins: 0, draws: 0, losses: 0, games: 0 }; }
  function analyzeHistory(allRows, playerId, model, options = {}) {
    const limit = Number.isInteger(options.limit) && options.limit > 0 ? options.limit : 200;
    const scope = options.scope === "all" ? "all" : "current";
    const unique = new Map();
    let malformed = 0, duplicate = 0;
    for (const row of Array.isArray(allRows) ? allRows : []) {
      if (!row || typeof row.id !== "string" || !slotFor(row, playerId)) { malformed++; continue; }
      if (unique.has(row.id)) { duplicate++; continue; }
      unique.set(row.id, row);
    }
    const sorted = [...unique.values()].sort((a, b) =>
      (Date.parse(b.created_at) || 0) - (Date.parse(a.created_at) || 0) || a.id.localeCompare(b.id));
    const groups = new Map();
    for (const row of sorted) {
      const m = row["player_" + slotFor(row, playerId)].model;
      const key = fingerprint(m) || "unknown";
      const entry = groups.get(key) || { key, model: m, series: 0 };
      entry.series++;
      groups.set(key, entry);
    }
    const selected = sorted.filter((r) => scope === "all" || sameModel(r["player_" + slotFor(r, playerId)].model, model));
    const settled = selected.filter((r) => r.status === "SETTLED");
    const p = {
      playerId, model, scope, limit, totalSeries: sorted.length, selectedSeries: selected.length,
      olderModelSeries: sorted.length - selected.length, pending: selected.filter((r) => r.status === "PENDING").length,
      voidSeries: selected.filter((r) => r.status === "VOID").length, duplicate, malformed,
      windowSeries: Math.min(limit, settled.length), normalSeries: 0, technicalSeries: 0,
      malformedSeries: 0, wins: 0, draws: 0, losses: 0, scoreSum: 0,
      matchWins: 0, matchDraws: 0, matchLosses: 0,
      colors: { black: emptyColor(), white: emptyColor() },
      knownOpeningSeries: 0, unknownOpeningSeries: 0, knownMatches: 0,
      startWins: 0, startDraws: 0, startLosses: 0, expectedLosses: 0,
      avoidableMatches: 0, affectedSeries: 0, forfeitedPoints: 0,
      liftSamples: [], evidence: [], series: [],
      opponentCount: 0, knownOpponentCount: 0, groups: [...groups.values()], normalOpponents: [],
      formulaChecks: 0, formulaMismatches: 0,
    };
    const opponents = new Set(), knownOpponents = new Set();
    for (const row of settled.slice(0, limit)) {
      const slot = slotFor(row, playerId), otherSlot = slot === "a" ? "b" : "a";
      const matches = row.matches;
      if (!Array.isArray(matches) || matches.length !== 2 ||
          !matches.every((m) => m && typeof m.id === "string" && m.id && ["a", "b"].includes(m.black_slot)) ||
          matches[0].id === matches[1].id ||
          new Set(matches.map((m) => m.black_slot)).size !== 2 ||
          !matches.every((m) => ["a", "b"].includes(m.black_slot))) {
        p.malformedSeries++; continue;
      }
      if (matches.some((m) => m.phase !== "COMPLETED" || m.end_reason !== "NORMAL")) {
        p.technicalSeries++; continue;
      }
      if (matches.some((m) => !outcome(m, slot))) { p.malformedSeries++; continue; }
      const actual = matches.map((m) => outcome(m, slot));
      const score = (actual[0].score + actual[1].score) / 2;
      const scoreA = slot === "a" ? score : 1 - score;
      if (row.settlement && finite(row.settlement.score_a) && Math.abs(row.settlement.score_a - scoreA) > 1e-6) {
        p.malformedSeries++; continue;
      }
      p.normalSeries++;
      p.scoreSum += score;
      p[score > 0.5 ? "wins" : score < 0.5 ? "losses" : "draws"]++;
      const opponent = row["player_" + otherSlot];
      if (opponent && opponent.id) opponents.add(opponent.id);
      const base = openingValue(row);
      if (base === null) p.unknownOpeningSeries++; else p.knownOpeningSeries++;
      let gap = 0, issues = 0;
      actual.forEach((item, i) => {
        const counts = p.colors[item.side === 1 ? "black" : "white"];
        counts.games++;
        counts[item.wdl === 1 ? "wins" : item.wdl === 0 ? "draws" : "losses"]++;
        p[item.wdl === 1 ? "matchWins" : item.wdl === 0 ? "matchDraws" : "matchLosses"]++;
        if (base === null) return;
        const initial = base * item.side;
        p.knownMatches++;
        p[initial === 1 ? "startWins" : initial === 0 ? "startDraws" : "startLosses"]++;
        if (initial === -1 && item.wdl === -1) p.expectedLosses++;
        if (item.wdl < initial) {
          const lost = (initial - item.wdl) / 2;
          gap += lost; issues++; p.avoidableMatches++;
          p.evidence.push({ seriesId: row.id, matchId: matches[i].id,
            createdAt: row.created_at, opponent: opponent && opponent.username,
            opponentId: opponent && opponent.id, color: item.side === 1 ? "black" : "white",
            initial, actual: item.wdl, forfeitedPoints: lost,
            opening: row.opening_actions, model: row["player_" + slot].model });
        }
      });
      if (base !== null) {
        if (opponent && opponent.id) knownOpponents.add(opponent.id);
        // One same-opening color-swapped series is one independent sample.
        // Its clipped forfeited score is in [0, 0.5]. Opponent errors can mask
        // a target's mistakes: zero observed forfeiture never proves perfection.
        p.liftSamples.push(gap / 2);
        p.forfeitedPoints += gap;
        if (issues) p.affectedSeries++;
      }
      p.series.push({ id: row.id, score, opponentId: opponent && opponent.id,
        opponentModel: opponent && opponent.model, openingKnown: base !== null, gap: gap / 2 });
      const t = row.settlement;
      if (t && [t.rating_a_before, t.rating_b_before, t.score_a, t.delta_a, row.k_factor].every(finite)) {
        p.formulaChecks++;
        const calc = elo(t.rating_a_before, t.rating_b_before, row.k_factor);
        if (!calc || roundEven(row.k_factor * (t.score_a - calc.expected)) !== t.delta_a) p.formulaMismatches++;
      }
    }
    p.opponentCount = opponents.size;
    p.knownOpponentCount = knownOpponents.size;
    p.normalOpponents = [...opponents];
    p.meanScore = p.normalSeries ? p.scoreSum / p.normalSeries : null;
    p.meanLift = p.liftSamples.length ? p.liftSamples.reduce((a, b) => a + b, 0) / p.liftSamples.length : null;
    p.heldOpeningRate = p.knownMatches ? (p.knownMatches - p.avoidableMatches) / p.knownMatches : null;
    return p;
  }
  function project(profile, calculator, options = {}) {
    if (!calculator) return { kind: "unknown", title: "缺少积分信息", explanation: "可手动填写我方和对方积分。" };
    const n = profile.knownOpeningSeries;
    let score = null, lower = null, upper = null, source, samples = n, scoreSamples = [];
    if (options.assumeOptimal !== false) {
      if (profile.meanLift !== null) {
        scoreSamples = profile.liftSamples.map((lift) => clamp(.5 + lift, .5, 1));
        score = clamp(0.5 + profile.meanLift, 0.5, 1);
        // Hoeffding bound for a [0, .5] per-series historical loss variable.
        // This is an error band on that scenario, not a probability of winning.
        const radius = 0.5 * Math.sqrt(Math.log(40) / (2 * n));
        lower = clamp(score - radius, 0.5, 1);
        upper = clamp(score + radius, 0.5, 1);
      }
      source = "我方稳定保住最优胜负值的情景";
    } else {
      const head = profile.series.filter((s) => s.opponentId === options.meId &&
        sameModel(s.opponentModel, options.myModel));
      samples = head.length;
      if (head.length) {
        scoreSamples = head.map((s) => 1 - s.score);
        score = 1 - head.reduce((sum, s) => sum + s.score, 0) / head.length;
        const radius = Math.sqrt(Math.log(40) / (2 * head.length));
        lower = clamp(score - radius, 0, 1); upper = clamp(score + radius, 0, 1);
      }
      source = "双方当前模型历史交手（" + head.length + "组）";
    }
    const gains = scoreSamples.map((s) => roundEven(calculator.k * (s - calculator.expected)));
    const meanGain = gains.length ? gains.reduce((a, b) => a + b, 0) / gains.length : null;
    const minGain = options.assumeOptimal !== false ? calculator.draw.rounded : calculator.swept.rounded;
    const maxGain = calculator.sweep.rounded;
    const gainRadius = gains.length ? (maxGain - minGain) * Math.sqrt(Math.log(40) / (2 * gains.length)) : null;
    const base = { source, score, lower, upper, samples,
      expectedDelta: meanGain,
      rawExpectedDelta: score === null ? null : calculator.k * (score - calculator.expected),
      lowDelta: meanGain === null ? null : clamp(meanGain - gainRadius, minGain, maxGain),
      highDelta: meanGain === null ? null : clamp(meanGain + gainRadius, minGain, maxGain),
      drawDelta: calculator.draw.rounded };
    if (options.self) return { ...base, kind: "self", title: "这是你自己的战绩", explanation: "可检查当前模型的失分记录；不作为对手推荐。" };
    if (score === null) return { ...base, kind: "unknown", title: "证据不足", explanation: "当前版本缺少可用的镜像开局样本，不能从旧模型战绩推断。" };
    if (samples < 8 || (options.assumeOptimal !== false && profile.knownOpponentCount < 2) || options.partial) {
      return { ...base, kind: "uncertain", title: options.partial ? "数据尚未读完" : "样本偏少，先观察",
        explanation: "暂不据少量战绩判定能稳定涨分。下方积分变化仍可参考。" };
    }
    if (base.expectedDelta <= 0) return { ...base, kind: "skip", title: "偏向跳过",
      explanation: profile.avoidableMatches === 0 ? "当前样本未见可避免失分；平手收益不足，不能把后手必败负局当作破绽。" :
        "虽然出现过可避免失分，但现有情景收益不足以覆盖你的积分门槛。" };
    if (options.assumeOptimal !== false && options.myVulnerable) return { ...base, kind: "uncertain", title: "仅在乐观假设下可能涨分",
      explanation: "我方当前模型也出现过可避免失分，稳定最优假设可能高估收益。" };
    if (base.lowDelta > 0) return { ...base, kind: "consider", title: "有正收益线索",
      explanation: "当前版本的可避免失分较多，情景误差带也高于回本门槛；实际能否复现仍取决于对局。" };
    return { ...base, kind: "uncertain", title: "可能有收益，证据仍不稳",
      explanation: "点估计为正，但保守误差带仍含掉分情景，不宜按稳赢计算。" };
  }
  return Object.freeze({ fingerprint, sameModel, openingValue, slotFor, outcome, elo,
    roundEven, analyzeHistory, project, BLACK_WDL });
});
