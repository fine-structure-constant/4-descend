"use strict";
const { test } = require("node:test");
const assert = require("node:assert/strict");
const C = require("../core.js");
const TARGET = "target", OTHER = "other";
const model = (version = "v1", code = "code1", weights = "weights1") => ({ version, code_sha256: code, models_sha256: weights });
const current = model();
function series(id, options = {}) {
  const slot = options.slot || "a";
  const target = { id: TARGET, username: "target", model: options.model || current };
  const other = { id: options.opponent || OTHER, username: "other", model: options.otherModel || model("other", "o", "w") };
  const winners = options.winners || [1, 1];
  return { id: "s" + id, status: options.status || "SETTLED", created_at: new Date(1700000000000 + id * 1000).toISOString(),
    opening_pairs: 1, opening_actions: options.opening || [1, 5],
    player_a: slot === "a" ? target : other, player_b: slot === "a" ? other : target,
    matches: winners.map((winner, i) => ({ id: "m" + id + "-" + i,
      black_slot: i === 0 ? "a" : "b", phase: "COMPLETED", winner, end_reason: "NORMAL" })) };
}
test("A normal losing white turn in a first-player-win opening is not a weakness", () => {
  const p = C.analyzeHistory([series(1)], TARGET, current);
  assert.equal(p.normalSeries, 1); assert.equal(p.expectedLosses, 1);
  assert.equal(p.avoidableMatches, 0); assert.equal(p.meanLift, 0);
  assert.equal(p.colors.black.wins, 1); assert.equal(p.colors.white.losses, 1);
});
test("Target slot b and black_slot are interpreted correctly", () => {
  const p = C.analyzeHistory([series(1, { slot: "b" })], TARGET, current);
  assert.equal(p.avoidableMatches, 0); assert.equal(p.meanScore, .5);
  assert.equal(p.colors.black.wins, 1); assert.equal(p.colors.white.losses, 1);
});
test("Losing a drawable start is evidence, irrespective of color", () => {
  const p = C.analyzeHistory([series(1, { opening: [2, 4], winners: [-1, 1] })], TARGET, current);
  assert.equal(p.avoidableMatches, 2); assert.equal(p.meanLift, .5);
  assert.equal(p.losses, 1);
});
test("Winning position to draw is half a match point forfeited", () => {
  const p = C.analyzeHistory([series(1, { winners: [0, 1] })], TARGET, current);
  assert.equal(p.avoidableMatches, 1); assert.equal(p.forfeitedPoints, .5);
  assert.equal(p.meanLift, .25); assert.equal(p.evidence[0].actual, 0);
});
test("Historical success against weak opponents does not imply exploitable loss", () => {
  const p = C.analyzeHistory([series(1, { winners: [1, -1] })], TARGET, current);
  assert.equal(p.meanScore, 1); assert.equal(p.meanLift, 0);
});
test("Old weak models are excluded, identical reuploads are retained", () => {
  const rows = [series(1, { model: model("old", "oldCode"), winners: [-1, 1] }),
    series(2), series(3, { model: model("alias") })];
  const p = C.analyzeHistory(rows, TARGET, current);
  assert.equal(p.normalSeries, 2); assert.equal(p.olderModelSeries, 1);
  assert.equal(p.avoidableMatches, 0);
});
test("Missing hashes require exact version identity", () => {
  assert.equal(C.sameModel({ version: "v1" }, current), true);
  assert.equal(C.sameModel({ version: "v2", number: 1 }, { version: "v1", number: 1 }), false);
  assert.equal(C.sameModel({}, {}), false);
});
test("Technical failures with null winner, pending, void and duplicates are separate", () => {
  const technical = series(2); technical.matches[0].phase = "FAILED";
  technical.matches[0].end_reason = "CPU_TIMEOUT"; technical.matches[0].winner = null;
  const p = C.analyzeHistory([series(1), series(1), technical, series(3, { status: "PENDING" }),
    series(4, { status: "VOID" })], TARGET, current);
  assert.equal(p.normalSeries, 1); assert.equal(p.technicalSeries, 1);
  assert.equal(p.duplicate, 1); assert.equal(p.pending, 1); assert.equal(p.voidSeries, 1);
});
test("Unknown opening theory must not be labelled as an error", () => {
  const row = series(1, { opening: [0, 1], winners: [-1, 1] });
  const p = C.analyzeHistory([row], TARGET, current);
  assert.equal(p.unknownOpeningSeries, 1); assert.equal(p.avoidableMatches, 0);
  assert.equal(p.meanLift, null);
  assert.equal(C.project(p, C.elo(1200, 1000)).score, null);
});
test("Incomplete pairs and inconsistent settlements are rejected", () => {
  const one = series(1); one.matches.pop();
  const wrong = series(2); wrong.settlement = { score_a: 1 };
  const p = C.analyzeHistory([one, wrong], TARGET, current);
  assert.equal(p.malformedSeries, 2); assert.equal(p.normalSeries, 0);
});
test("Recent settled window is selected after current model filtering", () => {
  const rows = [series(1, { winners: [-1, 1] }), series(2), series(3, { model: model("old", "old") })];
  const p = C.analyzeHistory(rows, TARGET, current, { limit: 1 });
  assert.equal(p.normalSeries, 1); assert.equal(p.avoidableMatches, 0);
});
test("All seven mirrored first moves agree with the previously exact-solved opening table", () => {
  assert.deepEqual([...C.BLACK_WDL], [1, 1, 0, 1, 0, 1, 1]);
  for (let c = 0; c < 7; c++) assert.equal(C.openingValue({ opening_pairs: 1, opening_actions: [c, 6 - c] }), C.BLACK_WDL[c]);
  assert.equal(C.openingValue({ opening_pairs: 2, opening_actions: [1, 5] }), null);
});
test("High Elo does not make a favorable board result profitable", () => {
  const e = C.elo(1267, 900, 32);
  assert.equal(e.firstPositiveScore, 1); assert.ok(e.cases[3].rounded < 0);
  assert.ok(e.draw.rounded < 0); assert.ok(e.sweep.rounded > 0);
});
test("Current real rating example has -1 on a tied series", () => {
  const e = C.elo(1267, 1237, 32);
  assert.equal(e.draw.rounded, -1); assert.equal(e.sweep.rounded, 15);
  assert.equal(e.swept.rounded, -17);
});
test("Python-style ties-to-even rounding, including negative halves", () => {
  assert.deepEqual([-.5, -1.5, -2.5, .5, 1.5, 2.5].map(C.roundEven), [0, -2, -2, 0, 2, 2]);
});
test("No evidence is not a 100% confidence statement or a win forecast", () => {
  const rows = Array.from({ length: 12 }, (_, i) => series(i, { opponent: "o" + i % 3 }));
  const p = C.analyzeHistory(rows, TARGET, current), d = C.project(p, C.elo(1300, 1000));
  assert.equal(d.score, .5); assert.ok(d.upper > .5); assert.equal(d.kind, "skip");
});
test("Rare profitable-looking evidence stays uncertain with small samples", () => {
  const p = C.analyzeHistory([series(1, { winners: [-1, 1] })], TARGET, current);
  assert.equal(C.project(p, C.elo(1000, 1000)).kind, "uncertain");
});
test("Head to head requires both current models and reverses opponent score", () => {
  const my = model("mine", "mycode", "myweights");
  const rows = [series(1, { opponent: "me", otherModel: my, winners: [-1, 1] }),
    series(2, { opponent: "me", otherModel: model("oldmine", "old"), winners: [1, -1] })];
  const p = C.analyzeHistory(rows, TARGET, current);
  const d = C.project(p, C.elo(1000, 1000), { assumeOptimal: false, meId: "me", myModel: my });
  assert.equal(d.score, 1); assert.ok(d.source.includes("1组"));
  assert.equal(d.samples, 1); assert.equal(d.kind, "uncertain");
});
test("Mean Elo profit rounds each series first, rather than rounding a mean score", () => {
  const rows = Array.from({ length: 24 }, (_, i) => series(i, {
    opponent: "other" + i % 3, winners: i === 0 ? [0, 1] : [1, 1]
  }));
  const p = C.analyzeHistory(rows, TARGET, current);
  const d = C.project(p, C.elo(1267, 1237));
  assert.ok(Math.abs(d.expectedDelta - (-2 / 3)) < 1e-10);
  assert.ok(Math.abs(d.expectedDelta - d.rawExpectedDelta) > .3);
});
test("Unknown openings do not inflate the diversity of exploitable samples", () => {
  const rows = Array.from({ length: 10 }, (_, i) => series(i, {
    opponent: i < 8 ? "oneKnown" : "unknown", opening: i < 8 ? [1, 5] : [0, 1]
  }));
  const p = C.analyzeHistory(rows, TARGET, current);
  assert.equal(p.opponentCount, 2); assert.equal(p.knownOpponentCount, 1);
  assert.equal(C.project(p, C.elo(1300, 1000)).kind, "uncertain");
});
test("Malformed match entries do not crash the entire profile", () => {
  const row = series(1); row.matches[0] = null;
  const p = C.analyzeHistory([row], TARGET, current);
  assert.equal(p.malformedSeries, 1);
});
