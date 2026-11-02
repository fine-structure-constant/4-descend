"use strict";
const key = "botlab_scout_settings_v1";
chrome.storage.local.get(key).then((values) => {
  document.getElementById("enabled").checked = values[key]?.enabled !== false;
});
document.getElementById("enabled").addEventListener("change", async (event) => {
  const values = await chrome.storage.local.get(key);
  await chrome.storage.local.set({ [key]: { ...values[key], enabled: event.target.checked } });
  document.getElementById("status").textContent = event.target.checked ? "已开启" : "已关闭";
});
document.getElementById("clear").addEventListener("click", async () => {
  await chrome.storage.local.remove("botlab_scout_cache_v1");
  document.getElementById("status").textContent = "缓存已清除，回到战绩页点击 ↻ 刷新。";
});
