"use strict";
let records = [], page = 1;
const size = 10;
const element = id => document.getElementById(id);
function render() {
  const list = records.filter(row => ["robot", "task", "status"].every(key => !element(key).value || row[key] === element(key).value) && row.id.toLowerCase().includes(element("search").value.toLowerCase()));
  const pages = Math.max(1, Math.ceil(list.length / size));
  page = Math.min(page, pages);
  element("rows").replaceChildren();
  list.slice((page - 1) * size, page * size).forEach((row, index) => {
    const tr = document.createElement("tr");
    const values = [(page - 1) * size + index + 1, row.id + "\n" + row.robot, row.task, row.status, row.steps + " / " + row.max_steps, row.video ? row.duration.toFixed(2) + "초 · 20fps" : "준비 중"];
    values[5] = row.video ? row.duration.toFixed(2) + "초 · 20fps" : "준비 중";
    values.forEach(value => { const td = document.createElement("td"); td.textContent = value; tr.append(td); });
    tr.children[1].className = "run-id";
    const td = document.createElement("td"), button = document.createElement("button");
    button.textContent = "재생"; button.disabled = !row.video;
    button.addEventListener("click", () => {
      element("player-title").textContent = row.id;
      element("video").src = row.video;
      element("video").poster = row.poster;
      element("detail").textContent = row.robot + " · " + row.task + " · " + row.status + " · " + row.type + " · " + row.cameras.map(camera => camera.replace("robot0_", "")).join(" → ");
      element("player").showModal();
    });
    td.append(button); tr.append(td); element("rows").append(tr);
  });
  if (!list.length) { const tr = document.createElement("tr"), td = document.createElement("td"); td.colSpan = 7; td.textContent = "해당하는 영상이 없습니다."; tr.append(td); element("rows").append(tr); }
  element("count").textContent = "총 " + list.length + "개 · 진행 상황은 15초마다 갱신됩니다.";
  element("page").textContent = page + " / " + pages;
  element("prev").disabled = page === 1; element("next").disabled = page === pages;
}
async function refresh() {
  try { const response = await fetch("media/catalog.json", {cache: "no-store"}); if (!response.ok) throw new Error(); records = (await response.json()).reverse(); render(); }
  catch { element("count").textContent = "목록을 불러오지 못했습니다. 잠시 후 다시 시도합니다."; }
}
["robot", "task", "status", "search"].forEach(id => element(id).addEventListener("input", () => { page = 1; render(); }));
element("prev").addEventListener("click", () => { page--; render(); });
element("next").addEventListener("click", () => { page++; render(); });
element("close-player").addEventListener("click", () => element("player").close());
element("player").addEventListener("close", () => { element("video").pause(); element("video").removeAttribute("src"); element("video").load(); });
refresh(); setInterval(refresh, 15000);
