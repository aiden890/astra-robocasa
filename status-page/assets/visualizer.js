"use strict";
const el = id => document.getElementById(id);
const conditions = {rgb:"RGB only",color:"RGB + depth image",pixel:"RGB + pixel Z",grid:"RGB + 16×16 grid Z"};
let records = [], outputIndex = {}, selected = null, calls = [], generation = 0, limit = 20;
const token = x => Number.isInteger(x) ? x.toLocaleString() : "미확인";
const clock = x => Number.isFinite(x) ? Math.floor(x/60)+":"+String(Math.floor(x%60)).padStart(2,"0") : "0:00";
const status = r => r.task_success === true || r.success === true ? "성공" : r.task_success === false || r.success === false ? "미성공" : ({complete:"완료",error:"오류",running:"진행 중"}[r.status] || r.status || "미확인");
const model = r => r.model || "모델 미기록";
const condition = r => conditions[r.condition] || r.type || "조건 미기록";
function sameOriginPath(path) {
  if (!path) return null;
  const url = new URL(path, location.origin+"/");
  return url.origin === location.origin && ["http:","https:"].includes(url.protocol) ? url.href : null;
}
function populate(id, values) {
  const current = el(id).value;
  el(id).replaceChildren(new Option("전체", ""));
  [...new Set(values)].sort().forEach(value=>el(id).add(new Option(value,value)));
  if ([...el(id).options].some(x=>x.value===current)) el(id).value=current;
}
function renderList() {
  const query=el("search").value.trim().toLowerCase();
  const filtered=records.filter(r=>[
    !el("task").value || r.task===el("task").value,
    !el("model").value || model(r)===el("model").value,
    !el("condition").value || condition(r)===el("condition").value,
    !el("result").value || status(r)===el("result").value,
    !el("with-output").checked || !!(r.visualization_data || outputIndex[r.id]),
    [r.id,r.task,r.scene_id,r.description,model(r),condition(r)].join(" ").toLowerCase().includes(query)
  ].every(Boolean));
  el("count").textContent=filtered.length+"개 영상";
  el("list").replaceChildren();
  filtered.slice(0,limit).forEach(r=>{
    const button=document.createElement("button");button.className="video-card"+(selected?.id===r.id?" selected":"");
    button.setAttribute("aria-label",r.task+" · "+r.id);button.setAttribute("aria-pressed",String(selected?.id===r.id));
    const title=document.createElement("strong"),line=document.createElement("small"),id=document.createElement("small");
    title.textContent=r.task+" · "+status(r);line.textContent=condition(r)+" · "+model(r);id.textContent=r.scene_id||r.id;
    button.append(title,line,id);button.addEventListener("click",()=>selectVideo(r));el("list").append(button);
  });
  if(!filtered.length)el("list").textContent="검색 조건에 맞는 영상이 없습니다.";
  el("more").hidden=filtered.length<=limit;
}
function clearOutput(message) {
  el("reason").textContent=message;
  ["input-token","output-token","cache-token","total-token"].forEach(id=>el(id).textContent="—");
  ["action","queries","repeat","call-meta","source-note"].forEach(id=>el(id).textContent="");
  el("action-phase").textContent="연결된 호출 없음";el("seek-call").disabled=true;
}
async function selectVideo(row) {
  const current=++generation;selected=row;calls=[];
  el("video").pause();el("video-error").hidden=true;
  el("title").textContent=row.task+" · "+(row.scene_id||row.id);
  el("meta").textContent=[model(row),condition(row),row.robot,row.steps!=null?row.steps.toLocaleString()+" native 스텝":null,row.excluded_from_current_target?"현재 평가 대상에서 제외된 보존 기록":null].filter(Boolean).join(" · ");
  el("result-badge").textContent=status(row);el("result-badge").className="badge "+(status(row)==="성공"?"success":status(row)==="미성공"?"failed":"");
  el("call").replaceChildren(new Option("현재 재생 위치에 맞추기","auto"));
  clearOutput("모델 출력 기록을 불러오는 중입니다.");
  const playbackURL=new URL(sameOriginPath(row.video));playbackURL.searchParams.set("playback","1");
  el("video").src=playbackURL.href;el("video").poster=sameOriginPath(row.poster)||"";el("video").load();
  el("video").playbackRate=Number(el("speed").value);
  const url=new URL(location.href);url.searchParams.set("id",row.id);history.replaceState(null,"",url);
  renderList();
  const dataPath=row.visualization_data||outputIndex[row.id];
  if(!dataPath){clearOutput("이 영상에는 연결된 모델 설명·토큰 기록이 없습니다.");el("sync-state").textContent="기록 없음";return}
  try {
    const response=await fetch(sameOriginPath(dataPath),{cache:"no-store"});if(!response.ok)throw Error();
    const data=await response.json();if(current!==generation)return;
    calls=(data.calls||[]).filter(c=>Number.isInteger(c.step)).sort((a,b)=>a.step-b.step||a.call.localeCompare(b.call));
    selected={...row,control_hz:data.control_hz||row.fps||20,timeline_note:data.timeline_note};
    calls.forEach(c=>el("call").add(new Option(c.call+" · 스텝 "+c.step+((c.response.queries||[]).length?" · 거리 조회":" · 행동"),c.call)));
    showOutput();
  } catch(e){if(current===generation){clearOutput("모델 출력 기록을 불러오지 못했습니다.");el("sync-state").textContent="기록 확인 불가"}}
}
function showOutput() {
  const video=el("video"),step=Math.floor((video.currentTime+1e-6)*(selected?.control_hz||selected?.fps||20));
  el("position").textContent=clock(video.currentTime)+" / "+clock(video.duration)+" · native 스텝 "+step;
  if(!calls.length)return;
  const manual=el("call").value!=="auto";
  const row=manual?calls.find(c=>c.call===el("call").value):[...calls].reverse().find(c=>c.step<=step&&!(c.response.queries||[]).length);
  el("sync-state").textContent=manual?"호출 직접 선택":"영상과 동기화";
  if(!row){clearOutput("현재 재생 위치에 대응하는 모델 응답이 없습니다.");return}
  const query=(row.response.queries||[]).length>0;
  el("reason").textContent=row.response.reason||"행동 이유 미제공";
  el("call-meta").textContent=row.call+" · 관측 스텝 "+row.step+(row.cli_seconds!=null?" · 응답 "+row.cli_seconds.toFixed(1)+"초":"");
  [["input-token","input_tokens"],["output-token","output_tokens"],["cache-token","cached_input_tokens"],["total-token","total_tokens"]].forEach(([id,key])=>el(id).textContent=token(row.usage?.[key]));
  el("action-phase").textContent=query?"거리 조회 · 로봇 정지 · 행동 값은 미적용":"시뮬레이터에 적용하는 행동";
  el("repeat").textContent=query?"":row.response.repeat+" native 스텝 반복";
  el("action").textContent=query?"거리 조회가 있으므로 이 응답의 action/repeat는 적용하지 않습니다.":JSON.stringify(row.response.action,null,2);
  const group=calls.filter(c=>c.step===row.step);
  const requests=group.flatMap(c=>c.response.queries||[]);
  const answers=row.query_answers?.length?row.query_answers:group.flatMap(c=>c.query_answers||[]);
  el("queries").textContent=JSON.stringify({requests,answers},null,2);
  el("query-box").open=requests.length>0;
  el("source-note").textContent=selected.timeline_note||"원본 response.json · receipt.json · native 관측 스텝 기준";
  el("seek-call").disabled=!manual;
}
async function loadLibrary() {
  try {
    const responses=await Promise.all([fetch("media/catalog.json",{cache:"no-store"}),fetch("media/supplemental-catalog.json",{cache:"no-store"}),fetch("/api/board",{cache:"no-store"}),fetch("media/model-output-index.json",{cache:"no-store"}),fetch("media/model-output-catalog.json",{cache:"no-store"})]);
    if(!responses[0].ok||!responses[2].ok)throw Error();
    const [base,extra,state,index,preserved]=await Promise.all(responses.map((r,i)=>r.ok?r.json():i===3?{}:[]));
    outputIndex=index;
    const deleted=new Set((state.deleted||[]).filter(x=>x.kind==="video").map(x=>x.id));
    records=[...new Map([...base,...extra,...preserved].filter(r=>r.video&&!deleted.has(r.id)&&sameOriginPath(r.video)).map(r=>[r.id,r])).values()];
    populate("task",records.map(r=>r.task));populate("model",records.map(model));populate("condition",records.map(condition));renderList();
    el("library-notice").textContent="영상 "+records.length+"개 · 삭제한 항목 제외";
    if(!selected){const id=new URL(location.href).searchParams.get("id");const first=records.find(r=>r.id===id)||records.find(r=>r.visualization_data||outputIndex[r.id])||records[0];if(first)await selectVideo(first)}
    else if(deleted.has(selected.id)){selected=null;calls=[];el("video").pause();el("video").removeAttribute("src");el("video").load();el("title").textContent="삭제된 영상";clearOutput("이 영상은 목록에서 삭제되었습니다.")}
  } catch(e){el("library-notice").textContent="영상 목록을 불러오지 못했습니다. 잠시 후 다시 확인합니다."}
}
["search","task","model","condition","result","with-output"].forEach(id=>el(id).addEventListener(id==="search"?"input":"change",()=>{limit=20;renderList()}));
el("more").addEventListener("click",()=>{limit+=20;renderList()});
el("video").addEventListener("timeupdate",showOutput);el("video").addEventListener("loadedmetadata",showOutput);el("video").addEventListener("seeked",showOutput);
el("video").addEventListener("error",()=>{el("video-error").hidden=false;el("video-error").textContent="영상을 불러오지 못했습니다. 파일 또는 연결 상태를 확인해 주세요."});
el("speed").addEventListener("change",()=>el("video").playbackRate=Number(el("speed").value));
el("call").addEventListener("change",showOutput);
el("seek-call").addEventListener("click",()=>{const row=calls.find(c=>c.call===el("call").value);if(row){el("video").currentTime=row.step/(selected.control_hz||20);el("call").value="auto";showOutput()}});
loadLibrary();setInterval(loadLibrary,30000);
