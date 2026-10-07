"use strict";
const el = id => document.getElementById(id);
const conditions = {rgb:"RGB only",color:"RGB + depth image",pixel:"RGB + pixel Z",grid:"RGB + 16×16 grid Z"};
let records = [], outputIndex = {}, selected = null, calls = [], generation = 0, limit = 20, inputIndex = {}, inputKey = "", previousStep = -1, seenQueries = new Set();
const token = x => Number.isInteger(x) ? x.toLocaleString() : "미확인";
const clock = x => Number.isFinite(x) ? Math.floor(x/60)+":"+String(Math.floor(x%60)).padStart(2,"0") : "0:00";
const status = r => r.user_assessment === "failed" || r.status === "user-stopped" ? "실패" : r.task_success === true || r.success === true ? "성공" : r.task_success === false || r.success === false ? "미성공" : ({complete:"완료",error:"오류",running:"진행 중"}[r.status] || r.status || "미확인");
const model = r => r.model || "모델 미기록";
const condition = r => {
  const label=conditions[r.condition] || r.type || "조건 미기록";
  const history=r.collection?.startsWith("history-five-task-")&&r.id.match(/-h([0-5])$/);
  return history?label+" · 과거 "+history[1]+"초":label;
};
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
  el("reason").textContent=message;inputKey="";el("input-images").replaceChildren();el("input-note").textContent="";
  ["input-token","output-token","cache-token","total-token"].forEach(id=>el(id).textContent="—");
  ["action","queries","repeat","call-meta","source-note"].forEach(id=>el(id).textContent="");
  el("action-legend").hidden=true;el("action-phase").textContent="연결된 호출 없음";el("seek-call").disabled=true;
}
async function selectVideo(row) {
  const current=++generation;
  const continuing=selected?.id===row.id&&selected?.cumulative_live;
  const savedTime=continuing?el("video").currentTime:0,savedPaused=continuing?el("video").paused:false,savedCall=continuing?el("call").value:"auto";
  if(row.live_clip&&row.visualization_data){try{const response=await fetch(new URL("playback.json",sameOriginPath(row.visualization_data)),{cache:"no-store"});if(response.ok){const playback=await response.json();row={...row,video:playback.video,video_start_step:0,cumulative_live:true,cumulative_version:playback.version,available_end_step:playback.end_step}}}catch(e){}}
  if(current!==generation)return;
  selected=row;calls=[];previousStep=Math.floor(savedTime*(row.control_hz||20));if(!continuing)seenQueries=new Set();
  el("depth-video").pause();
  el("depth-player").hidden=row.condition!=='color';
  el('pixel-query-panel').hidden=!['pixel','grid'].includes(row.condition);
  if(row.depth_video){
    const depthURL=new URL(sameOriginPath(row.depth_video));depthURL.searchParams.set('v',String(row.depth_version));
    el("depth-video").src=depthURL.href;el("depth-video").load();
  }else{el("depth-video").removeAttribute('src');el("depth-video").load();}
  el("depth-note").textContent=row.depth_video?"카메라 Z 시각화 · 밝을수록 가까움 · 기록 구간: 스텝 "+row.depth_start_step+"–"+row.depth_end_step+". 모델에 추가 전달하지 않는 표시용 영상입니다.":"Depth 영상은 기록 준비 중이거나 해당 과거 실행에 저장되지 않았습니다.";
  el("video").pause();el("video-error").hidden=true;
  el("title").textContent=row.task+" · "+(row.scene_id||row.id);
  el("meta").textContent=[model(row),condition(row),row.robot,row.steps!=null?row.steps.toLocaleString()+" native 스텝":null,row.excluded_from_current_target?"현재 평가 대상에서 제외된 보존 기록":null].filter(Boolean).join(" · ");
  el("result-badge").textContent=status(row);el("result-badge").className="badge "+(status(row)==="성공"?"success":status(row)==="미성공"?"failed":"");
  el("call").replaceChildren(new Option("현재 재생 위치에 맞추기","auto"));
  clearOutput("모델 출력 기록을 불러오는 중입니다.");
  const playbackURL=new URL(sameOriginPath(row.video));playbackURL.searchParams.set("playback",String(row.cumulative_version||row.live_version||1));
  el("video").src=playbackURL.href;el("video").poster=sameOriginPath(row.poster)||"";el("video").load();
  el("video").playbackRate=Number(el("speed").value);
  if(row.live_clip){el("video").muted=true;el("video").addEventListener("loadedmetadata",()=>{el("video").currentTime=Math.min(savedTime,Math.max(0,el("video").duration-0.05));if(!savedPaused)el("video").play().catch(()=>{});},{once:true});}
  const url=new URL(location.href);url.searchParams.set("id",row.id);history.replaceState(null,"",url);
  renderList();
  const dataPath=row.visualization_data||outputIndex[row.id];
  if(!dataPath){clearOutput("이 영상에는 연결된 모델 설명·토큰 기록이 없습니다.");el("sync-state").textContent="기록 없음";return}
  try {
    const response=await fetch(sameOriginPath(dataPath),{cache:"no-store"});if(!response.ok)throw Error();
    const data=await response.json();
    inputIndex={};try{const inputs=await fetch(new URL("inputs.json",sameOriginPath(dataPath)),{cache:"no-store"});if(inputs.ok)inputIndex=await inputs.json()}catch(e){}
    if(current!==generation)return;
    calls=(data.calls||[]).filter(c=>Number.isInteger(c.step)).sort((a,b)=>a.step-b.step||a.call.localeCompare(b.call));
    selected={...row,control_hz:data.control_hz||row.fps||20,timeline_note:data.timeline_note};
    calls.forEach(c=>el("call").add(new Option(c.call+" · 스텝 "+c.step+((c.response.queries||[]).length?" · 거리 조회":" · 행동"),c.call)));
    if([...el("call").options].some(o=>o.value===savedCall))el("call").value=savedCall;
    showOutput();
  } catch(e){if(current===generation){clearOutput("모델 출력 기록을 불러오지 못했습니다.");el("sync-state").textContent="기록 확인 불가"}}
}
function showOutput() {
  syncDepth();
  const video=el("video"),step=(selected?.video_start_step||0)+Math.floor((video.currentTime+1e-6)*(selected?.control_hz||selected?.fps||20));
  el("position").textContent=clock(video.currentTime)+" / "+clock(video.duration)+" · native 스텝 "+step;
  if(!calls.length)return;
  if(selected.condition==='pixel'&&el('pause-query').checked&&!video.paused&&el('call').value==='auto'){
    const crossed=calls.find(c=>c.step>previousStep&&c.step<=step&&(c.response.queries||[]).length&&!seenQueries.has(c.call));
    if(crossed){seenQueries.add(crossed.call);video.pause();video.currentTime=Math.max(0,crossed.step-(selected.video_start_step||0))/(selected.control_hz||20);el('call').value=crossed.call;}
  }
  previousStep=step;
  const latestQuery=[...calls].reverse().find(c=>c.step<=step&&(c.response.queries||[]).length);
  renderQuerySummary(el('call').value!=='auto'?calls.find(c=>c.call===el('call').value&&(c.response.queries||[]).length)||latestQuery:latestQuery);
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
  el("action").textContent=JSON.stringify(row.response.action,null,2)+(query?"\n거리 조회 응답: 이 벡터는 로봇에 적용하지 않습니다.":"");
  renderActionLegend(row.response.action,query);
  const group=calls.filter(c=>c.step===row.step);
  const requests=group.flatMap(c=>c.response.queries||[]);
  const answers=row.query_answers?.length?row.query_answers:group.flatMap(c=>c.query_answers||[]);
  el("queries").textContent=JSON.stringify({requests,answers},null,2);
  el("query-box").open=true;
  renderInputs(row,requests);
  el("source-note").textContent=selected.cumulative_live?"처음부터 누적된 실제 20fps 영상입니다. 인코더에 저장된 스텝 "+selected.available_end_step+"까지 재생할 수 있으며, 저장 중인 최신 프레임은 다음 갱신에 추가됩니다. 새 영상이 추가돼도 재생 위치를 유지합니다.":selected.live_clip?"현재 행동의 실제 프레임으로 만든 최신 20fps 영상입니다. 완료 후 전체 영상으로 바뀝니다.":selected.timeline_note||"원본 response.json · receipt.json · native 관측 스텝 기준";
  if(selected.video_capture_gap)el("source-note").textContent="재개 시 행동별 영상 연결에 누락 구간이 있어 검증된 스텝 "+selected.video_capture_gap.last_verified_step+"까지 재생합니다. 모델 조회·답변 기록은 계속 갱신됩니다. 완료 후 전체 녹화 영상으로 교체합니다.";
  el("seek-call").disabled=!manual;
}
function renderActionLegend(action,query){
  const supported=selected?.robot==='PandaOmron'&&Array.isArray(action)&&action.length===12;
  el('action-legend').hidden=!supported;el('action-components').replaceChildren();if(!supported)return;
  const meanings=[
    '손끝 이동 Δx · 로봇 베이스 좌표계 X축',
    '손끝 이동 Δy · 로봇 베이스 좌표계 Y축',
    '손끝 이동 Δz · 로봇 베이스 좌표계 Z축 (양수: 위)',
    '손끝 회전 Δrx · X축 회전벡터 성분',
    '손끝 회전 Δry · Y축 회전벡터 성분',
    '손끝 회전 Δrz · Z축 회전벡터 성분',
    '그리퍼 · +1 닫기 / −1 열기 / 0 개폐 목표 변화 없음',
    '이동 베이스 · 전후 방향 속도 입력',
    '이동 베이스 · 좌우 방향 속도 입력',
    '이동 베이스 · 수직축 회전(yaw) 속도 입력',
    '몸통 높이 관절 · JOINT_POSITION 입력',
    '모드 · ≤0: 현재 팔 위치 기준 / >0: 이동 베이스 추종용 팔 목표 기준'
  ];
  meanings.forEach((meaning,i)=>{const tr=document.createElement('tr');for(const value of [(i+1)+'번 / ['+i+']',String(action[i]),meaning]){const td=document.createElement('td');td.textContent=value;tr.append(td);}el('action-components').append(tr);});
  el('action-controller-note').textContent='PandaOmron · HYBRID_MOBILE_BASE / OSC_POSE · 이동 1.0은 목표 변화 0.05m, 회전 1.0은 0.5rad로 스케일됩니다. 실제 이동량은 물리·충돌에 따라 달라집니다. 회전은 Euler 각도가 아닌 회전벡터입니다. repeat만큼 20Hz에서 반복 적용합니다.'+(query?' 이 호출은 거리 조회이므로 모든 행동 성분과 repeat는 미적용입니다.':'');
}
function renderQuerySummary(row){
  el('query-rows').replaceChildren();
  el('query-position').textContent=row?row.call+' · 관측 스텝 '+row.step+' · 이 조회 결과는 다음 조회까지 유지됩니다.':'';
  el('query-empty').textContent=row?'원본 조회 요청과 환경 답변입니다. 빨간 좌표는 아래 모델 입력 이미지에서 확인할 수 있습니다.':'이 재생 위치까지 픽셀 조회 기록이 없습니다.';
  for(const q of row?.response.queries||[]){
    const answer=(row.query_answers||[]).find(a=>a.camera===q.camera&&a.observation_id===q.observation_id&&a.pixel_uv?.[0]===q.u&&a.pixel_uv?.[1]===q.v);
    const tr=document.createElement('tr');
    for(const value of [q.camera,'('+q.u+', '+q.v+')',Number.isFinite(answer?.pixel_depth_m)?(answer.pixel_depth_m*100).toFixed(2)+' cm ('+answer.pixel_depth_m.toFixed(4)+' m)':answer?.error||'답변 미확인']){const td=document.createElement('td');td.textContent=value;tr.append(td);}
    el('query-rows').append(tr);
  }
}
function syncDepth(){
  const depth=el('depth-video'),rgb=el('video');
  if(selected?.condition!=='color'||!selected?.depth_video||!Number.isFinite(depth.duration))return;
  const step=(selected.video_start_step||0)+rgb.currentTime*(selected.control_hz||20);
  const offset=(step-selected.depth_start_step)/(selected.control_hz||20);
  const available=offset>=0&&step<=selected.depth_end_step;
  depth.hidden=!available;
  if(!available){depth.pause();return;}
  if(Math.abs(depth.currentTime-offset)>0.15)depth.currentTime=Math.min(offset,depth.duration);
  depth.playbackRate=rgb.playbackRate;
  if(rgb.paused)depth.pause();else depth.play().catch(()=>{});
}
function renderInputs(row,requests) {
  const key=selected.id+":"+row.call;if(inputKey===key)return;inputKey=key;
  el("input-images").replaceChildren();
  const images=inputIndex[row.call]||row.input_images||[];
  el("input-note").textContent=(images.length?"해당 호출에 실제 전달된 원본 이미지입니다. Depth는 카메라 Z이며 밝기 범위는 이미지 하단에 표시됩니다.":"이 호출의 입력 이미지가 아직 게시되지 않았습니다.")+(requests.length?" 빨간 표시는 모델이 요청한 조회 위치입니다.":" 이 관측에서 모델의 거리 조회 요청은 없습니다.");
  for(const item of images){
    const figure=document.createElement("figure"),label=document.createElement("figcaption"),canvas=document.createElement("canvas"),img=new Image();
    label.textContent=item.camera+" · "+(item.kind==="depth"?"Depth 입력":"RGB 입력");figure.append(label,canvas);el("input-images").append(figure);
    img.onload=()=>{canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;const ctx=canvas.getContext("2d");ctx.drawImage(img,0,0);ctx.strokeStyle="#ff384f";ctx.fillStyle="#ff384f";ctx.lineWidth=2;
      const camera=item.camera.split(" (")[0];
      requests.filter(q=>q.camera===camera&&(!item.observation_id||q.observation_id===item.observation_id)).forEach(q=>{if(q.kind==="pixel"||q.u!=null){ctx.beginPath();ctx.arc(q.u,q.v,Math.max(4,q.radius||0),0,Math.PI*2);ctx.stroke();ctx.fillText("("+q.u+","+q.v+")",Math.min(q.u+6,canvas.width-65),Math.max(12,q.v-6))}});
    };img.src=sameOriginPath(item.url);
  }
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
    else if(selected.live_clip){
      const next=records.find(r=>r.id===selected.id);let changed=next?.live_version!==selected.live_version;
      if(next?.visualization_data){try{const resp=await fetch(new URL("playback.json",sameOriginPath(next.visualization_data)),{cache:"no-store"});if(resp.ok){const info=await resp.json();changed=changed||info.version!==selected.cumulative_version}}catch(e){}}
      if(next&&changed)await selectVideo(next);
    }
    else if(deleted.has(selected.id)){selected=null;calls=[];el("video").pause();el("video").removeAttribute("src");el("video").load();el("title").textContent="삭제된 영상";clearOutput("이 영상은 목록에서 삭제되었습니다.")}
  } catch(e){el("library-notice").textContent="영상 목록을 불러오지 못했습니다. 잠시 후 다시 확인합니다."}
}
["search","task","model","condition","result","with-output"].forEach(id=>el(id).addEventListener(id==="search"?"input":"change",()=>{limit=20;renderList()}));
el("more").addEventListener("click",()=>{limit+=20;renderList()});
el("video").addEventListener("timeupdate",showOutput);el("video").addEventListener("loadedmetadata",showOutput);el("video").addEventListener("seeked",showOutput);
el("video").addEventListener("play",()=>{el("call").value="auto";previousStep=Math.floor(el("video").currentTime*(selected?.control_hz||20));syncDepth();});el("video").addEventListener("pause",syncDepth);el("depth-video").addEventListener("loadedmetadata",syncDepth);
el("video").addEventListener("error",()=>{el("video-error").hidden=false;el("video-error").textContent="영상을 불러오지 못했습니다. 파일 또는 연결 상태를 확인해 주세요."});
el("speed").addEventListener("change",()=>el("video").playbackRate=Number(el("speed").value));
el("call").addEventListener("change",()=>{if(el("call").value!=="auto")el("video").pause();showOutput();});
el("seek-call").addEventListener("click",()=>{const row=calls.find(c=>c.call===el("call").value);if(row){el("video").currentTime=Math.max(0,row.step-(selected.video_start_step||0))/(selected.control_hz||20);el("call").value="auto";showOutput()}});
loadLibrary();setInterval(loadLibrary,5000);
