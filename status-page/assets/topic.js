"use strict";
const element=id=>document.getElementById(id),cards=new Map();let topic;
function render(){
 if(!topic)return;
 cards.forEach(card=>{card.hidden=true;});
 const visible=topic.records.filter(row=>["task","robot","status"].every(key=>!element("detail-"+key).value||row[key]===element("detail-"+key).value));
 topic.records.forEach((row,index)=>{
  let card=cards.get(row.id);
  if(!card){
   card=TopicLibrary.node("article",undefined,"card video-entry");const heading=TopicLibrary.node("div",undefined,"board-heading");heading.append(TopicLibrary.node("h2",(index+1)+". "+row.robot),Board.deleteButton("video",row.id,row.robot+" · "+row.id));card.append(heading);card.append(TopicLibrary.node("p",row.id,"run-id"));
   card.append(TopicLibrary.node("p",row.description || (row.task+" · "+(row.robot==="PandaOmron"?"기본 단일 팔 로봇":"양팔 로봇")+" · 공식 환경 지시문과 native 성공 판정으로 실행합니다."),"entry-description"));
   card.append(TopicLibrary.node("p",undefined,"entry-progress"));card.append(TopicLibrary.node("div",undefined,"media-slot"));card.append(TopicLibrary.node("p",row.cameras.map(c=>c.replace("robot0_","")).join(" → "),"vsub"));cards.set(row.id,card);element("video-list").append(card);
  }
  card.hidden=!visible.includes(row);card.style.order=index;card.querySelector("h2").textContent=(index+1)+". "+row.task+" · "+row.robot+(row.kind==="initial_scene"?" · 시드 "+row.seed:"");card.querySelector(".entry-progress").textContent=row.kind==="initial_scene"?"초기 상태 · 행동 실행 전 · 3개 카메라 · "+row.scene_id:row.status+" · "+row.steps+" / "+row.max_steps+(row.kind==="sensor_diagnostic"?"프레임":"스텝")+" · "+row.fps+"fps · 기록 "+row.duration.toFixed(2)+"초";
  const slot=card.querySelector(".media-slot");
  if(row.image&&!slot.querySelector("img")){slot.replaceChildren();const link=TopicLibrary.node("a",undefined,"initial-image-link");link.href=row.image;link.target="_blank";link.rel="noopener";const image=document.createElement("img");image.src=row.image;image.alt=row.task+" 시드 "+row.seed+" 초기 상태: 왼쪽, 오른쪽, 손목 카메라";image.loading="lazy";image.style.width="100%";image.style.height="auto";link.append(image);slot.append(link);const cameras=TopicLibrary.node("div",undefined,"tags");(row.camera_images||[]).forEach(view=>{const original=TopicLibrary.node("a",view.name.replace("robot0_","")+" 원본","download-link");original.href=view.image;original.target="_blank";original.rel="noopener";cameras.append(original);});slot.append(cameras);}
  else if(row.video&&!slot.querySelector("video")){slot.replaceChildren();const video=document.createElement("video");video.controls=true;video.playsInline=true;video.preload="metadata";video.src=row.video;video.poster=row.poster;video.setAttribute("aria-label",row.robot+" "+row.id+" 영상");const link=TopicLibrary.node("a","원본 MP4 열기","download-link");link.href=row.video;slot.append(video,link);}
  else if(!row.video&&!row.image&&!slot.firstChild)slot.append(TopicLibrary.node("div",row.status==="오류"?"실행 오류로 게시할 영상이 없습니다.":row.status==="대기"?"실행 대기 중입니다. 빈 슬롯이 생기면 자동으로 시작합니다.":"실행 중입니다. 완료 후 영상이 자동으로 추가됩니다.","media-pending"));
 });
 element("detail-count").textContent="관련 항목 "+visible.length+"개 · 영상 "+visible.filter(r=>r.video).length+"개 · 초기 이미지 "+visible.filter(r=>r.image).length+"개";
}
async function refresh(){
 try{const topics=await TopicLibrary.load(),id=new URLSearchParams(location.search).get("topic");topic=topics.find(t=>t.id===id||(t.aliases||[]).includes(id));if(!topic){element("topic-title").textContent="삭제되었거나 존재하지 않는 게시글입니다.";element("topic-description").textContent="게시판의 휴지통에서 복원할 수 있습니다.";element("delete-topic").disabled=true;element("video-list").replaceChildren();return;}
 element("recording-note").textContent=topic.records.some(row=>row.kind==="initial_scene")?"고정 평가 씬의 초기 관측입니다. 이미지를 클릭하면 크게 볼 수 있습니다. 태스크 수행 영상이나 성공 평가 결과가 아닙니다.":topic.records.some(row=>row.kind==="sensor_diagnostic")?"센서 렌더링 · 20fps · 태스크 평가 및 모델 추론 결과가 아닙니다.":"매 스텝 연속 기록 · 20fps · 영상 길이는 시뮬레이션 시간입니다. 모델 응답 대기 시간은 제외됩니다.";document.title="RoboCasa · Astra | "+topic.title;element("topic-title").textContent=topic.title;element("topic-description").textContent=topic.description;element("topic-meta").textContent=[topic.category,topic.task,topic.model,topic.environment,topic.date].filter(Boolean).join(" · ");element("topic-tags").replaceChildren(...topic.tags.map(tag=>TopicLibrary.node("span",tag,"tag")));TopicLibrary.options(element("detail-task"),topic.records.map(row=>row.task));TopicLibrary.options(element("detail-robot"),topic.records.map(row=>row.robot));render();}
 catch{element("detail-count").textContent="목록 갱신 실패. 기존 영상은 계속 재생할 수 있습니다.";}
}
element("delete-topic").addEventListener("click",async()=>{if(!topic)return;try{await Board.change("delete","topic",topic.id,topic.title);location.href="videos.html?deleted=1";}catch{Board.notice("삭제하지 못했습니다. 다시 시도해 주세요.");}});window.addEventListener("boardchanged",refresh);
["detail-task","detail-robot","detail-status"].forEach(id=>element(id).addEventListener("input",render));refresh();setInterval(refresh,15000);
