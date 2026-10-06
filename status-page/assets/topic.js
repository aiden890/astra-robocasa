"use strict";
const element=id=>document.getElementById(id),cards=new Map();let topic;
function render(){
 if(!topic)return;
 cards.forEach(card=>{card.hidden=true;});
 const visible=topic.records.filter(row=>["robot","status"].every(key=>!element("detail-"+key).value||row[key]===element("detail-"+key).value));
 topic.records.forEach((row,index)=>{
  let card=cards.get(row.id);
  if(!card){
   card=TopicLibrary.node("article",undefined,"card video-entry");const heading=TopicLibrary.node("div",undefined,"board-heading");heading.append(TopicLibrary.node("h2",(index+1)+". "+row.robot),Board.deleteButton("video",row.id,row.robot+" · "+row.id));card.append(heading);card.append(TopicLibrary.node("p",row.id,"run-id"));
   card.append(TopicLibrary.node("p",row.task+" · "+(row.robot==="PandaOmron"?"기본 단일 팔 로봇":"양팔 로봇")+" · 공식 환경 지시문과 native 성공 판정으로 실행합니다.","entry-description"));
   card.append(TopicLibrary.node("p",undefined,"entry-progress"));card.append(TopicLibrary.node("div",undefined,"media-slot"));card.append(TopicLibrary.node("p",row.cameras.map(c=>c.replace("robot0_","")).join(" → "),"vsub"));cards.set(row.id,card);element("video-list").append(card);
  }
  card.hidden=!visible.includes(row);card.querySelector(".entry-progress").textContent=row.status+" · "+row.steps+" / "+row.max_steps+"스텝 · "+row.fps+"fps · 기록 "+row.duration.toFixed(2)+"초";
  const slot=card.querySelector(".media-slot");
  if(row.video&&!slot.querySelector("video")){slot.replaceChildren();const video=document.createElement("video");video.controls=true;video.playsInline=true;video.preload="metadata";video.src=row.video;video.poster=row.poster;video.setAttribute("aria-label",row.robot+" "+row.id+" 영상");const link=TopicLibrary.node("a","원본 MP4 열기","download-link");link.href=row.video;slot.append(video,link);}
  else if(!row.video&&!slot.firstChild)slot.append(TopicLibrary.node("div",row.status==="오류"?"실행 오류로 게시할 영상이 없습니다.":row.status==="대기"?"실행 대기 중입니다. 빈 슬롯이 생기면 자동으로 시작합니다.":"실행 중입니다. 완료 후 영상이 자동으로 추가됩니다.","media-pending"));
 });
 element("detail-count").textContent="관련 실행 "+visible.length+"개 · 재생 가능 영상 "+visible.filter(r=>r.video).length+"개";
}
async function refresh(){
 try{const topics=await TopicLibrary.load(),id=new URLSearchParams(location.search).get("topic");topic=topics.find(t=>t.id===id);if(!topic){element("topic-title").textContent="삭제되었거나 존재하지 않는 게시글입니다.";element("topic-description").textContent="게시판의 휴지통에서 복원할 수 있습니다.";element("delete-topic").disabled=true;element("video-list").replaceChildren();return;}
 document.title="RoboCasa · Astra | "+topic.title;element("topic-title").textContent=topic.title;element("topic-description").textContent=topic.description;element("topic-meta").textContent=[topic.category,topic.task,topic.model,topic.environment,topic.date].filter(Boolean).join(" · ");element("topic-tags").replaceChildren(...topic.tags.map(tag=>TopicLibrary.node("span",tag,"tag")));TopicLibrary.options(element("detail-robot"),topic.records.map(row=>row.robot));render();}
 catch{element("detail-count").textContent="목록 갱신 실패. 기존 영상은 계속 재생할 수 있습니다.";}
}
element("delete-topic").addEventListener("click",async()=>{if(!topic)return;try{await Board.change("delete","topic",topic.id,topic.title);location.href="videos.html?deleted=1";}catch{Board.notice("삭제하지 못했습니다. 다시 시도해 주세요.");}});window.addEventListener("boardchanged",refresh);
["detail-robot","detail-status"].forEach(id=>element(id).addEventListener("input",render));refresh();setInterval(refresh,15000);
