"use strict";
const element=id=>document.getElementById(id),cards=new Map();let topic;
function render(){
 if(!topic)return;
 const visible=topic.records.filter(row=>["robot","status"].every(key=>!element("detail-"+key).value||row[key]===element("detail-"+key).value));
 topic.records.forEach((row,index)=>{
  let card=cards.get(row.id);
  if(!card){
   card=TopicLibrary.node("article",undefined,"card video-entry");card.append(TopicLibrary.node("h2",(index+1)+". "+row.robot));card.append(TopicLibrary.node("p",row.id,"run-id"));
   card.append(TopicLibrary.node("p",row.robot==="PandaOmron"?"기본 단일 팔 로봇으로 컵 집기부터 커피 시작까지 수행합니다.":"양팔 로봇으로 동일한 커피 태스크를 수행합니다. 기본 로봇과 다른 장면 초기화를 사용하므로 직접적인 성능 비교로 해석할 수 없습니다.","entry-description"));
   card.append(TopicLibrary.node("p",undefined,"entry-progress"));card.append(TopicLibrary.node("div",undefined,"media-slot"));card.append(TopicLibrary.node("p",row.cameras.map(c=>c.replace("robot0_","")).join(" → "),"vsub"));cards.set(row.id,card);element("video-list").append(card);
  }
  card.hidden=!visible.includes(row);card.querySelector(".entry-progress").textContent=row.status+" · "+row.steps+" / "+row.max_steps+"스텝 · "+row.fps+"fps · 기록 "+row.duration.toFixed(2)+"초";
  const slot=card.querySelector(".media-slot");
  if(row.video&&!slot.querySelector("video")){slot.replaceChildren();const video=document.createElement("video");video.controls=true;video.playsInline=true;video.preload="metadata";video.src=row.video;video.poster=row.poster;video.setAttribute("aria-label",row.robot+" "+row.id+" 영상");const link=TopicLibrary.node("a","원본 MP4 열기","download-link");link.href=row.video;slot.append(video,link);}
  else if(!row.video&&!slot.firstChild)slot.append(TopicLibrary.node("div",row.status==="오류"?"실행 오류로 게시할 영상이 없습니다.":"실행 중입니다. 완료 후 영상이 자동으로 추가됩니다.","media-pending"));
 });
 element("detail-count").textContent="관련 실행 "+visible.length+"개 · 재생 가능 영상 "+visible.filter(r=>r.video).length+"개";
}
async function refresh(){
 try{const topics=await TopicLibrary.load(),id=new URLSearchParams(location.search).get("topic");topic=topics.find(t=>t.id===id);if(!topic){element("topic-title").textContent="주제를 찾을 수 없습니다.";return;}
 document.title="RoboCasa · Astra | "+topic.title;element("topic-title").textContent=topic.title;element("topic-description").textContent=topic.description;element("topic-meta").textContent=[topic.category,topic.task,topic.model,topic.environment,topic.date].filter(Boolean).join(" · ");element("topic-tags").replaceChildren(...topic.tags.map(tag=>TopicLibrary.node("span",tag,"tag")));TopicLibrary.options(element("detail-robot"),topic.records.map(row=>row.robot));render();}
 catch{element("detail-count").textContent="목록 갱신 실패. 기존 영상은 계속 재생할 수 있습니다.";}
}
["detail-robot","detail-status"].forEach(id=>element(id).addEventListener("input",render));refresh();setInterval(refresh,15000);
