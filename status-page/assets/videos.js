"use strict";
let topics=[],page=1;
const size=10,element=id=>document.getElementById(id);
function render(){
 const query=element("search").value.trim().toLowerCase();
 const list=topics.filter(topic=>{
  if(["category","model","task"].some(key=>element(key).value&&topic[key]!==element(key).value))return false;
  if(!topic.records.some(row=>["robot","status"].every(key=>!element(key).value||row[key]===element(key).value))&&(element("robot").value||element("status").value))return false;
  return [topic.title,topic.description,topic.category,topic.model,topic.task,...topic.tags,...topic.records.flatMap(row=>[row.robot,row.id,row.status])].join(" ").toLowerCase().includes(query);
 });
 const pages=Math.max(1,Math.ceil(list.length/size));page=Math.min(page,pages);element("rows").replaceChildren();
 list.slice((page-1)*size,page*size).forEach((topic,index)=>{
  const tr=TopicLibrary.node("tr");tr.append(TopicLibrary.node("td",(page-1)*size+index+1));
  const title=TopicLibrary.node("td"),link=TopicLibrary.node("a",topic.title,"topic-title-link");link.href="topic.html?topic="+encodeURIComponent(topic.id);
  title.append(link,TopicLibrary.node("p",topic.description,"topic-summary"));const tags=TopicLibrary.node("div",undefined,"tags");topic.tags.forEach(tag=>tags.append(TopicLibrary.node("span",tag,"tag")));title.append(tags);tr.append(title);
  tr.append(TopicLibrary.node("td",topic.category+" · "+topic.task),TopicLibrary.node("td",[...new Set(topic.records.map(row=>row.robot))].join(" / ")||"등록 전"));
  const summary=["진행 중","성공","미성공","오류"].map(status=>{const n=topic.records.filter(row=>row.status===status).length;return n?status+" "+n:null;}).filter(Boolean).join(" · ");
  tr.append(TopicLibrary.node("td",summary||"등록 전"),TopicLibrary.node("td",topic.records.filter(row=>row.video).length+"개 영상 / "+topic.records.length+"개 실행"),TopicLibrary.node("td",topic.date||"미등록"));const manage=TopicLibrary.node("td"), remove=Board.deleteButton("topic",topic.id,topic.title);manage.append(remove);tr.append(manage);tr.addEventListener("click",event=>{if(!event.target.closest("a,button,select"))location.href=link.href;});element("rows").append(tr);
 });
 if(!list.length){const tr=TopicLibrary.node("tr"),td=TopicLibrary.node("td","해당하는 주제가 없습니다.");td.colSpan=8;tr.append(td);element("rows").append(tr);}
 element("count").textContent="총 "+list.length+"개 주제 · 관련 영상 "+list.reduce((n,t)=>n+t.records.filter(r=>r.video).length,0)+"개 · 15초마다 갱신";
 element("page").textContent=page+" / "+pages;element("prev").disabled=page===1;element("next").disabled=page===pages;
}
async function refresh(){try{topics=await TopicLibrary.load();["category","model","task"].forEach(key=>TopicLibrary.options(element(key),topics.map(t=>t[key])));TopicLibrary.options(element("robot"),topics.flatMap(t=>t.records.map(r=>r.robot)));render();}catch{element("count").textContent="목록을 불러오지 못했습니다. 잠시 후 다시 시도합니다.";}}
["category","model","robot","task","status","search"].forEach(id=>element(id).addEventListener("input",()=>{page=1;render();}));element("prev").addEventListener("click",()=>{page--;render();});element("next").addEventListener("click",()=>{page++;render();});window.addEventListener("boardchanged",refresh);refresh();setInterval(refresh,15000);
