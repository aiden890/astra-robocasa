"use strict";
const TopicLibrary = {
  async load() {
    const responses = await Promise.all([fetch("topics.json", {cache:"no-store"}), fetch("media/catalog.json", {cache:"no-store"}), fetch("/api/board", {cache:"no-store"})]);
    if (responses.some(response => !response.ok)) throw new Error("목록 요청 실패");
    const [topics, rawRecords, state] = await Promise.all(responses.map(response => response.json()));
    const records = rawRecords.filter(row => !state.deleted.some(item => item.kind === "video" && item.id === row.id));
    const matches = (topic, row) => (Object.keys(topic.match || {}).length > 0 && Object.entries(topic.match).every(([key,value]) => row[key] === value)) || (topic.run_ids || []).includes(row.id);
    const groups = topics.map(topic => ({...topic, records: records.filter(row => matches(topic,row))}));
    records.filter(row => !topics.some(topic => matches(topic,row))).forEach(row => groups.push({id:row.id,title:row.task+" · "+row.id,category:"미분류",description:"새 실행입니다. 주제 분류와 설명을 추가할 수 있습니다.",task:row.task,model:"",environment:"",date:"",tags:[],records:[row]}));
    return groups.filter(topic => !state.deleted.some(item => item.kind === "topic" && item.id === topic.id));
  },
  node(tag, text, className) {const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(className)node.className=className;return node;},
  options(select, values) {const selected=select.value;select.replaceChildren(new Option("전체",""));[...new Set(values.filter(Boolean))].sort().forEach(value=>select.add(new Option(value,value)));select.value=[...select.options].some(option=>option.value===selected)?selected:"";}
};
