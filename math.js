(function(root){
  'use strict';
  const sigmoid=x=>x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x));
  const activation=(name,x)=>{
    if(name==='relu')return {y:Math.max(0,x),d:x>0?1:0,note:x===0?'В нуле классическая производная ReLU не определена; здесь принят выбор PyTorch 0.':''};
    if(name==='sigmoid'){const y=sigmoid(x);return {y,d:y*(1-y),note:''};}
    if(name==='tanh'){const y=Math.tanh(x);return {y,d:1-y*y,note:''};}
    if(name==='leaky')return {y:x>=0?x:.1*x,d:x>=0?1:.1,note:x===0?'В нуле наклоны различаются; здесь выбрана правая производная.':''};
    throw new Error('Неизвестная активация');
  };
  const descent=(initial,rate,steps=12)=>{let w=initial;const points=[w];for(let i=0;i<steps;i++){w-=rate*(w-2);points.push(w);}return points;};
  const backprop=(x,w,b,target,rate)=>{const z=w*x+b,p=sigmoid(z),loss=-(target*Math.log(p)+(1-target)*Math.log(1-p)),dz=p-target,dw=dz*x,db=dz;return {z,p,loss,dz,dw,db,nextW:w-rate*dw,nextB:b-rate*db};};
  const correlate=(input,kernel,row,col)=>input.slice(row,row+kernel.length).reduce((s,line,i)=>s+kernel[i].reduce((a,k,j)=>a+k*line[col+j],0),0);
  const api={sigmoid,activation,descent,backprop,correlate};
  root.DLMath=api;
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(typeof window!=='undefined'?window:globalThis);
