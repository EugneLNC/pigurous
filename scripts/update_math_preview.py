from pathlib import Path

p = Path("preview/src/main.ts")
s = p.read_text(encoding="utf-8")
if "let loadSequence=0;" in s and "loadExample('/mathematical.json')" in s:
    print("Mathematical preview migration already applied")
    raise SystemExit(0)
s = s.replace(
    "if(score.metadata.render_profile==='performance-v1')",
    "if(['performance-v1','mathematical-v2'].includes(String(score.metadata.render_profile)))",
)
s = s.replace(
    "const names:Record<string,string>={orbit:'轮廓',ribbons:'波纹',tunnel:'纵深'};",
    "const names:Record<string,string>=score.metadata.render_profile==='mathematical-v2'?{orbit:'玫瑰基底',ribbons:'利萨如基底',tunnel:'旋轮基底'}:{orbit:'轮廓',ribbons:'波纹',tunnel:'纵深'};",
)
s = s.replace(
    "score.metadata.origin==='learned_actions'?'模型动作编排 · 几何转译监督'",
    "score.metadata.render_profile==='mathematical-v2'?(score.metadata.origin==='learned_actions'?'数学演出 · 模型控制':'数学动作 · 人工演示'):score.metadata.origin==='learned_actions'?'模型动作编排 · 几何转译监督'",
)
s = s.replace(
    "let performanceSurface:PerformanceSurface|undefined;",
    "let performanceSurface:PerformanceSurface|undefined;let loadSequence=0;",
)
s = s.replace(
    "if(f){load(JSON.parse(await f.text()));",
    "if(f){const sequence=++loadSequence;const data=JSON.parse(await f.text());if(sequence!==loadSequence)return;load(data);",
)
start = s.index("async function loadExample(")
s = (
    s[:start]
    + """async function loadExample(file:string){const sequence=++loadSequence;try{const response=await fetch(file);if(sequence!==loadSequence)return;if(!response.ok)throw Error('示例尚未生成');const data=await response.json();if(sequence!==loadSequence)return;load(data);message(score.metadata.render_profile==='mathematical-v2'?'已载入数学演出：原模型控制，柔和运动与闭合图形。关联 NWAD 音频可试听。':score.metadata.origin==='learned_actions'?'已载入原始模型示例；关联 NWAD 音频可试听。编排质量仍在验证中。':'已载入规则演出对照；关联 NWAD 音频可试听。');}catch(e){if(sequence===loadSequence)throw e;}}
for(const [label,file] of [['数学演出','/mathematical.json'],['原始模型','/model-reference.json'],['规则对照','/performance.json']]){const button=document.createElement('button');button.textContent=label;button.onclick=()=>loadExample(file).catch(e=>message(String(e)));document.querySelector('.imports')!.append(button);}
loadExample('/mathematical.json').catch(()=>loadExample('/learned.json')).catch(()=>loadExample('/performance.json')).catch(()=>fetch('/demo.json').then(v=>v.json()).then(load)).catch(e=>message(String(e)));
"""
)
p.write_text(s, encoding="utf-8")
