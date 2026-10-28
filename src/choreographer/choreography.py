"""Versioned independently executable geometric actions; never inspects audio."""
import colorsys
import copy
import json
import math
from pathlib import Path
import numpy as np
from .score import SceneScore, Entity, Event, TempoMap

PROFILE = "choreography-v3"
ACTION_VERSION = "choreography-actions-v2"
MATERIALS = ["stroke", "orbit", "ribbon", "tunnel", "spark", "stars", "rose", "lissajous", "hypotrochoid"]
VERBS = ["enter", "exit", "hold", "translate", "rotate", "scale", "draw", "morph", "split", "merge", "gather", "scatter"]
ROLES = ["hold", "build", "release", "transition"]
LAYOUTS = ["center", "bilateral", "ring", "diagonal", "field"]
FIELDS = ["x", "y", "scale", "rotation", "alpha", "hue", "shape", "bend", "spread", "copies", "reveal", "depth"]
DEFAULT = [.5, .5, .35, .5, 0., .52, .4, .5, 0., 0., 1., .3]
CHANNELS = {"enter":list(range(12)),"exit":[4],"hold":[],"translate":[0,1,7],"rotate":[3],
    "scale":[2],"draw":[10],"morph":[6],"split":[8,9],"merge":[8,9],"gather":[0,1,8],"scatter":[0,1,8]}
PLAN_SIZE = len(MATERIALS)+len(VERBS)+len(ROLES)+len(LAYOUTS)+8+16
PARAM_SIZE = len(FIELDS)
TAU = math.tau


def validate_plan(plan, analysis=None):
    if not isinstance(plan,dict):raise ValueError("Plan must be a JSON object")
    if plan.get("version") != "choreography-plan-v1":
        raise ValueError("Unsupported choreography plan")
    if not isinstance(plan.get("audio_sha256"), str) or not isinstance(plan.get("sections"),list) or not plan['sections']:
        raise ValueError("Plan lacks audio provenance or sections")
    if analysis:
        if plan["audio_sha256"] != analysis["audio_sha256"] or len(plan["sections"]) != len(analysis["sections"]):
            raise ValueError("Plan/audio structure mismatch")
    previous = None
    for i, s in enumerate(plan["sections"]):
        if not isinstance(s,dict):raise ValueError("Section must be a JSON object")
        if s.get("index") != i or not isinstance(s.get("motif_id"), str) or len(s["motif_id"]) > 64:
            raise ValueError("Invalid section or motif identity")
        if not 1 <= len(s.get("materials", [])) <= 4 or any(x not in MATERIALS for x in s["materials"]):
            raise ValueError("Invalid plan materials")
        if not 1 <= len(s.get("verbs", [])) <= 12 or any(x not in VERBS for x in s["verbs"]):
            raise ValueError("Invalid plan verbs")
        if s.get("role") not in ROLES or s.get("layout") not in LAYOUTS:
            raise ValueError("Invalid role or layout")
        for k in ("density", "intensity", "scale", "hue", "variation", "contrast"):
            if not isinstance(s.get(k), (float, int)) or not math.isfinite(s[k]) or not 0 <= s[k] <= 1:
                raise ValueError("Invalid plan parameter: " + k)
        if not isinstance(s.get("anchor"),list) or len(s['anchor'])!=2 or any(type(x) not in (int,float) or not math.isfinite(x) or abs(x)>1 for x in s['anchor']):
            raise ValueError("Invalid plan anchor")
        a, b = s.get("start_beat"), s.get("end_beat")
        if not isinstance(a, (int,float)) or not isinstance(b,(int,float)) or not math.isfinite(a+b) or b <= a:
            raise ValueError("Invalid plan time")
        if previous is not None and abs(a - previous) > 1e-4:
            raise ValueError("Plan sections must cover a continuous track")
        previous = b
        if analysis and any(abs(s[k] - analysis["sections"][i][k]) > 1e-4 for k in ("start_beat","end_beat")):
            raise ValueError("LLM changed analysis boundaries")
    return plan


def plan_vectors(plan):
    validate_plan(plan)
    motifs = list(dict.fromkeys(s["motif_id"] for s in plan["sections"]))
    out = np.zeros((len(plan["sections"]), PLAN_SIZE), np.float32)
    for i,s in enumerate(plan["sections"]):
        offset=0
        for choices,key in [(MATERIALS,"materials"),(VERBS,"verbs"),(ROLES,"role"),(LAYOUTS,"layout")]:
            values = s[key] if isinstance(s[key],list) else [s[key]]
            for v in values: out[i, offset+choices.index(v)] = 1
            offset += len(choices)
        out[i,offset:offset+8] = [s["density"],s["intensity"],s["scale"],s["hue"],*(np.asarray(s["anchor"])*.5+.5),s["variation"],s["contrast"]]
        out[i,offset+8+motifs.index(s["motif_id"])%16] = 1
    return out


def vectors_plan(vectors, analysis):
    sections=[]
    for v, a in zip(np.clip(vectors,0,1), analysis["sections"]):
        offset=0
        materials=[MATERIALS[int(np.argmax(v[:len(MATERIALS)]))]]
        # A background is learned independently, not a genre preset.
        ranking=np.argsort(-v[:len(MATERIALS)])
        if v[ranking[1]] > .45: materials.append(MATERIALS[int(ranking[1])])
        offset+=len(MATERIALS)
        scores=v[offset:offset+len(VERBS)]; verbs=[VERBS[int(j)] for j in np.argsort(-scores)[:4]]
        offset+=len(VERBS)
        role=ROLES[int(np.argmax(v[offset:offset+len(ROLES)]))]; offset+=len(ROLES)
        layout=LAYOUTS[int(np.argmax(v[offset:offset+len(LAYOUTS)]))]; offset+=len(LAYOUTS)
        density,intensity,scale,hue,x,y,variation,contrast=map(float,v[offset:offset+8])
        sections.append(dict(index=a["index"],start_beat=a["start_beat"],end_beat=a["end_beat"],
            motif_id="learned-"+str(int(np.argmax(v[offset+8:offset+24]))),materials=materials,verbs=verbs,
            role=role,layout=layout,density=density,intensity=intensity,scale=scale,hue=hue,
            anchor=[x*2-1,y*2-1],variation=variation,contrast=contrast))
    return validate_plan(dict(version="choreography-plan-v1",audio_sha256=analysis["audio_sha256"],
        sections=sections,source="student_model",duration=analysis["duration"]), analysis)


def validate_program(program, duration=None):
    if program.get("version") != ACTION_VERSION or program.get("profile") != PROFILE:
        raise ValueError("Unsupported action program")
    tempo=TempoMap(program["tempo_map"])
    duration=float(program["duration"] if duration is None else duration)
    ids=[g["id"] for g in program["groups"]]
    if len(set(ids)) != len(ids) or any(type(i) is not int or i<0 for i in ids):
        raise ValueError("Invalid group identities")
    previous=-math.inf
    hit_previous=-math.inf
    for at,strength in program.get("hits",[]):
        if not math.isfinite(at+strength) or not 0<=at<=duration or not 0<=strength<=1 or at<hit_previous:
            raise ValueError("Invalid generated accent timeline")
        hit_previous=at
    for c in program["calls"]:
        if c["object"] not in ids or c["material"] not in MATERIALS or c["verb"] not in VERBS:
            raise ValueError("Invalid action reference")
        if not all(math.isfinite(float(c[k])) for k in ("beat","duration")) or c["beat"]<previous-1e-7 or c["duration"]<0:
            raise ValueError("Invalid action chronology")
        if tempo.beat_to_time(c["beat"]) < -1e-4 or tempo.beat_to_time(c["beat"]+c["duration"]) > duration+1e-4:
            raise ValueError("Action outside audio")
        for key in ("from","to"):
            if len(c[key]) != PARAM_SIZE or any(not math.isfinite(x) or not 0<=x<=1 for x in c[key]):
                raise ValueError("Invalid action parameters")
        if c.get("curve","smooth") not in ("smooth","linear","step"):
            raise ValueError("Invalid action curve")
        if "blend_material" in c and type(c["blend_material"]) is not bool:
            raise ValueError("Invalid material transition flag")
        previous=c["beat"]
    return program


def score_program(program, metadata=None, plan=None):
    validate_program(program)
    tm=TempoMap(program["tempo_map"])
    score=SceneScore(program["duration"],program["tempo_map"],metadata=dict(metadata or {},
        render_profile=PROFILE,presentation=program,plan=plan))
    for g in program["groups"]:
        score.entities.append(Entity(g["id"],"circle",width=1,height=1,source={"presentation_layer":g.get("name",str(g["id"]))}))
        score.events.append(Event(tm.time_to_beat(0),"create",g["id"]))
    if plan:
        score.segments=[dict(start=s["start_beat"],end=s["end_beat"],mode=s["motif_id"]) for s in plan["sections"]]
    return score.validate()


def states(program,time):
    tm=TempoMap(program["tempo_map"]); beat=tm.time_to_beat(time)
    result={g["id"]:dict(values=list(DEFAULT),material="stroke",motif_id="",role="foreground") for g in program["groups"]}
    for c in program["calls"]:
        if c["beat"]>beat: break
        u=1. if c['duration']==0 else min(1,max(0,(beat-c["beat"])/c['duration']))
        curve=c.get("curve","smooth")
        q=u*u*(3-2*u) if curve=="smooth" else (float(u>=1) if curve=="step" else u)
        v=[a+(b-a)*q for a,b in zip(c["from"],c["to"])]
        if c["verb"]=="translate" and u<1:
            dx=c["to"][0]-c["from"][0];dy=c["to"][1]-c["from"][1]
            bend=(c["to"][7]-.5)*.8*math.sin(math.pi*u)
            v[0]=max(0,min(1,v[0]-dy*bend)); v[1]=max(0,min(1,v[1]+dx*bend))
        state=result[c["object"]]
        for channel in CHANNELS[c["verb"]]: state["values"][channel]=v[channel]
        if (c["verb"]=="morph" or (c["verb"]=="enter" and c.get("blend_material",False))) and c["material"]!=state["material"]:
            state.update(morph_from=state["material"],morph_progress=q,morph_shape=c["from"][6])
        elif c["verb"] in ("enter","morph"):
            state.pop("morph_from",None);state.pop("morph_progress",None)
        if c["verb"] in ("enter","morph"): state["material"]=c["material"]
        state.update(motif_id=c.get("motif_id",""),role=c.get("role","foreground"))
    return result


def contour(material,shape,phase,count=129):
    points=[]
    for j in range(count):
        u=j/(count-1);t=u*TAU
        if material=="stroke": x,y=(u*2-1), .12*shape*math.sin(t+phase)
        elif material=="ribbon":
            x=u*2-1; y=(.15+.35*shape)*math.sin(x*(2+4*shape)+phase)*max(0,1-x*x)
        elif material=="rose":
            r=.25+.65*math.cos(5*t); x,y=r*math.cos(t),r*math.sin(t)
            x+=.10*shape*math.cos(2*t+phase);y+=.10*shape*math.sin(3*t-phase)
        elif material=="lissajous": x,y=.85*math.sin(3*t+.8*shape),.65*math.sin(2*t)
        elif material=="hypotrochoid":
            t*=2;d=1.6+2*shape;x=(3*math.cos(t)+d*math.cos(1.5*t))/(3+d);y=(3*math.sin(t)-d*math.sin(1.5*t))/(3+d)
        else:
            r=.68+.10*shape*math.sin(5*t+phase)+.04*math.sin(9*t-phase)
            x,y=r*math.cos(t),r*math.sin(t)
        points.append((x,y))
    return points


def primitives(program,time):
    out=[];seed=program.get("seed",42)
    render_states=[]
    for ident,s in states(program,time).items():
        if s.get("morph_from") and s.get("morph_progress",1)<1:
            old=copy.deepcopy(s);old["material"]=s["morph_from"];old["values"][4]*=1-s["morph_progress"];old["values"][6]=s["morph_shape"]
            render_states.append((ident,old))
            s=copy.deepcopy(s);s["values"][4]*=s["morph_progress"]
        render_states.append((ident,s))
    for ident,s in render_states:
        x,y,size,angle,alpha,hue,shape,bend,spread,copies,reveal,depth=s["values"]
        if alpha < .003 or reveal < .001: continue
        size=.12+size*.95;angle=(angle-.5)*TAU*2;x=(x-.5)*2.7;y=(y-.5)*1.6
        material=s["material"];phase=time*(.20+shape*.8)+ident*.73
        rgb=colorsys.hsv_to_rgb(hue%1,.52,.98); count=1+copies*9
        for k in range(math.ceil(count)):
            visibility=min(1,max(0,count-k))
            theta=k*TAU/max(1,count)+ident*.17
            ox=math.cos(theta)*spread*.75 if k else 0.;oy=math.sin(theta)*spread*.45 if k else 0.
            c,sn=math.cos(angle),math.sin(angle)
            def transform(pts): return [(x+ox+size*(px*c-py*sn),y+oy+size*(px*sn+py*c)) for px,py in pts]
            def add(pts,a=1,stroke=1.7,dot=False):
                out.append((ident,transform(pts),rgb,alpha*visibility*a,stroke,dot))
            if material=="stars":
                for j in range(36):
                    n=j+seed*3+ident*137
                    a=(math.sin(n*127.1)*43758.5453)%1;b=(math.sin((n+41)*127.1)*43758.5453)%1
                    add([((a*2-1)*1.7,(b*2-1))],.35+.25*math.sin(phase+j)**2,1.7,True)
            elif material=="spark":
                for j in range(12):
                    a=j*TAU/12;r=.15+.65*shape
                    add([(r*math.cos(a),r*math.sin(a)),((r+.12)*math.cos(a),(r+.12)*math.sin(a))],.8,2.)
            else:
                rings=4 if material in ("orbit","tunnel","ribbon") else 1
                for ring in range(rings):
                    pts=contour(material,shape,phase+ring*.25)
                    factor=(.65+ring*.20+depth*.2) if material=="tunnel" else 1+ring*.045
                    pts=[(px*factor,py*factor+(ring-1.5)*.055 if material=="ribbon" else py*factor) for px,py in pts]
                    add(pts[:max(2,math.ceil(len(pts)*reveal))],1-ring*.14,1.5+(.8 if ring==0 else 0))
    # Accents are explicitly generated/model-predicted timestamps, never spectral rules.
    for i,(at,strength) in enumerate(program.get("hits",[])):
        age=time-at
        if not 0<=age<.55: continue
        ident=program.get("accent_object",program["groups"][0]["id"] if program["groups"] else 0)
        radius=.16+age*1.8
        out.append((ident,[(radius*math.cos(j*TAU/64),radius*math.sin(j*TAU/64)) for j in range(65)],(.9,.8,1.),strength*(1-age/.55)**2,1.5,False))
    return out
