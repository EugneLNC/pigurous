"""Local credential bridge, persistent reservations, validated cached music plans."""
import hashlib
import json
import os
import sqlite3
import subprocess
import uuid
from pathlib import Path
from .choreography import MATERIALS, VERBS, ROLES, LAYOUTS, ACTION_VERSION, validate_plan

PROMPT_VERSION = "music-director-v3"
TOKEN_LIMIT = 100_000_000


class Budget:
    def __init__(self, path, limit=TOKEN_LIMIT):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.limit = min(int(limit), TOKEN_LIMIT)
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS requests (id TEXT PRIMARY KEY, reserved INTEGER NOT NULL, actual INTEGER, response_id TEXT, status TEXT NOT NULL)")

    def reserve(self, tokens):
        ident=str(uuid.uuid4())
        with sqlite3.connect(self.path, timeout=30) as db:
            db.execute("BEGIN IMMEDIATE")
            used=db.execute("SELECT COALESCE(SUM(COALESCE(actual,reserved)),0) FROM requests").fetchone()[0]
            if used+tokens>self.limit: raise ValueError("LLM token budget exhausted")
            db.execute("INSERT INTO requests VALUES (?,?,NULL,NULL,'reserved')",(ident,int(tokens)))
        return ident

    def finish(self, ident, usage, response_id=None):
        # output_tokens already includes reasoning_tokens; never add the breakdown twice.
        actual=usage.get("total_tokens") if usage else None
        if actual is None and usage and "input_tokens" in usage and "output_tokens" in usage:
            actual=int(usage["input_tokens"])+int(usage["output_tokens"])
        if actual is not None and (type(actual) is not int or actual<0): raise ValueError("Invalid API usage")
        with sqlite3.connect(self.path) as db:
            row=db.execute("SELECT reserved FROM requests WHERE id=?",(ident,)).fetchone()
            if not row: raise ValueError("Unknown token reservation")
            db.execute("UPDATE requests SET actual=?,response_id=?,status=? WHERE id=?",(actual,response_id,"reported" if actual is not None else "unknown_usage",ident))

    def summary(self):
        with sqlite3.connect(self.path) as db:
            used,n,pending=db.execute("SELECT COALESCE(SUM(COALESCE(actual,reserved)),0),COUNT(*),SUM(actual IS NULL) FROM requests").fetchone()
        return dict(limit=self.limit,used_or_reserved=used,requests=n,unknown_usage=int(pending or 0))


def bridge_request(payload):
    script=Path(__file__).resolve().parents[2]/"scripts"/"deepseek_bridge.ps1"
    if os.name!="nt": raise RuntimeError("DeepSeek encrypted credential bridge runs on the local Windows host; transfer cached plans to the training server")
    result=subprocess.run(["powershell.exe","-NoProfile","-NonInteractive","-ExecutionPolicy","Bypass","-File",str(script)],
        input=json.dumps(payload,ensure_ascii=False),text=True,encoding="utf-8",capture_output=True,timeout=240)
    if result.returncode: raise RuntimeError("DeepSeek request failed: "+result.stderr.strip()[:300])
    return json.loads(result.stdout.lstrip("\ufeff"))


def prompt(analysis, candidate, style_brief=None):
    example=dict(index=0,start_beat=0,end_beat=16,motif_id="theme-a",materials=["ribbon","stars"],
        verbs=["draw","translate","split","gather"],role="build",layout="diagonal",density=.45,intensity=.4,
        scale=.55,hue=.6,anchor=[-.2,.1],variation=.35,contrast=.65)
    return """You are directing an abstract geometric music performance, not an FFT display.
Return ONLY one compact JSON object containing version, audio_sha256 and sections.
Do not repeat the supplied global vocabulary or include descriptions. Never output code,
URLs, shader source or markdown.
Do not use a genre-to-preset mapping. Shapes alone are not choreography: use entrances,
drawing, translations, grouping, separation, holds, build-ups, release and returning motifs.
Select a distinct primary and supporting visual role; avoid constant rotation or changing
every parameter all the time. Use observations and similar musical sections to organize
motif recurrence with variation. Do not invent instruments, vocals, emotions or exact
verse/chorus labels. There are no instrument transcriptions in the input.
Use exactly the supplied section indices and beat boundaries. At most four materials per
section. Different music should lead to different compositions and motion verbs.
Use normalized 0..1 density, intensity, scale, hue, variation, contrast; anchor x/y in -1..1.
Motif IDs are short reusable strings. Explain visual intention in optional description.
The plan controls macro composition. A separate audio-conditioned model will generate
the fine action timing and trajectories.
"""+json.dumps(dict(version="choreography-plan-v1",audio_sha256=analysis["audio_sha256"],
        action_version=ACTION_VERSION,schema_section=example,materials=MATERIALS,verbs=VERBS,roles=ROLES,layouts=LAYOUTS,
        candidate=candidate,style_brief=style_brief,analysis=analysis),ensure_ascii=False,separators=(",",":"))


def create_plan(analysis, output, provider="deepseek", candidate=0, cache="artifacts/llm/plans",
                ledger="artifacts/llm/token-ledger.sqlite", request=None, model="deepseek-v4-pro", max_output_tokens=16384, style_brief=None):
    if provider!="deepseek": raise ValueError("Unsupported planning provider")
    text=prompt(analysis,candidate,style_brief)
    digest=hashlib.sha256((PROMPT_VERSION+model+text).encode()).hexdigest()
    cached=Path(cache)/(digest+".json")
    budget=Budget(ledger)
    if cached.exists():
        result=validate_plan(json.loads(cached.read_text(encoding="utf-8")),analysis)
    else:
        failure=""
        for attempt in range(3):
            task=text+("\nCorrect the previous response. Validation error: "+failure if failure else "")
            # One UTF-8 byte per token is conservative for this JSON/text prompt.
            reservation=budget.reserve(len(task.encode("utf-8"))+max_output_tokens+4096)
            try:
                response=(request or bridge_request)(dict(model=model,input=task,instructions="Return the requested JSON plan only.",
                    reasoning={"effort":"high"},max_output_tokens=max_output_tokens,stream=False))
                budget.finish(reservation,response.get("usage"),response.get("id"))
                if response.get("status") in ("incomplete","failed"):
                    raise ValueError("Incomplete API response")
                answer=response.get("text",response.get("output_text",""))
                if not answer: raise ValueError("Empty API response")
                answer=answer.strip()
                if answer.startswith("```"): answer="\n".join(answer.splitlines()[1:-1])
                result=validate_plan(json.loads(answer),analysis)
                result.update(source="deepseek",model=model,resolved_model=response.get("model"),prompt_version=PROMPT_VERSION,cache_key=digest,
                    request_id=response.get("id"),usage=response.get("usage"),candidate=candidate,style_brief=style_brief)
                cached.parent.mkdir(parents=True,exist_ok=True)
                cached.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
                break
            except (RuntimeError,ValueError,KeyError,subprocess.TimeoutExpired) as exc:
                failure=str(exc)[:400]
        else: raise RuntimeError("DeepSeek plan failed after three attempts: "+failure)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return dict(output=str(output),cache_key=digest,budget=budget.summary())
