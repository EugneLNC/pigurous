import argparse, json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(
        prog="choreo", description="Music Choreographer research pipeline"
    )
    subs = p.add_subparsers(dest="command", required=True)
    an = subs.add_parser("analyze", help="Observed whole-song audio structure")
    an.add_argument("audio")
    an.add_argument("--features", required=True)
    an.add_argument("-o", "--output", required=True)
    pl = subs.add_parser("plan", help="Cached remote whole-song choreography plan")
    pl.add_argument("analysis")
    pl.add_argument("--provider", default="deepseek", choices=["deepseek"])
    pl.add_argument("--candidate", type=int, default=0)
    pl.add_argument("--style-brief", help="Optional visual direction; never a genre template")
    pl.add_argument("-o", "--output", required=True)
    st = subs.add_parser("student", help="Plan-conditioned compound action student")
    sts = st.add_subparsers(dest="student_command", required=True)
    sp = sts.add_parser("prepare")
    sp.add_argument("manifest")
    sp.add_argument("--plans", required=True)
    sp.add_argument("--reviews")
    sp.add_argument("-o", "--output", required=True)
    tr = sts.add_parser("train")
    tr.add_argument("manifest")
    tr.add_argument("-o", "--output", required=True)
    tr.add_argument("--initialize")
    tr.add_argument("--resume")
    tr.add_argument("--epochs", type=int, default=40)
    tr.add_argument("--batch-size", type=int, default=2)
    tr.add_argument("--accumulate", type=int, default=16)
    tr.add_argument("--patience", type=int, default=5)
    tr.add_argument("--review-boost", type=float, default=1.)
    tr.add_argument("--device", default="cpu")
    tr.add_argument("--seed", type=int, default=42)
    c = subs.add_parser("convert")
    c.add_argument("chart")
    c.add_argument("--audio", required=True)
    c.add_argument("-o", "--output", required=True)
    c.add_argument("--song-id", required=True)
    c.add_argument("--lines-only", action="store_true")
    f = subs.add_parser("features")
    f.add_argument("audio")
    f.add_argument("-o", "--output", required=True)
    f.add_argument("--backend", choices=["mert", "spectral"], default="mert")
    f.add_argument("--device", default="cpu")
    f.add_argument("--bpm", type=float)
    f.add_argument("--revision", default="main")
    d = subs.add_parser("prepare")
    d.add_argument("sources")
    d.add_argument("-o", "--output", required=True)
    d.add_argument("--backend", choices=["mert", "spectral"], default="mert")
    d.add_argument("--device", default="cpu")
    d.add_argument("--bpm", type=float)
    d.add_argument("--lines-only", action="store_true")
    t = subs.add_parser("train")
    t.add_argument("manifest")
    t.add_argument("-o", "--output", required=True)
    t.add_argument("--config", default="configs/research.json")
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--stage", type=int, choices=[1, 2, 3], default=1)
    t.add_argument("--batch-size", type=int, default=2)
    t.add_argument("--accumulate", type=int, default=16)
    t.add_argument("--device", default="cpu")
    t.add_argument("--resume")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--max-events", type=int, default=2048)
    t.add_argument("--patience", type=int, default=5)
    t.add_argument("--memory-limit-gb", type=float, default=30)
    g = subs.add_parser("generate")
    g.add_argument("audio")
    g.add_argument("--checkpoint")
    g.add_argument("--baseline", choices=["fixed"])
    g.add_argument("--features")
    g.add_argument("--planner", choices=["model", "llm"], default="model")
    g.add_argument("--plan")
    g.add_argument("--conditioning-features")
    g.add_argument("-o", "--output", required=True)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--device", default="cpu")
    g.add_argument("--bpm", type=float)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--max-events", type=int, default=2048)
    g.add_argument(
        "--condition", choices=["correct", "zero", "shuffle", "swap"], default="correct"
    )
    g.add_argument("--no-global", action="store_true")
    v = subs.add_parser(
        "polish", help="Compile an audio-directed presentation from a raw score"
    )
    v.add_argument("score")
    v.add_argument("--audio", required=True)
    v.add_argument("--features", required=True)
    v.add_argument("-o", "--output", required=True)
    v.add_argument("--seed", type=int, default=42)
    m = subs.add_parser(
        "math", help="Execute existing predicted controls with mathematical curves"
    )
    m.add_argument("score")
    m.add_argument("-o", "--output", required=True)
    r = subs.add_parser("render")
    r.add_argument("score")
    r.add_argument("-o", "--output", required=True)
    r.add_argument("--width", type=int, default=1920)
    r.add_argument("--height", type=int, default=1080)
    r.add_argument("--fps", type=float, default=60)
    r.add_argument("--audio")
    e = subs.add_parser("evaluate")
    e.add_argument("scores", nargs="+")
    e.add_argument("-o", "--output", required=True)
    e.add_argument("--votes")
    e.add_argument("--training-scores", nargs="*")
    e.add_argument(
        "--pairs", help="JSON list of song_id/full/local video pairs for blinded trials"
    )
    ac = subs.add_parser(
        "actions", help="Learned motion vocabulary and action director"
    )
    acts = ac.add_subparsers(dest="action", required=True)
    ap = acts.add_parser("prepare")
    ap.add_argument("sources", nargs="+")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--device", default="cuda")
    at = acts.add_parser("train")
    at.add_argument("manifest")
    at.add_argument("-o", "--output", required=True)
    at.add_argument("--epochs", type=int, default=40)
    at.add_argument("--device", default="cuda")
    at.add_argument("--batch-size", type=int, default=8)
    at.add_argument("--resume")
    at.add_argument("--initialize")
    ag = acts.add_parser("generate")
    ag.add_argument("--checkpoint", required=True)
    ag.add_argument("--features", required=True)
    ag.add_argument("--audio")
    ag.add_argument("-o", "--output", required=True)
    ag.add_argument("--device", default="cpu")
    ag.add_argument("--seed", type=int, default=42)
    ag.add_argument(
        "--condition", choices=["correct", "zero", "shuffle"], default="correct"
    )
    ae = acts.add_parser("evaluate")
    ae.add_argument("--checkpoint", required=True)
    ae.add_argument("--manifest", required=True)
    ae.add_argument("-o", "--output", required=True)
    ae.add_argument("--device", default="cpu")
    a = p.parse_args()
    args = vars(a)
    command = args.pop("command")
    if command == "math":
        from .math_presentation import mathematical_score

        print(json.dumps(mathematical_score(**args), ensure_ascii=False, indent=2))
        return
    if command == "analyze":
        from .music_analysis import analyze
        result = analyze(a.audio, a.features, a.output)
        print(json.dumps(dict(output=a.output, sections=len(result["sections"]))))
        return
    if command == "plan":
        from .llm_planning import create_plan
        print(json.dumps(create_plan(json.loads(Path(a.analysis).read_text(encoding="utf-8")), a.output, a.provider, a.candidate, style_brief=a.style_brief), ensure_ascii=False, indent=2))
        return
    if command == "student":
        operation = args.pop("student_command")
        if operation == "prepare":
            from .choreography_data import prepare_student as fn
        else:
            from .student_training import train as fn
        print(json.dumps(fn(**args), ensure_ascii=False, indent=2))
        return
    if command == "actions":
        action = args.pop("action")
        if action == "prepare":
            from .action_data import prepare as operation
        elif action == "train":
            from .action_training import train as operation
        else:
            from . import actions

            operation = getattr(actions, action)
        print(json.dumps(operation(**args), ensure_ascii=False, indent=2))
        return
    if command == "convert":
        from .audio import decode, sha256
        from .importer import convert_chart, reconstruction_audit

        score, native = convert_chart(
            a.chart, len(decode(a.audio)) / 24000, a.song_id, lines_only=a.lines_only
        )
        audit = reconstruction_audit(score, native)
        if not audit["passed"]:
            raise RuntimeError(json.dumps(audit))
        score.metadata.update(audit=audit, audio_sha256=sha256(a.audio))
        score.save(a.output)
        result = audit
    elif command == "features":
        from .audio import extract

        result = extract(a.audio, a.output, a.device, a.backend, a.bpm, a.revision)
    elif command == "prepare":
        from .dataset import prepare

        accepted, quarantine = prepare(
            a.sources, a.output, a.lines_only, a.backend, a.device, a.bpm
        )
        result = {"accepted": len(accepted), "quarantined": len(quarantine)}
    elif command == "train":
        from .training import train

        args["config"] = json.loads(Path(a.config).read_text())
        result = train(**args)
    elif command == "generate":
        from .audio import extract, Features, sha256
        from .generation import generate
        import torch

        cache = a.features or str(Path(a.output).with_suffix(".features.npz"))
        if not a.baseline and not a.checkpoint:
            p.error("generate requires --checkpoint or --baseline fixed")
        action_checkpoint = False
        checkpoint_meta = None
        student_checkpoint = False
        if a.checkpoint and not a.baseline:
            checkpoint_meta = torch.load(
                a.checkpoint, map_location="cpu", weights_only=False
            )
            action_checkpoint = checkpoint_meta.get("format") == "motion-actions-v1"
            student_checkpoint = checkpoint_meta.get("format") == "choreography-student-v1"
        if not Path(cache).exists():
            backend = "spectral"
            if not a.baseline:
                backend = (
                    "mert"
                    if action_checkpoint or student_checkpoint
                    else checkpoint_meta["provenance"]["feature_backend"]
                )
            if action_checkpoint or student_checkpoint:
                from .action_data import MERT_REVISION

                extract(a.audio, cache, a.device, backend, a.bpm, MERT_REVISION)
            else:
                extract(a.audio, cache, a.device, backend, a.bpm)
        if Features(cache).metadata["audio_sha256"] != sha256(a.audio):
            raise ValueError("Features do not belong to supplied audio")
        del checkpoint_meta
        if student_checkpoint:
            from .student_generation import generate as generate_student
            if a.no_global:
                p.error("Student --no-global is unavailable; compare model/LLM plan or audio conditions explicitly")
            result = generate_student(a.checkpoint, cache, a.output, a.planner, a.plan, a.audio, a.device, a.seed, a.condition, a.conditioning_features)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        if a.planner == "llm" or a.plan or a.condition == "swap" or a.conditioning_features:
            p.error("LLM plans require a choreography-student-v1 checkpoint")
        if action_checkpoint:
            from .actions import generate as generate_actions

            if a.no_global:
                p.error(
                    "--no-global is not implemented for the action checkpoint; use --condition zero/shuffle"
                )
            result = generate_actions(
                a.checkpoint, cache, a.output, a.device, a.seed, a.condition, a.audio
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        if a.baseline:
            from .baseline import fixed_mapping

            score = fixed_mapping(cache, a.output)
            print(
                json.dumps(
                    {
                        "baseline": "fixed",
                        "entities": len(score.entities),
                        "events": len(score.events),
                    }
                )
            )
            return
        score = generate(
            a.checkpoint,
            cache,
            a.output,
            a.seed,
            a.device,
            a.temperature,
            a.max_events,
            a.condition,
            a.no_global,
        )
        result = {
            "entities": len(score.entities),
            "events": len(score.events),
            "complete": score.metadata["complete_generation"],
        }
        if not result["complete"]:
            print(json.dumps(result))
            raise SystemExit(
                "Generation incomplete or empty; diagnostic SceneScore was saved. Do not use it as a successful full-song sample."
            )
    elif command == "polish":
        from .presentation import polish

        score = polish(**args)
        result = {
            "origin": score.metadata["origin"],
            "layers": len(score.entities),
            "onsets": len(score.metadata["presentation"]["hits"]),
            "output": a.output,
        }
    elif command == "render":
        from .render import render

        result = render(**args)
    else:
        from .evaluation import evaluate, blind_materials

        pairs = args.pop("pairs", None)
        if pairs:
            blind_materials(
                json.loads(Path(pairs).read_text(encoding="utf-8")),
                Path(a.output).parent / "blind",
            )

        result = evaluate(**args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
