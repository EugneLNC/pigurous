"""RPE easing and error-bounded piecewise linear sampling."""
import math
import numpy as np


def _bounce(x):
    n, d = 7.5625, 2.75
    if x < 1 / d:
        return n * x * x
    if x < 2 / d:
        return n * (x - 1.5 / d) ** 2 + 0.75
    if x < 2.5 / d:
        return n * (x - 2.25 / d) ** 2 + 0.9375
    return n * (x - 2.625 / d) ** 2 + 0.984375


def ease(kind, x):
    x = min(1.0, max(0.0, x))
    if kind in (0, 1):
        return x
    if kind == 2:
        return math.sin(x * math.pi / 2)
    if kind == 3:
        return 1 - math.cos(x * math.pi / 2)
    if kind == 4:
        return 1 - (1 - x) ** 2
    if kind == 5:
        return x * x
    if kind == 6:
        return (1 - math.cos(math.pi * x)) / 2
    if kind == 7:
        return 2 * x * x if x < 0.5 else 1 - (-2 * x + 2) ** 2 / 2
    if kind == 8:
        return 1 - (1 - x) ** 3
    if kind == 9:
        return x**3
    if kind == 10:
        return 1 - (1 - x) ** 4
    if kind == 11:
        return x**4
    if kind == 12:
        return 4 * x**3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2
    if kind == 13:
        return 8 * x**4 if x < 0.5 else 1 - (-2 * x + 2) ** 4 / 2
    if kind == 14:
        return 1 - (1 - x) ** 5
    if kind == 15:
        return x**5
    if kind == 16:
        return 1 if x == 1 else 1 - 2 ** (-10 * x)
    if kind == 17:
        return 0 if x == 0 else 2 ** (10 * x - 10)
    if kind == 18:
        return math.sqrt(max(0, 1 - (x - 1) ** 2))
    if kind == 19:
        return 1 - math.sqrt(max(0, 1 - x * x))
    if kind == 20:
        return 1 + 2.70158 * (x - 1) ** 3 + 1.70158 * (x - 1) ** 2
    if kind == 21:
        return 2.70158 * x**3 - 1.70158 * x * x
    if kind == 22:
        return (
            (1 - math.sqrt(max(0, 1 - (2 * x) ** 2))) / 2
            if x < 0.5
            else (math.sqrt(max(0, 1 - (-2 * x + 2) ** 2)) + 1) / 2
        )
    if kind == 23:
        c = 1.70158 * 1.525
        return (
            (2 * x) ** 2 * ((c + 1) * 2 * x - c) / 2
            if x < 0.5
            else ((2 * x - 2) ** 2 * ((c + 1) * (2 * x - 2) + c) + 2) / 2
        )
    if kind == 24:
        return (
            x
            if x in (0, 1)
            else 2 ** (-10 * x) * math.sin((10 * x - 0.75) * 2 * math.pi / 3) + 1
        )
    if kind == 25:
        return (
            x
            if x in (0, 1)
            else -(2 ** (10 * x - 10)) * math.sin((10 * x - 10.75) * 2 * math.pi / 3)
        )
    if kind == 26:
        return _bounce(x)
    if kind == 27:
        return 1 - _bounce(1 - x)
    if kind == 28:
        return (1 - _bounce(1 - 2 * x)) / 2 if x < 0.5 else (1 + _bounce(2 * x - 1)) / 2
    if kind == 29:
        if x in (0, 1):
            return x
        s = math.sin((20 * x - 11.125) * 2 * math.pi / 4.5)
        return (
            -(2 ** (20 * x - 10) * s) / 2
            if x < 0.5
            else (2 ** (-20 * x + 10) * s) / 2 + 1
        )
    raise ValueError(f"Unsupported easingType {kind}")


def bezier_ease(x, points):
    x1, y1, x2, y2 = points
    if not (0 <= x1 <= 1 and 0 <= x2 <= 1):
        raise ValueError("Bezier x controls must lie in [0,1]")

    def curve(t, a, b):
        return 3 * (1 - t) ** 2 * t * a + 3 * (1 - t) * t * t * b + t**3

    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if curve(mid, x1, x2) < x:
            lo = mid
        else:
            hi = mid
    return curve((lo + hi) / 2, y1, y2)


def event_value(ev, beat, to_time=lambda x: x):
    a, b = ev["startTime"], ev["endTime"]
    t0, t1 = to_time(a), to_time(b)
    u = min(1.0, max(0.0, (to_time(beat) - t0) / (t1 - t0))) if b > a else 1.0
    l, r = ev.get("easingLeft", 0.0), ev.get("easingRight", 1.0)
    fun = (
        (lambda x: bezier_ease(x, ev["bezierPoints"]))
        if ev.get("bezier")
        else (lambda x: ease(ev.get("easingType", 1), x))
    )
    fl, fr = fun(l), fun(r)
    f = (fun(l + (r - l) * u) - fl) / (fr - fl) if abs(fr - fl) > 1e-12 else u
    return (
        np.asarray(ev["start"]) + (np.asarray(ev["end"]) - np.asarray(ev["start"])) * f
    )


def adaptive_curve(fun, start, end, tolerance, max_step=0.125, depth=20):
    """Check quarter/midpoints; max_step bounds unsampled spans in beat space.

    This is a sampled numerical bound, not a proof for arbitrary oscillatory input.
    A separate dense-grid reconstruction audit reports the actual sampled error.
    """
    out = [(start, float(fun(start)))]

    def split(a, b, va, vb, left):
        xs = [a + (b - a) * q for q in (0.25, 0.5, 0.75)]
        vs = [float(fun(x)) for x in xs]
        error = max(
            abs(v - (va + (vb - va) * q)) for v, q in zip(vs, (0.25, 0.5, 0.75))
        )
        if (error > tolerance or b - a > max_step) and left:
            split(a, xs[1], va, vs[1], left - 1)
            split(xs[1], b, vs[1], vb, left - 1)
        elif error > tolerance:
            raise ValueError("Curve cannot meet reconstruction tolerance")
        else:
            out.append((b, vb))

    if end > start:
        split(start, end, out[0][1], float(fun(end)), depth)
    return simplify_curve(out, tolerance)


def simplify_curve(out, tolerance):
    # RDP removal is checked against all sampled points, never deletes discontinuities.
    arr = np.asarray(out)
    keep = {0, len(arr) - 1}
    pending = [(0, len(arr) - 1)]
    while pending:
        lo, hi = pending.pop()
        if hi <= lo + 1:
            continue
        u = (arr[lo : hi + 1, 0] - arr[lo, 0]) / (arr[hi, 0] - arr[lo, 0])
        errors = np.abs(
            arr[lo : hi + 1, 1] - (arr[lo, 1] + u * (arr[hi, 1] - arr[lo, 1]))
        )
        idx = int(errors.argmax()) + lo
        if errors.max() > tolerance:
            keep.add(idx)
            pending.extend([(lo, idx), (idx, hi)])
    return [tuple(arr[i]) for i in sorted(keep)]
