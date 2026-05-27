#!/usr/bin/env python3
"""
validate_jumps.py — batch jump detection validator for Svihej

Usage:
    python3 validate_jumps.py [video_name_filter]

Processes all validation videos and reports detection accuracy.
Target: ≥9/10 videos within ±10% of expected count.
"""

import cv2
import mediapipe as mp
import sys, os, time, math
from collections import defaultdict

# MediaPipe landmark indices
LS, RS = 11, 12   # shoulders
LH, RH = 23, 24   # hips
LK, RK = 25, 26   # knees
LA, RA = 27, 28   # ankles

VIDEOS_DIR = ("/Users/martingren/Library/CloudStorage/"
              "GoogleDrive-gren.martin89@gmail.com/Shared drives/"
              "VÝVOJ PRODUKTŮ/Počítadlo/Validační videa")

VIDEOS = [
    ("snozmo_18skoku.mp4",          18),
    ("Snozmo_rychle_18skoku.mp4",   18),
    ("Cross_21crossu.mp4",          42),   # 21 crossovers + 21 transition jumps = 42 body jumps
    ("vysoka_kolena_25skoku.mp4",   25),
    ("snozmo_rychle_27skoku.mp4",   27),
    ("snozmo_34skoku.mp4",          34),
    ("Boxer step_50skoku .mp4",     50),
    ("stridacka_89skoku.mov",       89),
    ("mix_176skoku.MP4",           176),
    ("mix_237plus_skoku.mp4",      237),
]

# ──────────────────────────────────────────────────────────────────────────────

def _sign(x):
    if x > 0: return 1
    if x < 0: return -1
    return 0

# Tunable thresholds — edit here and re-run
PARAMS = dict(
    TIME_MIN       = 100,    # ms — min half-period
    TIME_MAX       = 650,    # ms — max half-period
    MIN_SCORE      = 0.25,   # mediapipe visibility threshold
    KHIPDRV_LO     = 20,     # KhipDrv lower bound (original 35, lowered for high-energy styles)
    KHIPDRV_HI     = 371,    # KhipDrv upper bound
    KHIPDIFF_LO    = 0.3,    # Khipdiff lower bound
    KHIPDIFF_HI    = 2.1,    # Khipdiff upper bound (original 1.6, raised for asymmetric/high-knee)
    KRATIO_LO      = 0.5,    # tUp/tDown lower bound
    KRATIO_HI      = 1.6,    # tUp/tDown upper bound
    LEG_STRAIGHT   = 20,     # leg straightness minimum (original 45, lowered for high-knee style)
    BODY_MIN       = 0.05,   # hip-to-shoulder min (body too small)
    BODY_MAX       = 0.28,   # hip-to-shoulder max (body too big)
)

class JumpDetector:
    def __init__(self, fps=30, params=None):
        p = params or PARAMS
        self.fps = fps
        # SMA window: same formula as original app
        self.win = max(1, round(5 * fps / 30 + 0.3))

        self.h = []   # historyMoves, newest at index 0
        self.d = []   # historyPeaksSteps

        self.hipUpDrvInt   = 0.0
        self.hipDownDrvInt = 0.0
        self.KhipdiffTopLimitSet = False

        self.jump_count   = 0
        self.reject_log   = []        # list of (frame_ms, reason, KhipDrv, Khipdiff, Kratio)
        self.rejects      = defaultdict(int)

        self.TIME_MIN    = p['TIME_MIN']
        self.TIME_MAX    = p['TIME_MAX']
        self.MIN_SCORE   = p['MIN_SCORE']
        self.DRV_LO      = p['KHIPDRV_LO']
        self.DRV_HI      = p['KHIPDRV_HI']
        self.DIFF_LO     = p['KHIPDIFF_LO']
        self.DIFF_HI     = p['KHIPDIFF_HI']
        self.RATIO_LO    = p['KRATIO_LO']
        self.RATIO_HI    = p['KRATIO_HI']
        self.LEG_MIN     = p['LEG_STRAIGHT']
        self.BODY_MIN    = p['BODY_MIN']
        self.BODY_MAX    = p['BODY_MAX']

    def _sum(self, key, n):
        end = min(len(self.h), n)
        return sum(self.h[i][key] for i in range(end))

    def detect(self, lm, frame_ms):
        """
        lm: dict keys ls,rs,lh,rh,lk,rk,la,ra → each {'x','y','vis'}
        Returns True on confirmed jump.
        """
        min_vis = min(lm['lh']['vis'], lm['rh']['vis'],
                      lm['ls']['vis'], lm['rs']['vis'])
        if min_vis < self.MIN_SCORE:
            return False

        lh = lm['lh'];  rh = lm['rh']
        ls = lm['ls'];  rs = lm['rs']
        la = lm['la'];  ra = lm['ra']
        lk = lm['lk'];  rk = lm['rk']

        win = self.win
        m = max(ra['y'], la['y']) - (ls['y'] + rs['y']) / 2

        # Leg asymmetry → widen Khipdiff limit
        dL = lh['y'] - lk['y'];  dR = rh['y'] - rk['y']
        if dL > 0 and dR > 0 and max(dL / max(dR, 1e-9), dR / max(dL, 1e-9)) > 1.1:
            self.KhipdiffTopLimitSet = 2.2

        entry = {
            'hipy': 0.0, 'dhipy': 0.0, 'dhipyavg': 0.0,
            'hiply': lh['y'], 'hipry': rh['y'],
            'shly':  ls['y'], 'shry':  rs['y'],
            'knly':  lk['y'], 'knry':  rk['y'],
            'anly':  la['y'], 'anry':  ra['y'],
            'time':  frame_ms,
        }
        self.h.insert(0, entry)
        self.h[0]['hipy'] = (lh['y'] + rh['y'] + ls['y'] + rs['y']) / 4

        self.h[0]['dhipy']    = self.h[0]['hipy'] - self.h[1]['hipy'] if len(self.h) > 1 else 0.0
        self.h[0]['dhipyavg'] = self._sum('dhipy', win) / win

        if len(self.h) >= 60:
            self.h.pop()

        for e in self.d:
            e['frame'] += 1

        cur  = self.h[0]['dhipyavg']
        prev = self.h[1]['dhipyavg'] if len(self.h) > 1 else 0.0
        jumped = False

        # ── SIGN DECREASE (+→-): push-off ──────────────────────────────────
        if _sign(cur) < _sign(prev):
            if self.d:
                pv = 0.0;  pf = 0;  self.hipDownDrvInt = 0.0
                for e in range(1, self.d[0]['frame'] + 1):
                    if e < len(self.h):
                        v = self.h[e]['dhipyavg']
                        if v > pv: pv = v; pf = e
                        self.hipDownDrvInt += v
                self.d.insert(0, {'type': 3, 'frame': pf, 'value': pv})
            self.d.insert(0, {'type': 0, 'frame': 0, 'time': frame_ms})

            t_up = t_down = 0.0
            if len(self.d) > 4:
                t_up   = self.d[2]['time'] - self.d[4]['time']
                t_down = self.d[0]['time'] - self.d[2]['time']

            K_ratio  = t_up / max(t_down, 1e-9)
            time_ok  = (t_up + t_down) > 2 * self.TIME_MIN
            time_lim = (t_up + t_down) < 2 * self.TIME_MAX

            K_hipdiff = abs(self.hipDownDrvInt / self.hipUpDrvInt) if self.hipUpDrvInt != 0 else 0.0
            K_dlim    = self.KhipdiffTopLimitSet if self.KhipdiffTopLimitSet else self.DIFF_HI

            t_sum = max(t_up + t_down, 1e-9)
            K_drv = (self.hipDownDrvInt - self.hipUpDrvInt) / (t_sum * 30 * max(m, 1e-6)) * 1e7

            ok_drv   = self.DRV_LO  < K_drv    < self.DRV_HI
            ok_diff  = self.DIFF_LO < K_hipdiff < K_dlim
            ok_ratio = self.RATIO_LO < K_ratio  < self.RATIO_HI

            if ok_drv and time_ok and time_lim and ok_diff and ok_ratio:
                self.KhipdiffTopLimitSet = False
                abort = False; reason = ''

                leg_l = (lk['y']-lh['y']) / max(la['y']-lh['y'], 1e-3) * m * 1e3
                leg_r = (rk['y']-rh['y']) / max(ra['y']-rh['y'], 1e-3) * m * 1e3
                if leg_l < self.LEG_MIN or leg_r < self.LEG_MIN:
                    abort = True; reason = f'legs_c({min(leg_l,leg_r):.0f})'

                if not abort and (lh['y']>lk['y'] or lh['y']>la['y'] or rh['y']>rk['y'] or rh['y']>ra['y']):
                    abort = True; reason = 'leg_above_hip'

                if not abort and (lh['y']-ls['y'] > self.BODY_MAX or rh['y']-rs['y'] > self.BODY_MAX):
                    abort = True; reason = 'body_big'

                if not abort and (lh['y']-ls['y'] < self.BODY_MIN or rh['y']-rs['y'] < self.BODY_MIN):
                    abort = True; reason = 'body_small'

                if not abort:
                    jumped = True
                    self.jump_count += 1
                else:
                    self.rejects[reason] += 1
                    self.reject_log.append((frame_ms, reason, K_drv, K_hipdiff, K_ratio))
            else:
                if   not ok_drv:   reason = f'KDrv={K_drv:.0f}'
                elif not time_ok:  reason = f'tShort={t_up+t_down:.0f}'
                elif not time_lim: reason = f'tLong={t_up+t_down:.0f}'
                elif not ok_diff:  reason = f'Kdiff={K_hipdiff:.3f}'
                else:              reason = f'Kratio={K_ratio:.2f}'
                self.rejects[reason[:15]] += 1
                self.reject_log.append((frame_ms, reason, K_drv, K_hipdiff, K_ratio))

        # ── SIGN INCREASE (-→+): apex ──────────────────────────────────────
        if _sign(cur) > _sign(prev):
            if self.d:
                pv = 0.0;  pf = 0;  self.hipUpDrvInt = 0.0
                for e in range(1, self.d[0]['frame'] + 1):
                    if e < len(self.h):
                        v = self.h[e]['dhipyavg']
                        if v < pv: pv = v; pf = e
                        self.hipUpDrvInt += v
                self.d.insert(0, {'type': 1, 'frame': pf, 'value': pv})
            self.d.insert(0, {'type': 2, 'frame': 0, 'time': frame_ms})

        # Trim
        if len(self.d) > 8:
            self.d.pop()

        return jumped


def process_video(path, expected):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, f"Cannot open {path}"

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  {w}x{h} @ {fps:.1f}fps  {total} frames  SMA_win={max(1,round(5*fps/30+0.3))}")

    det = JumpDetector(fps=fps)

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_idx = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_ms  = frame_idx / fps * 1000.0
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res       = pose.process(frame_rgb)

        if res.pose_landmarks:
            raw = res.pose_landmarks.landmark
            def kp(i): l = raw[i]; return {'x': l.x, 'y': l.y, 'vis': l.visibility}
            lm = {'ls': kp(LS), 'rs': kp(RS), 'lh': kp(LH), 'rh': kp(RH),
                  'lk': kp(LK), 'rk': kp(RK), 'la': kp(LA), 'ra': kp(RA)}
            det.detect(lm, frame_ms)

        frame_idx += 1
        if frame_idx % 200 == 0:
            pct = frame_idx / total * 100 if total else 0
            print(f"    {frame_idx}/{total} ({pct:.0f}%)  jumps={det.jump_count}  t={time.time()-t0:.0f}s",
                  flush=True)

    cap.release()
    pose.close()
    return det, None


def run(params=None, filter_name=None, verbose=False):
    if params:
        PARAMS.update(params)

    rows    = []
    n_total = 0
    n_pass  = 0

    for fname, expected in VIDEOS:
        if filter_name and filter_name.lower() not in fname.lower():
            continue

        path = os.path.join(VIDEOS_DIR, fname)
        if not os.path.exists(path):
            print(f"  SKIP: {fname}")
            continue

        print(f"\n▶ {fname}  (expected={expected})")
        det, err = process_video(path, expected)
        if err:
            print(f"  ERROR: {err}")
            continue

        count     = det.jump_count
        lo, hi    = expected * 0.9, expected * 1.1
        # For 237+ video, only lower bound matters since exact count unknown
        passed    = (lo <= count <= hi) if fname != "mix_237plus_skoku.mp4" else (count >= lo)
        err_pct   = (count - expected) / expected * 100 if expected else 0

        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  → {count} detected  {err_pct:+.1f}%  {status}")

        # Top reject reasons
        if det.rejects:
            top = sorted(det.rejects.items(), key=lambda x: -x[1])[:4]
            print(f"     rejects: {top}")

        n_total += 1
        if passed: n_pass += 1
        rows.append((fname, expected, count, err_pct, passed))

    # Summary
    print("\n" + "="*72)
    print(f"  {'Video':<35} {'Exp':>5} {'Det':>5} {'Err%':>7}  Pass")
    print("  " + "-"*68)
    for fname, exp, det_c, err, ok in rows:
        print(f"  {fname[:35]:<35} {exp:>5} {det_c:>5} {err:>+7.1f}%  {'✓' if ok else '✗'}")
    print("="*72)
    pct = 100 * n_pass / n_total if n_total else 0
    print(f"  CELKEM: {n_pass}/{n_total}  ({pct:.0f}%)   CÍL 9/10: {'DOSAŽEN ✓' if n_pass >= 9 else 'NEDOSAŽEN ✗'}")
    print("="*72)
    return rows, n_pass, n_total


if __name__ == '__main__':
    filt = sys.argv[1] if len(sys.argv) > 1 else None
    run(filter_name=filt)
