#!/usr/bin/env python3
"""Create a standalone interactive view of the measured V12 photo RF waveform."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
IQ = ROOT / "work/photo_v12/packet_attempt2/derived_public/range_01.iq"
TRUTH = ROOT / "work/photo_v12/packet_attempt2/truth.json"
PACKET = ROOT / "work/photo_v12/packet_attempt2/packet.bits"
REPORT = ROOT / "work/photo_v12/packet_attempt2/derived_oracle_report.json"
OUT = ROOT / "work/blog_v12/RF_V12_PHOTO_EXPLORER.html"
FS = 100_000


def segment(env: np.ndarray, start: float, duration: float, down: int) -> dict:
    first = round(start * FS)
    last = round((start + duration) * FS)
    values = env[first:last:down]
    if not len(values):
        raise ValueError("empty signal segment")
    low, high = np.percentile(values, [1, 99])
    if high <= low:
        high = low + 1
    scaled = np.rint(np.clip((values - low) / (high - low), 0, 1) * 255).astype(np.uint8)
    return {"start": round(start, 6), "sampleRate": FS // down,
            "values": scaled.tolist(), "lo": round(float(low), 2),
            "hi": round(float(high), 2)}


def main() -> None:
    truth = json.loads(TRUTH.read_text())
    report = json.loads(REPORT.read_text())
    if not report["full_payload_valid"]:
        raise ValueError("RF payload must validate before making figure")
    raw = np.memmap(IQ, dtype="<i2", mode="r").reshape(-1, 2)
    if len(raw) != 27_400_000:
        raise ValueError("unexpected selected IQ size")
    env = np.hypot(raw[:, 0].astype(np.float32), raw[:, 1].astype(np.float32))
    t0 = float(report["packet_start_seconds"])
    rate = float(report["effective_chip_rate_start_to_end"])
    offsets = truth["stage_offsets_chips"]
    at = lambda key: t0 + int(offsets[key]) / rate
    chips = PACKET.read_text().strip()
    views = {
        "morse": segment(env, at("morse"), 5.0, 100),
        "anchor": segment(env, at("fast_anchor") - 0.02, 1.04, 1),
        "sync": segment(env, at("body_data_0") - 0.025, 0.05, 1),
        "body": segment(env, at("body_data_8") + 0.015, 0.30, 1),
    }
    for view in views.values():
        first_chip = max(0, int((view["start"] - t0) * rate) - 3)
        final_time = view["start"] + len(view["values"]) / view["sampleRate"]
        last_chip = min(len(chips), int((final_time - t0) * rate) + 4)
        view["firstChip"] = first_chip
        view["chips"] = chips[first_chip:last_chip]
    overview = env.reshape(-1, 1000).mean(axis=1)
    low, high = np.percentile(overview, [2, 98])
    overview = np.rint(np.clip((overview - low) / (high - low), 0, 1) * 255).astype(np.uint8)
    stages = [
        ("Beacon", t0, at("morse_sync")),
        ("Morse cue", at("morse_sync"), at("first_handoff")),
        ("Anchor", at("first_handoff"), at("header_0_0")),
        ("Repeated header", at("header_0_0"), at("second_handoff")),
        ("Unique body", at("body_0"), at("end_guard")),
        ("End marker", at("end_guard"), t0 + len(chips) / rate),
    ]
    payload = {"overview": overview.tolist(), "stages": stages,
               "views": views, "packetStart": t0, "chipRate": rate,
               "captureSeconds": 274,
               "bodyBps": report["data_segment_useful_bps"],
               "wholeBps": report["end_to_end_useful_bps"]}
    html = TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(json.dumps({"file": str(OUT), "bytes": OUT.stat().st_size,
                      "stages": len(stages), "views": list(views)}))


TEMPLATE = r"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><link rel="icon" href="data:,">
<title>A photo through measured RF</title>
<style>
:root{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,sans-serif;background:#08111d;color:#e6f0f5}
*{box-sizing:border-box}body{margin:0;padding:30px;max-width:1240px;margin:auto}
h1{font-size:clamp(27px,4vw,46px);letter-spacing:-.04em;margin:0 0 8px}
p{color:#9db7c7;line-height:1.5;margin:7px 0 20px}.eyebrow{color:#47d9d0;font-size:12px;font-weight:800;letter-spacing:.17em;text-transform:uppercase}
.cards{display:flex;gap:12px;flex-wrap:wrap;margin:24px 0}.card{background:#102234;border:1px solid #244255;border-radius:12px;padding:13px 18px;min-width:145px}.card b{display:block;font-size:23px;color:#fff}.card small{color:#91adbd}
.panel{background:#0d2030;border:1px solid #294457;border-radius:16px;padding:19px;margin:16px 0;box-shadow:0 8px 30px #0003}
.row{display:flex;justify-content:space-between;gap:16px;align-items:baseline;flex-wrap:wrap}.row h2{font-size:18px;margin:0 0 12px}
canvas{width:100%;display:block;border-radius:10px;background:#071725;touch-action:none}.timeline{height:150px}.detail{height:300px}
.controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:14px 0 8px}select,button{background:#173448;color:#e9f8fa;border:1px solid #3b6678;border-radius:9px;padding:9px 12px;font:inherit}button{cursor:pointer}button:hover{background:#24536a}
input[type=range]{flex:1;min-width:200px;accent-color:#50d7cb}.mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;color:#acd8dc;font-size:13px}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;color:#abc2cd;margin:10px 0}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
.note{font-size:13px;color:#8aabba}.warn{color:#e7c58d}
</style>
<div class="eyebrow">Measured SDR IQ · near-field MacBook experiment</div>
<h1>A JPEG carried by measured RF</h1>
<p>This interactive view uses the selected-band waveform derived from a physical SDR IQ recording. Scrub from the slow discovery cue into the single-copy photo packet; zoom in on the block sync word and measured fast chips.</p>
<div class="cards"><div class="card"><b>274 s</b><small>physical recording</small></div><div class="card"><b>6,500/s</b><small>physical chips</small></div><div class="card"><b>35,532 B</b><small>JPEG file bytes</small></div><div class="card"><b id="bodyrate">—</b><small>body file bits/s</small></div><div class="card"><b id="wholerate">—</b><small>whole packet file bits/s</small></div></div>
<p class="note">The 35,532-byte JPEG is sent once over an 18-block body lasting about 184 seconds; the final transport block has 864 zero-padding bytes. The packet first spends about 81 seconds on discovery, a repeated fast anchor, and header framing. A 64-bit sync word marks each body block. The whole-packet rate includes the setup and ending marker. Rates count file bytes, not measured source entropy.</p>
<section class="panel"><div class="row"><h2>Full capture</h2><span class="mono">RF magnitude · 10 ms bins</span></div><canvas id="timeline" class="timeline"></canvas><div class="legend"><span><i class="dot" style="background:#41d8d0"></i>measured RF</span><span><i class="dot" style="background:#b978d5"></i>protocol stages</span></div></section>
<section class="panel"><div class="row"><h2>Measured waveform</h2><span id="time" class="mono"></span></div><div class="controls"><select id="view"><option value="morse">Slow Morse cue</option><option value="anchor">Repeated NXT1 anchor</option><option value="sync">Body sync word</option><option value="body" selected>Unique fast body</option></select><select id="window"><option value="0.005" selected>5 ms</option><option value="0.02">20 ms</option><option value="0.05">50 ms</option><option value="0.2">200 ms</option><option value="0.5">500 ms</option></select><button id="play">▶ Animate</button><input id="scrub" type="range" min="0" max="1000" value="0"></div><canvas id="detail" class="detail"></canvas><div class="legend"><span><i class="dot" style="background:#4fe0d0"></i>measured magnitude</span><span><i class="dot" style="background:#e8ac5a"></i>transmitted chip state (source-side overlay)</span></div><p class="note">The gold trace uses the sender's frozen chip file for comparison; the cyan trace is measured SDR data. This is an explanatory view, separate from the spoiler-free blind challenge.</p></section>
<script>
const data=__PAYLOAD__;
const tl=document.getElementById('timeline'),cv=document.getElementById('detail'),sel=document.getElementById('view'),win=document.getElementById('window'),scrub=document.getElementById('scrub'),clock=document.getElementById('time'),play=document.getElementById('play');
document.getElementById('bodyrate').textContent=Math.round(data.bodyBps).toLocaleString();document.getElementById('wholerate').textContent=Math.round(data.wholeBps).toLocaleString();
function fit(canvas){const dpr=window.devicePixelRatio||1;canvas.width=Math.round(canvas.clientWidth*dpr);canvas.height=Math.round(canvas.clientHeight*dpr);return canvas.getContext('2d')}
function line(ctx,color,width,points){ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.stroke()}
function timeline(){const c=fit(tl),w=tl.width,h=tl.height;c.clearRect(0,0,w,h);data.stages.forEach((s,i)=>{const x0=s[1]/data.captureSeconds*w,x1=s[2]/data.captureSeconds*w;c.fillStyle=i%2?'#55356844':'#6f4c8b33';c.fillRect(x0,0,x1-x0,h);if(x1-x0>80){c.fillStyle='#bda9cb';c.font=`${Math.max(10,h*.075)}px sans-serif`;c.fillText(s[0],x0+4,16)}});const a=data.overview,pts=[];for(let x=0;x<w;x++){const j=Math.min(a.length-1,Math.floor(x/w*a.length));pts.push([x,h-16-(a[j]/255)*(h-38)])}line(c,'#45d8d0',Math.max(1,w/1100),pts);c.fillStyle='#aac1ca';c.font=`${Math.max(10,h*.08)}px monospace`;const tick=data.captureSeconds>150?20:10;for(let sec=0;sec<data.captureSeconds;sec+=tick)c.fillText(`${sec}s`,sec/data.captureSeconds*w+3,h-3)}
function view(){return data.views[sel.value]}function windowSec(){return Number(win.value)}function currentStart(){const v=view(),dur=v.values.length/v.sampleRate,span=Math.max(0,dur-windowSec());return v.start+Number(scrub.value)/1000*span}
function detail(){const c=fit(cv),w=cv.width,h=cv.height;c.clearRect(0,0,w,h);const v=view(),start=currentStart(),span=windowSec(),first=Math.max(0,Math.floor((start-v.start)*v.sampleRate)),last=Math.min(v.values.length,Math.ceil((start+span-v.start)*v.sampleRate));const vals=v.values,pts=[];for(let x=0;x<w;x++){const j=Math.min(last-1,first+Math.floor(x/w*(last-first)));pts.push([x,h-28-vals[j]/255*(h-68)])}line(c,'#49dfd0',Math.max(1.2,w/900),pts);
const chipPoints=[];for(let x=0;x<w;x++){const t=start+x/w*span;const chip=Math.floor((t-data.packetStart)*data.chipRate);const idx=chip-v.firstChip;const state=idx>=0&&idx<v.chips.length&&v.chips[idx]==='1'?1:0;chipPoints.push([x,h-28-state*(h-68)])}line(c,'#e8ac5a99',Math.max(1,w/1500),chipPoints);c.strokeStyle='#426173';c.lineWidth=1;for(let t=Math.ceil(start*100)/100;t<start+span;t+=.01){let x=(t-start)/span*w;c.beginPath();c.moveTo(x,0);c.lineTo(x,h);c.stroke()}c.fillStyle='#b5cbd3';c.font=`${Math.max(11,h*.043)}px monospace`;c.fillText('0',4,h-6);c.fillText(`${Math.round(span*1000)} ms`,w-90,h-6);clock.textContent=`${start.toFixed(5)}–${(start+span).toFixed(5)} s · ${sel.options[sel.selectedIndex].text}`}
let playing=false,last=0;function tick(now){if(playing){if(last){const v=view(),dur=v.values.length/v.sampleRate,span=Math.max(.001,dur-windowSec());let pos=Number(scrub.value)/1000+(now-last)/1000*.10/span;if(pos>=1)pos=0;scrub.value=Math.round(pos*1000)}last=now;detail();requestAnimationFrame(tick)}}
play.onclick=()=>{playing=!playing;play.textContent=playing?'❚❚ Pause':'▶ Animate';last=0;if(playing)requestAnimationFrame(tick)};sel.onchange=()=>{scrub.value=0;win.value=sel.value==='morse'?'0.5':sel.value==='anchor'||sel.value==='sync'?'0.02':'0.005';detail()};win.onchange=()=>{scrub.value=0;detail()};scrub.oninput=detail;window.onresize=()=>{timeline();detail()};timeline();detail();
</script></html>"""


if __name__ == "__main__":
    main()
