"""
Frame-accurate "stretch the video to fit the dubbed audio" mode.

Instead of speeding up the dubbed speech, every segment of the video whose
dubbed audio is longer than its original slot is slowed down (setpts), so the
dub plays at natural speed and stays in sync with the picture.

Accuracy rules used here:
  * The whole timeline is computed in integer FRAMES of a constant frame rate
    (CFR), so per-clip rounding errors can never accumulate.
  * Every rendered clip has exactly the number of frames it is supposed to
    have (-frames:v), so the concatenated video has exactly the planned length.
  * The background (original) audio is retimed with the same piece boundaries
    and padded/trimmed to the exact length of each piece.
  * Subtitles / dub placement use the same time map (original time -> new time).
"""
import os
import re
import json
import shutil
import logging
import subprocess
from fractions import Fraction

logger = logging.getLogger(__name__)

VIDEO_PIECES_DIR = "stretch_tmp"


def _run(cmd):
    logger.debug(" ".join(cmd))
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="ignore")[-2000:])
    return proc.stdout.decode(errors="ignore")


def probe_video(path):
    """fps (as Fraction), duration (s) and frame count of the first video."""
    out = _run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration",
        "-of", "json", path,
    ])
    info = json.loads(out)
    stream = info["streams"][0]

    def to_frac(value):
        try:
            frac = Fraction(value)
            return frac if frac > 0 else None
        except (ZeroDivisionError, ValueError):
            return None

    fps = to_frac(stream.get("avg_frame_rate", "0/1")) or to_frac(
        stream.get("r_frame_rate", "0/1")
    ) or Fraction(25, 1)

    duration = None
    for candidate in (stream.get("duration"), info["format"].get("duration")):
        try:
            duration = float(candidate)
            break
        except (TypeError, ValueError):
            continue
    if duration is None:
        raise RuntimeError("Could not read the video duration")

    return fps, duration


class TimeMap:
    """Piecewise-linear map from the original timeline to the new one."""

    def __init__(self, pieces, fps):
        self.pieces = pieces  # list of dicts, see build_pieces
        self.fps = fps

    @property
    def total_frames(self):
        return sum(p["new_frames"] for p in self.pieces)

    @property
    def total_seconds(self):
        return float(self.total_frames / self.fps)

    def segment_new_starts(self, n_segments, segments=None):
        """Exact new start (s) of each segment piece, by segment index."""
        starts, base = {}, 0
        for p in self.pieces:
            if p["kind"] == "seg":
                starts[p["seg_index"]] = float(base / self.fps)
            base += p["new_frames"]
        result = []
        for i in range(n_segments):
            if i in starts:
                result.append(starts[i])
            elif segments is not None:  # segment dropped from the plan
                result.append(self(segments[i]["start"]))
            else:
                result.append(0.0)
        return result

    def __call__(self, t):
        """Original time (s) -> new time (s)."""
        f = float(t) * float(self.fps)
        new_frame_base = 0
        for p in self.pieces:
            if f <= p["src_end"] or p is self.pieces[-1]:
                span = p["src_end"] - p["src_start"]
                ratio = 0.0 if span <= 0 else (f - p["src_start"]) / span
                ratio = min(max(ratio, 0.0), 1.0)
                return float(
                    (new_frame_base + ratio * p["new_frames"]) / self.fps
                )
            new_frame_base += p["new_frames"]
        return float(new_frame_base / self.fps)


def build_pieces(
    segments,
    tts_durations,
    video_duration,
    fps,
    max_video_stretch=2.0,
    min_gap_seconds=0.0,
    slack_seconds=0.04,
):
    """
    Split the video into pieces [gap, segment, gap, segment, ..., tail].

    Each segment piece gets new_frames = ceil((tts + slack) * fps) if that is
    longer than its original frames (never shorter: the video is only
    extended). The stretch factor is capped by max_video_stretch; whatever the
    cap leaves over is reported in `audio_speedup` so the caller can speed up
    only that remainder in the audio, keeping the sync exact.
    """
    fps_f = float(fps)
    total_src_frames = int(round(video_duration * fps_f))
    pieces = []
    cursor = 0  # source frame cursor

    def frame_of(t):
        return int(round(float(t) * fps_f))

    ordered = sorted(range(len(segments)), key=lambda i: segments[i]["start"])
    prev_end_frame = 0
    for idx in ordered:
        seg = segments[idx]
        s = max(frame_of(seg["start"]), prev_end_frame)
        e = max(frame_of(seg["end"]), s + 1)
        e = min(e, total_src_frames)
        s = min(s, e - 1) if e > 0 else 0
        if e <= s:
            continue

        if s > cursor:
            pieces.append({
                "kind": "gap", "src_start": cursor, "src_end": s,
                "new_frames": s - cursor, "seg_index": None,
                "audio_speedup": 1.0,
            })

        src_frames = e - s
        tts = float(tts_durations[idx])
        if tts <= src_frames / fps_f:
            tts_eff = 0.0  # fits in the original slot: video untouched
            new_frames = src_frames
        else:
            tts_eff = tts + slack_seconds
            new_frames = max(src_frames, int(-(-(tts_eff * fps_f) // 1)))
        cap_frames = int(-(-(src_frames * max_video_stretch) // 1))
        if new_frames > cap_frames:
            new_frames = cap_frames
        # audio must fit inside new_frames exactly
        slot_seconds = new_frames / fps_f
        audio_speedup = max(1.0, tts_eff / slot_seconds)

        pieces.append({
            "kind": "seg", "src_start": s, "src_end": e,
            "new_frames": new_frames, "seg_index": idx,
            "audio_speedup": audio_speedup,
        })
        cursor = e
        prev_end_frame = e

    if cursor < total_src_frames:
        pieces.append({
            "kind": "gap", "src_start": cursor, "src_end": total_src_frames,
            "new_frames": total_src_frames - cursor, "seg_index": None,
            "audio_speedup": 1.0,
        })
    return TimeMap(pieces, fps)


def _fps_str(fps):
    return f"{fps.numerator}/{fps.denominator}"


def render_video(video_in, video_out, tmap, crf=18, preset="medium"):
    """
    Render every piece with an exact frame count, then concat.
    Source frames [src_start, src_end) -> exactly new_frames output frames.
    """
    fps = tmap.fps
    work = VIDEO_PIECES_DIR
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)

    list_path = os.path.join(work, "list.txt")
    # integer ticks per frame -> no sub-tick drift when clips are concatenated
    timescale = fps.numerator * max(1, -(-90000 // fps.numerator))
    with open(list_path, "w") as list_file:
        for n, p in enumerate(tmap.pieces):
            clip = os.path.join(work, f"v_{n:05d}.mp4")
            src_frames = p["src_end"] - p["src_start"]
            new_frames = p["new_frames"]
            factor = new_frames / src_frames
            vf = (
                f"trim=start_frame=0:end_frame={src_frames},"
                f"setpts=(PTS-STARTPTS)*{factor:.12f},"
                f"fps={_fps_str(fps)}:round=down,"
                # ffmpeg's fps filter can end 1-2 frames short on stretched
                # clips (last frame's duration isn't scaled): clone the last
                # frame, -frames:v then cuts at the exact planned count.
                f"tpad=stop_mode=clone:stop=-1"
            )
            cmd = ["ffmpeg", "-y", "-loglevel", "error"]
            if p["src_start"] > 0:
                # seek half a frame early: the first kept frame is exactly
                # source frame `src_start` whatever the container rounding
                cmd += ["-ss", f"{(p['src_start'] - 0.5) / float(fps):.6f}"]
            cmd += [
                "-i", video_in, "-an", "-vf", vf,
                "-frames:v", str(new_frames),
                "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                "-pix_fmt", "yuv420p", "-r", _fps_str(fps),
                "-video_track_timescale", str(timescale), clip,
            ]
            _run(cmd)
            list_file.write(f"file '{os.path.abspath(clip)}'\n")

    _run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
        "-i", list_path, "-c", "copy", video_out,
    ])
    shutil.rmtree(work, ignore_errors=True)
    return video_out


def render_background_audio(audio_in, audio_out, tmap, sample_rate=44100):
    """
    Retime the original/background audio with the same pieces. Each piece is
    slowed with atempo (pitch preserved) and padded/trimmed to the EXACT
    length of the corresponding video piece.
    """
    fps = float(tmap.fps)
    work = VIDEO_PIECES_DIR + "_aud"
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    list_path = os.path.join(work, "list.txt")

    with open(list_path, "w") as list_file:
        for n, p in enumerate(tmap.pieces):
            clip = os.path.join(work, f"a_{n:05d}.wav")
            t0 = p["src_start"] / fps
            t1 = p["src_end"] / fps
            new_len = p["new_frames"] / fps
            src_len = t1 - t0
            tempo = src_len / new_len  # <= 1.0 -> slower
            chain = "asetpts=PTS-STARTPTS"
            if abs(tempo - 1.0) > 1e-6:
                # atempo only accepts 0.5..100 per instance: chain if needed
                remaining = tempo
                while remaining < 0.5:
                    chain += ",atempo=0.5"
                    remaining /= 0.5
                chain += f",atempo={remaining:.8f}"
            chain += (
                f",apad=whole_dur={new_len:.6f},"
                f"atrim=0:{new_len:.6f}"
            )
            _run([
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", f"{t0:.6f}", "-t", f"{src_len:.6f}", "-i", audio_in,
                "-af", chain, "-ar", str(sample_rate), "-ac", "2",
                "-c:a", "pcm_s16le", clip,
            ])
            list_file.write(f"file '{os.path.abspath(clip)}'\n")

    _run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
        "-i", list_path, "-c", "copy", audio_out,
    ])
    shutil.rmtree(work, ignore_errors=True)
    return audio_out


def speedup_to_fit(audio_in, audio_out, speedup):
    """Speed up a TTS clip by the exact leftover factor (>1.0)."""
    if speedup <= 1.0 + 1e-6:
        if audio_in != audio_out:
            shutil.copy2(audio_in, audio_out)
        return audio_out
    # atempo accepts 0.5..100 in modern ffmpeg; chain for old builds
    factors, remaining = [], speedup
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    factors.append(remaining)
    chain = ",".join(f"atempo={f:.8f}" for f in factors)
    _run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", audio_in,
        "-filter:a", chain, audio_out,
    ])
    return audio_out


_SRT_TIME = re.compile(
    r"(?:(\d+):)?(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*"
    r"(?:(\d+):)?(\d{2}):(\d{2})[,.](\d{3})"
)


def _to_s(h, m, s, ms):
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _fmt(t, comma=True):
    t = max(0.0, t)
    ms_total = int(round(t * 1000))
    h, rem = divmod(ms_total, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    sep = "," if comma else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def remap_srt_text(srt_text, tmap):
    """Rewrite every cue time of an SRT/VTT text through the time map."""
    def repl(match):
        a = _to_s(*match.group(1, 2, 3, 4))
        b = _to_s(*match.group(5, 6, 7, 8))
        comma = "," in match.group(0)
        return f"{_fmt(tmap(a), comma)} --> {_fmt(tmap(b), comma)}"
    return _SRT_TIME.sub(repl, srt_text)


def remap_subtitle_file(path_in, path_out, tmap):
    with open(path_in, encoding="utf-8") as f:
        text = f.read()
    with open(path_out, "w", encoding="utf-8") as f:
        f.write(remap_srt_text(text, tmap))
    return path_out


def remap_subtitle_set(tmap, fmt, out_dir="subs_stretched"):
    """
    Remap sub_tra / sub_ori (srt always, plus `fmt` when it is vtt/ass) into
    out_dir, keeping the same file names. Returns out_dir.
    """
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    for name in ("sub_tra", "sub_ori"):
        src_srt = f"{name}.srt"
        if not os.path.exists(src_srt):
            continue
        dst_srt = os.path.join(out_dir, src_srt)
        remap_subtitle_file(src_srt, dst_srt, tmap)
        if fmt == "ass":
            _run([
                "ffmpeg", "-y", "-loglevel", "error", "-i", dst_srt,
                os.path.join(out_dir, f"{name}.ass"),
            ])
        elif fmt == "vtt" and os.path.exists(f"{name}.vtt"):
            remap_subtitle_file(
                f"{name}.vtt", os.path.join(out_dir, f"{name}.vtt"), tmap
            )
    return out_dir
