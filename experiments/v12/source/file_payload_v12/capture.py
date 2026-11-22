"""Capture one frozen V12 binary-file packet using silent headless SDRplay.

The IQ and metadata go beside the frozen candidate in capture_public/.
Logs stay there in private_capture/.
Hardware starts only with the explicit --capture option. This script counts
chips but does not interpret or decode the packet. The default RF center is
near the strong band established by the prior physical survey, with that band
about 50 kHz off center to avoid the receiver's DC bin. This is a new direct
SDR recording, not a filtered derivative.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SENDER = ROOT.parent / "robust_sync_v11" / "fast_sender_raw"
HEADLESS = Path("/Applications/SDRconnect.app/Contents/MacOS/SDRconnect_headless")

CENTER_HZ = 1_455_100
SAMPLE_RATE_HZ = 2_000_000
BYTES_PER_COMPLEX_SAMPLE = 4  # Signed, little-endian int16 I followed by int16 Q.
CHIP_RATE = 6_500
WORKERS = 4
WORKER_MIB = 128
LEAD_MS = 1_000
PORT = 9876
RETUNE_SETTLE_SECONDS = 0.7
IQ_SETTLE_SECONDS = 0.5
POST_SENDER_SECONDS = 0.5
CAPTURE_MARGIN_SECONDS = 5.0


@dataclass(frozen=True)
class CaptureConfig:
    centers_hz: tuple[int, ...]
    log_dir: Path
    public_dir: Path
    packet_path: Path
    manifest_path: Path
    packet_sha256: str


def port_open() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", PORT), timeout=0.2):
            return True
    except OSError:
        return False


def range_name(index: int) -> str:
    return f"range_{index:02d}"


def private_paths(config: CaptureConfig) -> list[Path]:
    paths = [config.log_dir / "headless.log", config.log_dir / "capture_summary.json"]
    for index in range(1, len(config.centers_hz) + 1):
        name = range_name(index)
        paths.extend((config.log_dir / f"{name}.sender.log",
                      config.log_dir / f"{name}.sender.stdout",
                      config.log_dir / f".{name}.staging.iq"))
    return paths


def preflight(config: CaptureConfig) -> dict[str, int | float]:
    for executable in (HEADLESS, SENDER):
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise FileNotFoundError(f"required executable is unavailable: {executable}")
    if port_open():
        raise RuntimeError(f"port {PORT} is in use; will not attach to another SDRconnect")
    if not config.packet_path.is_file():
        raise FileNotFoundError(config.packet_path)
    frozen = json.loads(config.manifest_path.read_text())
    if (frozen.get("candidate_kind") != "off_air_binary_file_packet" or
            frozen.get("packet_sha256") != config.packet_sha256 or
            frozen.get("packet_file") != config.packet_path.name):
        raise ValueError("capture configuration and frozen manifest disagree")
    source = config.packet_path.parent / frozen["source_file"]
    truth = config.packet_path.parent / frozen["truth_file"]
    if source.stat().st_size != frozen["source_bytes"]:
        raise ValueError("frozen source byte count mismatch")
    if hashlib.sha256(source.read_bytes()).hexdigest() != frozen["source_sha256"]:
        raise ValueError("frozen source file SHA-256 mismatch")
    if hashlib.sha256(truth.read_bytes()).hexdigest() != frozen["truth_sha256"]:
        raise ValueError("frozen truth SHA-256 mismatch")
    if hashlib.sha256(config.packet_path.read_bytes()).hexdigest() != config.packet_sha256:
        raise ValueError("frozen packet SHA-256 mismatch")

    # Count only physical chips. No packet, coding, or message fields are read.
    chips = config.packet_path.read_text(encoding="ascii").strip()
    if not chips or any(ch not in "01" for ch in chips):
        raise ValueError("packet file must contain only raw binary chips")
    if len(chips) != frozen["packet_chips"]:
        raise ValueError("frozen packet chip count mismatch")
    air_seconds = len(chips) / CHIP_RATE
    if not 1.0 <= air_seconds <= 300.0 or abs(air_seconds - frozen["air_seconds"]) > 1e-6:
        raise ValueError(f"unexpected packet airtime {air_seconds:.3f}s")
    capture_seconds = math.ceil(air_seconds + LEAD_MS / 1_000 + CAPTURE_MARGIN_SECONDS)
    target_samples = capture_seconds * SAMPLE_RATE_HZ
    target_bytes = target_samples * BYTES_PER_COMPLEX_SAMPLE

    public_paths = [config.public_dir / "metadata.json"] + [
        config.public_dir / f"{range_name(index)}.iq"
        for index in range(1, len(config.centers_hz) + 1)
    ]
    for path in public_paths + private_paths(config):
        if path.exists():
            raise FileExistsError(f"capture output already exists: {path}")
    required_bytes = target_bytes * len(config.centers_hz) + 64 * 2**20
    available_bytes = shutil.disk_usage(ROOT).free
    if available_bytes < required_bytes:
        raise OSError(f"insufficient disk space: need {required_bytes:,} bytes, "
                      f"have {available_bytes:,} bytes free")

    return {
        "physical_chips": len(chips),
        "on_air_seconds": air_seconds,
        "capture_seconds_per_range": capture_seconds,
        "complex_samples_per_range": target_samples,
        "bytes_per_range": target_bytes,
        "total_iq_bytes": target_bytes * len(config.centers_hz),
    }


def wait_for_server(process: subprocess.Popen[bytes], config: CaptureConfig) -> None:
    log_path = config.log_dir / "headless.log"
    for _ in range(50):
        if process.poll() is not None:
            raise RuntimeError(f"SDRconnect exited {process.returncode}; see {log_path}")
        # A simultaneous process could take the port. Require evidence that our
        # process itself started its WebSocket server before connecting.
        if port_open() and "Websocket Server listening" in log_path.read_text(errors="replace"):
            return
        time.sleep(0.2)
    raise TimeoutError(f"SDRconnect did not open port {PORT}; see {log_path}")


async def initialize_device() -> None:
    import websockets

    async with websockets.connect(f"ws://127.0.0.1:{PORT}") as ws:
        found: dict[str, str] = {}
        for _ in range(30):
            for name in ("valid_devices", "active_device"):
                await ws.send(json.dumps({"event_type": "get_property", "property": name, "value": ""}))
            found = {}
            while len(found) < 2:
                message = await asyncio.wait_for(ws.recv(), timeout=3)
                if isinstance(message, bytes):
                    continue
                item = json.loads(message)
                if item.get("event_type") == "get_property_response" and item.get("property") in (
                    "valid_devices", "active_device"
                ):
                    found[item["property"]] = str(item.get("value", ""))
            if "RSP1B" in found["valid_devices"] and "RSP1B" in found["active_device"]:
                return
            await asyncio.sleep(0.5)
        raise RuntimeError(f"RSP1B not ready: {found}")


async def send(ws, event: str, prop: str = "", value: object = "") -> None:
    await ws.send(json.dumps({"event_type": event, "property": prop, "value": str(value)}))


async def discard(ws, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        try:
            await asyncio.wait_for(ws.recv(), max(0.001, end - time.monotonic()))
        except asyncio.TimeoutError:
            break


async def read_property(ws, name: str, timeout: float = 3.0) -> str:
    await send(ws, "get_property", name)
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        message = await asyncio.wait_for(
            ws.recv(), timeout=max(0.001, deadline - asyncio.get_running_loop().time()))
        if isinstance(message, bytes):
            continue
        item = json.loads(message)
        if (item.get("event_type") == "get_property_response" and
                item.get("property") == name):
            return str(item.get("value", ""))
    raise TimeoutError(f"SDRconnect did not return {name}")


async def assert_numeric_property(ws, name: str, expected: int) -> None:
    value = await read_property(ws, name)
    try:
        actual = float(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} returned nonnumeric readback {value!r}") from exc
    if not math.isfinite(actual) or abs(actual - expected) > 1.0:
        raise RuntimeError(f"{name} readback {value!r} differs from {expected}")


def sender_command(index: int, config: CaptureConfig) -> list[str]:
    return [
        str(SENDER), "--rate", str(CHIP_RATE), "--encoding", "raw",
        "--payload-file", str(config.packet_path),
        "--workers", str(WORKERS), "--buffer-mib", str(WORKER_MIB),
        "--lead-ms", str(LEAD_MS), "--mode", "park", "--clock", "busy",
        "--worker-qos", "default", "--clock-qos", "default",
        "--log", str(config.log_dir / f"{range_name(index)}.sender.log"),
    ]


def stop_sender(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def stop_headless(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


async def capture_one(ws, headless: subprocess.Popen[bytes], index: int,
                      center_hz: int, target_samples: int, air_seconds: float,
                      config: CaptureConfig) -> dict:
    name = range_name(index)
    stage_path = config.log_dir / f".{name}.staging.iq"
    stdout_path = config.log_dir / f"{name}.sender.stdout"
    await send(ws, "set_property", "device_center_frequency", center_hz)
    await discard(ws, RETUNE_SETTLE_SECONDS)
    await assert_numeric_property(ws, "device_center_frequency", center_hz)
    await send(ws, "set_property", "audio_mute", "true")
    await send(ws, "audio_stream_enable", value="false")
    await discard(ws, 0.2)
    mute_value = await read_property(ws, "audio_mute")
    if mute_value.lower() != "true":
        raise RuntimeError(f"headless audio mute reset during {name}: {mute_value!r}")
    if hashlib.sha256(config.packet_path.read_bytes()).hexdigest() != config.packet_sha256:
        raise ValueError("frozen packet changed before sender launch")
    await send(ws, "iq_stream_enable", value="true")
    try:
        await discard(ws, IQ_SETTLE_SECONDS)
        digest = hashlib.sha256()
        written_samples = 0
        frames = 0
        sender_done_samples: int | None = None
        deadline = time.monotonic() + air_seconds + LEAD_MS / 1_000 + 10
        with stdout_path.open("xb") as stdout, stage_path.open("xb") as iq_file:
            sender = subprocess.Popen(sender_command(index, config), stdout=stdout,
                                      stderr=subprocess.STDOUT)
            try:
                while written_samples < target_samples:
                    if headless.poll() is not None:
                        raise RuntimeError(f"SDRconnect exited during {name}; "
                                           f"see {config.log_dir / 'headless.log'}")
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"{name} exceeded its sender/capture deadline")
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=0.2)
                    except asyncio.TimeoutError:
                        message = None
                    if isinstance(message, bytes) and len(message) >= 2 and int.from_bytes(
                        message[:2], "little"
                    ) == 2:
                        payload = message[2:]
                        if len(payload) % BYTES_PER_COMPLEX_SAMPLE:
                            raise ValueError(f"{name} received a partial complex IQ sample")
                        if payload:
                            needed = (target_samples - written_samples) * BYTES_PER_COMPLEX_SAMPLE
                            chunk = payload[:needed]
                            iq_file.write(chunk)
                            digest.update(chunk)
                            written_samples += len(chunk) // BYTES_PER_COMPLEX_SAMPLE
                            frames += 1
                    if sender.poll() is not None and sender_done_samples is None:
                        sender_done_samples = written_samples
                        if sender.returncode:
                            raise RuntimeError(f"sender failed in {name} (exit {sender.returncode}); "
                                               f"see {stdout_path}")
                if sender_done_samples is None:
                    raise RuntimeError(f"{name} reached its sample budget before sender exited")
                if target_samples - sender_done_samples < POST_SENDER_SECONDS * SAMPLE_RATE_HZ:
                    raise RuntimeError(f"{name} has less than {POST_SENDER_SECONDS}s of post-sender IQ")
            finally:
                stop_sender(sender)
        expected_bytes = target_samples * BYTES_PER_COMPLEX_SAMPLE
        if stage_path.stat().st_size != expected_bytes:
            raise RuntimeError(f"{name} byte count is incomplete")
        return {"file": f"{name}.iq", "center_frequency_hz": center_hz,
                "sha256": digest.hexdigest(), "frames": frames}
    finally:
        await send(ws, "iq_stream_enable", value="false")
        await discard(ws, 0.2)


async def capture_all(headless: subprocess.Popen[bytes], plan: dict[str, int | float],
                      config: CaptureConfig) -> list[dict]:
    import websockets

    async with websockets.connect(f"ws://127.0.0.1:{PORT}", max_size=None,
                                  max_queue=None) as ws:
        await send(ws, "set_property", "audio_mute", "true")
        await send(ws, "audio_stream_enable", value="false")
        await send(ws, "device_stream_enable", value="true")
        try:
            await discard(ws, 0.8)
            await send(ws, "set_property", "device_sample_rate", SAMPLE_RATE_HZ)
            await send(ws, "set_property", "lna_state", 0)
            await discard(ws, 0.7)
            await assert_numeric_property(ws, "device_sample_rate", SAMPLE_RATE_HZ)
            # SDRconnect can reset audio properties when a device stream starts.
            # Set mute again after startup and verify it before any IQ or sender.
            await send(ws, "set_property", "audio_mute", "true")
            await send(ws, "audio_stream_enable", value="false")
            await discard(ws, 0.2)
            mute_value = await read_property(ws, "audio_mute")
            if mute_value.lower() != "true":
                raise RuntimeError(f"headless audio mute could not be verified: {mute_value!r}")
            results = []
            for index, center_hz in enumerate(config.centers_hz, start=1):
                result = await capture_one(ws, headless, index, center_hz,
                                           int(plan["complex_samples_per_range"]),
                                           float(plan["on_air_seconds"]), config)
                results.append(result)
                print(json.dumps({"captured": result["file"],
                                  "center_frequency_hz": center_hz,
                                  "complex_samples": plan["complex_samples_per_range"]}), flush=True)
            return results
        finally:
            await send(ws, "device_stream_enable", value="false")
            await discard(ws, 0.2)


def publish(plan: dict[str, int | float], results: list[dict],
            config: CaptureConfig) -> Path:
    if len(results) != len(config.centers_hz):
        raise RuntimeError("every requested range must complete before publication")
    config.public_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "iq_format": "interleaved signed 16-bit little-endian I,Q",
        "complex_samples_per_range": plan["complex_samples_per_range"],
        "capture_seconds_per_range": plan["capture_seconds_per_range"],
        "ranges": [{key: item[key] for key in ("file", "center_frequency_hz", "sha256")}
                   for item in results],
    }
    # Hard-link creation refuses existing destinations, unlike a POSIX rename.
    # The IQ stays in private until the capture and byte count pass.
    for item in results:
        name = Path(item["file"]).stem
        stage_path = config.log_dir / f".{name}.staging.iq"
        os.link(stage_path, config.public_dir / item["file"])
        stage_path.unlink()
    metadata_path = config.public_dir / "metadata.json"
    with metadata_path.open("x", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2)
        output.write("\n")
    with (config.log_dir / "capture_summary.json").open("x", encoding="utf-8") as output:
        json.dump({"physical_chips": plan["physical_chips"],
                   "on_air_seconds": plan["on_air_seconds"], "ranges": results},
                  output, indent=2)
        output.write("\n")
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-dir", type=Path,
                        default=ROOT / "private_candidate",
                        help="directory containing frozen packet.bits, source.bin, and manifest")
    parser.add_argument("--center-hz", type=int, default=CENTER_HZ,
                        help="direct SDR center frequency in Hz")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--capture", action="store_true",
                      help="start the sender and SDR after validating the capture plan")
    mode.add_argument("--dry-run", action="store_true",
                      help="print the capture plan without starting hardware (the default)")
    args = parser.parse_args()
    packet_dir = args.packet_dir.expanduser().resolve()
    manifest = packet_dir / "frozen_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"freeze this packet first: {manifest}")
    frozen = json.loads(manifest.read_text())
    config = CaptureConfig((args.center_hz,), packet_dir / "private_capture",
                           packet_dir / "capture_public",
                           packet_dir / "packet.bits", manifest,
                           frozen["packet_sha256"])
    plan = preflight(config)
    if not args.capture:
        print(json.dumps({"ready": True, **plan,
                          "centers_hz_in_capture_order": config.centers_hz,
                          "public_dir": str(config.public_dir)},
                         indent=2))
        return

    try:
        import websockets  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("capture requires websockets; run with uv --with websockets") from exc

    config.log_dir.mkdir(parents=True, exist_ok=True)
    with (config.log_dir / "headless.log").open("xb") as log:
        headless = subprocess.Popen([str(HEADLESS), f"--websocket_port={PORT}"],
                                    stdin=subprocess.PIPE, stdout=log,
                                    stderr=subprocess.STDOUT)
        try:
            wait_for_server(headless, config)
            asyncio.run(initialize_device())
            time.sleep(0.5)
            results = asyncio.run(capture_all(headless, plan, config))
        finally:
            stop_headless(headless)
    metadata_path = publish(plan, results, config)
    print(json.dumps({"metadata": str(metadata_path),
                      "range_files": [str(config.public_dir / item["file"]) for item in results],
                      "complex_samples_per_range": plan["complex_samples_per_range"]},
                     indent=2))


if __name__ == "__main__":
    main()
