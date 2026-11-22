# Covert Channels without Coordination

Private working repository for the interactive article and the Mac-to-SDRplay RF experiment. Article prose is intentionally left as placeholders in `site/index.html`.

## Site

`site/` is a buildless static page. It contains the measured V12 signal explorer and a WebAssembly memory-workload sender. Enter 1–34 UTF-8 bytes and click **Transmit repeatedly**. Four 16 MiB Web Workers send the same coded packet until **Stop** or the page is hidden. The browser mode uses 32 ms chips; it is separate from the faster native CLI below. Reception depends on device, antenna position, and SDR setup. Keep the page visible during a test.

The measured explorer includes the recovered photo and sender-state overlay. Do not hand it to a model taking a blind challenge; use the separate spoiler-free IQ ZIP instead.

## Native CLI (macOS)

```sh
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/covert-channel emit --text 'Hello over RF'
.venv/bin/covert-channel decode /path/to/capture.iq --profile fast6500
```

`emit` compiles the included C sender with `cc` on first use. It uses four memory workers and a 6,500-symbol/s Manchester-coded packet with interleaving, FEC, and CRC. The frame carries up to 1,020 UTF-8 bytes. `--rate 6250` selects the more conservative profile; use `--profile baseline6250` when decoding it. The transmitter uses no display or audio signal.

`decode` expects raw interleaved little-endian int16 I/Q sampled at 2 MS/s, with the SDR centered at 1.400 MHz. It discovers timing from the public preamble and prints text only if all four FEC blocks pass parity and CRC. The SDRplay capture step is currently performed by the existing headless SDRconnect workflow; this CLI consumes its saved IQ file. For a supervised send-and-capture experiment, `python -m covert_channel.fast_link --message-file message.txt --iq-out capture.iq` uses a running, selected SDRconnect headless receiver at `127.0.0.1:9876`. Keep receiver audio disabled and muted.

The native 6,500-symbol/s profile previously recovered 64/64 blocks in 16 independent captures using the two-harmonic receiver, but that finite result is not a reliability guarantee for another setup. The browser sender has a different protocol and a much lower rate.

## Source

`covert_channel/` contains the native sender, packet framing, interleaver, FEC, and IQ decoder. `site/` contains the browser transmitter and interactive capture viewer. Large raw recordings are intentionally outside Git; see the separately archived challenge ZIPs in the experiment workspace.
