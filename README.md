# Covert Channels without Coordination

An interactive article draft (`site/`) and a Mac-to-SDRplay RF experiment. The article text is left for the author.

## CLI

On macOS, build with `make` (requires Apple command-line tools and `uv`). The resulting command has no Python or virtual-environment setup at runtime:

```sh
make
./covert-channel encode --text 'Hello over RF'
./covert-channel decode capture.iq
```

`encode` also accepts `--text-file message.txt` and carries up to 1,020 UTF-8 bytes. `decode` reads raw interleaved int16 IQ sampled at 2 MS/s with the SDR centered at 1.400 MHz. Both commands default to the 6,500-symbol/s profile; use `--rate 6250` on both for the fallback profile. The encoder uses memory workloads, with no display or audio signaling. The decoder prints the message only after all four blocks pass FEC and CRC.

The browser demo has its own slower packet format. `experiments/v12/` contains the separate blind-reader file-transfer research source. Raw IQ recordings are kept outside Git.
