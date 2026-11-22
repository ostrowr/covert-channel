/* Matched to work/receiver_experiments/ook_text125.py::encode_packet. */
"use strict";

export const SIGNAL_CHECK_CHIPS = "0000000011111111000011110000111101010101010101011111000000001111";
export const CHIP_MS = 125;
export const LEAD_MS = 750;
export const MAX_MESSAGE_BYTES_125 = 3;
export const MAX_MESSAGE_BYTES_64 = 13;
export const MAX_MESSAGE_BYTES_32 = 34;
const PREAMBLE = SIGNAL_CHECK_CHIPS.slice(0, 32);
const DATA_POSITIONS = [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15];
const PARITY_POSITIONS = [1, 2, 4, 8];

function bitsOf(value, width) {
  let bits = "";
  for (let shift = width - 1; shift >= 0; shift--) bits += (value >>> shift) & 1;
  return bits;
}

function crc16Ccitt(bytes) {
  let crc = 0xffff;
  for (const byte of bytes) {
    crc ^= byte << 8;
    for (let bit = 0; bit < 8; bit++) {
      crc = crc & 0x8000 ? ((crc << 1) ^ 0x1021) & 0xffff : (crc << 1) & 0xffff;
    }
  }
  return crc;
}

function hamming1511(data) {
  if (!/^[01]{11}$/.test(data)) throw new Error("Hamming input must contain 11 bits");
  const word = Array(16).fill(0);
  DATA_POSITIONS.forEach((position, index) => { word[position] = Number(data[index]); });
  for (const parity of PARITY_POSITIONS) {
    let value = 0;
    for (let position = 1; position <= 15; position++) {
      if (position !== parity && (position & parity)) value ^= word[position];
    }
    word[parity] = value;
  }
  return word.slice(1).join("");
}

function diagonalInterleave(coded) {
  const words = coded.length / 15;
  if (!Number.isInteger(words) || words < 1) throw new Error("Invalid Hamming frame");
  const remaining = words - 1;
  let physical = coded.slice(0, 15);
  for (let column = 0; column < 15; column++) {
    for (let number = 0; number < remaining; number++) {
      const row = (number + column) % remaining;
      physical += coded[15 + 15 * row + column];
    }
  }
  return physical;
}

function encodeMessage(message, chipMs, maxBytes, interleaved = false) {
  const payload = new TextEncoder().encode(message);
  if (payload.length < 1 || payload.length > maxBytes) {
    throw new RangeError(`Message must contain 1–${maxBytes} UTF-8 bytes`);
  }
  const contents = new Uint8Array(payload.length + 1);
  contents[0] = payload.length;
  contents.set(payload, 1);
  const crc = crc16Ccitt(contents);
  let data = "";
  for (const byte of contents) data += bitsOf(byte, 8);
  data += bitsOf(crc, 16);
  data = data.padEnd(Math.ceil(data.length / 11) * 11, "0");
  let coded = "";
  for (let offset = 0; offset < data.length; offset += 11) {
    coded += hamming1511(data.slice(offset, offset + 11));
  }
  if (interleaved) coded = diagonalInterleave(coded);
  const chips = PREAMBLE + coded;
  if (LEAD_MS + chips.length * chipMs >= 15_000) throw new Error("Packet exceeds 15 seconds");
  return { chips, chipMs, leadMs: LEAD_MS, utf8Bytes: payload.length };
}

export function encodeText125(message) {
  return encodeMessage(message, CHIP_MS, MAX_MESSAGE_BYTES_125);
}

export function encodeText64(message) {
  return encodeMessage(message, 64, MAX_MESSAGE_BYTES_64);
}

export function encodeText32Interleaved(message) {
  return encodeMessage(message, 32, MAX_MESSAGE_BYTES_32, true);
}

export function encodeSignalCheck() {
  return { chips: SIGNAL_CHECK_CHIPS, chipMs: CHIP_MS, leadMs: LEAD_MS };
}
