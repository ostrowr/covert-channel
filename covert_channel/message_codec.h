/* Native encoder for the four-block text wire format used by blind_decode.py. */
#ifndef MESSAGE_CODEC_H
#define MESSAGE_CODEC_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "interleave_order.h"

enum { MESSAGE_CAPACITY = 1020, MESSAGE_BLOCKS = 4, DATA_BITS = 2048,
       SYSTEMATIC_BITS = 2064, PARITY_BITS = 496, CODE_BITS = 2560,
       MESSAGE_WIRE_BITS = MESSAGE_BLOCKS * CODE_BITS };

static const uint16_t ldpc_shifts[5][3] = {
    {89, 269, 480}, {143, 178, 404}, {42, 172, 311},
    {82, 92, 185}, {84, 87, 179}
};

static int valid_utf8(const uint8_t *s, size_t n) {
    for (size_t i = 0; i < n;) {
        uint8_t c = s[i];
        if (c < 0x80) { if (!c) return 0; ++i; continue; }
        size_t extra;
        uint32_t code;
        if (c >= 0xC2 && c <= 0xDF) { extra = 1; code = c & 0x1F; }
        else if (c >= 0xE0 && c <= 0xEF) { extra = 2; code = c & 0x0F; }
        else if (c >= 0xF0 && c <= 0xF4) { extra = 3; code = c & 0x07; }
        else return 0;
        if (i + extra >= n) return 0;
        for (size_t j = 1; j <= extra; ++j) {
            uint8_t next = s[i + j];
            if ((next & 0xC0) != 0x80) return 0;
            code = (code << 6) | (next & 0x3F);
        }
        if ((extra == 2 && code < 0x800) || (extra == 3 && code < 0x10000) ||
            (code >= 0xD800 && code <= 0xDFFF) || code > 0x10FFFF) return 0;
        i += extra + 1;
    }
    return 1;
}

static void encode_message(const uint8_t *message, size_t message_len,
                           uint8_t wire[MESSAGE_WIRE_BITS]) {
    uint8_t frame[1024] = {0};
    uint8_t coded[MESSAGE_BLOCKS][CODE_BITS];
    frame[0] = (uint8_t)(message_len >> 24);
    frame[1] = (uint8_t)(message_len >> 16);
    frame[2] = (uint8_t)(message_len >> 8);
    frame[3] = (uint8_t)message_len;
    memcpy(frame + 4, message, message_len);
    for (size_t block = 0; block < MESSAGE_BLOCKS; ++block) {
        uint8_t *word = coded[block];
        for (size_t bit = 0; bit < DATA_BITS; ++bit)
            word[bit] = (frame[block * 256 + bit / 8] >> (7 - bit % 8)) & 1;
        uint16_t crc = 0xFFFF;
        for (size_t bit = 0; bit < DATA_BITS; ++bit) {
            unsigned top = (crc >> 15) & 1;
            crc = (uint16_t)(crc << 1);
            if (top ^ word[bit]) crc ^= 0x1021;
        }
        for (size_t bit = 0; bit < 16; ++bit)
            word[DATA_BITS + bit] = (crc >> (15 - bit)) & 1;
        uint8_t previous = 0;
        for (size_t row = 0; row < PARITY_BITS; ++row) {
            uint8_t acc = previous;
            for (size_t group = 0; group < 5; ++group)
                for (size_t k = 0; k < 3; ++k) {
                    size_t index = group * PARITY_BITS +
                        (row + PARITY_BITS - ldpc_shifts[group][k]) % PARITY_BITS;
                    if (index < SYSTEMATIC_BITS) acc ^= word[index];
                }
            word[SYSTEMATIC_BITS + row] = acc;
            previous = acc;
        }
    }
    for (size_t col = 0; col < CODE_BITS; ++col)
        for (size_t block = 0; block < MESSAGE_BLOCKS; ++block)
            wire[col * MESSAGE_BLOCKS + block] = coded[block][interleave_order[col]];
}

#endif
