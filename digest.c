
#include <stdint.h>
#include <string.h>
#include <immintrin.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>

#include "simdutf_c.h"

#define NOINLINE __attribute__((noinline))

typedef struct timespec Timespec;
static void clock_ns(Timespec* t) {
  clock_gettime(CLOCK_MONOTONIC, t);
}
static uint64_t elapsed_ns(Timespec start, Timespec stop) {
  return (uint64_t)(stop.tv_sec - start.tv_sec) * 1000000000LL + (uint64_t)(stop.tv_nsec - start.tv_nsec);
}

void printbits_32(uint32_t x) {
    for (int i = 0; i < 32; i++) {
        if ((x >> (31 - i)) & 1) {
            printf("1");
        } else {
            printf("0");
        }
    }
    printf("\n");
}

void printbits_64(uint64_t x) {
    for (int i = 0; i < 64; i++) {
        if ((x >> (63 - i)) & 1) {
            printf("1");
        } else {
            printf("0");
        }
        if ((i + 1) % 4 == 0 && (i + 1) % 8 != 0) {
            printf("_");
        }
        if ((i + 1) % 8 == 0) {
            printf(" ");
        }
    }
    printf("\n");
}

void printhex(const char *x, int n) {
    for (int i = 0; i < n; i++) {
        printf("%02x ", (uint8_t)x[i]);
    }
    printf("\n");
}

// -- this code derived from  https://stackoverflow.com/questions/21622212/how-to-perform-the-inverse-of-mm256-movemask-epi8-vpmovmskb
// my step by step
// from the original movemask, the lsb of mask is for lane 0, msb is for lane 31
//   lane 31|                                     |lane 0
// mask is  dddd_dddd cccc_cccc bbbb_bbbb aaaa_aaaa
// vmask gets
//       lane 3  | lane 2 | lane 1 | lane 0
// vmask: 0x03...| 0x02...| 0x01...| 0x00...
// vmask: D{8}   | C{8}   | B{8}   | A{8}  # replicate each byte 8 times
// in each lane, bit_mask picks out each bit, call the bits of A: ZYXW_VUTS
//           ZYXWVUTS_ZYXWVUTS_ZYXWVUTS_ZYXWVUTS_ZYXWVUTS_ZYXWVUTS_ZYXWVUTS_ZYXWVUTS
// bit_mask: 01111111_10111111_11011111_11101111_11110111_11111011_11111101_11111110
// bit_mask: Z....... .Y...... ..X..... ...W.... ....V... .....U.. ......T. .......S  # (. == 1 for readability)
// now cmp each byte against 0xff
// cmp leaves 0xff for match and 0x00 for no match. Will only match if that one bit was 1
__m256i expand_mask(const uint32_t mask) {
  __m256i vmask = _mm256_set1_epi32(mask);
  const __m256i shuffle = _mm256_setr_epi64x(0x0000000000000000, 0x0101010101010101, 0x0202020202020202, 0x0303030303030303);
  vmask = _mm256_shuffle_epi8(vmask, shuffle);
  const __m256i bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
  vmask = _mm256_or_si256(vmask, bit_mask);
  return _mm256_cmpeq_epi8(vmask, _mm256_set1_epi64x(-1));
}
// -- end SO code

// -- this code dervied from https://github.com/lemire/Code-used-on-Daniel-Lemire-s-blog/blob/master/2018/01/09/simdinterleave.c
typedef struct {
  uint32_t lo;
  uint32_t hi;
} uint32_2;

uint64_t interleave(uint32_t lo, uint32_t  hi) {
  return _pdep_u64(lo, 0x5555555555555555) |
         _pdep_u64(hi, 0xaaaaaaaaaaaaaaaa);
}

uint32_2 deinterleave(uint64_t input) {
  uint32_2 ret;
  ret.lo = _pext_u64(input, 0x5555555555555555);
  ret.hi = _pext_u64(input, 0xaaaaaaaaaaaaaaaa);
  return ret;
}
// -- end lemire code

// we can either store the 2 bits in little or big endian
void NOINLINE encode_40_be(const char x[32], char ret[40]) {
    uint64_t acc = 0;
    for (int i = 0; i < 32; i++) {
        acc <<= 2;
        if (x[i] == 0) {
            ret[i] = 0xfe;
            acc |= 2;
        } else if (x[i] == '/') {
            ret[i] = 0xfe;
            acc |= 3;
        } else {
            ret[i] = x[i];
            acc |= 1;
        }
    }
    memcpy(ret + 32, &acc, 8);
}

void NOINLINE decode_40_be(const char x[40], char ret[32]) {
    uint64_t acc;
    memcpy(&acc, x + 32, 8);
    for (int i = 31; i >= 0; i--) {
        int b = acc & 0b11;
        acc >>= 2;
        if (b == 2) {
            ret[i] = '\0';
        } else if (b == 3) {
            ret[i] = '/';
        } else if (b == 1) {
            ret[i] = x[i];
        } else {
            assert(false && "unexpected bits");
        }
    }
}

void NOINLINE encode_40_le(const char x[32], char ret[40]) {
    uint64_t acc = 0;
    for (int i = 0; i < 32; i++) {
        if (x[i] == 0) {
            ret[i] = 0xfe;
            acc |= ((uint64_t)2 << i * 2);
        } else if (x[i] == '/') {
            ret[i] = 0xfe;
            acc |= ((uint64_t)3 << i * 2);
        } else {
            ret[i] = x[i];
            acc |= ((uint64_t)1 << i * 2);
        }
    }
    memcpy(ret + 32, &acc, 8);
}

void NOINLINE decode_40_le(const char x[40], char ret[32]) {
    uint64_t acc;
    memcpy(&acc, x + 32, 8);
    for (int i = 0; i < 32; i++) {
        int b = (acc >> (i * 2)) & 0b11;
        if (b == 2) {
            ret[i] = '\0';
        } else if (b == 3) {
            ret[i] = '/';
        } else if (b == 1) {
            ret[i] = x[i];
        } else {
            assert(false && "unexpected bits");
        }
    }
}

// each position gets 2 bits in the mask
// HL - where H is the mask_hi bit and L is the mask_lo bit
// 00 - not used
// 01 - okay
// 10 - zero
// 11 - slash
// this matches encode_40_le's output
void NOINLINE encode_40_simd(const char x[32], char ret[40]) {
    __m256i y = _mm256_loadu_si256((__m256i*) x);
    __m256i is_zero = _mm256_cmpeq_epi8(y, _mm256_set1_epi8(0));
    __m256i is_slash = _mm256_cmpeq_epi8(y, _mm256_set1_epi8('/'));
    __m256i is_zero_or_slash = _mm256_or_si256(is_zero, is_slash);

    y = _mm256_blendv_epi8(y, _mm256_set1_epi8(0xfe), is_zero_or_slash);
    _mm256_storeu_si256((__m256i*) ret, y);

    uint32_t mask_zero = _mm256_movemask_epi8(is_zero);
    uint32_t mask_slash = _mm256_movemask_epi8(is_slash);
    uint32_t mask_hi = mask_zero | mask_slash;
    uint32_t mask_lo = ~mask_zero;
    uint64_t mask = interleave(mask_lo, mask_hi);

    memcpy(ret + 32, &mask, 8);
}

void NOINLINE decode_40_simd(const char x[40], char ret[32]) {
    uint64_t mask;
    memcpy(&mask, x + 32, 8);
    uint32_2 masks = deinterleave(mask);
    uint32_t mask_slash = masks.hi & masks.lo;
    uint32_t mask_zero = masks.hi & ~masks.lo;

    // TODO you can check the validity by checking that ~masks.hi & ~masks.lo is 0

    __m256i is_slash = expand_mask(mask_slash);
    __m256i is_zero = expand_mask(mask_zero);

    __m256i y = _mm256_loadu_si256((__m256i*) x);
    y = _mm256_blendv_epi8(y, _mm256_set1_epi8(0), is_zero);
    y = _mm256_blendv_epi8(y, _mm256_set1_epi8('/'), is_slash);
    _mm256_storeu_si256((__m256i*) ret, y);
}

// note that encode_37 and encode_37_simd use opposite endian for
// the bits. encode_37 stores them like
// low mem -> high mem
// encode_37:      1ddd_dddd 1ccc_cccd 1bbb_bbcc 1aaa_abbb 1xxx_aaaa
// encode_37_simd: 1aaa_aaaa 1bbb_bbba 1ccc_ccbb 1ddd_dccc 1xxx_dddd
// so to be compatible one of them needs a __builtin_bswap32 on bits
// I see no perf difference with the bswap
#define ENCODE_37_BSWAP

void NOINLINE encode_37(const char x[32], char ret[37]) {
    uint32_t bits = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t a;
        memcpy(&a, x + i * 8, 8);
        bits <<= 8;
        // grab the msb of each byte
        // 0x80 == 0b1000_0000
        bits |= _pext_u64(a, 0x8080808080808080);
        // set the msb of each byte
        a |= 0x8080808080808080;
        memcpy(ret + i * 8, &a, 8);
    }
#ifdef ENCODE_37_BSWAP
    bits = __builtin_bswap32(bits);
#endif
    // for 64 bit a, b, c, d
    // bits = aaaa_aaaa bbbb_bbbb cccc_cccc dddd_dddd
    // put 28 bits into the low 7 bits of each byte
    // 1aaa_abbb 1bbb_bbcc 1ccc_cccd 1ddd_dddd
    // 0x7f = 0b0111_1111
    uint32_t b = _pdep_u32(bits, 0x7f7f7f7f) | 0x80808080;
    // 1xxx_aaaa
    uint8_t c = bits >> 28 | 0x80;
    memcpy(ret + 32, &b, 4);
    memcpy(ret + 36, &c, 1);
}

void NOINLINE decode_37(const char x[37], char ret[32]) {
    uint32_t b;
    uint8_t c;
    memcpy(&b, x + 32, 4);
    memcpy(&c, x + 36, 1);
    uint32_t bits = _pext_u32(b, 0x7f7f7f7f) | (uint32_t)c << 28;
#ifdef ENCODE_37_BSWAP
    bits = __builtin_bswap32(bits);
#endif
    for (int i = 3; i >= 0; i--) {
        uint64_t a;
        memcpy(&a, x + i * 8, 8);
        // mask off msb of each byte
        a &= 0x7f7f7f7f7f7f7f7f;
        // spread the byte in b to the msb of each byte
        a |= _pdep_u64(bits & 0xff, 0x8080808080808080);
        bits >>= 8;
        memcpy(ret + i * 8, &a, 8);
    }
}

void NOINLINE encode_37_simd(const char x[32], char ret[37]) {
    __m256i y = _mm256_loadu_si256((__m256i*) x);
    // msb of each byte is same as checking for byte < 0
    // we can only use gt, so invert when we get the mask
    __m256i gt_zero = _mm256_cmpgt_epi8(y, _mm256_set1_epi8(-1));
    uint32_t bits = ~_mm256_movemask_epi8(gt_zero);

    y = _mm256_or_si256(y, _mm256_set1_epi8(0x80));
    _mm256_storeu_si256((__m256i*) ret, y);

    uint32_t b = _pdep_u32(bits, 0x7f7f7f7f);
    b |= 0x80808080;
    uint8_t c = bits >> 28 | 0x80;
    memcpy(ret + 32, &b, 4);
    memcpy(ret + 36, &c, 1);
}

void NOINLINE decode_37_simd(const char x[37], char ret[32]) {
    uint32_t b;
    uint8_t c;
    memcpy(&b, x + 32, 4);
    memcpy(&c, x + 36, 1);
    uint32_t bits = _pext_u32(b, 0x7f7f7f7f) | (uint32_t)c << 28;

    __m256i y = _mm256_loadu_si256((__m256i*) x);

    // expand_mask puts 0xff in each byte where the bit is 1
    // so mask off the lower 7 to just get the high bit
    __m256i msb = _mm256_and_si256(expand_mask(bits), _mm256_set1_epi8(0x80));

    y = _mm256_or_si256(
            /*_mm256_and_si256(y, _mm256_set1_epi8(0x7f)),*/
            _mm256_and_si256(y, _mm256_set1_epi64x(0x7f7f7f7f7f7f7f7f)),
            msb
            );
    _mm256_storeu_si256((__m256i*) ret, y);
}

void NOINLINE decode_37_simd_mullo(const char x[37], char ret[32]) {
    uint32_t b;
    uint8_t c;
    memcpy(&b, x + 32, 4);
    memcpy(&c, x + 36, 1);
    uint32_t bits = _pext_u32(b, 0x7f7f7f7f) | (uint32_t)c << 28;

    __m256i y = _mm256_loadu_si256((__m256i*) x);

    // alternatively to fixing up the expand_mask, I want
    //      ZYXWVUTS_ZYXWVUTS_ZYXWVUTS_ZYXWVUTS_ZYXWVUTS_ZYXWVUTS_ZYXWVUTS_ZYXWVUTS
    // msb: Z....... Y....... X....... W....... V....... U....... T....... S.......  # (. == 0 for readability)
    // but no slli_epi8 :(
    // the 16 bit number: ZYXWVUTS_ZYXWVUTS
    // to get             T......._S.......
    // we need (x << 7) | (x << 14) which is the same as mullo_epi16 with 2**7 + 2**14 == 0x4080
    // this is bits 0,1 to 7,15 == (7, 14) == 0x4080
    //              2,3 to 7,15 == (5, 12) == 0x1020
    //              4,5 to 7,15 == (3, 10) == 0x0408
    //              6,7 to 7,15 == (1, 8)  == 0x0102
    //              which we combine to 0x0102040810204080
    // setup is same as expand_mask
    __m256i vbits = _mm256_set1_epi32(bits);
    const __m256i shuffle = _mm256_setr_epi64x(0x0000000000000000, 0x0101010101010101, 0x0202020202020202, 0x0303030303030303);
    vbits = _mm256_shuffle_epi8(vbits, shuffle);
    const __m256i shift = _mm256_set1_epi64x(0x0102040810204080);
    __m256i msb = _mm256_and_si256(_mm256_mullo_epi16(vbits, shift), _mm256_set1_epi8(0x80));

    y = _mm256_or_si256(
            /*_mm256_and_si256(y, _mm256_set1_epi8(0x7f)),*/
            _mm256_and_si256(y, _mm256_set1_epi64x(0x7f7f7f7f7f7f7f7f)),
            msb
            );
    _mm256_storeu_si256((__m256i*) ret, y);
}

// this version grabs the msb of each 64 bit
// x = a b c d e f g h, i j k l m n o p, ...
// the first 64 bit value hgfedcba
// grab the msb h000_0000 g000_0000 f000_0000 e000_0000 d000_0000 c000_0000 b000_0000 a000_0000
// shift by 7           h         g         f         e         d         c         b         a  # puts to lsb
// next by 6           p         o         n         m         l         k         j         i
// etc. so that each low nibble gets the bits
// set the msb of these 8 bytes so that it can't be '\0' or '/'
// no simd or pdep/pext required
void NOINLINE encode_40_alt(const char x[32], char ret[40]) {
    uint64_t bits = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t a;
        memcpy(&a, x + i * 8, 8);
        uint64_t msb = a & 0x8080808080808080;
        bits |= msb >> (7 - i);
        a |= 0x8080808080808080;
        memcpy(ret + i * 8, &a, 8);
    }
    bits |= 0x8080808080808080;
    memcpy(ret + 32, &bits, 8);
}

void NOINLINE decode_40_alt(const char x[40], char ret[32]) {
    uint64_t bits;
    memcpy(&bits, x + 32, 8);
    for (int i = 0; i < 4; i++) {
        uint64_t a;
        memcpy(&a, x + i * 8, 8);
        uint64_t msb = (bits << (7 - i)) & 0x8080808080808080;
        a = (a & 0x7f7f7f7f7f7f7f7f) | msb;
        memcpy(ret + i * 8, &a, 8);
    }
}

void NOINLINE encode_40_alt_simd(const char x[32], char ret[40]) {
    __m256i y = _mm256_loadu_si256((__m256i*) x);

    const __m256i shifts = _mm256_set_epi64x(4, 5, 6, 7); // maybe use a cvtepu8_epi64?
                                                          //
    __m256i z = _mm256_srlv_epi64(_mm256_and_si256(y, _mm256_set1_epi64x(0x8080808080808080)), shifts);
    // or reduce the 4 u64
    // clang chooses to do 1 shuf then an extractf128 and or
    // 3 2   1 0
    // d c | b a
    // c d | a b
    z = _mm256_or_si256(z, _mm256_permute4x64_epi64(z, 0b10110001));
    // dc dc | ba ba
    // ba ba | dc dc
    z = _mm256_or_si256(z, _mm256_permute4x64_epi64(z, 0b00001010));

    y = _mm256_or_si256(y, _mm256_set1_epi64x(0x8080808080808080));
    _mm256_storeu_si256((__m256i*)ret, y);

    uint64_t bits = _mm256_extract_epi64(z, 0);
    bits |= 0x8080808080808080;
    memcpy(ret + 32, &bits, 8);
}

void NOINLINE decode_40_alt_simd(const char x[40], char ret[32]) {
    __m256i y = _mm256_loadu_si256((__m256i*) x);

    uint64_t bits;
    memcpy(&bits, x + 32, 8);
    __m256i vbits = _mm256_set1_epi64x(bits);

    const __m256i shifts = _mm256_set_epi64x(4, 5, 6, 7); // maybe use a cvtepu8_epi64?
    vbits = _mm256_sllv_epi64(vbits, shifts);
    vbits = _mm256_and_si256(vbits, _mm256_set1_epi64x(0x8080808080808080));

    y = _mm256_and_si256(y, _mm256_set1_epi64x(0x7f7f7f7f7f7f7f7f));
    y = _mm256_or_si256(y, vbits);

    _mm256_storeu_si256((__m256i*)ret, y);
}

// a utf-8 2 byte sequence is 110xxxxx 10xxxxxx
// this is 11 bits, so a 256 bit digest needs 24 pairs == 48 bytes
// a 3 byte sequence is 110xxxxx 10xxxxxx 10xxxxxx
// this is 17 bits, need 16 triples == 48 bytes
// a 3 byte sequence is 110xxxxx 10xxxxxx 10xxxxxx 10xxxxxx
// this is 23 bits, need 4 quads == 48 bytes
// they all use the same number of bytes
// the 17 bit sequence is appealing because if we only use 16 of those bits, we fit perfectly in 16 triples
// and 16 is a nicer number than any of the others
// does require pdep/pext
// boo I don't think this works because of overlong sequence, if a character can be encoded in less than the 3 bytes then it must be
void NOINLINE encode_48_utf8(const char x[32], char ret[48]) {
    // we transform 64 bits into 4 triples == 12 bytes
    //      0       1       2        3        4         5       6        7
    // 110xxxxx 10xxxxxx 10xxxxxx 110xxxxx 10xxxxxx 10xxxxxx 110xxxxx 10xxxxxx # this has 45 bits
    //      8       9       a        b
    // 10xxxxxx 110xxxxx 10xxxxxx 10xxxxxx  # this has 23 bits, only need 19 bits
    //
    // 0x3f gives the dep mask for 10xx_xxxx == 0011_1111
    // 0x1f gives the dep mask for 110x_xxxx == 0001_1111
    // 0x80 gives the bits for 10xx_xxxx
    // 0xc0 gives the bits for 110x_xxxx
    // written in reverse
    // in byte order:
    const uint64_t bits8 = 0x80c08080c08080c0;
    const uint32_t bits4 = 0x8080c080;
    const uint64_t dep_mask8 = 0x3f1f3f3f1f3f3f1f;
    const uint32_t dep_mask4 = 0x3f3f1f3f;
    assert(__builtin_popcountll(dep_mask8) == 45);
    assert(__builtin_popcountll(dep_mask4) == 23);
    for (int i = 0; i < 4; i++) {
        uint64_t a;
        memcpy(&a, x + i * 8, 8);
        uint64_t o = _pdep_u64(a, dep_mask8) | bits8;
        uint32_t p = _pdep_u32(a >> 45, dep_mask4) | bits4;
        memcpy(ret, &o, 8);
        memcpy(ret + 8, &p, 4);
        ret += 12;
    }
}

void NOINLINE decode_48_utf8(const char x[48], char ret[32]) {
    const uint64_t dep_mask8 = 0x3f1f3f3f1f3f3f1f;
    const uint32_t dep_mask4 = 0x3f3f1f3f;
    for (int i = 0; i < 4; i++) {
        uint64_t o, a;
        uint32_t p;
        memcpy(&o, x, 8);
        memcpy(&p, x + 8, 4);
        x += 12;
        a = _pext_u64(o, dep_mask8) | (uint64_t)_pext_u32(p, dep_mask4) << 45;
        memcpy(ret + i * 8, &a, 8);
    }
}

// -- following section of code adapted from https://github.com/zbjornson/fast-hex/blob/master/src/hex.cc under MIT
static const __attribute__((aligned(32))) char  HEX_LUTR[32] = {
  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'
};

inline static __m256i hex(__m256i value) {
    return _mm256_shuffle_epi8(_mm256_load_si256((__m256i*)HEX_LUTR), value);
}

static const char __attribute__((aligned(32))) ROT2[32] = {
  -1, 0, -1, 2, -1, 4, -1, 6, -1, 8, -1, 10, -1, 12, -1, 14,
  -1, 0, -1, 2, -1, 4, -1, 6, -1, 8, -1, 10, -1, 12, -1, 14
};

inline static __m256i byte2nib(__m128i val) {
    __m256i doubled = _mm256_cvtepu8_epi16(val);
    __m256i hi = _mm256_srli_epi16(doubled, 4);
    __m256i lo = _mm256_shuffle_epi8(doubled, _mm256_load_si256((__m256i*)ROT2));
    __m256i bytes = _mm256_or_si256(hi, lo);
    bytes = _mm256_and_si256(bytes, _mm256_set1_epi8(0b1111));
    return bytes;
}

// adapted from encodeHexVec
void NOINLINE encode_hex(const char x[32], char ret[65]) {
    const __m128i* input128 = (const __m128i*)x;
    __m256i* output256 = (__m256i*)ret;

  for (size_t i = 0; i < 2; i++) {
    __m128i av = _mm_lddqu_si128(&input128[i]);
    __m256i nibs = byte2nib(av);
    __m256i hexed = hex(nibs);
    _mm256_storeu_si256(&output256[i], hexed);
  }
  ret[64] = 0;
}

inline static __m256i unhexBitManip(const __m256i value) {
    __m256i and15 = _mm256_and_si256(value, _mm256_set1_epi16(15));
    __m256i sr6 = _mm256_srai_epi16(value, 6);
    __m256i mul = _mm256_maddubs_epi16(sr6, _mm256_set1_epi16(9)); // this has a latency of 5
    __m256i add = _mm256_add_epi16(mul, and15);
    return add;
}

inline static __m256i nib2byte(__m256i a1, __m256i b1, __m256i a2, __m256i b2) {
    __m256i a4_1 = _mm256_slli_epi16(a1, 4);
    __m256i a4_2 = _mm256_slli_epi16(a2, 4);
    __m256i a4orb_1 = _mm256_or_si256(a4_1, b1);
    __m256i a4orb_2 = _mm256_or_si256(a4_2, b2);
    __m256i pck1 = _mm256_packus_epi16(a4orb_1, a4orb_2); // lo1 lo2 hi1 hi2
    __m256i pck64 = _mm256_permute4x64_epi64(pck1, 0b11011000); // 0213
    return pck64;
}

void NOINLINE decode_hex(uint8_t* __restrict__ src, const uint8_t* __restrict__ dest) {
    const __m256i A_MASK = _mm256_setr_epi8(
    0, -1, 2, -1, 4, -1, 6, -1, 8, -1, 10, -1, 12, -1, 14, -1,
    0, -1, 2, -1, 4, -1, 6, -1, 8, -1, 10, -1, 12, -1, 14, -1);
    const __m256i B_MASK = _mm256_setr_epi8(
    1, -1, 3, -1, 5, -1, 7, -1, 9, -1, 11, -1, 13, -1, 15, -1,
    1, -1, 3, -1, 5, -1, 7, -1, 9, -1, 11, -1, 13, -1, 15, -1);

    const __m256i* val3 = (__m256i*)src;
    __m256i* dec256 = (__m256i*)dest;

    __m256i av1 = _mm256_loadu_si256(val3 + 0); // 32 nibbles, 16 bytes
    __m256i av2 = _mm256_loadu_si256(val3 + 1);
                                                // Separate high and low nibbles and extend into 16-bit elements
    __m256i a1 = _mm256_shuffle_epi8(av1, A_MASK);
    __m256i b1 = _mm256_shuffle_epi8(av1, B_MASK);
    __m256i a2 = _mm256_shuffle_epi8(av2, A_MASK);
    __m256i b2 = _mm256_shuffle_epi8(av2, B_MASK);

    // Convert ASCII values to nibbles
    a1 = unhexBitManip(a1);
    a2 = unhexBitManip(a2);
    b1 = unhexBitManip(b1);
    b2 = unhexBitManip(b2);

    // Nibbles to bytes
    __m256i bytes = nib2byte(a1, b1, a2, b2);

    _mm256_storeu_si256(dec256, bytes);
}
// -- end fast-hex code

int check(char x[32], char* y, size_t n, char z[32]) {
    for (size_t i = 0; i < n; i++) {
        if (y[i] == '\0') {
            printf("ERR, byte %ld is null\n", i);
            return 0;
        } else if (y[i] == '/') {
            printf("ERR, byte %ld is '/'\n", i);
            return 0;
        }
    }
    if (memcmp(x, z, 32) == 0) {
        return 1;
    } else {
        printf("ERR, roundtrip fail\n");
        printhex(x, 32);
        printhex(y, n);
        printhex(z, 32);
        return 0;
    }
}

int main(int argc, char **argv) {
    char x[32], xx[32];
    char y[48];
    char z[65];

    memset(x, 1, 32);
    x[0] = 0;
    x[5] = '\xf1';
    x[10] = 0;
    x[15] = '\xf3';
    x[20] = '/';
    x[31] = '/';

#ifdef TEST

    printf("input\n");
    printhex(x, 32);

#define CASE(n, encoder, decoder) \
    memset(y, 0, n); \
    memset(z, 0, 32); \
    printf(#encoder "\n"); \
    encoder(x, y); \
    decoder(y, z); \
    assert(check(x, y, n, z)); \
    printhex(y, n); \
    printhex(z, 32);

    CASE(40, encode_40_le, decode_40_le);
    CASE(40, encode_40_be, decode_40_be);
    CASE(40, encode_40_simd, decode_40_simd);
    CASE(40, encode_40_alt, decode_40_alt);
    CASE(40, encode_40_alt_simd, decode_40_alt_simd);
    // check interop
    CASE(40, encode_40_alt_simd, decode_40_alt);
    CASE(40, encode_40_alt, decode_40_alt_simd);

    CASE(37, encode_37, decode_37);
    CASE(37, encode_37_simd, decode_37_simd);
    CASE(37, encode_37_simd, decode_37_simd_mullo);
    // check interop between the simd version and not
    CASE(37, encode_37, decode_37_simd);
    CASE(37, encode_37_simd, decode_37);

    /*CASE(48, encode_48_utf8, decode_48_utf8);*/
    // the utf8 is not valid :(
    /*assert(simdutf_validate_utf8(y, 48));*/

    char test[32];
    for (int i = 0; i < 32; i++) {test[i] = i;}
    encode_hex(test, z);
    printf("\nhextest: %s\n", z);
    memset(test, 0, 32);
    decode_hex((uint8_t*)z, (uint8_t*)test);
    for (int i = 0; i < 32; i++) {
        assert(test[i] == i);
        printf("%d ", test[i]);
    }
    printf("\n\n");

    /*printbits_32(1);*/
    /*printbits_32(0xff00ff00);*/

    return 0;
#endif

    Timespec start, stop;

    int iters = 10 * 1000000;

    uint64_t acc;

#define BENCH(encoder, decoder) \
    acc = 0; \
    clock_ns(&start); \
    for (int i = 0; i < iters; i++) { \
        encoder(x, y); \
        acc += y[33]; \
    } \
    clock_ns(&stop);\
    printf("%20s acc=%lx elapsed=%ld per_iter=%.2f\n", #encoder, acc, elapsed_ns(start, stop), (double)elapsed_ns(start, stop) / iters); \
    acc = 0; \
    clock_ns(&start); \
    for (int i = 0; i < iters; i++) { \
        decoder(y, xx); \
        acc += xx[9]; \
    } \
    clock_ns(&stop); \
    printf("%20s acc=%lx elapsed=%ld per_iter=%.2f\n", #decoder, acc, elapsed_ns(start, stop), (double)elapsed_ns(start, stop) / iters);

    BENCH(encode_40_le, decode_40_le);
    BENCH(encode_40_be, decode_40_be);
    BENCH(encode_40_simd, decode_40_simd);
    BENCH(encode_40_alt, decode_40_alt);
    BENCH(encode_40_alt_simd, decode_40_alt_simd);

    BENCH(encode_37, decode_37);
    BENCH(encode_37_simd, decode_37_simd);
    /*BENCH(encode_48_utf8, decode_48_utf8);*/

    acc = 0;
    clock_ns(&start);
    for (int i = 0; i < iters; i++) {
        decode_37_simd_mullo(x, y);
        acc += y[33];
    }
    clock_ns(&stop);
    printf("%20s acc=%lx elapsed=%ld per_iter=%.2f\n", "decode_37_simd_mullo", acc, elapsed_ns(start, stop), (double)elapsed_ns(start, stop) / iters);

    acc = 0;
    clock_ns(&start);
    for (int i = 0; i < iters; i++) {
        encode_hex(x, z);
        acc += z[39];
    }
    clock_ns(&stop);
    printf("%20s acc=%lx elapsed=%ld per_iter=%.2f\n", "encode_hex", acc, elapsed_ns(start, stop), (double)elapsed_ns(start, stop) / iters);

    acc = 0;
    clock_ns(&start);
    for (int i = 0; i < iters; i++) {
        z[i & 0b11111] = 'a';  // optimizer was defeating us otherwise
        decode_hex((uint8_t*)z, (uint8_t*)y);
        acc += y[3];
    }
    clock_ns(&stop);
    printf("%20s acc=%lx elapsed=%ld per_iter=%.2f\n", "decode_hex", acc, elapsed_ns(start, stop), (double)elapsed_ns(start, stop) / iters);

    acc = 0;
    size_t l; // this will be 44
    clock_ns(&start);
    for (int i = 0; i < iters; i++) {
        l = simdutf_binary_to_base64(x, 32, z, SIMDUTF_BASE64_DEFAULT);
        acc += l;
    }
    clock_ns(&stop);
    printf("%20s acc=%lx elapsed=%ld per_iter=%.2f\n", "base64_encode", acc, elapsed_ns(start, stop), (double)elapsed_ns(start, stop) / iters);

    acc = 0;
    clock_ns(&start);
    for (int i = 0; i < iters; i++) {
        simdutf_result r = simdutf_base64_to_binary(z, l, xx, SIMDUTF_BASE64_DEFAULT, SIMDUTF_LAST_CHUNK_LOOSE);
        acc += r.error;
    }
    clock_ns(&stop);
    printf("%20s acc=%lx elapsed=%ld per_iter=%.2f\n", "base64_decode", acc, elapsed_ns(start, stop), (double)elapsed_ns(start, stop) / iters);

}

//  ls digest.c | entr -c bash -c './build.sh &&  llvm-objdump --disassemble-symbols=xform_simd,xform_invert_simd -Mintel a.out && ./a.out'
/*
0000000000401400 <xform_simd>:
  401400: c5 fe 6f 07                  	vmovdqu	ymm0, ymmword ptr [rdi]
  401404: c5 fd 74 0d 54 1c 00 00      	vpcmpeqb	ymm1, ymm0, ymmword ptr [rip + 0x1c54] # 0x403060 <__dso_handle+0x58>
  40140c: c5 e9 ef d2                  	vpxor	xmm2, xmm2, xmm2
  401410: c5 fd 74 d2                  	vpcmpeqb	ymm2, ymm0, ymm2
  401414: c5 ed eb c9                  	vpor	ymm1, ymm2, ymm1
  401418: c4 e3 7d 4c 05 5e 1c 00 00 10	vpblendvb	ymm0, ymm0, ymmword ptr [rip + 0x1c5e], ymm1 # 0x403080 <__dso_handle+0x78>
  401422: c5 fe 7f 06                  	vmovdqu	ymmword ptr [rsi], ymm0
  401426: c5 fd d7 c2                  	vpmovmskb	eax, ymm2
  40142a: c5 fd d7 c9                  	vpmovmskb	ecx, ymm1
  40142e: f7 d0                        	not	eax
  401430: 48 ba 55 55 55 55 55 55 55 55	movabs	rdx, 0x5555555555555555
  40143a: c4 e2 fb f5 c2               	pdep	rax, rax, rdx
  40143f: 48 ba aa aa aa aa aa aa aa aa	movabs	rdx, -0x5555555555555556
  401449: c4 e2 f3 f5 ca               	pdep	rcx, rcx, rdx
  40144e: 48 09 c1                     	or	rcx, rax
  401451: 48 89 4e 20                  	mov	qword ptr [rsi + 0x20], rcx
  401455: c5 f8 77                     	vzeroupper
  401458: c3                           	ret
  401459: 0f 1f 80 00 00 00 00         	nop	dword ptr [rax]

0000000000401460 <xform_invert_simd>:
  401460: 48 8b 47 20                  	mov	rax, qword ptr [rdi + 0x20]
  401464: 48 b9 55 55 55 55 55 55 55 55	movabs	rcx, 0x5555555555555555
  40146e: c4 e2 fa f5 c9               	pext	rcx, rax, rcx
  401473: 48 ba aa aa aa aa aa aa aa aa	movabs	rdx, -0x5555555555555556
  40147d: c4 e2 fa f5 c2               	pext	rax, rax, rdx
  401482: c4 e2 70 f2 d0               	andn	edx, ecx, eax
  401487: 21 c8                        	and	eax, ecx
  401489: c5 f9 6e c0                  	vmovd	xmm0, eax
  40148d: c4 e3 fd 00 c0 44            	vpermq	ymm0, ymm0, 0x44        # ymm0 = ymm0[0,1,0,1]
  401493: c5 fd 6f 0d 85 1b 00 00      	vmovdqa	ymm1, ymmword ptr [rip + 0x1b85] # 0x403020 <__dso_handle+0x18>
  40149b: c4 e2 7d 00 c1               	vpshufb	ymm0, ymm0, ymm1
  4014a0: c4 e2 7d 59 15 97 1c 00 00   	vpbroadcastq	ymm2, qword ptr [rip + 0x1c97] # 0x403140 <__dso_handle+0x138>
  4014a9: c5 fd eb c2                  	vpor	ymm0, ymm0, ymm2
  4014ad: c5 e5 76 db                  	vpcmpeqd	ymm3, ymm3, ymm3
  4014b1: c5 f9 6e e2                  	vmovd	xmm4, edx
  4014b5: c4 e3 fd 00 e4 44            	vpermq	ymm4, ymm4, 0x44        # ymm4 = ymm4[0,1,0,1]
  4014bb: c4 e2 5d 00 c9               	vpshufb	ymm1, ymm4, ymm1
  4014c0: c5 f5 eb ca                  	vpor	ymm1, ymm1, ymm2
  4014c4: c5 f5 74 cb                  	vpcmpeqb	ymm1, ymm1, ymm3
  4014c8: c5 f5 df 0f                  	vpandn	ymm1, ymm1, ymmword ptr [rdi]
  4014cc: c5 fd 74 c3                  	vpcmpeqb	ymm0, ymm0, ymm3
  4014d0: c4 e3 75 4c 05 86 1b 00 00 00	vpblendvb	ymm0, ymm1, ymmword ptr [rip + 0x1b86], ymm0 # 0x403060 <__dso_handle+0x58>
  4014da: c5 fe 7f 06                  	vmovdqu	ymmword ptr [rsi], ymm0
  4014de: c5 f8 77                     	vzeroupper
  4014e1: c3                           	ret

reg  acc=ffffffffe65eb880 elapsed=254017475 per_iter=25.40
simd acc=ffffffffe65eb880 elapsed=21736239 per_iter=2.17
invert  acc=989680 elapsed=22620208 per_iter=2.26
hex  acc=1d34ce80 elapsed=21306444 per_iter=2.13
hexdec  acc=ffffffffccbd7301 elapsed=155027268 per_iter=15.50
binary_to_base64  acc=1a39de00 elapsed=104796149 per_iter=10.48
base64_to_binary  acc=0 elapsed=369421927 per_iter=36.94
*/
