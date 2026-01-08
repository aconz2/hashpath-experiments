
#include <stdint.h>
#include <string.h>
#include <immintrin.h>
#include <stdio.h>
#include <time.h>
#define NOINLINE __attribute__((noinline))

typedef struct timespec Timespec;
static void clock_ns(Timespec* t) {
  clock_gettime(CLOCK_MONOTONIC, t);
}
static uint64_t elapsed_ns(Timespec start, Timespec stop) {
  return (uint64_t)(stop.tv_sec - start.tv_sec) * 1000000000LL + (uint64_t)(stop.tv_nsec - start.tv_nsec);
}

// -- this code derived from  https://stackoverflow.com/questions/21622212/how-to-perform-the-inverse-of-mm256-movemask-epi8-vpmovmskb
__m256i expand_mask(const uint32_t mask) {
  __m256i vmask = _mm256_set1_epi32(mask);
  const __m256i shuffle = _mm256_setr_epi64x(0x0000000000000000,
      0x0101010101010101, 0x0202020202020202, 0x0303030303030303);
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

void NOINLINE xform(const uint8_t x[32], uint8_t ret[40]) {
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
        if ((x >> (64 - i)) & 1) {
            printf("1");
        } else {
            printf("0");
        }
    }
    printf("\n");
}

void printhex(const uint8_t *x, int n) {
    for (int i = 0; i < n; i++) {
        printf("%02x ", (uint8_t)x[i]);
    }
    printf("\n\n");
}

// each position gets 2 bits in the mask
// 00 - not used
// 01 - okay
// 10 - zero
// 11 - slash
//
void NOINLINE xform_simd(const uint8_t x[32], uint8_t ret[40]) {
    __m256i y = _mm256_loadu_si256((__m256i*) x);
    __m256i is_zero = _mm256_cmpeq_epi8(y, _mm256_set1_epi8(0));
    __m256i is_slash = _mm256_cmpeq_epi8(y, _mm256_set1_epi8('/'));
    __m256i is_zero_or_slash = _mm256_or_si256(is_zero, is_slash);
    __m256i is_ok = _mm256_xor_si256(is_zero_or_slash, _mm256_set1_epi8(0xff));

    y = _mm256_blendv_epi8(y, _mm256_set1_epi8(0xfe), is_zero_or_slash);
    _mm256_storeu_si256((__m256i*) ret, y);

    uint32_t mask_hi = _mm256_movemask_epi8(is_zero_or_slash);
    /*uint32_t mask_lo = _mm256_movemask_epi8(is_slash_or_ok);*/
    uint32_t mask_lo = _mm256_movemask_epi8(_mm256_or_si256(is_ok, is_slash));
    /*uint32_t mask_ok = _mm256_movemask_epi8(is_ok);*/
    uint64_t mask = interleave(mask_lo, mask_hi);
    /*if (mask != interleave_u32_u32(mask_lo, mask_hi)) {*/
    /*    printf("ERROR\n");*/
    /*}*/

    /*printf("mask_ok: ");*/
    /*printbits_32(mask_ok);*/
    /*printf("mask_hi: ");*/
    /*printbits_32(mask_hi);*/
    /*printf("mask_lo: ");*/
    /*printbits_32(mask_lo);*/
    /*printf("mask   : ");*/
    /*printbits_64(mask);*/

    memcpy(ret + 32, &mask, 8);
}

void NOINLINE xform_invert_simd(const uint8_t x[40], uint8_t ret[32]) {
    uint64_t mask;
    memcpy(&mask, ret + 32, 8);
    uint32_2 masks = deinterleave(mask);
    // TODO
    __m256i is_zero_or_slash = expand_mask(masks.hi);
    __m256i is_ok_or_slash = expand_mask(masks.lo);
    /*__m256i is_slash = expand_mask(is_slash_mask);*/
    __m256i y = _mm256_loadu_si256((__m256i*) x);
    /*y = _mm256_blendv_epi8(y, _mm256_set1_epi8(0), is_zero);*/
    /*y = _mm256_blendv_epi8(y, _mm256_set1_epi8('/'), is_slash);*/
    _mm256_storeu_si256((__m256i*) ret, y);
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
void NOINLINE encode_hex(const uint8_t x[32], char ret[65]) {
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

// -- end fast-hex code

int main() {
    uint8_t x[32];
    memset(x, 1, 32);
    uint8_t y[40];
    uint8_t z[65];


    x[0] = 0;
    x[10] = 0;
    x[20] = '/';
    x[31] = '/';

    printhex(x, 32);

    xform(x, y);
    printhex(y, 40);
    memset(y, 0, 40);

    xform_simd(x, y);
    printhex(y, 40);

    xform_invert_simd(y, z);
    printhex(z, 32);

    uint8_t test[32];
    for (int i = 0; i < 32; i++) {test[i] = i;}
    encode_hex(test, z);
    printf("test: %s\n", z);

    /*printbits_32(1);*/
    /*printbits_32(0xff00ff00);*/

    /*return 0;*/

    Timespec start, stop;

    int iters = 10 * 1000000;

    uint64_t acc = 0;
    clock_ns(&start);
    for (int i = 0; i < iters; i++) {
        xform(x, y);
        acc += y[39];
    }
    clock_ns(&stop);
    printf("reg  acc=%lx elapsed=%ld per_iter=%.2f\n", acc, elapsed_ns(start, stop), (double)elapsed_ns(start, stop) / iters);

    acc = 0;
    clock_ns(&start);
    for (int i = 0; i < iters; i++) {
        xform_simd(x, y);
        acc += y[39];
    }
    clock_ns(&stop);
    printf("simd acc=%lx elapsed=%ld per_iter=%.2f\n", acc, elapsed_ns(start, stop), (double)elapsed_ns(start, stop) / iters);

    acc = 0;
    clock_ns(&start);
    for (int i = 0; i < iters; i++) {
        encode_hex(y, (char*)z);
        acc += z[39];
    }
    clock_ns(&stop);
    printf("hex  acc=%lx elapsed=%ld per_iter=%.2f\n", acc, elapsed_ns(start, stop), (double)elapsed_ns(start, stop) / iters);

    acc = 0;
    clock_ns(&start);
    for (int i = 0; i < iters; i++) {
        xform_invert_simd(y, z);
        acc += z[10];
    }
    clock_ns(&stop);
    printf("invert  acc=%lx elapsed=%ld per_iter=%.2f\n", acc, elapsed_ns(start, stop), (double)elapsed_ns(start, stop) / iters);
}

//  ls digest.c | entr -c bash -c 'clang -Wall -march=native -O2 digest.c &&  llvm-objdump --disassemble-symbols=xform,xform_simd,xform_invert_simd -Mintel a.out && ./a.out'
/*
 *00000000004011e0 <xformsimd>:
  4011e0: c5 fe 6f 07                  	vmovdqu	ymm0, ymmword ptr [rdi]
  4011e4: c5 f1 ef c9                  	vpxor	xmm1, xmm1, xmm1
  4011e8: c5 fd 74 c9                  	vpcmpeqb	ymm1, ymm0, ymm1
  4011ec: c5 fd 74 15 2c 0e 00 00      	vpcmpeqb	ymm2, ymm0, ymmword ptr [rip + 0xe2c] # 0x402020 <__dso_handle+0x18>
  4011f4: c5 f5 eb da                  	vpor	ymm3, ymm1, ymm2
  4011f8: c4 e3 7d 4c 05 3e 0e 00 00 30	vpblendvb	ymm0, ymm0, ymmword ptr [rip + 0xe3e], ymm3 # 0x402040 <__dso_handle+0x38>
  401202: c5 fd d7 c1                  	vpmovmskb	eax, ymm1
  401206: c5 fd d7 ca                  	vpmovmskb	ecx, ymm2
  40120a: c5 fe 7f 06                  	vmovdqu	ymmword ptr [rsi], ymm0
  40120e: 89 46 20                     	mov	dword ptr [rsi + 0x20], eax
  401211: 89 4e 24                     	mov	dword ptr [rsi + 0x24], ecx
  401214: c5 f8 77                     	vzeroupper
  401217: c3                           	ret
  401218: 0f 1f 84 00 00 00 00 00      	nop	dword ptr [rax + rax]

output:

00 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 2f

fe 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 fe 01 00 00 00 00 00 00 80

fe 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 fe 01 00 00 00 00 00 00 80

reg  acc=ffffffffb3b4c000 elapsed=268857426 per_iter=26.89
simd acc=ffffffffb3b4c000 elapsed=20204660 per_iter=2.02
*/
