
#include <stdint.h>
#include <string.h>
#include <immintrin.h>
#include <stdio.h>
#include <time.h>

#include "simdutf_c.h"

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

void NOINLINE xform(const char x[32], char ret[40]) {
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

void printhex(const char *x, int n) {
    for (int i = 0; i < n; i++) {
        printf("%02x ", (uint8_t)x[i]);
    }
    printf("\n\n");
}

// each position gets 2 bits in the mask
// HL - where H is the mask_hi bit and L is the mask_lo bit
// 00 - not used
// 01 - okay
// 10 - zero
// 11 - slash
//
void NOINLINE xform_simd(const char x[32], char ret[40]) {
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

void NOINLINE xform_invert_simd(const char x[40], char ret[32]) {
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

#ifndef NO_MADDUBS
    __m256i sr6 = _mm256_srai_epi16(value, 6);
    __m256i mul = _mm256_maddubs_epi16(sr6, _mm256_set1_epi16(9)); // this has a latency of 5
#else
    // ... while this I think has a latency of 4, but worse throughput(?).
    // (x >> 6) * 9 is x * 8 + x:
    // ((x >> 6) << 3) + (x >> 6)
    // We need & 0b11 to emulate 8-bit operations (narrowest shift is 16b) -- or a left shift
    // (((x >> 6) & 0b11) << 3) + ((x >> 6) & 0b11)
    // or
    // tmp = (x >> 6) & 0b11
    // tmp << 3 + tmp
    // there's no carry due to the mask+shift combo, so + is |
    // tmp << 3 | tmp
    __m256i sr6_lo2 = _mm256_and_si256(_mm256_srli_epi16(value, 6), _mm256_set1_epi16(0b11));
    __m256i sr6_lo2_sl3 = _mm256_slli_epi16(sr6_lo2, 3);
    __m256i mul = _mm256_or_si256(sr6_lo2_sl3, sr6_lo2);
#endif

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

int main(int argc, char **argv) {
    char x[32], xx[32];
    memset(x, 1, 32);
    char y[40];
    char z[65];


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

    memset(z, 0, 40);
    xform_invert_simd(y, z);
    printhex(z, 32);

    char test[32];
    for (int i = 0; i < 32; i++) {test[i] = i;}
    encode_hex(test, z);
    printf("hextest: %s\n", z);
    memset(test, 0, 32);
    decode_hex((uint8_t*)z, (uint8_t*)test);
    for (int i = 0; i < 32; i++) {printf("%d ", test[i]);}
    printf("\n\n");

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
        xform_invert_simd(y, xx);
        acc += xx[9];
    }
    clock_ns(&stop);
    printf("invert  acc=%lx elapsed=%ld per_iter=%.2f\n", acc, elapsed_ns(start, stop), (double)elapsed_ns(start, stop) / iters);

    acc = 0;
    clock_ns(&start);
    for (int i = 0; i < iters; i++) {
        encode_hex(x, z);
        acc += z[39];
    }
    clock_ns(&stop);
    printf("hex  acc=%lx elapsed=%ld per_iter=%.2f\n", acc, elapsed_ns(start, stop), (double)elapsed_ns(start, stop) / iters);

    acc = 0;
    clock_ns(&start);
    for (int i = 0; i < iters; i++) {
        z[i & 0b11111] = 'a';  // optimizer was defeating us otherwise
        decode_hex((uint8_t*)z, (uint8_t*)y);
        acc += y[3];
    }
    clock_ns(&stop);
    printf("hexdec  acc=%lx elapsed=%ld per_iter=%.2f\n", acc, elapsed_ns(start, stop), (double)elapsed_ns(start, stop) / iters);

    acc = 0;
    size_t l; // this will be 44
    clock_ns(&start);
    for (int i = 0; i < iters; i++) {
        l = simdutf_binary_to_base64(x, 32, z, SIMDUTF_BASE64_DEFAULT);
        acc += l;
    }
    clock_ns(&stop);
    printf("binary_to_base64  acc=%lx elapsed=%ld per_iter=%.2f\n", acc, elapsed_ns(start, stop), (double)elapsed_ns(start, stop) / iters);

    acc = 0;
    clock_ns(&start);
    for (int i = 0; i < iters; i++) {
        simdutf_result r = simdutf_base64_to_binary(z, l, xx, SIMDUTF_BASE64_DEFAULT, SIMDUTF_LAST_CHUNK_LOOSE);
        acc += r.error;
    }
    clock_ns(&stop);
    printf("base64_to_binary  acc=%lx elapsed=%ld per_iter=%.2f\n", acc, elapsed_ns(start, stop), (double)elapsed_ns(start, stop) / iters);
}

//  ls digest.c | entr -c bash -c 'clang -Wall -march=native -O2 digest.c &&  llvm-objdump --disassemble-symbols=xform,xform_simd,xform_invert_simd -Mintel a.out && ./a.out'
//  ls digest.c | entr -c bash -c 'clang -Wall -march=native -O2 -lstdc++ simdutf.o digest.c &&  llvm-objdump --disassemble-symbols=xform,xform_simd,xform_invert_simd -Mintel a.out && ./a.out'
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
reg  acc=7ef53880 elapsed=290780392 per_iter=29.08
simd acc=7ef53880 elapsed=29865673 per_iter=2.99
invert  acc=989680 elapsed=31452898 per_iter=3.15
hex  acc=1d34ce80 elapsed=28834164 per_iter=2.88
hexdec  acc=6553ed01 elapsed=113337212 per_iter=11.33
*/
