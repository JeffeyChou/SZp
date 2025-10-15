/**
 *  @file szp_TypeManager.c
 *  @author Jiajun Huang <jiajunhuang19990916@gmail.com>, Sheng Di <sdi1@anl.gov>
 *  @date Oct, 2023
 */

#include <stdio.h>
#include <stdlib.h>
#include "szp.h"

// Conditionally include the header for AVX2 intrinsics
// This block will only be active if compiled with -mavx2 or equivalent
#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace szp{

size_t Jiajun_save_fixed_length_bits(unsigned int *unsignintArray, size_t intArrayLength, unsigned char *result, unsigned int bit_count)
{

	unsigned int byte_count = 0;
	unsigned int remainder_bit = 0;
	size_t i, j;
	byte_count = bit_count / 8; 
	remainder_bit = bit_count % 8;
	size_t byteLength = 0;
		
	size_t byte_offset = byte_count * intArrayLength;
	if (remainder_bit == 0)
	{
		byteLength = byte_offset;
	}
	else
	{
		byteLength = byte_count * intArrayLength + (remainder_bit * intArrayLength - 1) / 8 + 1;
	}

	size_t n = 0;
	unsigned int tmp;
	i = 0;
	if (byte_count > 0)
	{
		while (n < intArrayLength)
		{
			j = 0;
			tmp = unsignintArray[n];
			tmp >>= remainder_bit;
			while (i < byteLength && j < byte_count)
			{
				result[i] = (unsigned char)tmp;
				i++;
				tmp >>= 8; // right shift by 8 bits to store the next full byte
				j++;
			}
			n++;
		}
	}
	if (remainder_bit > 0)
	{
		if (byte_count > 0)
		{
			for (i = 0; i < intArrayLength; i++)
			{
				unsignintArray[i] <<= (32 - remainder_bit);
				unsignintArray[i] >>= (32 - remainder_bit);
			}
		}
		

		switch (remainder_bit)
		{
		case 1:
			Jiajun_convertUInt2Byte_fast_1b_args_avx2(unsignintArray, intArrayLength, result + byte_offset);
			break;
		case 2:
			Jiajun_convertUInt2Byte_fast_2b_args_avx2(unsignintArray, intArrayLength, result + byte_offset);
			break;
		case 3:
			Jiajun_convertUInt2Byte_fast_3b_args_avx2(unsignintArray, intArrayLength, result + byte_offset);
			break;
		case 4:
			Jiajun_convertUInt2Byte_fast_4b_args_avx2(unsignintArray, intArrayLength, result + byte_offset);
			break;
		case 5:
			Jiajun_convertUInt2Byte_fast_5b_args_avx2(unsignintArray, intArrayLength, result + byte_offset);
			break;
		case 6:
			Jiajun_convertUInt2Byte_fast_6b_args_avx2(unsignintArray, intArrayLength, result + byte_offset);
			break;
		case 7:
			Jiajun_convertUInt2Byte_fast_7b_args_avx2(unsignintArray, intArrayLength, result + byte_offset);
			break;
		default:
			printf("Error: try to save %d bits\n", remainder_bit);
		}
	}

	return byteLength;
}

size_t convertInt2Byte_fast_1b_args(unsigned char *intArray, size_t intArrayLength, unsigned char *result)
{
	size_t byteLength = 0;
	size_t i, j;
	if (intArrayLength % 8 == 0)
		byteLength = intArrayLength / 8;
	else
		byteLength = intArrayLength / 8 + 1;

	size_t n = 0;
	unsigned int tmp, type;
	for (i = 0; i < byteLength; i++)
	{
		tmp = 0;
		for (j = 0; j < 8 && n < intArrayLength; j++)
		{
			type = intArray[n];
			
			tmp = (tmp | (type << (7 - j)));
			n++;
		}
		result[i] = (unsigned char)tmp;
	}
	return byteLength;
}

size_t Jiajun_convertUInt2Byte_fast_1b_args(unsigned int *intArray, size_t intArrayLength, unsigned char *result)
{
	size_t byteLength = (intArrayLength + 7) / 8;
	size_t n = 0;
	for (size_t i = 0; i < byteLength; i++) {
		unsigned char tmp = 0;
		for (int j = 0; j < 8 && n < intArrayLength; j++) {
			tmp |= (intArray[n++] & 1) << (7 - j);
		}
		result[i] = tmp;
	}
	return byteLength;
}

// A lookup table to reverse the bits of a byte.
// This is extremely fast as it's just a memory access.
static constexpr unsigned char bit_reverse_table[256] = {
    0x00, 0x80, 0x40, 0xC0, 0x20, 0xA0, 0x60, 0xE0, 0x10, 0x90, 0x50, 0xD0, 0x30, 0xB0, 0x70, 0xF0,
    0x08, 0x88, 0x48, 0xC8, 0x28, 0xA8, 0x68, 0xE8, 0x18, 0x98, 0x58, 0xD8, 0x38, 0xB8, 0x78, 0xF8,
    0x04, 0x84, 0x44, 0xC4, 0x24, 0xA4, 0x64, 0xE4, 0x14, 0x94, 0x54, 0xD4, 0x34, 0xB4, 0x74, 0xF4,
    0x0C, 0x8C, 0x4C, 0xCC, 0x2C, 0xAC, 0x6C, 0xEC, 0x1C, 0x9C, 0x5C, 0xDC, 0x3C, 0xBC, 0x7C, 0xFC,
    0x02, 0x82, 0x42, 0xC2, 0x22, 0xA2, 0x62, 0xE2, 0x12, 0x92, 0x52, 0xD2, 0x32, 0xB2, 0x72, 0xF2,
    0x0A, 0x8A, 0x4A, 0xCA, 0x2A, 0xAA, 0x6A, 0xEA, 0x1A, 0x9A, 0x5A, 0xDA, 0x3A, 0xBA, 0x7A, 0xFA,
    0x06, 0x86, 0x46, 0xC6, 0x26, 0xA6, 0x66, 0xE6, 0x16, 0x96, 0x56, 0xD6, 0x36, 0xB6, 0x76, 0xF6,
    0x0E, 0x8E, 0x4E, 0xCE, 0x2E, 0xAE, 0x6E, 0xEE, 0x1E, 0x9E, 0x5E, 0xDE, 0x3E, 0xBE, 0x7E, 0xFE,
    0x01, 0x81, 0x41, 0xC1, 0x21, 0xA1, 0x61, 0xE1, 0x11, 0x91, 0x51, 0xD1, 0x31, 0xB1, 0x71, 0xF1,
    0x09, 0x89, 0x49, 0xC9, 0x29, 0xA9, 0x69, 0xE9, 0x19, 0x99, 0x59, 0xD9, 0x39, 0xB9, 0x79, 0xF9,
    0x05, 0x85, 0x45, 0xC5, 0x25, 0xA5, 0x65, 0xE5, 0x15, 0x95, 0x55, 0xD5, 0x35, 0xB5, 0x75, 0xF5,
    0x0D, 0x8D, 0x4D, 0xCD, 0x2D, 0xAD, 0x6D, 0xED, 0x1D, 0x9D, 0x5D, 0xDD, 0x3D, 0xBD, 0x7D, 0xFD,
    0x03, 0x83, 0x43, 0xC3, 0x23, 0xA3, 0x63, 0xE3, 0x13, 0x93, 0x53, 0xD3, 0x33, 0xB3, 0x73, 0xF3,
    0x0B, 0x8B, 0x4B, 0xCB, 0x2B, 0xAB, 0x6B, 0xEB, 0x1B, 0x9B, 0x5B, 0xDB, 0x3B, 0xBB, 0x7B, 0xFB,
    0x07, 0x87, 0x47, 0xC7, 0x27, 0xA7, 0x67, 0xE7, 0x17, 0x97, 0x57, 0xD7, 0x37, 0xB7, 0x77, 0xF7,
    0x0F, 0x8F, 0x4F, 0xCF, 0x2F, 0xAF, 0x6F, 0xEF, 0x1F, 0x9F, 0x5F, 0xDF, 0x3F, 0xBF, 0x7F, 0xFF
};

size_t Jiajun_convertUInt2Byte_fast_1b_args_avx2(unsigned int *intArray, size_t intArrayLength, unsigned char *result) {
#ifdef __AVX2__
    size_t n = 0; // Input element index
    size_t i = 0; // Output byte index
    const size_t simd_step = 32; // Process 32 uints -> 4 bytes

    const __m256i zero = _mm256_setzero_si256();
    const __m256i mask1 = _mm256_set1_epi32(0x1);
    const __m256i perm_mask = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

    while (n + simd_step <= intArrayLength) {
        // Step 1: Load 32 uints
        __m256i v0 = _mm256_loadu_si256((__m256i*)(intArray + n + 0));
        __m256i v1 = _mm256_loadu_si256((__m256i*)(intArray + n + 8));
        __m256i v2 = _mm256_loadu_si256((__m256i*)(intArray + n + 16));
        __m256i v3 = _mm256_loadu_si256((__m256i*)(intArray + n + 24));

        // Step 2 (FIX #1): Apply the & 1 mask to all 32 integers
        v0 = _mm256_and_si256(v0, mask1);
        v1 = _mm256_and_si256(v1, mask1);
        v2 = _mm256_and_si256(v2, mask1);
        v3 = _mm256_and_si256(v3, mask1);

        // Step 3: Pack down to 32 bytes (now guaranteed to be 0 or 1)
        __m256i p16_0 = _mm256_packus_epi32(v0, v1);
        __m256i p16_1 = _mm256_packus_epi32(v2, v3);
        __m256i p8_scrambled = _mm256_packus_epi16(p16_0, p16_1);

        // Step 4 (FIX #2): Fix the byte order using the correct permutation
        __m256i ordered_bytes = _mm256_permutevar8x32_epi32(p8_scrambled, perm_mask);

        // Step 5: Set the MSB of bytes that are 1
        __m256i msb_set = _mm256_sub_epi8(zero, ordered_bytes);

        // Step 6: Extract the MSB of each of the 32 bytes into a 32-bit integer.
        uint32_t packed_bits = _mm256_movemask_epi8(msb_set);
        
        // Step 7 (FIX #3): The bits in each byte are reversed. Fix them.
        result[i + 0] = bit_reverse_table[(packed_bits >> 0) & 0xFF];
        result[i + 1] = bit_reverse_table[(packed_bits >> 8) & 0xFF];
        result[i + 2] = bit_reverse_table[(packed_bits >> 16) & 0xFF];
        result[i + 3] = bit_reverse_table[(packed_bits >> 24) & 0xFF];
        
        n += simd_step;
        i += 4; // We produced 4 bytes
    }
    
    // Handle the remainder with the original scalar function
    if (n < intArrayLength) {
        Jiajun_convertUInt2Byte_fast_1b_args(intArray + n, intArrayLength - n, result + i);
    }
    
    return (intArrayLength + 7) / 8;
#else
    return Jiajun_convertUInt2Byte_fast_1b_args(intArray, intArrayLength, result);
#endif
}

size_t Jiajun_convertUInt2Byte_fast_2b_args(unsigned int *timeStepType, size_t timeStepTypeLength, unsigned char *result)
{
    size_t i = 0, n = 0;
    size_t byteLength = (timeStepTypeLength * 2 + 7) / 8;

    // Fast path: Process 4 elements at a time
    while (n + 4 <= timeStepTypeLength) {
        result[i] = ((timeStepType[n + 0] & 0x3) << 6) |
                    ((timeStepType[n + 1] & 0x3) << 4) |
                    ((timeStepType[n + 2] & 0x3) << 2) |
                     (timeStepType[n + 3] & 0x3);
        n += 4;
        i += 1;
    }

    // Handle remaining 1, 2, or 3 elements
    if (n < timeStepTypeLength) {
        unsigned char tmp = 0;
        int shift = 6;
        for (; n < timeStepTypeLength; n++) {
            tmp |= (timeStepType[n] & 0x3) << shift;
            shift -= 2;
        }
        result[i] = tmp;
    }

    return byteLength;
}


size_t Jiajun_convertUInt2Byte_fast_2b_args_avx2(unsigned int *timeStepType, size_t timeStepTypeLength, unsigned char *result) {
#ifdef __AVX2__
    size_t n = 0; // Input element index
    size_t i = 0; // Output byte index
    size_t byteLength = (timeStepTypeLength * 2 + 7) / 8;
    
    const __m256i mask2 = _mm256_set1_epi32(0x3);
    const __m256i perm_mask = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

    const __m256i mul_shuffle = _mm256_setr_epi8(
        1 << 6, 1 << 4, 1 << 2, 1, 1 << 6, 1 << 4, 1 << 2, 1,
        1 << 6, 1 << 4, 1 << 2, 1, 1 << 6, 1 << 4, 1 << 2, 1,
        1 << 6, 1 << 4, 1 << 2, 1, 1 << 6, 1 << 4, 1 << 2, 1,
        1 << 6, 1 << 4, 1 << 2, 1, 1 << 6, 1 << 4, 1 << 2, 1
    );

    while (n + 32 <= timeStepTypeLength) {
        // Steps 1 & 2: Load, mask, pack, permute, and maddubs (This part is correct)
        __m256i v0 = _mm256_loadu_si256((__m256i const*)(timeStepType + n));
        __m256i v1 = _mm256_loadu_si256((__m256i const*)(timeStepType + n + 8));
        __m256i v2 = _mm256_loadu_si256((__m256i const*)(timeStepType + n + 16));
        __m256i v3 = _mm256_loadu_si256((__m256i const*)(timeStepType + n + 24));
        v0 = _mm256_and_si256(v0, mask2); v1 = _mm256_and_si256(v1, mask2);
        v2 = _mm256_and_si256(v2, mask2); v3 = _mm256_and_si256(v3, mask2);
        __m256i p16_0 = _mm256_packus_epi32(v0, v1);
        __m256i p16_1 = _mm256_packus_epi32(v2, v3);
        __m256i p8_scrambled = _mm256_packus_epi16(p16_0, p16_1);
        __m256i packed_bytes = _mm256_permutevar8x32_epi32(p8_scrambled, perm_mask);
        __m256i madd_res = _mm256_maddubs_epi16(packed_bytes, mul_shuffle);
        __m256i hadd_res = _mm256_hadd_epi16(madd_res, madd_res);

        // Step 3: We have the correct results, but they are duplicated and in the wrong places.
        // hadd_res = [R0,R1,R2,R3, R0,R1,R2,R3 | R4,R5,R6,R7, R4,R5,R6,R7]
        // We need to bring [R0,R1,R2,R3] from the low lane and [R4,R5,R6,R7] from the high lane together.
        
        // A shuffle on 64-bit qwords is perfect for this. We want the first qword from each lane.
        // The mask _MM_SHUFFLE(d,c,b,a) selects qwords for the output [a,b,c,d].
        // We want output qword 0 to be input qword 0, and output qword 1 to be input qword 2.
        __m256i final_16b_words = _mm256_permute4x64_epi64(hadd_res, _MM_SHUFFLE(0, 0, 2, 0));
        // final_16b_words now contains [R0,R1,R2,R3, R4,R5,R6,R7 | garbage ]

        // Step 4: Pack the 16-bit words (containing our bytes) down to 8-bit bytes.
        // We only care about the low 128-bit lane which now has all 8 results.
        __m128i final_128b = _mm256_castsi256_si128(final_16b_words);
        __m128i final_8_bytes = _mm_packus_epi16(final_128b, final_128b);
        // final_8_bytes now contains [B0,B1,B2,B3,B4,B5,B6,B7 | duplicate]

        // Step 5: Store the final 8 bytes.
        _mm_storel_epi64((__m128i*)(result + i), final_8_bytes);

        n += 32;
        i += 8;
    }

    // Handle the remainder with the reference scalar function.
    if (n < timeStepTypeLength) {
        Jiajun_convertUInt2Byte_fast_2b_args(timeStepType + n, timeStepTypeLength - n, result + i);
    }

    return byteLength;
#else
    // Fallback if not compiling with AVX2
    return Jiajun_convertUInt2Byte_fast_2b_args(timeStepType, timeStepTypeLength, result);
#endif
}

size_t Jiajun_convertUInt2Byte_fast_3b_args(unsigned int *timeStepType, size_t timeStepTypeLength, unsigned char *result)
{
	size_t i = 0, n = 0;
	size_t byteLength = (timeStepTypeLength * 3 + 7) / 8;

	// Fast path for processing 8 elements at a time
	while (n + 8 <= timeStepTypeLength) {
		result[i + 0] = ((timeStepType[n + 0] & 0x7) << 5) | ((timeStepType[n + 1] & 0x7) << 2) | ((timeStepType[n + 2] & 0x7) >> 1);
		result[i + 1] = ((timeStepType[n + 2] & 0x7) << 7) | ((timeStepType[n + 3] & 0x7) << 4) | ((timeStepType[n + 4] & 0x7) << 1) | ((timeStepType[n + 5] & 0x7) >> 2);
		result[i + 2] = ((timeStepType[n + 5] & 0x7) << 6) | ((timeStepType[n + 6] & 0x7) << 3) | (timeStepType[n + 7] & 0x7);
		n += 8;
		i += 3;
	}

	// Handle remaining elements
	if (n < timeStepTypeLength) {
		unsigned char tmp = 0;
		size_t k = 0;
		for (; n < timeStepTypeLength; n++)
		{
			k = n % 8;
			switch (k)
			{
			case 0:
				tmp = (timeStepType[n] & 0x7) << 5;
				break;
			case 1:
				tmp |= (timeStepType[n] & 0x7) << 2;
				break;
			case 2:
				tmp |= (timeStepType[n] & 0x7) >> 1;
				result[i++] = tmp;
				tmp = (timeStepType[n] & 0x7) << 7;
				break;
			case 3:
				tmp |= (timeStepType[n] & 0x7) << 4;
				break;
			case 4:
				tmp |= (timeStepType[n] & 0x7) << 1;
				break;
			case 5:
				tmp |= (timeStepType[n] & 0x7) >> 2;
				result[i++] = tmp;
				tmp = (timeStepType[n] & 0x7) << 6;
				break;
			case 6:
				tmp |= (timeStepType[n] & 0x7) << 3;
				break;
			case 7:
				tmp |= (timeStepType[n] & 0x7);
				result[i++] = tmp;
				tmp = 0;
				break;
			}
		}
		if ((timeStepTypeLength % 8) != 0) {
			result[i] = tmp;
		}
	}

	return byteLength;
}

size_t Jiajun_convertUInt2Byte_fast_3b_args_avx2(unsigned int *timeStepType, size_t timeStepTypeLength, unsigned char *result) {
#ifdef __AVX2__
    size_t i = 0, n = 0;
    size_t byteLength = (timeStepTypeLength * 3 + 7) / 8;

    __m256i mask3 = _mm256_set1_epi32(0x7);

    // This mask is used to fix the lane-crossing issue of pack instructions.
    // It reorders the 32-bit integers to restore the correct sequential order.
    __m256i perm_mask = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

    while (n + 32 <= timeStepTypeLength) {
        // Load 32 integers
        __m256i v0 = _mm256_loadu_si256((__m256i const*)(timeStepType + n));
        __m256i v1 = _mm256_loadu_si256((__m256i const*)(timeStepType + n + 8));
        __m256i v2 = _mm256_loadu_si256((__m256i const*)(timeStepType + n + 16));
        __m256i v3 = _mm256_loadu_si256((__m256i const*)(timeStepType + n + 24));

        // Mask down to 3 bits each
        v0 = _mm256_and_si256(v0, mask3);
        v1 = _mm256_and_si256(v1, mask3);
        v2 = _mm256_and_si256(v2, mask3);
        v3 = _mm256_and_si256(v3, mask3);

        // Narrow 32-bit → 16-bit values (creates lane-scrambled data)
        __m256i pack01 = _mm256_packus_epi32(v0, v1);
        __m256i pack23 = _mm256_packus_epi32(v2, v3);
        
        // Narrow 16-bit → 8-bit values (still lane-scrambled)
        __m256i pack16_scrambled = _mm256_packus_epi16(pack01, pack23);

        // Permute the 32-bit chunks to restore the correct sequential order of bytes.
        __m256i pack16_ordered = _mm256_permutevar8x32_epi32(pack16_scrambled, perm_mask);

        // Spill the CORRECTLY ORDERED data to the stack
        alignas(32) uint8_t vals[32];
        _mm256_store_si256((__m256i*)vals, pack16_ordered);

        // Pack every 8 values → 3 bytes (this part is unchanged and now correct)
        for (int blk = 0; blk < 4; blk++) {
            const uint8_t *in = vals + blk * 8;
            result[i + 0] = (in[0] << 5) | (in[1] << 2) | (in[2] >> 1);
            result[i + 1] = (in[2] << 7) | (in[3] << 4) | (in[4] << 1) | (in[5] >> 2);
            result[i + 2] = (in[5] << 6) | (in[6] << 3) | (in[7]);
            i += 3;
        }

        n += 32;
    }

    // Handle remainder (scalar fallback)
    if (n < timeStepTypeLength) {
        unsigned char tmp = 0;
        size_t k = 0;
        for (; n < timeStepTypeLength; n++) {
            k = n % 8;
            switch (k) {
            case 0:
                tmp = (timeStepType[n] & 0x7) << 5;
                break;
            case 1:
                tmp |= (timeStepType[n] & 0x7) << 2;
                break;
            case 2:
                tmp |= (timeStepType[n] & 0x7) >> 1;
                result[i++] = tmp;
                tmp = (timeStepType[n] & 0x7) << 7;
                break;
            case 3:
                tmp |= (timeStepType[n] & 0x7) << 4;
                break;
            case 4:
                tmp |= (timeStepType[n] & 0x7) << 1;
                break;
            case 5:
                tmp |= (timeStepType[n] & 0x7) >> 2;
                result[i++] = tmp;
                tmp = (timeStepType[n] & 0x7) << 6;
                break;
            case 6:
                tmp |= (timeStepType[n] & 0x7) << 3;
                break;
            case 7:
                tmp |= (timeStepType[n] & 0x7);
                result[i++] = tmp;
                tmp = 0;
                break;
            }
        }
        if ((timeStepTypeLength % 8) != 0) {
            result[i] = tmp;
        }
    }

    return byteLength;
#else // Fallback if not compiling with AVX2
    return Jiajun_convertUInt2Byte_fast_3b_args(timeStepType, timeStepTypeLength, result);
#endif
}

size_t Jiajun_convertUInt2Byte_fast_4b_args(unsigned int *timeStepType, size_t timeStepTypeLength, unsigned char *result)
{
	size_t i = 0, byteLength = 0, n = 0;
	if (timeStepTypeLength % 2 == 0)
		byteLength = timeStepTypeLength * 4 / 8;
	else
		byteLength = timeStepTypeLength * 4 / 8 + 1;

	for (n = 0; n < timeStepTypeLength;)
	{
		unsigned char tmp = 0;
		for (int j = 0; j < 2 && n < timeStepTypeLength; j++)
		{
			unsigned int type = timeStepType[n];
			if (j == 0)
				tmp = tmp | (type << 4);
			else 
				tmp = tmp | type;
			n++;
		}
		(result)[i++] = tmp;
	}
	return byteLength;
}

size_t Jiajun_convertUInt2Byte_fast_4b_args_avx2(unsigned int *timeStepType, size_t timeStepTypeLength, unsigned char *result)
{
#ifdef __AVX2__
    size_t i = 0;
    const size_t simd_step = 32; // Process 32 elements at a time
    const __m256i mult = _mm256_setr_epi8(
        16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1,
        16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1);

    while (i + simd_step <= timeStepTypeLength) {
        __m256i v0 = _mm256_loadu_si256((__m256i*)(timeStepType + i + 0));
        __m256i v1 = _mm256_loadu_si256((__m256i*)(timeStepType + i + 8));
        __m256i v2 = _mm256_loadu_si256((__m256i*)(timeStepType + i + 16));
        __m256i v3 = _mm256_loadu_si256((__m256i*)(timeStepType + i + 24));
        __m256i p16_0 = _mm256_packus_epi32(v0, v1);
        __m256i p16_1 = _mm256_packus_epi32(v2, v3);
        __m256i p8 = _mm256_packus_epi16(p16_0, p16_1);
        __m256i permuted_bytes = _mm256_permute4x64_epi64(p8, _MM_SHUFFLE(3, 1, 2, 0));
        __m256i packed_16bit = _mm256_maddubs_epi16(permuted_bytes, mult);
        __m128i lo_lane = _mm256_castsi256_si128(packed_16bit);
        __m128i hi_lane = _mm256_extracti128_si256(packed_16bit, 1);
        __m128i final_bytes = _mm_packus_epi16(lo_lane, hi_lane);
        _mm_storeu_si128((__m128i*)(result + i / 2), final_bytes);
        i += simd_step;
    }
    size_t remainder_bytes = 0;
    if (i < timeStepTypeLength) {
        remainder_bytes = Jiajun_convertUInt2Byte_fast_4b_args(timeStepType + i, timeStepTypeLength - i, result + i / 2);
    }
    return i / 2 + remainder_bytes;
#else
    // If __AVX2__ is not defined, just call the scalar version directly.
    return Jiajun_convertUInt2Byte_fast_4b_args(timeStepType, timeStepTypeLength, result);
#endif
}

size_t Jiajun_convertUInt2Byte_fast_5b_args(unsigned int *timeStepType, size_t timeStepTypeLength, unsigned char *result)
{
	size_t i = 0, n = 0;
	size_t byteLength = (timeStepTypeLength * 5 + 7) / 8;

	// Fast path for processing 8 elements at a time
	while (n + 8 <= timeStepTypeLength) {
		result[i + 0] = ((timeStepType[n + 0] & 0x1F) << 3) | ((timeStepType[n + 1] & 0x1F) >> 2);
		result[i + 1] = ((timeStepType[n + 1] & 0x1F) << 6) | ((timeStepType[n + 2] & 0x1F) << 1) | ((timeStepType[n + 3] & 0x1F) >> 4);
		result[i + 2] = ((timeStepType[n + 3] & 0x1F) << 4) | ((timeStepType[n + 4] & 0x1F) >> 1);
		result[i + 3] = ((timeStepType[n + 4] & 0x1F) << 7) | ((timeStepType[n + 5] & 0x1F) << 2) | ((timeStepType[n + 6] & 0x1F) >> 3);
		result[i + 4] = ((timeStepType[n + 6] & 0x1F) << 5) | (timeStepType[n + 7] & 0x1F);
		n += 8;
		i += 5;
	}

	// Handle remaining elements
	if (n < timeStepTypeLength) {
		unsigned char tmp = 0;
		size_t k = 0;
		for (; n < timeStepTypeLength; n++)
		{
			k = n % 8;
			switch (k)
			{
			case 0:
				tmp = (timeStepType[n] & 0x1F) << 3;
				break;
			case 1:
				tmp |= (timeStepType[n] & 0x1F) >> 2;
				result[i++] = tmp;
				tmp = (timeStepType[n] & 0x1F) << 6;
				break;
			case 2:
				tmp |= (timeStepType[n] & 0x1F) << 1;
				break;
			case 3:
				tmp |= (timeStepType[n] & 0x1F) >> 4;
				result[i++] = tmp;
				tmp = (timeStepType[n] & 0x1F) << 4;
				break;
			case 4:
				tmp |= (timeStepType[n] & 0x1F) >> 1;
				result[i++] = tmp;
				tmp = (timeStepType[n] & 0x1F) << 7;
				break;
			case 5:
				tmp |= (timeStepType[n] & 0x1F) << 2;
				break;
			case 6:
				tmp |= (timeStepType[n] & 0x1F) >> 3;
				result[i++] = tmp;
				tmp = (timeStepType[n] & 0x1F) << 5;
				break;
			case 7:
				tmp |= (timeStepType[n] & 0x1F);
				result[i++] = tmp;
				tmp = 0;
				break;
			}
		}
		if ((timeStepTypeLength % 8) != 0) {
			result[i] = tmp;
		}
	}

	return byteLength;
}

size_t Jiajun_convertUInt2Byte_fast_5b_args_avx2(unsigned int *timeStepType, size_t timeStepTypeLength, unsigned char *result)
{
#ifdef __AVX2__
    size_t i = 0, n = 0;
    size_t byteLength = (timeStepTypeLength * 5 + 7) / 8;

    // Mask to keep the 5 least significant bits (0x1F = 0b00011111)
    __m256i mask5 = _mm256_set1_epi32(0x1F);
    // Permutation mask to fix the lane-crossing issue of pack instructions
    __m256i perm_mask = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

    // Main AVX2 loop: Process 32 integers at a time
    while (n + 32 <= timeStepTypeLength) {
        // 1. Load 32 integers into four 256-bit registers
        __m256i v0 = _mm256_loadu_si256((__m256i const*)(timeStepType + n));
        __m256i v1 = _mm256_loadu_si256((__m256i const*)(timeStepType + n + 8));
        __m256i v2 = _mm256_loadu_si256((__m256i const*)(timeStepType + n + 16));
        __m256i v3 = _mm256_loadu_si256((__m256i const*)(timeStepType + n + 24));

        // 2. Mask all 32 integers in parallel
        v0 = _mm256_and_si256(v0, mask5);
        v1 = _mm256_and_si256(v1, mask5);
        v2 = _mm256_and_si256(v2, mask5);
        v3 = _mm256_and_si256(v3, mask5);

        // 3. Narrow 32-bit integers down to 8-bit bytes (scrambled order)
        __m256i pack01 = _mm256_packus_epi32(v0, v1);
        __m256i pack23 = _mm256_packus_epi32(v2, v3);
        __m256i pack16_scrambled = _mm256_packus_epi16(pack01, pack23);

        // 4. Fix the scrambled order using a permutation
        __m256i pack16_ordered = _mm256_permutevar8x32_epi32(pack16_scrambled, perm_mask);

        // 5. Store the 32 correctly ordered bytes to a temporary array
        alignas(32) uint8_t vals[32];
        _mm256_store_si256((__m256i*)vals, pack16_ordered);

        // 6. Perform the complex bit-packing using simple scalar code on the temp array
        //    This loop runs 4 times, since 32 integers / 8 per block = 4 blocks.
        for (int blk = 0; blk < 4; blk++) {
            const uint8_t *in = vals + blk * 8;
            result[i + 0] = (in[0] << 3) | (in[1] >> 2);
            result[i + 1] = (in[1] << 6) | (in[2] << 1) | (in[3] >> 4);
            result[i + 2] = (in[3] << 4) | (in[4] >> 1);
            result[i + 3] = (in[4] << 7) | (in[5] << 2) | (in[6] >> 3);
            result[i + 4] = (in[6] << 5) | (in[7]);
            i += 5; // Move output pointer by 5 bytes
        }

        n += 32; // Move input pointer by 32 integers
    }

    // Handle the remainder using the original, reliable scalar function
    if (n < timeStepTypeLength) {
        size_t remaining_len = timeStepTypeLength - n;
        Jiajun_convertUInt2Byte_fast_5b_args(timeStepType + n, remaining_len, result + i);
    }

    return byteLength;
#else // Fallback if not compiling with AVX2
    return Jiajun_convertUInt2Byte_fast_5b_args(timeStepType, timeStepTypeLength, result);
#endif
}

size_t Jiajun_convertUInt2Byte_fast_6b_args(unsigned int *timeStepType, size_t timeStepTypeLength, unsigned char *result)
{
	size_t i = 0, n = 0;
	size_t byteLength = (timeStepTypeLength * 6 + 7) / 8;

	// Fast path for processing 4 elements at a time
	while (n + 4 <= timeStepTypeLength) {
		result[i + 0] = ((timeStepType[n + 0] & 0x3F) << 2) | ((timeStepType[n + 1] & 0x3F) >> 4);
		result[i + 1] = ((timeStepType[n + 1] & 0x3F) << 4) | ((timeStepType[n + 2] & 0x3F) >> 2);
		result[i + 2] = ((timeStepType[n + 2] & 0x3F) << 6) | (timeStepType[n + 3] & 0x3F);
		n += 4;
		i += 3;
	}

	// Handle remaining elements
	if (n < timeStepTypeLength) {
		unsigned char tmp = 0;
		size_t k = 0;
		for (; n < timeStepTypeLength; n++)
		{
			k = n % 4;
			switch (k)
			{
			case 0:
				tmp = (timeStepType[n] & 0x3F) << 2;
				break;
			case 1:
				tmp |= (timeStepType[n] & 0x3F) >> 4;
				result[i++] = tmp;
				tmp = (timeStepType[n] & 0x3F) << 4;
				break;
			case 2:
				tmp |= (timeStepType[n] & 0x3F) >> 2;
				result[i++] = tmp;
				tmp = (timeStepType[n] & 0x3F) << 6;
				break;
			case 3:
				tmp |= (timeStepType[n] & 0x3F);
				result[i++] = tmp;
				tmp = 0;
				break;
			}
		}
		if ((timeStepTypeLength % 4) != 0) {
			result[i] = tmp;
		}
	}

	return byteLength;
}

size_t Jiajun_convertUInt2Byte_fast_6b_args_avx2(unsigned int *timeStepType, size_t timeStepTypeLength, unsigned char *result)
{
#ifdef __AVX2__
    size_t i = 0, n = 0;
    size_t byteLength = (timeStepTypeLength * 6 + 7) / 8;

    // Mask to keep the 6 least significant bits (0x3F = 0b00111111)
    __m256i mask6 = _mm256_set1_epi32(0x3F);
    // Permutation mask to fix the lane-crossing issue of pack instructions
    __m256i perm_mask = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

    // Main AVX2 loop: Process 32 integers at a time
    while (n + 32 <= timeStepTypeLength) {
        // 1. Load, Mask, Narrow, Permute, and Store (same pattern as before)
        __m256i v0 = _mm256_loadu_si256((__m256i const*)(timeStepType + n));
        __m256i v1 = _mm256_loadu_si256((__m256i const*)(timeStepType + n + 8));
        __m256i v2 = _mm256_loadu_si256((__m256i const*)(timeStepType + n + 16));
        __m256i v3 = _mm256_loadu_si256((__m256i const*)(timeStepType + n + 24));

        v0 = _mm256_and_si256(v0, mask6);
        v1 = _mm256_and_si256(v1, mask6);
        v2 = _mm256_and_si256(v2, mask6);
        v3 = _mm256_and_si256(v3, mask6);

        __m256i pack01 = _mm256_packus_epi32(v0, v1);
        __m256i pack23 = _mm256_packus_epi32(v2, v3);
        __m256i pack16_scrambled = _mm256_packus_epi16(pack01, pack23);
        __m256i pack16_ordered = _mm256_permutevar8x32_epi32(pack16_scrambled, perm_mask);

        alignas(32) uint8_t vals[32];
        _mm256_store_si256((__m256i*)vals, pack16_ordered);

        // 2. Perform scalar bit-packing on the temp array
        //    This loop runs 8 times, since 32 integers / 4 per block = 8 blocks.
        for (int blk = 0; blk < 8; blk++) {
            const uint8_t *in = vals + blk * 4;
            result[i + 0] = (in[0] << 2) | (in[1] >> 4);
            result[i + 1] = (in[1] << 4) | (in[2] >> 2);
            result[i + 2] = (in[2] << 6) | (in[3]);
            i += 3; // Move output pointer by 3 bytes
        }

        n += 32; // Move input pointer by 32 integers
    }

    // Handle the remainder using the original, reliable scalar function
    if (n < timeStepTypeLength) {
        size_t remaining_len = timeStepTypeLength - n;
        Jiajun_convertUInt2Byte_fast_6b_args(timeStepType + n, remaining_len, result + i);
    }

    return byteLength;
#else // Fallback if not compiling with AVX2
    return Jiajun_convertUInt2Byte_fast_6b_args(timeStepType, timeStepTypeLength, result);
#endif
}

size_t Jiajun_convertUInt2Byte_fast_7b_args(unsigned int *timeStepType,
size_t timeStepTypeLength,
unsigned char *result)
{
	size_t i = 0, n = 0;
        size_t byteLength =
            (timeStepTypeLength * 7 + 7) / 8; // same as original

        // Fast path: process 8 elements at a time
        while (n + 8 <= timeStepTypeLength) {
            unsigned char t0 = (unsigned char)timeStepType[n++];
            unsigned char t1 = (unsigned char)timeStepType[n++];
            unsigned char t2 = (unsigned char)timeStepType[n++];
            unsigned char t3 = (unsigned char)timeStepType[n++];
            unsigned char t4 = (unsigned char)timeStepType[n++];
            unsigned char t5 = (unsigned char)timeStepType[n++];
            unsigned char t6 = (unsigned char)timeStepType[n++];
            unsigned char t7 = (unsigned char)timeStepType[n++];

            result[i++] = (t0 << 1) | (t1 >> 6);
            result[i++] = (t1 << 2) | (t2 >> 5);
            result[i++] = (t2 << 3) | (t3 >> 4);
            result[i++] = (t3 << 4) | (t4 >> 3);
            result[i++] = (t4 << 5) | (t5 >> 2);
            result[i++] = (t5 << 6) | (t6 >> 1);
            result[i++] = (t6 << 7) | (t7 >> 0);
        }

        // Handle remaining elements (if not a multiple of 8)
        if (n < timeStepTypeLength) {
            unsigned char t[8] = {0};
            size_t rem = timeStepTypeLength - n;
            for (size_t j = 0; j < rem; ++j)
                t[j] = (unsigned char)timeStepType[n + j];
            // The following matches the above pattern, but only writes as many
            // bytes as needed
            if (rem > 0)
                result[i++] = (t[0] << 1) | (t[1] >> 6);
            if (rem > 1)
                result[i++] = (t[1] << 2) | (t[2] >> 5);
            if (rem > 2)
                result[i++] = (t[2] << 3) | (t[3] >> 4);
            if (rem > 3)
                result[i++] = (t[3] << 4) | (t[4] >> 3);
            if (rem > 4)
                result[i++] = (t[4] << 5) | (t[5] >> 2);
            if (rem > 5)
                result[i++] = (t[5] << 6) | (t[6] >> 1);
            if (rem > 6)
                result[i++] = (t[6] << 7);
        }

	return byteLength;
}

size_t Jiajun_convertUInt2Byte_fast_7b_args_avx2(unsigned int *timeStepType, size_t timeStepTypeLength, unsigned char *result)
{
#ifdef __AVX2__
    size_t i = 0, n = 0;
    size_t byteLength = (timeStepTypeLength * 7 + 7) / 8;

    __m256i mask7 = _mm256_set1_epi32(0x7F);
    __m256i perm_mask = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

    // Process 32 integers at a time (4 blocks of 8)
    while (n + 32 <= timeStepTypeLength) {
        // 1. Load, Mask, Narrow, and Permute (same as before)
        // This brings 32 correctly ordered 7-bit values into a single AVX register.
        __m256i v0 = _mm256_loadu_si256((__m256i const*)(timeStepType + n));
        __m256i v1 = _mm256_loadu_si256((__m256i const*)(timeStepType + n + 8));
        __m256i v2 = _mm256_loadu_si256((__m256i const*)(timeStepType + n + 16));
        __m256i v3 = _mm256_loadu_si256((__m256i const*)(timeStepType + n + 24));
        v0 = _mm256_and_si256(v0, mask7);
        v1 = _mm256_and_si256(v1, mask7);
        v2 = _mm256_and_si256(v2, mask7);
        v3 = _mm256_and_si256(v3, mask7);
        __m256i pack01 = _mm256_packus_epi32(v0, v1);
        __m256i pack23 = _mm256_packus_epi32(v2, v3);
        __m256i pack16_scrambled = _mm256_packus_epi16(pack01, pack23);
        __m256i packed_bytes = _mm256_permutevar8x32_epi32(pack16_scrambled, perm_mask);

        // 2. EFFICIENT EXTRACTION (The Fix)
        // Instead of storing to the stack, extract 64-bit chunks directly.
        // This moves data from AVX registers to general-purpose registers.
        uint64_t chunk0 = _mm256_extract_epi64(packed_bytes, 0);
        uint64_t chunk1 = _mm256_extract_epi64(packed_bytes, 1);
        uint64_t chunk2 = _mm256_extract_epi64(packed_bytes, 2);
        uint64_t chunk3 = _mm256_extract_epi64(packed_bytes, 3);

        // 3. Fast Scalar Packing on the extracted 64-bit integers
        // The compiler is extremely good at optimizing this simple pointer casting and access.
        const uint8_t* in;

        // Block 0
        in = (const uint8_t*)&chunk0;
        result[i + 0] = (in[0] << 1) | (in[1] >> 6);
        result[i + 1] = (in[1] << 2) | (in[2] >> 5);
        result[i + 2] = (in[2] << 3) | (in[3] >> 4);
        result[i + 3] = (in[3] << 4) | (in[4] >> 3);
        result[i + 4] = (in[4] << 5) | (in[5] >> 2);
        result[i + 5] = (in[5] << 6) | (in[6] >> 1);
        result[i + 6] = (in[6] << 7) | (in[7]);
        
        // Block 1
        in = (const uint8_t*)&chunk1;
        result[i + 7] = (in[0] << 1) | (in[1] >> 6);
        result[i + 8] = (in[1] << 2) | (in[2] >> 5);
        result[i + 9] = (in[2] << 3) | (in[3] >> 4);
        result[i + 10] = (in[3] << 4) | (in[4] >> 3);
        result[i + 11] = (in[4] << 5) | (in[5] >> 2);
        result[i + 12] = (in[5] << 6) | (in[6] >> 1);
        result[i + 13] = (in[6] << 7) | (in[7]);

        // Block 2
        in = (const uint8_t*)&chunk2;
        result[i + 14] = (in[0] << 1) | (in[1] >> 6);
        result[i + 15] = (in[1] << 2) | (in[2] >> 5);
        result[i + 16] = (in[2] << 3) | (in[3] >> 4);
        result[i + 17] = (in[3] << 4) | (in[4] >> 3);
        result[i + 18] = (in[4] << 5) | (in[5] >> 2);
        result[i + 19] = (in[5] << 6) | (in[6] >> 1);
        result[i + 20] = (in[6] << 7) | (in[7]);

        // Block 3
        in = (const uint8_t*)&chunk3;
        result[i + 21] = (in[0] << 1) | (in[1] >> 6);
        result[i + 22] = (in[1] << 2) | (in[2] >> 5);
        result[i + 23] = (in[2] << 3) | (in[3] >> 4);
        result[i + 24] = (in[3] << 4) | (in[4] >> 3);
        result[i + 25] = (in[4] << 5) | (in[5] >> 2);
        result[i + 26] = (in[5] << 6) | (in[6] >> 1);
        result[i + 27] = (in[6] << 7) | (in[7]);

        i += 28; // 4 blocks * 7 bytes/block
        n += 32;
    }

    // Handle the remainder using the original scalar function
    if (n < timeStepTypeLength) {
        size_t remaining_len = timeStepTypeLength - n;
        Jiajun_convertUInt2Byte_fast_7b_args(timeStepType + n, remaining_len, result + i);
    }

    return byteLength;
#else
    return Jiajun_convertUInt2Byte_fast_7b_args(timeStepType, timeStepTypeLength, result);
#endif
}

size_t Jiajun_extract_fixed_length_bits(unsigned char *result, size_t intArrayLength, unsigned int *unsignintArray, unsigned int bit_count)
{
	unsigned int byte_count = 0;
	unsigned int remainder_bit = 0;
	
	size_t i, j;
	byte_count = bit_count / 8; 
	remainder_bit = bit_count % 8;
	size_t byteLength = 0;
	
	
	size_t n = 0;
	unsigned int tmp1, tmp2;
	i = 0;
	size_t byte_offset = byte_count * intArrayLength;

	if (remainder_bit == 0)
	{
		byteLength = byte_offset;
	}
	else
	{
		byteLength = byte_count * intArrayLength + (remainder_bit * intArrayLength - 1) / 8 + 1;
	}

	
	if (remainder_bit > 0)
	{
		switch (remainder_bit)
		{
		case 1:
			Jiajun_convertByte2UInt_fast_1b_args_avx2(intArrayLength, result + byte_offset, (intArrayLength - 1) / 8 + 1, unsignintArray);
			break;
		case 2:
			Jiajun_convertByte2UInt_fast_2b_args_avx2(intArrayLength, result + byte_offset, (intArrayLength * 2 - 1) / 8 + 1, unsignintArray);
			break;
		case 3:
			Jiajun_convertByte2UInt_fast_3b_args_avx2(intArrayLength, result + byte_offset, (intArrayLength * 3 - 1) / 8 + 1, unsignintArray);
			break;
		case 4:
			Jiajun_convertByte2UInt_fast_4b_args_avx2(intArrayLength, result + byte_offset, (intArrayLength * 4 - 1) / 8 + 1, unsignintArray);
			break;
		case 5:
			Jiajun_convertByte2UInt_fast_5b_args_avx2(intArrayLength, result + byte_offset, (intArrayLength * 5 - 1) / 8 + 1, unsignintArray);
			break;
		case 6:
			Jiajun_convertByte2UInt_fast_6b_args_avx2(intArrayLength, result + byte_offset, (intArrayLength * 6 - 1) / 8 + 1, unsignintArray);
			break;
		case 7:
			Jiajun_convertByte2UInt_fast_7b_args_avx2(intArrayLength, result + byte_offset, (intArrayLength * 7 - 1) / 8 + 1, unsignintArray);
			break;
		default:
			printf("Error: try to extract %d bits\n", remainder_bit);
		}
	}
	if (byte_count > 0)
	{
		if(remainder_bit == 0)
		{
			memset(unsignintArray, 0 , intArrayLength * sizeof(unsigned int));
		}
		while (i < intArrayLength)
		{
			j = 0;
			tmp1 = 0;
			tmp2 = 0;
			while (j < byte_count && n < byteLength)
			{
				tmp1 = result[n];
				tmp1 <<= (8 * j);
				tmp2 = (tmp2 | tmp1);
				n++;
				j++;
			}
			tmp2 <<= remainder_bit; 
			unsignintArray[i] = (unsignintArray[i] | tmp2);
			i++;
		}
	}

	return byteLength;
}

void Jiajun_convertByte2UInt_fast_1b_args(size_t intArrayLength, unsigned char *byteArray, size_t byteArrayLength, unsigned int *intArray)
{
	size_t n = 0, i;
	unsigned int tmp;
	for (i = 0; i < byteArrayLength - 1; i++)
	{
		tmp = byteArray[i];
		intArray[n++] = (tmp & 0x80) >> 7;
		intArray[n++] = (tmp & 0x40) >> 6;
		intArray[n++] = (tmp & 0x20) >> 5;
		intArray[n++] = (tmp & 0x10) >> 4;
		intArray[n++] = (tmp & 0x08) >> 3;
		intArray[n++] = (tmp & 0x04) >> 2;
		intArray[n++] = (tmp & 0x02) >> 1;
		intArray[n++] = (tmp & 0x01) >> 0;
	}

	tmp = byteArray[i];
	if (n == intArrayLength)
		return;
	intArray[n++] = (tmp & 0x80) >> 7;
	if (n == intArrayLength)
		return;
	intArray[n++] = (tmp & 0x40) >> 6;
	if (n == intArrayLength)
		return;
	intArray[n++] = (tmp & 0x20) >> 5;
	if (n == intArrayLength)
		return;
	intArray[n++] = (tmp & 0x10) >> 4;
	if (n == intArrayLength)
		return;
	intArray[n++] = (tmp & 0x08) >> 3;
	if (n == intArrayLength)
		return;
	intArray[n++] = (tmp & 0x04) >> 2;
	if (n == intArrayLength)
		return;
	intArray[n++] = (tmp & 0x02) >> 1;
	if (n == intArrayLength)
		return;
	intArray[n++] = (tmp & 0x01) >> 0;
}

void Jiajun_convertByte2UInt_fast_1b_args_avx2(size_t intArrayLength, unsigned char *byteArray,
                                               size_t byteArrayLength, unsigned int *intArray) {
#ifdef __AVX2__
    size_t n = 0; // output uint index
    size_t i = 0; // input byte index
    const size_t simd_out_step = 32; // 32 uints (from 4 bytes) per iteration
    const size_t simd_in_step = 4;   // 4 input bytes

    // Constants for the unpacking logic
    const __m256i mask_1 = _mm256_set1_epi32(1);
    const __m256i shifts = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);

    // Main AVX2 loop: Process 4 bytes -> 32 uints per iteration
    while (n + simd_out_step <= intArrayLength && i + simd_in_step <= byteArrayLength) {
        // --- Process 4 bytes independently ---

        // Byte 0 -> intArray[n+0] to [n+7]
        __m256i b0_broadcast = _mm256_set1_epi32(byteArray[i + 0]);
        __m256i b0_shifted = _mm256_srlv_epi32(b0_broadcast, shifts);
        __m256i b0_unpacked = _mm256_and_si256(b0_shifted, mask_1);
        _mm256_storeu_si256((__m256i*)(intArray + n), b0_unpacked);

        // Byte 1 -> intArray[n+8] to [n+15]
        __m256i b1_broadcast = _mm256_set1_epi32(byteArray[i + 1]);
        __m256i b1_shifted = _mm256_srlv_epi32(b1_broadcast, shifts);
        __m256i b1_unpacked = _mm256_and_si256(b1_shifted, mask_1);
        _mm256_storeu_si256((__m256i*)(intArray + n + 8), b1_unpacked);

        // Byte 2 -> intArray[n+16] to [n+23]
        __m256i b2_broadcast = _mm256_set1_epi32(byteArray[i + 2]);
        __m256i b2_shifted = _mm256_srlv_epi32(b2_broadcast, shifts);
        __m256i b2_unpacked = _mm256_and_si256(b2_shifted, mask_1);
        _mm256_storeu_si256((__m256i*)(intArray + n + 16), b2_unpacked);

        // Byte 3 -> intArray[n+24] to [n+31]
        __m256i b3_broadcast = _mm256_set1_epi32(byteArray[i + 3]);
        __m256i b3_shifted = _mm256_srlv_epi32(b3_broadcast, shifts);
        __m256i b3_unpacked = _mm256_and_si256(b3_shifted, mask_1);
        _mm256_storeu_si256((__m256i*)(intArray + n + 24), b3_unpacked);

        n += simd_out_step;
        i += simd_in_step;
    }

    // Handle any remaining bytes with the scalar function
    if (n < intArrayLength) {
        Jiajun_convertByte2UInt_fast_1b_args(intArrayLength - n, byteArray + i, byteArrayLength - i, intArray + n);
    }
#else
    // Fallback if not compiling with AVX2
    Jiajun_convertByte2UInt_fast_1b_args(intArrayLength, byteArray, byteArrayLength, intArray);
#endif
}

void Jiajun_convertByte2UInt_fast_2b_args(size_t stepLength, unsigned char *byteArray, size_t byteArrayLength, unsigned int *intArray)
{

	size_t i, n = 0;

	int mod4 = stepLength % 4;
	if (mod4 == 0)
	{
		for (i = 0; i < byteArrayLength; i++)
		{
			unsigned char tmp = byteArray[i];
			intArray[n++] = (tmp & 0xC0) >> 6;
			intArray[n++] = (tmp & 0x30) >> 4;
			intArray[n++] = (tmp & 0x0C) >> 2;
			intArray[n++] = tmp & 0x03;
		}
	}
	else
	{
		size_t t = byteArrayLength - 1;
		for (i = 0; i < t; i++)
		{
			unsigned char tmp = byteArray[i];
			intArray[n++] = (tmp & 0xC0) >> 6;
			intArray[n++] = (tmp & 0x30) >> 4;
			intArray[n++] = (tmp & 0x0C) >> 2;
			intArray[n++] = tmp & 0x03;
		}
		unsigned char tmp = byteArray[i];
		switch (mod4)
		{
		case 1:
			intArray[n++] = (tmp & 0xC0) >> 6;
			break;
		case 2:
			intArray[n++] = (tmp & 0xC0) >> 6;
			intArray[n++] = (tmp & 0x30) >> 4;
			break;
		case 3:
			intArray[n++] = (tmp & 0xC0) >> 6;
			intArray[n++] = (tmp & 0x30) >> 4;
			intArray[n++] = (tmp & 0x0C) >> 2;
			break;
		}
	}
}

void Jiajun_convertByte2UInt_fast_2b_args_avx2(size_t intArrayLength, unsigned char *byteArray, size_t byteArrayLength, unsigned int *intArray) {
#ifdef __AVX2__
    size_t n = 0; // output uint index
    size_t i = 0; // input byte index
    const size_t simd_out_step = 32; // Process 8 bytes -> 32 uints
    const size_t simd_in_step = 8;

    // Constants for the unpacking logic
    const __m256i mask_3 = _mm256_set1_epi32(0x03);
    const __m256i shifts = _mm256_setr_epi32(6, 4, 2, 0, 6, 4, 2, 0);

    while (n + simd_out_step <= intArrayLength && i + simd_in_step <= byteArrayLength) {
        // --- Process bytes 0 and 1 ---
        __m256i b01_replicated = _mm256_setr_epi32(
            byteArray[i + 0], byteArray[i + 0], byteArray[i + 0], byteArray[i + 0],
            byteArray[i + 1], byteArray[i + 1], byteArray[i + 1], byteArray[i + 1]);
        __m256i b01_shifted = _mm256_srlv_epi32(b01_replicated, shifts);
        __m256i b01_unpacked = _mm256_and_si256(b01_shifted, mask_3);
        _mm256_storeu_si256((__m256i*)(intArray + n), b01_unpacked);

        // --- Process bytes 2 and 3 ---
        __m256i b23_replicated = _mm256_setr_epi32(
            byteArray[i + 2], byteArray[i + 2], byteArray[i + 2], byteArray[i + 2],
            byteArray[i + 3], byteArray[i + 3], byteArray[i + 3], byteArray[i + 3]);
        __m256i b23_shifted = _mm256_srlv_epi32(b23_replicated, shifts);
        __m256i b23_unpacked = _mm256_and_si256(b23_shifted, mask_3);
        _mm256_storeu_si256((__m256i*)(intArray + n + 8), b23_unpacked);
        
        // --- Process bytes 4 and 5 ---
        __m256i b45_replicated = _mm256_setr_epi32(
            byteArray[i + 4], byteArray[i + 4], byteArray[i + 4], byteArray[i + 4],
            byteArray[i + 5], byteArray[i + 5], byteArray[i + 5], byteArray[i + 5]);
        __m256i b45_shifted = _mm256_srlv_epi32(b45_replicated, shifts);
        __m256i b45_unpacked = _mm256_and_si256(b45_shifted, mask_3);
        _mm256_storeu_si256((__m256i*)(intArray + n + 16), b45_unpacked);

        // --- Process bytes 6 and 7 ---
        __m256i b67_replicated = _mm256_setr_epi32(
            byteArray[i + 6], byteArray[i + 6], byteArray[i + 6], byteArray[i + 6],
            byteArray[i + 7], byteArray[i + 7], byteArray[i + 7], byteArray[i + 7]);
        __m256i b67_shifted = _mm256_srlv_epi32(b67_replicated, shifts);
        __m256i b67_unpacked = _mm256_and_si256(b67_shifted, mask_3);
        _mm256_storeu_si256((__m256i*)(intArray + n + 24), b67_unpacked);

        n += simd_out_step;
        i += simd_in_step;
    }

    // Handle the remainder with the scalar function
    if (n < intArrayLength) {
        Jiajun_convertByte2UInt_fast_2b_args(intArrayLength - n, byteArray + i, byteArrayLength - i, intArray + n);
    }
#else
    Jiajun_convertByte2UInt_fast_2b_args(intArrayLength, byteArray, byteArrayLength, intArray);
#endif
}

void Jiajun_convertByte2UInt_fast_3b_args(size_t stepLength, unsigned char *byteArray, size_t byteArrayLength, unsigned int *intArray)
{
	size_t i = 0, n = 0;

	// Fast path for processing 8 elements (3 bytes) at a time
	while (n + 8 <= stepLength && i + 3 <= byteArrayLength) {
		unsigned char b0 = byteArray[i + 0];
		unsigned char b1 = byteArray[i + 1];
		unsigned char b2 = byteArray[i + 2];

		intArray[n + 0] = b0 >> 5;
		intArray[n + 1] = (b0 >> 2) & 0x7;
		intArray[n + 2] = ((b0 & 0x3) << 1) | (b1 >> 7);
		intArray[n + 3] = (b1 >> 4) & 0x7;
		intArray[n + 4] = (b1 >> 1) & 0x7;
		intArray[n + 5] = ((b1 & 0x1) << 2) | (b2 >> 6);
		intArray[n + 6] = (b2 >> 3) & 0x7;
		intArray[n + 7] = b2 & 0x7;

		n += 8;
		i += 3;
	}

	// Handle remaining elements
	if (n < stepLength) {
		if (i >= byteArrayLength) return; // Bounds check
		unsigned char tmp = byteArray[i];
		size_t ii = 0;
		for (; n < stepLength;) 
		{
			switch (n % 8)
			{
			case 0:
				intArray[n++] = (tmp & 0xE0) >> 5;
				break;
			case 1:
				intArray[n++] = (tmp & 0x1C) >> 2;
				break;
			case 2:
				ii = (tmp & 0x03) << 1;
				i++;
				if (i >= byteArrayLength) { if(n < stepLength) intArray[n++] = ii; return; }
				tmp = byteArray[i];
				ii |= (tmp & 0x80) >> 7;
				intArray[n++] = ii;
				break;
			case 3:
				intArray[n++] = (tmp & 0x70) >> 4;
				break;
			case 4:
				intArray[n++] = (tmp & 0x0E) >> 1;
				break;
			case 5:
				ii = (tmp & 0x01) << 2;
				i++;
				if (i >= byteArrayLength) { if(n < stepLength) intArray[n++] = ii; return; }
				tmp = byteArray[i];
				ii |= (tmp & 0xC0) >> 6;
				intArray[n++] = ii;
				break;
			case 6:
				intArray[n++] = (tmp & 0x38) >> 3;
				break;
			case 7:
				intArray[n++] = (tmp & 0x07);
				i++;
				if (i < byteArrayLength) tmp = byteArray[i];
				break;
			}
		}
	}
}

void Jiajun_convertByte2UInt_fast_3b_args_avx2(size_t stepLength, unsigned char *byteArray, size_t byteArrayLength, unsigned int *intArray) {
#ifdef __AVX2__
    size_t n = 0; // Output index (in uints)
    size_t i = 0; // Input index (in bytes)
    const size_t simd_out_step = 64; // We produce 64 uints
    const size_t simd_in_step = 24;  // from 24 bytes (24*8/3=64)

    const __m256i mask_7 = _mm256_set1_epi32(0x7);

    // The loop must check if 32 bytes are available because _mm256_loadu_si256 reads 32 bytes,
    // even though we only logically process 24 of them.
    while (n + simd_out_step <= stepLength && i + 32 <= byteArrayLength) {
        // Step 1: Safely load 24 bytes onto the stack. This simplifies gathering.
        alignas(32) uint8_t bytes[32];
        memcpy(bytes, byteArray + i, 24);

        // Step 2: Build the source vectors directly. This is clear and correct.
        // Note: _mm256_set_epi32 takes arguments in reverse order (lane 7, 6, ..., 0).
        __m256i v_b0 = _mm256_set_epi32(bytes[21], bytes[18], bytes[15], bytes[12], bytes[9], bytes[6], bytes[3], bytes[0]);
        __m256i v_b1 = _mm256_set_epi32(bytes[22], bytes[19], bytes[16], bytes[13], bytes[10], bytes[7], bytes[4], bytes[1]);
        __m256i v_b2 = _mm256_set_epi32(bytes[23], bytes[20], bytes[17], bytes[14], bytes[11], bytes[8], bytes[5], bytes[2]);
        
        // Step 3: Calculate the 8 unpacked values for each of the 8 groups in parallel.
        // This logic directly mirrors the scalar implementation.
        __m256i res0 = _mm256_srli_epi32(v_b0, 5); // No mask needed here
        __m256i res1 = _mm256_and_si256(_mm256_srli_epi32(v_b0, 2), mask_7);
        
        __m256i tmp2_a = _mm256_slli_epi32(_mm256_and_si256(v_b0, _mm256_set1_epi32(0x3)), 1);
        __m256i tmp2_b = _mm256_srli_epi32(v_b1, 7);
        __m256i res2 = _mm256_or_si256(tmp2_a, tmp2_b);

        __m256i res3 = _mm256_and_si256(_mm256_srli_epi32(v_b1, 4), mask_7);
        __m256i res4 = _mm256_and_si256(_mm256_srli_epi32(v_b1, 1), mask_7);
        
        __m256i tmp5_a = _mm256_slli_epi32(_mm256_and_si256(v_b1, _mm256_set1_epi32(0x1)), 2);
        __m256i tmp5_b = _mm256_srli_epi32(v_b2, 6);
        __m256i res5 = _mm256_or_si256(tmp5_a, tmp5_b);

        __m256i res6 = _mm256_and_si256(_mm256_srli_epi32(v_b2, 3), mask_7);
        __m256i res7 = _mm256_and_si256(v_b2, mask_7);

        // Step 4: Transpose the 8x8 matrix of results to fix the order.
        // After this block, res0 will contain [int0, int1, ..., int7], res1 will contain [int8, ..., int15], etc.
        __m256i t0, t1, t2, t3, t4, t5, t6, t7;
        __m256i tt0, tt1, tt2, tt3, tt4, tt5, tt6, tt7;

        t0 = _mm256_unpacklo_epi32(res0, res1); t1 = _mm256_unpackhi_epi32(res0, res1);
        t2 = _mm256_unpacklo_epi32(res2, res3); t3 = _mm256_unpackhi_epi32(res2, res3);
        t4 = _mm256_unpacklo_epi32(res4, res5); t5 = _mm256_unpackhi_epi32(res4, res5);
        t6 = _mm256_unpacklo_epi32(res6, res7); t7 = _mm256_unpackhi_epi32(res6, res7);

        tt0 = _mm256_unpacklo_epi64(t0, t2); tt1 = _mm256_unpackhi_epi64(t0, t2);
        tt2 = _mm256_unpacklo_epi64(t1, t3); tt3 = _mm256_unpackhi_epi64(t1, t3);
        tt4 = _mm256_unpacklo_epi64(t4, t6); tt5 = _mm256_unpackhi_epi64(t4, t6);
        tt6 = _mm256_unpacklo_epi64(t5, t7); tt7 = _mm256_unpackhi_epi64(t5, t7);

        res0 = _mm256_permute2x128_si256(tt0, tt4, 0x20);
        res1 = _mm256_permute2x128_si256(tt1, tt5, 0x20);
        res2 = _mm256_permute2x128_si256(tt2, tt6, 0x20);
        res3 = _mm256_permute2x128_si256(tt3, tt7, 0x20);
        res4 = _mm256_permute2x128_si256(tt0, tt4, 0x31);
        res5 = _mm256_permute2x128_si256(tt1, tt5, 0x31);
        res6 = _mm256_permute2x128_si256(tt2, tt6, 0x31);
        res7 = _mm256_permute2x128_si256(tt3, tt7, 0x31);

        // Step 5: Store the correctly ordered results.
        _mm256_storeu_si256((__m256i*)(intArray + n + 0),  res0);
        _mm256_storeu_si256((__m256i*)(intArray + n + 8),  res1);
        _mm256_storeu_si256((__m256i*)(intArray + n + 16), res2);
        _mm256_storeu_si256((__m256i*)(intArray + n + 24), res3);
        _mm256_storeu_si256((__m256i*)(intArray + n + 32), res4);
        _mm256_storeu_si256((__m256i*)(intArray + n + 40), res5);
        _mm256_storeu_si256((__m256i*)(intArray + n + 48), res6);
        _mm256_storeu_si256((__m256i*)(intArray + n + 56), res7);

        n += simd_out_step;
        i += simd_in_step;
    }
    
    if (n < stepLength) {
        Jiajun_convertByte2UInt_fast_3b_args(stepLength - n, byteArray + i, byteArrayLength - i, intArray + n);
    }
#else
    Jiajun_convertByte2UInt_fast_3b_args(stepLength, byteArray, byteArrayLength, intArray);
#endif
}

void Jiajun_convertByte2UInt_fast_4b_args(size_t stepLength, unsigned char *byteArray, size_t byteArrayLength, unsigned int *intArray)
{
	size_t i = 0, n = 0;
	unsigned char tmp;
	for (i = 0; i < byteArrayLength; i++)
	{
		tmp = byteArray[i];
		intArray[n++] = (tmp & 0xF0) >> 4;
		if (n == stepLength)
			break;
		intArray[n++] = tmp & 0x0F;
	}
}

void Jiajun_convertByte2UInt_fast_4b_args_avx2(size_t stepLength, unsigned char *byteArray, size_t byteArrayLength, unsigned int *intArray)
{
#ifdef __AVX2__
    size_t i = 0; // input byte index
    size_t n = 0; // output uint index
    const size_t simd_in_step = 16;
    const size_t simd_out_step = 32;
    const __m128i mask_0F = _mm_set1_epi8(0x0F);

    while (i + simd_in_step <= byteArrayLength && n + simd_out_step <= stepLength) {
        __m128i packed_bytes = _mm_loadu_si128((__m128i*)(byteArray + i));
        __m128i high_nibbles = _mm_and_si128(_mm_srli_epi16(packed_bytes, 4), mask_0F);
        __m128i low_nibbles = _mm_and_si128(packed_bytes, mask_0F);
        __m128i interleaved_lo = _mm_unpacklo_epi8(high_nibbles, low_nibbles);
        __m128i interleaved_hi = _mm_unpackhi_epi8(high_nibbles, low_nibbles);
        _mm256_storeu_si256((__m256i*)(intArray + n + 0), _mm256_cvtepu8_epi32(interleaved_lo));
        _mm256_storeu_si256((__m256i*)(intArray + n + 8), _mm256_cvtepu8_epi32(_mm_srli_si128(interleaved_lo, 8)));
        _mm256_storeu_si256((__m256i*)(intArray + n + 16), _mm256_cvtepu8_epi32(interleaved_hi));
        _mm256_storeu_si256((__m256i*)(intArray + n + 24), _mm256_cvtepu8_epi32(_mm_srli_si128(interleaved_hi, 8)));
        i += simd_in_step;
        n += simd_out_step;
    }
    if (n < stepLength) {
        Jiajun_convertByte2UInt_fast_4b_args(stepLength - n, byteArray + i, byteArrayLength - i, intArray + n);
    }
#else
    // If __AVX2__ is not defined, just call the scalar version directly.
    Jiajun_convertByte2UInt_fast_4b_args(stepLength, byteArray, byteArrayLength, intArray);
#endif
}

void Jiajun_convertByte2UInt_fast_5b_args(size_t stepLength, unsigned char *byteArray, size_t byteArrayLength, unsigned int *intArray)
{
	size_t i = 0, n = 0;

	// Fast path for processing 8 elements (5 bytes) at a time
	while (n + 8 <= stepLength && i + 5 <= byteArrayLength) {
		unsigned char b0 = byteArray[i + 0];
		unsigned char b1 = byteArray[i + 1];
		unsigned char b2 = byteArray[i + 2];
		unsigned char b3 = byteArray[i + 3];
		unsigned char b4 = byteArray[i + 4];

		intArray[n + 0] = b0 >> 3;
		intArray[n + 1] = ((b0 & 0x07) << 2) | (b1 >> 6);
		intArray[n + 2] = (b1 >> 1) & 0x1F;
		intArray[n + 3] = ((b1 & 0x01) << 4) | (b2 >> 4);
		intArray[n + 4] = ((b2 & 0x0F) << 1) | (b3 >> 7);
		intArray[n + 5] = (b3 >> 2) & 0x1F;
		intArray[n + 6] = ((b3 & 0x03) << 3) | (b4 >> 5);
		intArray[n + 7] = b4 & 0x1F;
		
		n += 8;
		i += 5;
	}

	// Handle remaining elements
	if (n < stepLength) {
		if (i >= byteArrayLength) return; // Bounds check
		unsigned char tmp = byteArray[i];
		size_t ii = 0;
		for (; n < stepLength;)
		{
			switch (n % 8)
			{
			case 0:
				intArray[n++] = (tmp & 0xF8) >> 3;
				break;
			case 1:
				ii = (tmp & 0x07) << 2;
				i++;
				if (i >= byteArrayLength) { if(n < stepLength) intArray[n++] = ii; return; }
				tmp = byteArray[i];
				ii |= (tmp & 0xC0) >> 6;
				intArray[n++] = ii;
				break;
			case 2:
				intArray[n++] = (tmp & 0x3E) >> 1;
				break;
			case 3:
				ii = (tmp & 0x01) << 4;
				i++;
				if (i >= byteArrayLength) { if(n < stepLength) intArray[n++] = ii; return; }
				tmp = byteArray[i];
				ii |= (tmp & 0xF0) >> 4;
				intArray[n++] = ii;
				break;
			case 4:
				ii = (tmp & 0x0F) << 1;
				i++;
				if (i >= byteArrayLength) { if(n < stepLength) intArray[n++] = ii; return; }
				tmp = byteArray[i];
				ii |= (tmp & 0x80) >> 7;
				intArray[n++] = ii;
				break;
			case 5:
				intArray[n++] = (tmp & 0x7C) >> 2;
				break;
			case 6:
				ii = (tmp & 0x03) << 3;
				i++;
				if (i >= byteArrayLength) { if(n < stepLength) intArray[n++] = ii; return; }
				tmp = byteArray[i];
				ii |= (tmp & 0xE0) >> 5;
				intArray[n++] = ii;
				break;
			case 7:
				intArray[n++] = (tmp & 0x1F);
				i++;
				if (i < byteArrayLength) tmp = byteArray[i];
				break;
			}
		}
	}
}

void Jiajun_convertByte2UInt_fast_5b_args_avx2(size_t stepLength, unsigned char *byteArray, size_t byteArrayLength, unsigned int *intArray)
{
#ifdef __AVX2__
    size_t n = 0; // Output index (in uints)
    size_t i = 0; // Input index (in bytes)
    const size_t simd_out_step = 64; // We produce 64 uints per loop
    const size_t simd_in_step = 40;  // from 40 bytes (40*8 / 5 = 64)

    const __m256i mask_1f = _mm256_set1_epi32(0x1F); // Mask for 5 bits

    // The loop must check if 40 bytes are available for processing.
    while (n + simd_out_step <= stepLength && i + simd_in_step <= byteArrayLength) {
        // Step 1: Safely load 40 bytes onto the stack. This simplifies gathering.
        alignas(32) uint8_t bytes[40];
        memcpy(bytes, byteArray + i, 40);

        // Step 2: Build the source vectors directly from the stack array.
        // _mm256_set_epi32 takes arguments in reverse order (lane 7, 6, ..., 0).
        __m256i v_b0 = _mm256_set_epi32(bytes[35], bytes[30], bytes[25], bytes[20], bytes[15], bytes[10], bytes[5], bytes[0]);
        __m256i v_b1 = _mm256_set_epi32(bytes[36], bytes[31], bytes[26], bytes[21], bytes[16], bytes[11], bytes[6], bytes[1]);
        __m256i v_b2 = _mm256_set_epi32(bytes[37], bytes[32], bytes[27], bytes[22], bytes[17], bytes[12], bytes[7], bytes[2]);
        __m256i v_b3 = _mm256_set_epi32(bytes[38], bytes[33], bytes[28], bytes[23], bytes[18], bytes[13], bytes[8], bytes[3]);
        __m256i v_b4 = _mm256_set_epi32(bytes[39], bytes[34], bytes[29], bytes[24], bytes[19], bytes[14], bytes[9], bytes[4]);

        // Step 3: Calculate the 8 unpacked values for each of the 8 groups in parallel.
        // This logic directly mirrors the scalar implementation.
        __m256i res0 = _mm256_srli_epi32(v_b0, 3);

        __m256i tmp1_a = _mm256_slli_epi32(_mm256_and_si256(v_b0, _mm256_set1_epi32(0x07)), 2);
        __m256i tmp1_b = _mm256_srli_epi32(v_b1, 6);
        __m256i res1 = _mm256_or_si256(tmp1_a, tmp1_b);

        __m256i res2 = _mm256_and_si256(_mm256_srli_epi32(v_b1, 1), mask_1f);

        __m256i tmp3_a = _mm256_slli_epi32(_mm256_and_si256(v_b1, _mm256_set1_epi32(0x01)), 4);
        __m256i tmp3_b = _mm256_srli_epi32(v_b2, 4);
        __m256i res3 = _mm256_or_si256(tmp3_a, tmp3_b);

        __m256i tmp4_a = _mm256_slli_epi32(_mm256_and_si256(v_b2, _mm256_set1_epi32(0x0F)), 1);
        __m256i tmp4_b = _mm256_srli_epi32(v_b3, 7);
        __m256i res4 = _mm256_or_si256(tmp4_a, tmp4_b);

        __m256i res5 = _mm256_and_si256(_mm256_srli_epi32(v_b3, 2), mask_1f);

        __m256i tmp6_a = _mm256_slli_epi32(_mm256_and_si256(v_b3, _mm256_set1_epi32(0x03)), 3);
        __m256i tmp6_b = _mm256_srli_epi32(v_b4, 5);
        __m256i res6 = _mm256_or_si256(tmp6_a, tmp6_b);

        __m256i res7 = _mm256_and_si256(v_b4, mask_1f);

        // Step 4: Transpose the 8x8 matrix of results to fix the order.
        // This is a standard, efficient algorithm for transposing 8x8 32-bit elements in AVX2.
        __m256i t0, t1, t2, t3, t4, t5, t6, t7;
        __m256i tt0, tt1, tt2, tt3, tt4, tt5, tt6, tt7;

        t0 = _mm256_unpacklo_epi32(res0, res1); t1 = _mm256_unpackhi_epi32(res0, res1);
        t2 = _mm256_unpacklo_epi32(res2, res3); t3 = _mm256_unpackhi_epi32(res2, res3);
        t4 = _mm256_unpacklo_epi32(res4, res5); t5 = _mm256_unpackhi_epi32(res4, res5);
        t6 = _mm256_unpacklo_epi32(res6, res7); t7 = _mm256_unpackhi_epi32(res6, res7);

        tt0 = _mm256_unpacklo_epi64(t0, t2); tt1 = _mm256_unpackhi_epi64(t0, t2);
        tt2 = _mm256_unpacklo_epi64(t1, t3); tt3 = _mm256_unpackhi_epi64(t1, t3);
        tt4 = _mm256_unpacklo_epi64(t4, t6); tt5 = _mm256_unpackhi_epi64(t4, t6);
        tt6 = _mm256_unpacklo_epi64(t5, t7); tt7 = _mm256_unpackhi_epi64(t5, t7);

        res0 = _mm256_permute2x128_si256(tt0, tt4, 0x20);
        res1 = _mm256_permute2x128_si256(tt1, tt5, 0x20);
        res2 = _mm256_permute2x128_si256(tt2, tt6, 0x20);
        res3 = _mm256_permute2x128_si256(tt3, tt7, 0x20);
        res4 = _mm256_permute2x128_si256(tt0, tt4, 0x31);
        res5 = _mm256_permute2x128_si256(tt1, tt5, 0x31);
        res6 = _mm256_permute2x128_si256(tt2, tt6, 0x31);
        res7 = _mm256_permute2x128_si256(tt3, tt7, 0x31);

        // Step 5: Store the correctly ordered results.
        _mm256_storeu_si256((__m256i*)(intArray + n + 0),  res0);
        _mm256_storeu_si256((__m256i*)(intArray + n + 8),  res1);
        _mm256_storeu_si256((__m256i*)(intArray + n + 16), res2);
        _mm256_storeu_si256((__m256i*)(intArray + n + 24), res3);
        _mm256_storeu_si256((__m256i*)(intArray + n + 32), res4);
        _mm256_storeu_si256((__m256i*)(intArray + n + 40), res5);
        _mm256_storeu_si256((__m256i*)(intArray + n + 48), res6);
        _mm256_storeu_si256((__m256i*)(intArray + n + 56), res7);

        n += simd_out_step;
        i += simd_in_step;
    }

    // Handle the remainder with the correct and safe scalar function
    if (n < stepLength) {
        Jiajun_convertByte2UInt_fast_5b_args(stepLength - n, byteArray + i, byteArrayLength - i, intArray + n);
    }
#else
    // Fallback if AVX2 is not available
    Jiajun_convertByte2UInt_fast_5b_args(stepLength, byteArray, byteArrayLength, intArray);
#endif
}

void Jiajun_convertByte2UInt_fast_5b_args_avx2_slow(
    size_t stepLength,
    unsigned char *byteArray,
    size_t byteArrayLength,
    unsigned int *intArray)
{
#ifdef __AVX2__
    size_t i = 0, n = 0;

    // Process 32 values (8 groups of 5 bytes) per SIMD batch
    while (n + 32 <= stepLength && i + 20 <= byteArrayLength) {
        // Load 20 bytes (for 32 values) into AVX2 registers
        __m128i b0 = _mm_loadu_si128((__m128i const *)(byteArray + i));      // first 16 bytes
        __m128i b1 = _mm_loadl_epi64((__m128i const *)(byteArray + i + 16)); // next 4 bytes

        // Combine into one 256-bit vector (20 bytes used, rest garbage ignored)
        __m256i bytes = _mm256_castsi128_si256(b0);
        bytes = _mm256_inserti128_si256(bytes, b1, 1);

        // Store into aligned temp array for scalar extraction
        alignas(32) unsigned char tmp[32];
        _mm256_store_si256((__m256i*)tmp, bytes);

        // Decode 4 groups of 5 bytes -> 8 integers each = 32 integers total
        for (int blk = 0; blk < 4; blk++) {
            const unsigned char *in = tmp + blk * 5;
            unsigned int *out = intArray + n + blk * 8;

            out[0] = in[0] >> 3;
            out[1] = ((in[0] & 0x07) << 2) | (in[1] >> 6);
            out[2] = (in[1] >> 1) & 0x1F;
            out[3] = ((in[1] & 0x01) << 4) | (in[2] >> 4);
            out[4] = ((in[2] & 0x0F) << 1) | (in[3] >> 7);
            out[5] = (in[3] >> 2) & 0x1F;
            out[6] = ((in[3] & 0x03) << 3) | (in[4] >> 5);
            out[7] = in[4] & 0x1F;
        }

        n += 32;
        i += 20;
    }

    // Handle the remainder (non-multiple of 32 integers)
    if (n < stepLength) {
        Jiajun_convertByte2UInt_fast_5b_args(stepLength - n, (unsigned char*)(byteArray + i),
                                             byteArrayLength - i, intArray + n);
    }

#else
    // Fallback: scalar implementation
    Jiajun_convertByte2UInt_fast_5b_args(stepLength, (unsigned char*)byteArray,
                                         byteArrayLength, intArray);
#endif
}

void Jiajun_convertByte2UInt_fast_6b_args(size_t stepLength, unsigned char *byteArray, size_t byteArrayLength, unsigned int *intArray)
{
	size_t i = 0, n = 0;

	// Fast path for processing 4 elements (3 bytes) at a time
	while (n + 4 <= stepLength && i + 3 <= byteArrayLength) {
		unsigned char b0 = byteArray[i + 0];
		unsigned char b1 = byteArray[i + 1];
		unsigned char b2 = byteArray[i + 2];

		intArray[n + 0] = b0 >> 2;
		intArray[n + 1] = ((b0 & 0x03) << 4) | (b1 >> 4);
		intArray[n + 2] = ((b1 & 0x0F) << 2) | (b2 >> 6);
		intArray[n + 3] = b2 & 0x3F;
		
		n += 4;
		i += 3;
	}

	// Handle remaining elements
	if (n < stepLength) {
		if (i >= byteArrayLength) return; // Bounds check
		unsigned char tmp = byteArray[i];
		size_t ii = 0;
		for (; n < stepLength;)
		{
			switch (n % 4)
			{
			case 0:
				intArray[n++] = (tmp & 0xFC) >> 2;
				break;
			case 1:
				ii = (tmp & 0x03) << 4;
				i++;
				if (i >= byteArrayLength) { if(n < stepLength) intArray[n++] = ii; return; }
				tmp = byteArray[i];
				ii |= (tmp & 0xF0) >> 4;
				intArray[n++] = ii;
				break;
			case 2:
				ii = (tmp & 0x0F) << 2;
				i++;
				if (i >= byteArrayLength) { if(n < stepLength) intArray[n++] = ii; return; }
				tmp = byteArray[i];
				ii |= (tmp & 0xC0) >> 6;
				intArray[n++] = ii;
				break;
			case 3:
				intArray[n++] = (tmp & 0x3F);
				i++;
				if (i < byteArrayLength) tmp = byteArray[i];
				break;
			}
		}
	}
}

void Jiajun_convertByte2UInt_fast_6b_args_avx2(size_t stepLength, unsigned char *byteArray, size_t byteArrayLength, unsigned int *intArray)
{
#ifdef __AVX2__
    size_t i = 0, n = 0;
    const size_t simd_out_step = 32; // We produce 32 uints per loop
    const size_t simd_in_step = 24;  // from 24 bytes (24*8 / 6 = 32)

    // Pre-calculate masks
    const __m256i mask_3f = _mm256_set1_epi32(0x3F); // Mask for 6 bits
    const __m256i mask_0f = _mm256_set1_epi32(0x0F); // Mask for 4 bits
    const __m256i mask_03 = _mm256_set1_epi32(0x03); // Mask for 2 bits

    while (n + simd_out_step <= stepLength && i + simd_in_step <= byteArrayLength) {
        // Step 1: Load 24 bytes onto the stack.
        alignas(32) uint8_t bytes[24];
        memcpy(bytes, byteArray + i, 24);

        // Step 2: Build the source vectors. This part remains correct.
        __m256i v_b0 = _mm256_set_epi32(bytes[21], bytes[18], bytes[15], bytes[12], bytes[9], bytes[6], bytes[3], bytes[0]);
        __m256i v_b1 = _mm256_set_epi32(bytes[22], bytes[19], bytes[16], bytes[13], bytes[10], bytes[7], bytes[4], bytes[1]);
        __m256i v_b2 = _mm256_set_epi32(bytes[23], bytes[20], bytes[17], bytes[14], bytes[11], bytes[8], bytes[5], bytes[2]);

        // Step 3: Calculate the 4 unpacked values per group. This part remains correct.
        __m256i res0 = _mm256_srli_epi32(v_b0, 2);
        __m256i res1 = _mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(v_b0, mask_03), 4), _mm256_srli_epi32(v_b1, 4));
        __m256i res2 = _mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(v_b1, mask_0f), 2), _mm256_srli_epi32(v_b2, 6));
        __m256i res3 = _mm256_and_si256(v_b2, mask_3f);

        // --- Step 4: DEFINITIVELY CORRECTED Transpose logic for a 4x8 matrix ---
        // Intermediate unpacking stage 1: Unpack adjacent pairs
        __m256i t0 = _mm256_unpacklo_epi32(res0, res1); // [..., r1g1, r0g1, r1g0, r0g0] -> [..., int5, int4, int1, int0]
        __m256i t1 = _mm256_unpackhi_epi32(res0, res1); // High halves
        __m256i t2 = _mm256_unpacklo_epi32(res2, res3); // [..., r3g1, r2g1, r3g0, r2g0] -> [..., int7, int6, int3, int2]
        __m256i t3 = _mm256_unpackhi_epi32(res2, res3); // High halves

        // Intermediate unpacking stage 2: Unpack pairs of pairs to form quads
        __m256i o0 = _mm256_unpacklo_epi64(t0, t2); // [..., int3, int2, int1, int0]
        __m256i o1 = _mm256_unpackhi_epi64(t0, t2); // [..., int7, int6, int5, int4]
        __m256i o2 = _mm256_unpacklo_epi64(t1, t3); // [..., int11, int10, int9, int8]
        __m256i o3 = _mm256_unpackhi_epi64(t1, t3); // [..., int15, int14, int13, int12]
        
        // Final assembly of output vectors by combining 128-bit lanes
        __m256i out0 = _mm256_permute2x128_si256(o0, o1, 0x20); // [o1_lo | o0_lo] -> [int7..4 | int3..0]
        __m256i out1 = _mm256_permute2x128_si256(o2, o3, 0x20); // [o3_lo | o2_lo] -> [int15..12 | int11..8]
        __m256i out2 = _mm256_permute2x128_si256(o0, o1, 0x31); // [o1_hi | o0_hi] -> [int23..20 | int19..16]
        __m256i out3 = _mm256_permute2x128_si256(o2, o3, 0x31); // [o3_hi | o2_hi] -> [int31..28 | int27..24]

        // Step 5: Store the correctly ordered results.
        _mm256_storeu_si256((__m256i*)(intArray + n + 0),  out0);
        _mm256_storeu_si256((__m256i*)(intArray + n + 8),  out1);
        _mm256_storeu_si256((__m256i*)(intArray + n + 16), out2);
        _mm256_storeu_si256((__m256i*)(intArray + n + 24), out3);
        
        n += simd_out_step;
        i += simd_in_step;
    }

	if (n < stepLength) {
		Jiajun_convertByte2UInt_fast_6b_args(stepLength - n, byteArray + i, byteArrayLength - i, intArray + n);
	}
#else
	Jiajun_convertByte2UInt_fast_6b_args(stepLength, byteArray, byteArrayLength, intArray);
#endif
}

void Jiajun_convertByte2UInt_fast_7b_args(size_t stepLength,
unsigned char *byteArray,
size_t byteArrayLength,
unsigned int *intArray)
{
	size_t n = 0, i = 0;
        while (n + 8 <= stepLength) {
            unsigned char t0 = byteArray[i++];
            unsigned char t1 = byteArray[i++];
            unsigned char t2 = byteArray[i++];
            unsigned char t3 = byteArray[i++];
            unsigned char t4 = byteArray[i++];
            unsigned char t5 = byteArray[i++];
            unsigned char t6 = byteArray[i++];

            intArray[n++] = (t0 & 0xFE) >> 1;
            intArray[n++] = ((t0 & 0x01) << 6) | ((t1 & 0xFC) >> 2);
            intArray[n++] = ((t1 & 0x03) << 5) | ((t2 & 0xF8) >> 3);
            intArray[n++] = ((t2 & 0x07) << 4) | ((t3 & 0xF0) >> 4);
            intArray[n++] = ((t3 & 0x0F) << 3) | ((t4 & 0xE0) >> 5);
            intArray[n++] = ((t4 & 0x1F) << 2) | ((t5 & 0xC0) >> 6);
            intArray[n++] = ((t5 & 0x3F) << 1) | ((t6 & 0x80) >> 7);
            intArray[n++] = t6 & 0x7F;
        }
        // Handle remaining elements (if stepLength is not a multiple of 8)
        if (n < stepLength) {
	unsigned char tmp = byteArray[i];
	int k = 0;
            while (n < stepLength) {
                switch (k) {
		case 0:
			intArray[n++] = (tmp & 0xFE) >> 1;
			break;
		case 1:
			intArray[n++] = (tmp & 0x01) << 6;
			i++;
			tmp = byteArray[i];
			intArray[n - 1] |= (tmp & 0xFC) >> 2;
						break;
		case 2:
			intArray[n++] = (tmp & 0x03) << 5;
			i++;
			tmp = byteArray[i];
			intArray[n - 1] |= (tmp & 0xF8) >> 3;
						break;
		case 3:
			intArray[n++] = (tmp & 0x07) << 4;
			i++;
			tmp = byteArray[i];
			intArray[n - 1] |= (tmp & 0xF0) >> 4;
						break;
		case 4:
			intArray[n++] = (tmp & 0x0F) << 3;
			i++;
			tmp = byteArray[i];
			intArray[n - 1] |= (tmp & 0xE0) >> 5;
						break;
		case 5:
			intArray[n++] = (tmp & 0x1F) << 2;
			i++;
			tmp = byteArray[i];
			intArray[n - 1] |= (tmp & 0xC0) >> 6;
						break;
		case 6:
			intArray[n++] = (tmp & 0x3F) << 1;
			i++;
			tmp = byteArray[i];
			intArray[n - 1] |= (tmp & 0x80) >> 7;
						break;
		case 7:
			intArray[n++] = tmp & 0x7F;
			i++;
if (n < stepLength)
			tmp = byteArray[i];
			break;
}
                k = (k + 1) % 8;
		}
	}
}

void Jiajun_convertByte2UInt_fast_7b_args_avx2(size_t stepLength,
                                              unsigned char *byteArray,
                                              size_t byteArrayLength,
                                              unsigned int *intArray)
{
#ifdef __AVX2__
    size_t n = 0, i = 0;
    const size_t simd_out_step = 64; // We produce 64 uints per loop
    const size_t simd_in_step = 56;  // from 56 bytes (56*8 / 7 = 64)

    // Pre-calculate all necessary masks based on the scalar logic
    const __m256i mask_7f = _mm256_set1_epi32(0x7F);
    const __m256i mask_fe = _mm256_set1_epi32(0xFE);
    const __m256i mask_01 = _mm256_set1_epi32(0x01);
    const __m256i mask_fc = _mm256_set1_epi32(0xFC);
    const __m256i mask_03 = _mm256_set1_epi32(0x03);
    const __m256i mask_f8 = _mm256_set1_epi32(0xF8);
    const __m256i mask_07 = _mm256_set1_epi32(0x07);
    const __m256i mask_f0 = _mm256_set1_epi32(0xF0);
    const __m256i mask_0f = _mm256_set1_epi32(0x0F);
    const __m256i mask_e0 = _mm256_set1_epi32(0xE0);
    const __m256i mask_1f = _mm256_set1_epi32(0x1F);
    const __m256i mask_c0 = _mm256_set1_epi32(0xC0);
    const __m256i mask_3f = _mm256_set1_epi32(0x3F);
    const __m256i mask_80 = _mm256_set1_epi32(0x80);
    
    // The loop must check if 56 bytes are available for processing.
    while (n + simd_out_step <= stepLength && i + simd_in_step <= byteArrayLength) {
        // Step 1: Safely load 56 bytes onto the stack for easy gathering.
        alignas(64) uint8_t bytes[56];
        memcpy(bytes, byteArray + i, 56);

        // Step 2: Build the source vectors directly from the stack array.
        // _mm256_set_epi32 takes arguments in reverse order (lane 7, 6, ..., 0).
        __m256i v_b0 = _mm256_set_epi32(bytes[49], bytes[42], bytes[35], bytes[28], bytes[21], bytes[14], bytes[7], bytes[0]);
        __m256i v_b1 = _mm256_set_epi32(bytes[50], bytes[43], bytes[36], bytes[29], bytes[22], bytes[15], bytes[8], bytes[1]);
        __m256i v_b2 = _mm256_set_epi32(bytes[51], bytes[44], bytes[37], bytes[30], bytes[23], bytes[16], bytes[9], bytes[2]);
        __m256i v_b3 = _mm256_set_epi32(bytes[52], bytes[45], bytes[38], bytes[31], bytes[24], bytes[17], bytes[10], bytes[3]);
        __m256i v_b4 = _mm256_set_epi32(bytes[53], bytes[46], bytes[39], bytes[32], bytes[25], bytes[18], bytes[11], bytes[4]);
        __m256i v_b5 = _mm256_set_epi32(bytes[54], bytes[47], bytes[40], bytes[33], bytes[26], bytes[19], bytes[12], bytes[5]);
        __m256i v_b6 = _mm256_set_epi32(bytes[55], bytes[48], bytes[41], bytes[34], bytes[27], bytes[20], bytes[13], bytes[6]);

        // Step 3: Calculate the 8 unpacked values for each of the 8 groups in parallel.
        // This logic directly mirrors the scalar implementation.
        __m256i res0 = _mm256_srli_epi32(_mm256_and_si256(v_b0, mask_fe), 1);
        __m256i res1 = _mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(v_b0, mask_01), 6), _mm256_srli_epi32(_mm256_and_si256(v_b1, mask_fc), 2));
        __m256i res2 = _mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(v_b1, mask_03), 5), _mm256_srli_epi32(_mm256_and_si256(v_b2, mask_f8), 3));
        __m256i res3 = _mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(v_b2, mask_07), 4), _mm256_srli_epi32(_mm256_and_si256(v_b3, mask_f0), 4));
        __m256i res4 = _mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(v_b3, mask_0f), 3), _mm256_srli_epi32(_mm256_and_si256(v_b4, mask_e0), 5));
        __m256i res5 = _mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(v_b4, mask_1f), 2), _mm256_srli_epi32(_mm256_and_si256(v_b5, mask_c0), 6));
        __m256i res6 = _mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(v_b5, mask_3f), 1), _mm256_srli_epi32(_mm256_and_si256(v_b6, mask_80), 7));
        __m256i res7 = _mm256_and_si256(v_b6, mask_7f);

        // Step 4: Transpose the 8x8 matrix of results to fix the order.
        __m256i t0, t1, t2, t3, t4, t5, t6, t7;
        __m256i tt0, tt1, tt2, tt3, tt4, tt5, tt6, tt7;

        t0 = _mm256_unpacklo_epi32(res0, res1); t1 = _mm256_unpackhi_epi32(res0, res1);
        t2 = _mm256_unpacklo_epi32(res2, res3); t3 = _mm256_unpackhi_epi32(res2, res3);
        t4 = _mm256_unpacklo_epi32(res4, res5); t5 = _mm256_unpackhi_epi32(res4, res5);
        t6 = _mm256_unpacklo_epi32(res6, res7); t7 = _mm256_unpackhi_epi32(res6, res7);

        tt0 = _mm256_unpacklo_epi64(t0, t2); tt1 = _mm256_unpackhi_epi64(t0, t2);
        tt2 = _mm256_unpacklo_epi64(t1, t3); tt3 = _mm256_unpackhi_epi64(t1, t3);
        tt4 = _mm256_unpacklo_epi64(t4, t6); tt5 = _mm256_unpackhi_epi64(t4, t6);
        tt6 = _mm256_unpacklo_epi64(t5, t7); tt7 = _mm256_unpackhi_epi64(t5, t7);

        res0 = _mm256_permute2x128_si256(tt0, tt4, 0x20);
        res1 = _mm256_permute2x128_si256(tt1, tt5, 0x20);
        res2 = _mm256_permute2x128_si256(tt2, tt6, 0x20);
        res3 = _mm256_permute2x128_si256(tt3, tt7, 0x20);
        res4 = _mm256_permute2x128_si256(tt0, tt4, 0x31);
        res5 = _mm256_permute2x128_si256(tt1, tt5, 0x31);
        res6 = _mm256_permute2x128_si256(tt2, tt6, 0x31);
        res7 = _mm256_permute2x128_si256(tt3, tt7, 0x31);

        // Step 5: Store the correctly ordered results.
        _mm256_storeu_si256((__m256i*)(intArray + n + 0),  res0);
        _mm256_storeu_si256((__m256i*)(intArray + n + 8),  res1);
        _mm256_storeu_si256((__m256i*)(intArray + n + 16), res2);
        _mm256_storeu_si256((__m256i*)(intArray + n + 24), res3);
        _mm256_storeu_si256((__m256i*)(intArray + n + 32), res4);
        _mm256_storeu_si256((__m256i*)(intArray + n + 40), res5);
        _mm256_storeu_si256((__m256i*)(intArray + n + 48), res6);
        _mm256_storeu_si256((__m256i*)(intArray + n + 56), res7);

        n += simd_out_step;
        i += simd_in_step;
    }

    // Handle remaining elements with the scalar function
    if (n < stepLength) {
        Jiajun_convertByte2UInt_fast_7b_args(stepLength - n, byteArray + i, byteArrayLength - i, intArray + n);
    }
#else
    Jiajun_convertByte2UInt_fast_7b_args(stepLength, byteArray, byteArrayLength, intArray);
#endif
}

size_t convertIntArray2ByteArray_fast_1b_args(unsigned char *intArray,
size_t intArrayLength,
unsigned char *result)
{
	size_t byteLength = 0;
		if (intArrayLength % 8 == 0)
		byteLength = intArrayLength / 8;
	else
		byteLength = intArrayLength / 8 + 1;

	size_t n = 0;
	// Branchless version for all full bytes except possibly the last
        size_t i;
	for (i = 0; i + 1 < byteLength; i++) {
            result[i] = ((intArray[n + 0] & 1) << 7) | //
                        ((intArray[n + 1] & 1) << 6) | //
                        ((intArray[n + 2] & 1) << 5) | //
                        ((intArray[n + 3] & 1) << 4) | //
                        ((intArray[n + 4] & 1) << 3) | //
                        ((intArray[n + 5] & 1) << 2) | //
                        ((intArray[n + 6] & 1) << 1) | //
                        ((intArray[n + 7] & 1) << 0);  //
            n += 8;
        }
        // Handle the last (possibly partial) byte
        if (i < byteLength) {
            uint8_t tmp = 0;
            if (n < intArrayLength)
                tmp |= (intArray[n++] & 1) << 7;
            if (n < intArrayLength)
                tmp |= (intArray[n++] & 1) << 6;
            if (n < intArrayLength)
                tmp |= (intArray[n++] & 1) << 5;
            if (n < intArrayLength)
                tmp |= (intArray[n++] & 1) << 4;
            if (n < intArrayLength)
                tmp |= (intArray[n++] & 1) << 3;
            if (n < intArrayLength)
                tmp |= (intArray[n++] & 1) << 2;
            if (n < intArrayLength)
                tmp |= (intArray[n++] & 1) << 1;
            if (n < intArrayLength)
                tmp |= (intArray[n++] & 1) << 0;
            result[i] = tmp;
	}
	return byteLength;
}

size_t convertIntArray2ByteArray_fast_1b(unsigned char *intArray, size_t intArrayLength, unsigned char **result)
{
	size_t byteLength = 0;
	size_t i, j;
	if (intArrayLength % 8 == 0)
		byteLength = intArrayLength / 8;
	else
		byteLength = intArrayLength / 8 + 1;

	if (byteLength > 0)
		*result = (unsigned char *)malloc(byteLength * sizeof(unsigned char));
	else
		*result = NULL;
	size_t n = 0;
	int tmp, type;
	for (i = 0; i < byteLength; i++)
	{
		tmp = 0;
		for (j = 0; j < 8 && n < intArrayLength; j++)
		{
			type = intArray[n];
			if (type == 1)
				tmp = (tmp | (1 << (7 - j)));
			n++;
		}
		(*result)[i] = (unsigned char)tmp;
	}
	return byteLength;
}

size_t convertIntArray2ByteArray_fast_1b_to_result(unsigned char *intArray, size_t intArrayLength, unsigned char *result)
{
	size_t byteLength = 0;
	size_t i, j;
	if (intArrayLength % 8 == 0)
		byteLength = intArrayLength / 8;
	else
		byteLength = intArrayLength / 8 + 1;

	size_t n = 0;
	int tmp, type;
	for (i = 0; i < byteLength; i++)
	{
		tmp = 0;
		for (j = 0; j < 8 && n < intArrayLength; j++)
		{
			type = intArray[n];
			if (type == 1)
				tmp = (tmp | (1 << (7 - j)));
			n++;
		}
		result[i] = (unsigned char)tmp;
	}
	return byteLength;
}

void convertByteArray2IntArray_fast_1b_args(size_t intArrayLength, unsigned char *byteArray, size_t byteArrayLength, unsigned char *intArray)
{
	size_t n = 0, i;
	unsigned int tmp;
	for (i = 0; i < byteArrayLength - 1; i++)
	{
		tmp = byteArray[i];
		intArray[n++] = (tmp & 0x80) >> 7;
		intArray[n++] = (tmp & 0x40) >> 6;
		intArray[n++] = (tmp & 0x20) >> 5;
		intArray[n++] = (tmp & 0x10) >> 4;
		intArray[n++] = (tmp & 0x08) >> 3;
		intArray[n++] = (tmp & 0x04) >> 2;
		intArray[n++] = (tmp & 0x02) >> 1;
		intArray[n++] = (tmp & 0x01) >> 0;
	}

	tmp = byteArray[i];
	if (n == intArrayLength)
		return;
	intArray[n++] = (tmp & 0x80) >> 7;
	if (n == intArrayLength)
		return;
	intArray[n++] = (tmp & 0x40) >> 6;
	if (n == intArrayLength)
		return;
	intArray[n++] = (tmp & 0x20) >> 5;
	if (n == intArrayLength)
		return;
	intArray[n++] = (tmp & 0x10) >> 4;
	if (n == intArrayLength)
		return;
	intArray[n++] = (tmp & 0x08) >> 3;
	if (n == intArrayLength)
		return;
	intArray[n++] = (tmp & 0x04) >> 2;
	if (n == intArrayLength)
		return;
	intArray[n++] = (tmp & 0x02) >> 1;
	if (n == intArrayLength)
		return;
	intArray[n++] = (tmp & 0x01) >> 0;
}

void convertByteArray2IntArray_fast_1b(size_t intArrayLength, unsigned char *byteArray, size_t byteArrayLength, unsigned char **intArray)
{
	if (intArrayLength > byteArrayLength * 8)
	{
		printf("Error: intArrayLength > byteArrayLength*8\n");
		printf("intArrayLength=%zu, byteArrayLength = %zu", intArrayLength, byteArrayLength);
		exit(0);
	}
	if (intArrayLength > 0)
		*intArray = (unsigned char *)malloc(intArrayLength * sizeof(unsigned char));
	else
		*intArray = NULL;

	size_t n = 0, i;
	int tmp;
	for (i = 0; i < byteArrayLength - 1; i++)
	{
		tmp = byteArray[i];
		(*intArray)[n++] = (tmp & 0x80) >> 7;
		(*intArray)[n++] = (tmp & 0x40) >> 6;
		(*intArray)[n++] = (tmp & 0x20) >> 5;
		(*intArray)[n++] = (tmp & 0x10) >> 4;
		(*intArray)[n++] = (tmp & 0x08) >> 3;
		(*intArray)[n++] = (tmp & 0x04) >> 2;
		(*intArray)[n++] = (tmp & 0x02) >> 1;
		(*intArray)[n++] = (tmp & 0x01) >> 0;
	}

	tmp = byteArray[i];
	if (n == intArrayLength)
		return;
	(*intArray)[n++] = (tmp & 0x80) >> 7;
	if (n == intArrayLength)
		return;
	(*intArray)[n++] = (tmp & 0x40) >> 6;
	if (n == intArrayLength)
		return;
	(*intArray)[n++] = (tmp & 0x20) >> 5;
	if (n == intArrayLength)
		return;
	(*intArray)[n++] = (tmp & 0x10) >> 4;
	if (n == intArrayLength)
		return;
	(*intArray)[n++] = (tmp & 0x08) >> 3;
	if (n == intArrayLength)
		return;
	(*intArray)[n++] = (tmp & 0x04) >> 2;
	if (n == intArrayLength)
		return;
	(*intArray)[n++] = (tmp & 0x02) >> 1;
	if (n == intArrayLength)
		return;
	(*intArray)[n++] = (tmp & 0x01) >> 0;
}

inline size_t convertIntArray2ByteArray_fast_2b_args(unsigned char *timeStepType, size_t timeStepTypeLength, unsigned char *result)
{
	register unsigned char tmp = 0;
	size_t i, byteLength = 0;
	if (timeStepTypeLength % 4 == 0)
		byteLength = timeStepTypeLength * 2 / 8;
	else
		byteLength = timeStepTypeLength * 2 / 8 + 1;
	size_t n = 0;
	if (timeStepTypeLength % 4 == 0)
	{
		for (i = 0; i < byteLength; i++)
		{
			tmp = 0;

			tmp |= timeStepType[n++] << 6;
			tmp |= timeStepType[n++] << 4;
			tmp |= timeStepType[n++] << 2;
			tmp |= timeStepType[n++];

			

			result[i] = tmp;
		}
	}
	else
	{
		size_t byteLength_ = byteLength - 1;
		for (i = 0; i < byteLength_; i++)
		{
			tmp = 0;

			tmp |= timeStepType[n++] << 6;
			tmp |= timeStepType[n++] << 4;
			tmp |= timeStepType[n++] << 2;
			tmp |= timeStepType[n++];

			

			result[i] = tmp;
		}
		tmp = 0;
		int mod4 = timeStepTypeLength % 4;
		for (int j = 0; j < mod4; j++)
		{
			unsigned char type = timeStepType[n++];
			tmp = tmp | type << (6 - (j << 1));
		}
		result[i] = tmp;
	}

	
	return byteLength;
}

/**
 * little endian
 * [01|10|11|00|....]-->[01|10|11|00][....]
 * @param timeStepType
 * @return
 */
size_t convertIntArray2ByteArray_fast_2b(unsigned char *timeStepType, size_t timeStepTypeLength, unsigned char **result)
{
	size_t i, j, byteLength = 0;
	if (timeStepTypeLength % 4 == 0)
		byteLength = timeStepTypeLength * 2 / 8;
	else
		byteLength = timeStepTypeLength * 2 / 8 + 1;
	if (byteLength > 0)
		*result = (unsigned char *)malloc(byteLength * sizeof(unsigned char));
	else
		*result = NULL;
	size_t n = 0;
	for (i = 0; i < byteLength; i++)
	{
		int tmp = 0;
		for (j = 0; j < 4 && n < timeStepTypeLength; j++)
		{
			int type = timeStepType[n];
			switch (type)
			{
			case 0:

				break;
			case 1:
				tmp = (tmp | (1 << (6 - j * 2)));
				break;
			case 2:
				tmp = (tmp | (2 << (6 - j * 2)));
				break;
			case 3:
				tmp = (tmp | (3 << (6 - j * 2)));
				break;
			default:
				printf("Error: wrong timestep type...: type[%zu]=%d\n", n, type);
				exit(0);
			}
			n++;
		}
		(*result)[i] = (unsigned char)tmp;
	}
	return byteLength;
}

void convertByteArray2IntArray_fast_2b(size_t stepLength, unsigned char *byteArray, size_t byteArrayLength, unsigned char **intArray)
{
	if (stepLength > byteArrayLength * 4)
	{
		printf("Error: stepLength > byteArray.length*4\n");
		printf("stepLength=%zu, byteArray.length=%zu\n", stepLength, byteArrayLength);
		exit(0);
	}
	if (stepLength > 0)
		*intArray = (unsigned char *)malloc(stepLength * sizeof(unsigned char));
	else
		*intArray = NULL;
	size_t i, n = 0;

	int mod4 = stepLength % 4;
	if (mod4 == 0)
	{
		for (i = 0; i < byteArrayLength; i++)
		{
			unsigned char tmp = byteArray[i];
			(*intArray)[n++] = (tmp & 0xC0) >> 6;
			(*intArray)[n++] = (tmp & 0x30) >> 4;
			(*intArray)[n++] = (tmp & 0x0C) >> 2;
			(*intArray)[n++] = tmp & 0x03;
		}
	}
	else
	{
		size_t t = byteArrayLength - 1;
		for (i = 0; i < t; i++)
		{
			unsigned char tmp = byteArray[i];
			(*intArray)[n++] = (tmp & 0xC0) >> 6;
			(*intArray)[n++] = (tmp & 0x30) >> 4;
			(*intArray)[n++] = (tmp & 0x0C) >> 2;
			(*intArray)[n++] = tmp & 0x03;
		}
		unsigned char tmp = byteArray[i];
		switch (mod4)
		{
		case 1:
			(*intArray)[n++] = (tmp & 0xC0) >> 6;
			break;
		case 2:
			(*intArray)[n++] = (tmp & 0xC0) >> 6;
			(*intArray)[n++] = (tmp & 0x30) >> 4;
			break;
		case 3:
			(*intArray)[n++] = (tmp & 0xC0) >> 6;
			(*intArray)[n++] = (tmp & 0x30) >> 4;
			(*intArray)[n++] = (tmp & 0x0C) >> 2;
			break;
		}
	}
}

size_t convertIntArray2ByteArray_fast_3b(unsigned char *timeStepType, size_t timeStepTypeLength, unsigned char **result)
{
	size_t i = 0, k = 0, byteLength = 0, n = 0;
	if (timeStepTypeLength % 8 == 0)
		byteLength = timeStepTypeLength * 3 / 8;
	else
		byteLength = timeStepTypeLength * 3 / 8 + 1;

	if (byteLength > 0)
		*result = (unsigned char *)malloc(byteLength * sizeof(unsigned char));
	else
		*result = NULL;
	unsigned char tmp = 0;
	for (n = 0; n < timeStepTypeLength; n++)
	{
		k = n % 8;
		switch (k)
		{
		case 0:
			tmp = tmp | (timeStepType[n] << 5);
			break;
		case 1:
			tmp = tmp | (timeStepType[n] << 2);
			break;
		case 2:
			tmp = tmp | (timeStepType[n] >> 1);
			(*result)[i++] = tmp;
			tmp = 0 | (timeStepType[n] << 7);
			break;
		case 3:
			tmp = tmp | (timeStepType[n] << 4);
			break;
		case 4:
			tmp = tmp | (timeStepType[n] << 1);
			break;
		case 5:
			tmp = tmp | (timeStepType[n] >> 2);
			(*result)[i++] = tmp;
			tmp = 0 | (timeStepType[n] << 6);
			break;
		case 6:
			tmp = tmp | (timeStepType[n] << 3);
			break;
		case 7:
			tmp = tmp | (timeStepType[n] << 0);
			(*result)[i++] = tmp;
			tmp = 0;
			break;
		}
	}
	if (k != 7) 
		(*result)[i] = tmp;

	return byteLength;
}

void convertByteArray2IntArray_fast_3b(size_t stepLength, unsigned char *byteArray, size_t byteArrayLength, unsigned char **intArray)
{
	if (stepLength > byteArrayLength * 8 / 3)
	{
		printf("Error: stepLength > byteArray.length*8/3, impossible case unless bugs elsewhere.\n");
		printf("stepLength=%zu, byteArray.length=%zu\n", stepLength, byteArrayLength);
		exit(0);
	}
	if (stepLength > 0)
		*intArray = (unsigned char *)malloc(stepLength * sizeof(unsigned char));
	else
		*intArray = NULL;
	size_t i = 0, ii = 0, n = 0;
	unsigned char tmp = byteArray[i];
	for (n = 0; n < stepLength;)
	{
		switch (n % 8)
		{
		case 0:
			(*intArray)[n++] = (tmp & 0xE0) >> 5;
			break;
		case 1:
			(*intArray)[n++] = (tmp & 0x1C) >> 2;
			break;
		case 2:
			ii = (tmp & 0x03) << 1;
			i++;
			tmp = byteArray[i];
			ii |= (tmp & 0x80) >> 7;
			(*intArray)[n++] = ii;
			break;
		case 3:
			(*intArray)[n++] = (tmp & 0x70) >> 4;
			break;
		case 4:
			(*intArray)[n++] = (tmp & 0x0E) >> 1;
			break;
		case 5:
			ii = (tmp & 0x01) << 2;
			i++;
			tmp = byteArray[i];
			ii |= (tmp & 0xC0) >> 6;
			(*intArray)[n++] = ii;
			break;
		case 6:
			(*intArray)[n++] = (tmp & 0x38) >> 3;
			break;
		case 7:
			(*intArray)[n++] = (tmp & 0x07);
			i++;
			tmp = byteArray[i];
			break;
		}
	}
}

size_t convertIntArray2ByteArray_fast_4b(unsigned char *timeStepType, size_t timeStepTypeLength, unsigned char **result)
{
	size_t i = 0, byteLength = 0, n = 0;
	if (timeStepTypeLength % 2 == 0)
		byteLength = timeStepTypeLength * 4 / 8;
	else
		byteLength = timeStepTypeLength * 4 / 8 + 1;

	if (byteLength > 0)
		*result = (unsigned char *)malloc(byteLength * sizeof(unsigned char));
	else
		*result = NULL;
	for (n = 0; n < timeStepTypeLength; n++)
	{
		unsigned char tmp = 0;
		for (int j = 0; j < 2 && n < timeStepTypeLength; j++)
		{
			int type = timeStepType[n];
			if (j == 0)
				tmp = tmp | (type << 4);
			else 
				tmp = tmp | type;
			n++;
		}
		(*result)[i++] = tmp;
	}
	return byteLength;
}

void convertByteArray2IntArray_fast_4b(size_t stepLength, unsigned char *byteArray, size_t byteArrayLength, unsigned char **intArray)
{
	if (stepLength > byteArrayLength * 2)
	{
		printf("Error: stepLength > byteArray.length*2, impossible case unless bugs elsewhere.\n");
		printf("stepLength=%zu, byteArray.length=%zu\n", stepLength, byteArrayLength);
		exit(0);
	}
	if (stepLength > 0)
		*intArray = (unsigned char *)malloc(stepLength * sizeof(unsigned char));
	else
		*intArray = NULL;
	size_t i = 0, n = 0;
	unsigned char tmp;
	for (i = 0; i < byteArrayLength; i++)
	{
		tmp = byteArray[i];
		(*intArray)[n++] = (tmp & 0xF0) >> 4;
		if (n == stepLength)
			break;
		(*intArray)[n++] = tmp & 0x0F;
	}
}

size_t convertIntArray2ByteArray_fast_5b(unsigned char *timeStepType, size_t timeStepTypeLength, unsigned char **result)
{
	size_t i = 0, k = 0, byteLength = 0, n = 0;
	if (timeStepTypeLength % 8 == 0)
		byteLength = timeStepTypeLength * 5 / 8;
	else
		byteLength = timeStepTypeLength * 5 / 8 + 1;

	if (byteLength > 0)
		*result = (unsigned char *)malloc(byteLength * sizeof(unsigned char));
	else
		*result = NULL;
	unsigned char tmp = 0;
	for (n = 0; n < timeStepTypeLength; n++)
	{
		k = n % 8;
		switch (k)
		{
		case 0:
			tmp = tmp | (timeStepType[n] << 3);
			break;
		case 1:
			tmp = tmp | (timeStepType[n] >> 2);
			(*result)[i++] = tmp;
			tmp = 0 | (timeStepType[n] << 6);
			break;
		case 2:
			tmp = tmp | (timeStepType[n] << 1);
			break;
		case 3:
			tmp = tmp | (timeStepType[n] >> 4);
			(*result)[i++] = tmp;
			tmp = 0 | (timeStepType[n] << 4);
			break;
		case 4:
			tmp = tmp | (timeStepType[n] >> 1);
			(*result)[i++] = tmp;
			tmp = 0 | (timeStepType[n] << 7);
			break;
		case 5:
			tmp = tmp | (timeStepType[n] << 2);
			break;
		case 6:
			tmp = tmp | (timeStepType[n] >> 3);
			(*result)[i++] = tmp;
			tmp = 0 | (timeStepType[n] << 5);
			break;
		case 7:
			tmp = tmp | (timeStepType[n] << 0);
			(*result)[i++] = tmp;
			tmp = 0;
			break;
		}
	}
	if (k != 7) 
		(*result)[i] = tmp;

	return byteLength;
}

void convertByteArray2IntArray_fast_5b(size_t stepLength, unsigned char *byteArray, size_t byteArrayLength, unsigned char **intArray)
{
	if (stepLength > byteArrayLength * 8 / 5)
	{
		printf("Error: stepLength > byteArray.length*8/5, impossible case unless bugs elsewhere.\n");
		printf("stepLength=%zu, byteArray.length=%zu\n", stepLength, byteArrayLength);
		exit(0);
	}
	if (stepLength > 0)
		*intArray = (unsigned char *)malloc(stepLength * sizeof(unsigned char));
	else
		*intArray = NULL;
	size_t i = 0, ii = 0, n = 0;
	unsigned char tmp = byteArray[i];
	for (n = 0; n < stepLength;)
	{
		switch (n % 8)
		{
		case 0:
			(*intArray)[n++] = (tmp & 0xF8) >> 3;
			break;
		case 1:
			ii = (tmp & 0x07) << 2;
			i++;
			tmp = byteArray[i];
			ii |= (tmp & 0xC0) >> 6;
			(*intArray)[n++] = ii;
			break;
		case 2:
			(*intArray)[n++] = (tmp & 0x3E) >> 1;
			break;
		case 3:
			ii = (tmp & 0x01) << 4;
			i++;
			tmp = byteArray[i];
			ii |= (tmp & 0xF0) >> 4;
			(*intArray)[n++] = ii;
			break;
		case 4:
			ii = (tmp & 0x0F) << 1;
			i++;
			tmp = byteArray[i];
			ii |= (tmp & 0x80) >> 7;
			(*intArray)[n++] = ii;
			break;
		case 5:
			(*intArray)[n++] = (tmp & 0x7C) >> 2;
			break;
		case 6:
			ii = (tmp & 0x03) << 3;
			i++;
			tmp = byteArray[i];
			ii |= (tmp & 0xE0) >> 5;
			(*intArray)[n++] = ii;
			break;
		case 7:
			(*intArray)[n++] = (tmp & 0x1F);
			i++;
			tmp = byteArray[i];
			break;
		}
	}
}

size_t convertIntArray2ByteArray_fast_6b(unsigned char *timeStepType, size_t timeStepTypeLength, unsigned char **result)
{
	size_t i = 0, k = 0, byteLength = 0, n = 0;
	if (timeStepTypeLength % 8 == 0)
		byteLength = timeStepTypeLength * 6 / 8;
	else
		byteLength = timeStepTypeLength * 6 / 8 + 1;

	if (byteLength > 0)
		*result = (unsigned char *)malloc(byteLength * sizeof(unsigned char));
	else
		*result = NULL;
	unsigned char tmp = 0;
	for (n = 0; n < timeStepTypeLength; n++)
	{
		k = n % 4;
		switch (k)
		{
		case 0:
			tmp = tmp | (timeStepType[n] << 2);
			break;
		case 1:
			tmp = tmp | (timeStepType[n] >> 4);
			(*result)[i++] = tmp;
			tmp = 0 | (timeStepType[n] << 4);
			break;
		case 2:
			tmp = tmp | (timeStepType[n] >> 2);
			(*result)[i++] = tmp;
			tmp = 0 | (timeStepType[n] << 6);
			break;
		case 3:
			tmp = tmp | (timeStepType[n] << 0);
			(*result)[i++] = tmp;
			tmp = 0;
			break;
		}
	}
	if (k != 3) 
		(*result)[i] = tmp;

	return byteLength;
}

void convertByteArray2IntArray_fast_6b(size_t stepLength, unsigned char *byteArray, size_t byteArrayLength, unsigned char **intArray)
{
	if (stepLength > byteArrayLength * 8 / 6)
	{
		printf("Error: stepLength > byteArray.length*8/6, impossible case unless bugs elsewhere.\n");
		printf("stepLength=%zu, byteArray.length=%zu\n", stepLength, byteArrayLength);
		exit(0);
	}
	if (stepLength > 0)
		*intArray = (unsigned char *)malloc(stepLength * sizeof(unsigned char));
	else
		*intArray = NULL;
	size_t i = 0, ii = 0, n = 0;
	unsigned char tmp = byteArray[i];
	for (n = 0; n < stepLength;)
	{
		switch (n % 4)
		{
		case 0:
			(*intArray)[n++] = (tmp & 0xFC) >> 2;
			break;
		case 1:
			ii = (tmp & 0x03) << 4;
			i++;
			tmp = byteArray[i];
			ii |= (tmp & 0xF0) >> 4;
			(*intArray)[n++] = ii;
			break;
		case 2:
			ii = (tmp & 0x0F) << 2;
			i++;
			tmp = byteArray[i];
			ii |= (tmp & 0xC0) >> 6;
			(*intArray)[n++] = ii;
			break;
		case 3:
			(*intArray)[n++] = (tmp & 0x3F);
			i++;
			tmp = byteArray[i];
			break;
		}
	}
}

size_t convertIntArray2ByteArray_fast_7b(unsigned char *timeStepType, size_t timeStepTypeLength, unsigned char **result)
{
	size_t i = 0, k = 0, byteLength = 0, n = 0;
	if (timeStepTypeLength % 8 == 0)
		byteLength = timeStepTypeLength * 7 / 8;
	else
		byteLength = timeStepTypeLength * 7 / 8 + 1;

	if (byteLength > 0)
		*result = (unsigned char *)malloc(byteLength * sizeof(unsigned char));
	else
		*result = NULL;
	unsigned char tmp = 0;
	for (n = 0; n < timeStepTypeLength; n++)
	{
		k = n % 8;
		switch (k)
		{
		case 0:
			tmp = tmp | (timeStepType[n] << 1);
			break;
		case 1:
			tmp = tmp | (timeStepType[n] >> 6);
			(*result)[i++] = tmp;
			tmp = 0 | (timeStepType[n] << 2);
			break;
		case 2:
			tmp = tmp | (timeStepType[n] >> 5);
			(*result)[i++] = tmp;
			tmp = 0 | (timeStepType[n] << 3);
			break;
		case 3:
			tmp = tmp | (timeStepType[n] >> 4);
			(*result)[i++] = tmp;
			tmp = 0 | (timeStepType[n] << 4);
			break;
		case 4:
			tmp = tmp | (timeStepType[n] >> 3);
			(*result)[i++] = tmp;
			tmp = 0 | (timeStepType[n] << 5);
			break;
		case 5:
			tmp = tmp | (timeStepType[n] >> 2);
			(*result)[i++] = tmp;
			tmp = 0 | (timeStepType[n] << 6);
			break;
		case 6:
			tmp = tmp | (timeStepType[n] >> 1);
			(*result)[i++] = tmp;
			tmp = 0 | (timeStepType[n] << 7);
			break;
		case 7:
			tmp = tmp | (timeStepType[n] << 0);
			(*result)[i++] = tmp;
			tmp = 0;
			break;
		}
	}
	if (k != 7) 
		(*result)[i] = tmp;

	return byteLength;
}

void convertByteArray2IntArray_fast_7b(size_t stepLength, unsigned char *byteArray, size_t byteArrayLength, unsigned char **intArray)
{
	if (stepLength > byteArrayLength * 8 / 7)
	{
		printf("Error: stepLength > byteArray.length*8/7, impossible case unless bugs elsewhere.\n");
		printf("stepLength=%zu, byteArray.length=%zu\n", stepLength, byteArrayLength);
		exit(0);
	}
	if (stepLength > 0)
		*intArray = (unsigned char *)malloc(stepLength * sizeof(unsigned char));
	else
		*intArray = NULL;
	size_t i = 0, ii = 0, n = 0;
	unsigned char tmp = byteArray[i];
	for (n = 0; n < stepLength;)
	{
		switch (n % 8)
		{
		case 0:
			(*intArray)[n++] = (tmp & 0xFE) >> 1;
			break;
		case 1:
			ii = (tmp & 0x01) << 6;
			i++;
			tmp = byteArray[i];
			ii |= (tmp & 0xFC) >> 2;
			(*intArray)[n++] = ii;
			break;
		case 2:
			ii = (tmp & 0x03) << 5;
			i++;
			tmp = byteArray[i];
			ii |= (tmp & 0xF8) >> 3;
			(*intArray)[n++] = ii;
			break;
		case 3:
			ii = (tmp & 0x07) << 4;
			i++;
			tmp = byteArray[i];
			ii |= (tmp & 0xF0) >> 4;
			(*intArray)[n++] = ii;
			break;
		case 4:
			ii = (tmp & 0x0F) << 3;
			i++;
			tmp = byteArray[i];
			ii |= (tmp & 0xE0) >> 5;
			(*intArray)[n++] = ii;
			break;
		case 5:
			ii = (tmp & 0x1F) << 2;
			i++;
			tmp = byteArray[i];
			ii |= (tmp & 0xC0) >> 6;
			(*intArray)[n++] = ii;
			break;
		case 6:
			ii = (tmp & 0x3F) << 1;
			i++;
			tmp = byteArray[i];
			ii |= (tmp & 0x80) >> 7;
			(*intArray)[n++] = ii;
			break;
		case 7:
			(*intArray)[n++] = (tmp & 0x7F);
			i++;
			tmp = byteArray[i];
			break;
		}
	}
}

inline int getLeftMovingSteps(size_t k, unsigned char resiBitLength)
{
	return 8 - k % 8 - resiBitLength;
}

}
