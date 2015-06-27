#include <iostream>
#include <algorithm>
#include <chrono>
#include <cassert>
#include <immintrin.h>
#include <omp.h>
#include "avx_impl.h"
#include "../common/x86.h"
#include "../common/aligned_buffer.h"

namespace waifu2x {

namespace {

static const int BLOCK_WIDTH = 64;
static const int BLOCK_HEIGHT = 16;

template <bool ENABLE_FMA>
inline __m256 madd(__m256 a, __m256 b, __m256 c){
	return _mm256_add_ps(_mm256_mul_ps(a, b), c);
}
template <>
inline __m256 madd<true>(__m256 a, __m256 b, __m256 c){
	return _mm256_fmadd_ps(a, b, c);
}

template <bool ENABLE_AVX2>
inline __m256 alignr4(__m256 a, __m256 b){
	return _mm256_blend_ps(
		_mm256_permute_ps(b, 0x39), _mm256_permute_ps(a, 0x39), 0x88);
}
template <>
inline __m256 alignr4<true>(__m256 a, __m256 b){
	return _mm256_castsi256_ps(_mm256_alignr_epi8(
		_mm256_castps_si256(a), _mm256_castps_si256(b), 4));
}

template <bool ENABLE_AVX2>
inline __m256 alignr8(__m256 a, __m256 b){
	return _mm256_blend_ps(
		_mm256_permute_ps(b, 0x4e), _mm256_permute_ps(a, 0x4e), 0xcc);
}
template <>
inline __m256 alignr8<true>(__m256 a, __m256 b){
	return _mm256_castsi256_ps(_mm256_alignr_epi8(
		_mm256_castps_si256(a), _mm256_castps_si256(b), 8));
}

template <bool ENABLE_FMA, bool ENABLE_AVX2>
inline void convolute_add(
	float *dst, const float *src, int width, int height, int pitch,
	const float *kernel)
{
	const __m256 k0 = _mm256_broadcast_ss(kernel + 0);
	const __m256 k1 = _mm256_broadcast_ss(kernel + 1);
	const __m256 k2 = _mm256_broadcast_ss(kernel + 2);
	const __m256 k3 = _mm256_broadcast_ss(kernel + 3);
	const __m256 k4 = _mm256_broadcast_ss(kernel + 4);
	const __m256 k5 = _mm256_broadcast_ss(kernel + 5);
	const __m256 k6 = _mm256_broadcast_ss(kernel + 6);
	const __m256 k7 = _mm256_broadcast_ss(kernel + 7);
	const __m256 k8 = _mm256_broadcast_ss(kernel + 8);
	for(int y = 0; y < height; y += 4){
		const float *line0 = src + y * pitch;
		const float *line1 = src + (y + 1) * pitch;
		const float *line2 = src + (y + 2) * pitch;
		const float *line3 = src + (y + 3) * pitch;
		const float *line4 = src + (y + 4) * pitch;
		const float *line5 = src + (y + 5) * pitch;
		float *dline1 = dst + y * pitch;
		float *dline2 = dst + (y + 1) * pitch;
		float *dline3 = dst + (y + 2) * pitch;
		float *dline4 = dst + (y + 3) * pitch;
		__m256 cur0 = _mm256_load_ps(line0);
		__m256 cur1 = _mm256_load_ps(line1);
		__m256 cur2 = _mm256_load_ps(line2);
		__m256 cur3 = _mm256_load_ps(line3);
		__m256 cur4 = _mm256_load_ps(line4);
		__m256 cur5 = _mm256_load_ps(line5);
		for(int x = 0; x < width; x += 8){
			__m256 sum1 = _mm256_load_ps(dline1 + x);
			__m256 sum2 = _mm256_load_ps(dline2 + x);
			__m256 sum3 = _mm256_load_ps(dline3 + x);
			__m256 sum4 = _mm256_load_ps(dline4 + x);
			{ // line0
				const __m256 next = _mm256_load_ps(line0 + x + 8);
				sum1 = madd<ENABLE_FMA>(cur0, k0, sum1);
				const __m256 temp = _mm256_permute2f128_ps(cur0, next, 0x21);
				const __m256 shift1 = alignr4<ENABLE_AVX2>(temp, cur0);
				sum1 = madd<ENABLE_FMA>(shift1, k1, sum1);
				const __m256 shift2 = alignr8<ENABLE_AVX2>(temp, cur0);
				sum1 = madd<ENABLE_FMA>(shift2, k2, sum1);
				cur0 = next;
			}
			{ // line1
				const __m256 next = _mm256_load_ps(line1 + x + 8);
				sum1 = madd<ENABLE_FMA>(cur1, k3, sum1);
				sum2 = madd<ENABLE_FMA>(cur1, k0, sum2);
				const __m256 temp = _mm256_permute2f128_ps(cur1, next, 0x21);
				const __m256 shift1 = alignr4<ENABLE_AVX2>(temp, cur1);
				sum1 = madd<ENABLE_FMA>(shift1, k4, sum1);
				sum2 = madd<ENABLE_FMA>(shift1, k1, sum2);
				const __m256 shift2 = alignr8<ENABLE_AVX2>(temp, cur1);
				sum1 = madd<ENABLE_FMA>(shift2, k5, sum1);
				sum2 = madd<ENABLE_FMA>(shift2, k2, sum2);
				cur1 = next;
			}
			{ // line2
				const __m256 next = _mm256_load_ps(line2 + x + 8);
				sum1 = madd<ENABLE_FMA>(cur2, k6, sum1);
				sum2 = madd<ENABLE_FMA>(cur2, k3, sum2);
				sum3 = madd<ENABLE_FMA>(cur2, k0, sum3);
				const __m256 temp = _mm256_permute2f128_ps(cur2, next, 0x21);
				const __m256 shift1 = alignr4<ENABLE_AVX2>(temp, cur2);
				sum1 = madd<ENABLE_FMA>(shift1, k7, sum1);
				sum2 = madd<ENABLE_FMA>(shift1, k4, sum2);
				sum3 = madd<ENABLE_FMA>(shift1, k1, sum3);
				const __m256 shift2 = alignr8<ENABLE_AVX2>(temp, cur2);
				sum1 = madd<ENABLE_FMA>(shift2, k8, sum1);
				sum2 = madd<ENABLE_FMA>(shift2, k5, sum2);
				sum3 = madd<ENABLE_FMA>(shift2, k2, sum3);
				cur2 = next;
			}
			{ // line3
				const __m256 next = _mm256_load_ps(line3 + x + 8);
				sum2 = madd<ENABLE_FMA>(cur3, k6, sum2);
				sum3 = madd<ENABLE_FMA>(cur3, k3, sum3);
				sum4 = madd<ENABLE_FMA>(cur3, k0, sum4);
				const __m256 temp = _mm256_permute2f128_ps(cur3, next, 0x21);
				const __m256 shift1 = alignr4<ENABLE_AVX2>(temp, cur3);
				sum2 = madd<ENABLE_FMA>(shift1, k7, sum2);
				sum3 = madd<ENABLE_FMA>(shift1, k4, sum3);
				sum4 = madd<ENABLE_FMA>(shift1, k1, sum4);
				const __m256 shift2 = alignr8<ENABLE_AVX2>(temp, cur3);
				sum2 = madd<ENABLE_FMA>(shift2, k8, sum2);
				sum3 = madd<ENABLE_FMA>(shift2, k5, sum3);
				sum4 = madd<ENABLE_FMA>(shift2, k2, sum4);
				cur3 = next;
			}
			{ // line4
				const __m256 next = _mm256_load_ps(line4 + x + 8);
				sum3 = madd<ENABLE_FMA>(cur4, k6, sum3);
				sum4 = madd<ENABLE_FMA>(cur4, k3, sum4);
				const __m256 temp = _mm256_permute2f128_ps(cur4, next, 0x21);
				const __m256 shift1 = alignr4<ENABLE_AVX2>(temp, cur4);
				sum3 = madd<ENABLE_FMA>(shift1, k7, sum3);
				sum4 = madd<ENABLE_FMA>(shift1, k4, sum4);
				const __m256 shift2 = alignr8<ENABLE_AVX2>(temp, cur4);
				sum3 = madd<ENABLE_FMA>(shift2, k8, sum3);
				sum4 = madd<ENABLE_FMA>(shift2, k5, sum4);
				cur4 = next;
			}
			{ // line5
				const __m256 next = _mm256_load_ps(line5 + x + 8);
				sum4 = madd<ENABLE_FMA>(cur5, k6, sum4);
				const __m256 temp = _mm256_permute2f128_ps(cur5, next, 0x21);
				const __m256 shift1 = alignr4<ENABLE_AVX2>(temp, cur5);
				sum4 = madd<ENABLE_FMA>(shift1, k7, sum4);
				const __m256 shift2 = alignr8<ENABLE_AVX2>(temp, cur5);
				sum4 = madd<ENABLE_FMA>(shift2, k8, sum4);
				cur5 = next;
			}
			_mm256_store_ps(dline1 + x, sum1);
			_mm256_store_ps(dline2 + x, sum2);
			_mm256_store_ps(dline3 + x, sum3);
			_mm256_store_ps(dline4 + x, sum4);
		}
	}
}

template <bool ENABLE_FMA, bool ENABLE_AVX2>
inline void compute_block(
	std::vector<AlignedBuffer<float>> &out_planes,
	const std::vector<AlignedBuffer<float>> &in_planes,
	const Model &model, const int step, int width, int height,
	int io_pitch, int io_offset, int buf_pitch, int buf_slice,
	AlignedBuffer<float> &in_buffer, AlignedBuffer<float> &out_buffer)
{
	const int num_in_planes = static_cast<int>(model.num_in_planes(step));
	const int num_out_planes = static_cast<int>(model.num_out_planes(step));
	for(int op = 0; op < num_out_planes; ++op){
		const __m256 v_bias = _mm256_set1_ps(model.bias(step, op));
		float *ptr = out_buffer.data() + buf_slice * op;
		for(int i = 0; i < buf_slice; i += 8){
			_mm256_store_ps(ptr + i, v_bias);
		}
	}
	for(int ip = 0; ip < num_in_planes; ++ip){
		const float *src_ptr =
			in_planes[ip].data() + io_offset;
		float *dst_ptr = in_buffer.data() + buf_slice * ip;
		for(int i = 0; i < height + 2; ++i){
			const float *src_line = src_ptr + i * io_pitch;
			float *dst_line = dst_ptr + i * buf_pitch;
			for(int j = 0; j < width + 8; j += 8){
				_mm256_store_ps(
					dst_line + j, _mm256_load_ps(src_line + j));
			}
		}
	}
	for(int op = 0; op < num_out_planes; ++op){
		for(int ip = 0; ip < num_in_planes; ++ip){
			convolute_add<ENABLE_FMA, ENABLE_AVX2>(
				out_buffer.data() + op * buf_slice,
				in_buffer.data() + ip * buf_slice,
				width, height, buf_pitch,
				model.coeffs(step, op, ip));
		}
		const __m256 neg_coeff = _mm256_set1_ps(0.1f);
		const float *src_ptr = out_buffer.data() + buf_slice * op;
		float *dst_ptr = out_planes[op].data() + io_offset;
		for(int i = 0; i < height; ++i){
			const float *src_line = src_ptr + i * buf_pitch;
			float *dst_line = dst_ptr + i * io_pitch;
			for(int j = 0; j < width; j += 8){
				const __m256 v = _mm256_load_ps(src_line + j);
				const __m256 mask =
					_mm256_cmp_ps(v, _mm256_setzero_ps(), _CMP_LT_OQ);
				const __m256 mv = _mm256_mul_ps(v, neg_coeff);
				_mm256_store_ps(
					dst_line + j, _mm256_blendv_ps(v, mv, mask));
			}
		}
	}
}
 
template <bool ENABLE_FMA, bool ENABLE_AVX2>
inline std::vector<AlignedBuffer<float>> compute_step(
	const Model &model, int step,
	const std::vector<AlignedBuffer<float>> &in_planes,
	int width, int height, int pitch, int num_threads)
{
	const int num_in_planes = static_cast<int>(model.num_in_planes(step));
	const int num_out_planes = static_cast<int>(model.num_out_planes(step));
	const int num_y_blocks = (height + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT;
	const int num_x_blocks = (width + BLOCK_WIDTH - 1) / BLOCK_WIDTH;

	std::vector<AlignedBuffer<float>> out_planes(num_out_planes);
	for(int i = 0; i < num_out_planes; ++i){
		out_planes[i] = std::move(AlignedBuffer<float>(pitch * (height + 2)));
	}

	const int block_pitch = BLOCK_WIDTH + 8;
	const int block_slice = block_pitch * (BLOCK_HEIGHT + 4);

#pragma omp parallel num_threads(num_threads)
	{
		AlignedBuffer<float> in_buffer(block_slice * num_in_planes);
		AlignedBuffer<float> out_buffer(block_slice * num_out_planes);
#pragma omp for
		for(int yb = 0; yb < num_y_blocks; ++yb){
			const int y0 = yb * BLOCK_HEIGHT;
			const int block_height = std::min(BLOCK_HEIGHT, height - y0);
			for(int xb = 0; xb < num_x_blocks; ++xb){
				const int x0 = xb * BLOCK_WIDTH;
				const int block_width = std::min(BLOCK_WIDTH, width - x0);
				const int io_offset = y0 * pitch + x0;
				compute_block<ENABLE_FMA, ENABLE_AVX2>(
					out_planes, in_planes, model, step,
					block_width, block_height, pitch, io_offset,
					block_pitch, block_slice, in_buffer, out_buffer);
			}
		}
	}
	return out_planes;
}


}

template <bool ENABLE_FMA, bool ENABLE_AVX2>
void AVXImpl<ENABLE_FMA, ENABLE_AVX2>::prepare(const Model &model){
	m_model = model;
}

template <bool ENABLE_FMA, bool ENABLE_AVX2>
void AVXImpl<ENABLE_FMA, ENABLE_AVX2>::process(
	float *dst, const float *src, int io_width, int io_height, int io_pitch,
	int x0, int y0, int block_width, int block_height, bool verbose)
{
	namespace chrono = std::chrono;
	const int num_steps = static_cast<int>(m_model.num_steps());
	const int width = block_width + 2 * num_steps;
	const int height = block_height + 2 * num_steps;
	const int pitch = (width + 15) & ~7;
	const int image_size = pitch * (height + 2);

	std::vector<AlignedBuffer<float>> in_planes(1);
	in_planes[0] = AlignedBuffer<float>(image_size);
#pragma omp parallel for num_threads(m_num_threads)
	for(int y = 0; y < height; ++y){
		const int ty = std::min(
			std::max(0, y0 + y - num_steps), io_height - 1);
		for(int x = 0; x < width; ++x){
			const int tx = std::min(
				std::max(0, x0 + x - num_steps), io_width - 1);
			in_planes[0][y * pitch + x] = src[ty * io_pitch + tx];
		}
	}

	for(int i = 0; i < num_steps; ++i){
		const auto begin_time = chrono::system_clock::now();
		if(verbose){
			std::cerr << "  Step " << i << "/" << num_steps;
		}
		const int p = (i + 1) * 2;
		auto out_planes = compute_step<ENABLE_FMA, ENABLE_AVX2>(
			m_model, i, in_planes, width - p, height - p, pitch,
			m_num_threads);
		out_planes.swap(in_planes);
		if(verbose){
			const auto end_time = chrono::system_clock::now();
			const auto duration =
				chrono::duration_cast<chrono::milliseconds>(end_time - begin_time);
			std::cerr << " " << duration.count() << " [ms]" << std::endl;
		}
	}
	assert(in_planes.size() == 1);

#pragma omp parallel for num_threads(m_num_threads)
	for(int y = 0; y < height - 2 * num_steps; ++y){
		for(int x = 0; x < width - 2 * num_steps; ++x){
			dst[(y0 + y) * io_pitch + (x0 + x)] = in_planes[0][y * pitch + x];
		}
	}
}

// Explicit instantiation
template class AVXImpl<false, false>;
template class AVXImpl<true, false>;
template class AVXImpl<true, true>;

}
