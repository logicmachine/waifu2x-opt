#include <iostream>
#include <algorithm>
#include <chrono>
#include <cassert>
#include <immintrin.h>
#include <omp.h>
#include "dft_avx_impl.h"
#include "dft_table.h"
#include "../common/x86.h"
#include "../common/aligned_buffer.h"

namespace waifu2x {

namespace {

static const int SUBBLOCK_WIDTH  = 16;
static const int SUBBLOCK_HEIGHT = 16;
static const int BLOCK_WIDTH  = 4;
static const int BLOCK_HEIGHT = 2;
static const int OUT_BLOCK_SIZE = 2;

template <bool ENABLE_FMA>
inline __m256 madd(__m256 a, __m256 b, __m256 c){
	return _mm256_add_ps(_mm256_mul_ps(a, b), c);
}
template <>
inline __m256 madd<true>(__m256 a, __m256 b, __m256 c){
	return _mm256_fmadd_ps(a, b, c);
}

template <bool ENABLE_FMA>
inline __m256 msub(__m256 a, __m256 b, __m256 c){
	return _mm256_sub_ps(_mm256_mul_ps(a, b), c);
}
template <>
inline __m256 msub<true>(__m256 a, __m256 b, __m256 c){
	return _mm256_fmsub_ps(a, b, c);
}

template <bool ENABLE_FMA>
inline __m256 complex_mul_re(__m256 a_re, __m256 a_im, __m256 b_re, __m256 b_im){
	return msub<ENABLE_FMA>(a_re, b_re, _mm256_mul_ps(a_im, b_im));
}
template <bool ENABLE_FMA>
inline __m256 complex_mul_im(__m256 a_re, __m256 a_im, __m256 b_re, __m256 b_im){
	return madd<ENABLE_FMA>(a_re, b_im, _mm256_mul_ps(a_im, b_re));
}

template <bool ENABLE_FMA>
inline __m256 complex_madd_re(
	__m256 a_re, __m256 a_im, __m256 b_re, __m256 b_im, __m256 c_re)
{
	return msub<ENABLE_FMA>(a_re, b_re, msub<ENABLE_FMA>(a_im, b_im, c_re));
}
template <bool ENABLE_FMA>
inline __m256 complex_madd_im(
	__m256 a_re, __m256 a_im, __m256 b_re, __m256 b_im, __m256 c_im)
{
	return madd<ENABLE_FMA>(a_re, b_im, madd<ENABLE_FMA>(a_im, b_re, c_im));
}

template <bool ENABLE_FMA>
inline void horizontal_dft15(float *data){
	__m256 re = _mm256_setzero_ps(), im = _mm256_setzero_ps();
	for(int i = 0; i < 15; ++i){
		const __m256 x = _mm256_broadcast_ss(data + i);
		const __m256 re_coeff = _mm256_load_ps(RDFT15_TABLE + i * 16);
		const __m256 im_coeff = _mm256_load_ps(RDFT15_TABLE + i * 16 + 8);
		re = madd<ENABLE_FMA>(x, re_coeff, re);
		im = madd<ENABLE_FMA>(x, im_coeff, im);
	}
	_mm256_store_ps(data,     re);
	_mm256_store_ps(data + 8, im);
}
template <bool ENABLE_FMA>
inline void horizontal_idft15(float *data){
	__m256 lo = _mm256_setzero_ps(), hi = _mm256_setzero_ps();
	for(int i = 0; i < 16; ++i){
		const __m256 y = _mm256_broadcast_ss(data + i);
		const __m256 lo_coeff = _mm256_load_ps(IRDFT15_TABLE + i * 16);
		const __m256 hi_coeff = _mm256_load_ps(IRDFT15_TABLE + i * 16 + 8);
		lo = madd<ENABLE_FMA>(y, lo_coeff, lo);
		hi = madd<ENABLE_FMA>(y, hi_coeff, hi);
	}
	_mm256_store_ps(data,     lo);
	_mm256_store_ps(data + 8, hi);
}

template <bool ENABLE_FMA>
inline void vertical_dft16(float *data){
	ALIGNED(32) float buf[16 * 16];
	float *src = data, *dst = buf;
	for(int n = 16, s = 1; n > 1; n >>= 2, s <<= 2){
		const int k = n >> 2;
		for(int p = 0; p < k; ++p){
			const __m256 w1p_re = _mm256_broadcast_ss(COS_TABLE + p * s);
			const __m256 w1p_im = _mm256_broadcast_ss(SIN_TABLE + ((p * s) ^ 8));
			const __m256 w2p_re = complex_mul_re<ENABLE_FMA>(w1p_re, w1p_im, w1p_re, w1p_im);
			const __m256 w2p_im = complex_mul_im<ENABLE_FMA>(w1p_re, w1p_im, w1p_re, w1p_im);
			const __m256 w3p_re = complex_mul_re<ENABLE_FMA>(w2p_re, w2p_im, w1p_re, w1p_im);
			const __m256 w3p_im = complex_mul_im<ENABLE_FMA>(w2p_re, w2p_im, w1p_re, w1p_im);
			for(int q = 0; q < s; ++q){
				const float *sptr = src + (q + s * p) * 16;
				const __m256 a_re = _mm256_load_ps(sptr), a_im = _mm256_load_ps(sptr + 8);
				sptr += s * k * 16;
				const __m256 b_re = _mm256_load_ps(sptr), b_im = _mm256_load_ps(sptr + 8);
				sptr += s * k * 16;
				const __m256 c_re = _mm256_load_ps(sptr), c_im = _mm256_load_ps(sptr + 8);
				sptr += s * k * 16;
				const __m256 d_re = _mm256_load_ps(sptr), d_im = _mm256_load_ps(sptr + 8);
				const __m256  apc_re = _mm256_add_ps(a_re, c_re),  apc_im = _mm256_add_ps(a_im, c_im);
				const __m256  amc_re = _mm256_sub_ps(a_re, c_re),  amc_im = _mm256_sub_ps(a_im, c_im);
				const __m256  bpd_re = _mm256_add_ps(b_re, d_re),  bpd_im = _mm256_add_ps(b_im, d_im);
				const __m256 jbmd_re = _mm256_sub_ps(d_im, b_im), jbmd_im = _mm256_sub_ps(b_re, d_re);
				float *dptr = dst + (q + s * 4 * p) * 16;
				_mm256_store_ps(dptr,     _mm256_add_ps(apc_re, bpd_re));
				_mm256_store_ps(dptr + 8, _mm256_add_ps(apc_im, bpd_im));
				dptr += s * 16;
				const __m256 t1_re = _mm256_sub_ps(amc_re, jbmd_re), t1_im = _mm256_sub_ps(amc_im, jbmd_im);
				_mm256_store_ps(dptr,     complex_mul_re<ENABLE_FMA>(w1p_re, w1p_im, t1_re, t1_im));
				_mm256_store_ps(dptr + 8, complex_mul_im<ENABLE_FMA>(w1p_re, w1p_im, t1_re, t1_im));
				dptr += s * 16;
				const __m256 t2_re = _mm256_sub_ps(apc_re, bpd_re), t2_im = _mm256_sub_ps(apc_im, bpd_im);
				_mm256_store_ps(dptr,     complex_mul_re<ENABLE_FMA>(w2p_re, w2p_im, t2_re, t2_im));
				_mm256_store_ps(dptr + 8, complex_mul_im<ENABLE_FMA>(w2p_re, w2p_im, t2_re, t2_im));
				dptr += s * 16;
				const __m256 t3_re = _mm256_add_ps(amc_re, jbmd_re), t3_im = _mm256_add_ps(amc_im, jbmd_im);
				_mm256_store_ps(dptr,     complex_mul_re<ENABLE_FMA>(w3p_re, w3p_im, t3_re, t3_im));
				_mm256_store_ps(dptr + 8, complex_mul_im<ENABLE_FMA>(w3p_re, w3p_im, t3_re, t3_im));
			}
		}
		std::swap(src, dst);
	}
}
template <bool ENABLE_FMA>
inline void vertical_idft16(float *data){
	ALIGNED(32) float buf[16 * 16];
	float *src = data, *dst = buf;
	for(int n = 16, s = 1; n > 1; n >>= 2, s <<= 2){
		const int k = n >> 2;
		for(int p = 0; p < k; ++p){
			const __m256 w1p_re = _mm256_broadcast_ss(COS_TABLE + p * s);
			const __m256 w1p_im = _mm256_broadcast_ss(SIN_TABLE + p * s);
			const __m256 w2p_re = complex_mul_re<ENABLE_FMA>(w1p_re, w1p_im, w1p_re, w1p_im);
			const __m256 w2p_im = complex_mul_im<ENABLE_FMA>(w1p_re, w1p_im, w1p_re, w1p_im);
			const __m256 w3p_re = complex_mul_re<ENABLE_FMA>(w2p_re, w2p_im, w1p_re, w1p_im);
			const __m256 w3p_im = complex_mul_im<ENABLE_FMA>(w2p_re, w2p_im, w1p_re, w1p_im);
			for(int q = 0; q < s; ++q){
				const float *sptr = src + (q + s * p) * 16;
				const __m256 a_re = _mm256_load_ps(sptr), a_im = _mm256_load_ps(sptr + 8);
				sptr += s * k * 16;
				const __m256 b_re = _mm256_load_ps(sptr), b_im = _mm256_load_ps(sptr + 8);
				sptr += s * k * 16;
				const __m256 c_re = _mm256_load_ps(sptr), c_im = _mm256_load_ps(sptr + 8);
				sptr += s * k * 16;
				const __m256 d_re = _mm256_load_ps(sptr), d_im = _mm256_load_ps(sptr + 8);
				const __m256  apc_re = _mm256_add_ps(a_re, c_re),  apc_im = _mm256_add_ps(a_im, c_im);
				const __m256  amc_re = _mm256_sub_ps(a_re, c_re),  amc_im = _mm256_sub_ps(a_im, c_im);
				const __m256  bpd_re = _mm256_add_ps(b_re, d_re),  bpd_im = _mm256_add_ps(b_im, d_im);
				const __m256 jbmd_re = _mm256_sub_ps(d_im, b_im), jbmd_im = _mm256_sub_ps(b_re, d_re);
				float *dptr = dst + (q + s * 4 * p) * 16;
				_mm256_store_ps(dptr,     _mm256_add_ps(apc_re, bpd_re));
				_mm256_store_ps(dptr + 8, _mm256_add_ps(apc_im, bpd_im));
				dptr += s * 16;
				const __m256 t1_re = _mm256_add_ps(amc_re, jbmd_re), t1_im = _mm256_add_ps(amc_im, jbmd_im);
				_mm256_store_ps(dptr,     complex_mul_re<ENABLE_FMA>(w1p_re, w1p_im, t1_re, t1_im));
				_mm256_store_ps(dptr + 8, complex_mul_im<ENABLE_FMA>(w1p_re, w1p_im, t1_re, t1_im));
				dptr += s * 16;
				const __m256 t2_re = _mm256_sub_ps(apc_re, bpd_re), t2_im = _mm256_sub_ps(apc_im, bpd_im);
				_mm256_store_ps(dptr,     complex_mul_re<ENABLE_FMA>(w2p_re, w2p_im, t2_re, t2_im));
				_mm256_store_ps(dptr + 8, complex_mul_im<ENABLE_FMA>(w2p_re, w2p_im, t2_re, t2_im));
				dptr += s * 16;
				const __m256 t3_re = _mm256_sub_ps(amc_re, jbmd_re), t3_im = _mm256_sub_ps(amc_im, jbmd_im);
				_mm256_store_ps(dptr,     complex_mul_re<ENABLE_FMA>(w3p_re, w3p_im, t3_re, t3_im));
				_mm256_store_ps(dptr + 8, complex_mul_im<ENABLE_FMA>(w3p_re, w3p_im, t3_re, t3_im));
			}
		}
		std::swap(src, dst);
	}
	const __m256 divisor = _mm256_set1_ps(1.0f / 16.0f);
	for(int i = 0; i < 16; ++i){
		float *p = data + i * 16;
		_mm256_store_ps(p,     _mm256_mul_ps(_mm256_load_ps(p),     divisor));
		_mm256_store_ps(p + 8, _mm256_mul_ps(_mm256_load_ps(p + 8), divisor));
	}
}

template <bool ENABLE_FMA>
inline void block_dft2d(float *ptr){
	for(int i = 0; i < SUBBLOCK_HEIGHT; ++i){
		horizontal_dft15<ENABLE_FMA>(ptr + i * 16);
	}
	vertical_dft16<ENABLE_FMA>(ptr);
}
template <bool ENABLE_FMA>
inline void block_idft2d(float *ptr){
	vertical_idft16<ENABLE_FMA>(ptr);
	for(int i = 0; i < SUBBLOCK_HEIGHT; ++i){
		horizontal_idft15<ENABLE_FMA>(ptr + i * 16);
	}
}

template <bool ENABLE_FMA>
inline void transform_input(
	float *dst, const AlignedBuffer<float> &src,
	int width, int height, int pitch, int x0, int y0)
{
	const int inner_width = SUBBLOCK_WIDTH - 3;
	const int inner_height = SUBBLOCK_HEIGHT - 2;
	const int slice_size = SUBBLOCK_WIDTH * SUBBLOCK_HEIGHT;
	for(int by = 0; by < BLOCK_HEIGHT; ++by){
		const int y1 = y0 + by * inner_height;
		for(int bx = 0; bx < BLOCK_WIDTH; ++bx){
			const int bid = by * BLOCK_WIDTH + bx;
			const int x1 = x0 + bx * inner_width;
			const float *sptr = src.data() + y1 * pitch + x1;
			float *dptr = dst + bid * slice_size;
			if(y1 + SUBBLOCK_HEIGHT > height || x1 + SUBBLOCK_WIDTH > width){
				const __m256 zero = _mm256_setzero_ps();
				for(int i = 0; i < slice_size; i += 8){ _mm256_store_ps(dptr + i, zero); }
				const int ymax = std::min(SUBBLOCK_HEIGHT, height - y1);
				const int xmax = std::min(SUBBLOCK_WIDTH, width - x1);
				for(int y = 0; y < ymax; ++y){
					for(int x = 0; x < xmax; ++x){
						dptr[y * SUBBLOCK_WIDTH + x] = sptr[y * pitch + x];
					}
				}
			}else{
				for(int y = 0; y < SUBBLOCK_HEIGHT; ++y){
					const __m256 lo = _mm256_loadu_ps(sptr + y * pitch);
					const __m256 hi = _mm256_loadu_ps(sptr + y * pitch + 8);
					_mm256_store_ps(dptr + y * SUBBLOCK_WIDTH,     lo);
					_mm256_store_ps(dptr + y * SUBBLOCK_WIDTH + 8, hi);
				}
			}
			block_dft2d<ENABLE_FMA>(dptr);
		}
	}
}
template <bool ENABLE_FMA>
inline void transform_output(
	AlignedBuffer<float> &dst, float *src,
	int width, int height, int pitch, int x0, int y0)
{
	const int inner_width = SUBBLOCK_WIDTH - 3;
	const int inner_height = SUBBLOCK_HEIGHT - 2;
	const int slice_size = SUBBLOCK_WIDTH * SUBBLOCK_HEIGHT;
	const __m256 neg_coeff = _mm256_set1_ps(0.1f);
	for(int by = 0; by < BLOCK_HEIGHT; ++by){
		const int y1 = y0 + by * inner_height;
		const int ymax = std::min(inner_height, height - y1);
		for(int bx = 0; bx < BLOCK_WIDTH; ++bx){
			const int bid = by * BLOCK_WIDTH + bx;
			const int x1 = x0 + bx * inner_width;
			const int xmax = std::min(inner_width, width - x1);
			float *sptr = src + bid * slice_size;
			float *dptr = dst.data() + y1 * pitch + x1;
			block_idft2d<ENABLE_FMA>(sptr);
			for(int y = 0; y < ymax; ++y){
				float *sline = sptr + y * SUBBLOCK_WIDTH;
				float *dline = dptr + y * pitch;
				const __m256 lo = _mm256_load_ps(sline);
				const __m256 hi = _mm256_load_ps(sline + 8);
				const __m256 mask_lo = _mm256_cmp_ps(lo, _mm256_setzero_ps(), _CMP_LT_OQ);
				const __m256 mask_hi = _mm256_cmp_ps(hi, _mm256_setzero_ps(), _CMP_LT_OQ);
				const __m256 mul_lo = _mm256_mul_ps(lo, neg_coeff);
				const __m256 mul_hi = _mm256_mul_ps(hi, neg_coeff);
				_mm256_store_ps(sline,     _mm256_blendv_ps(lo, mul_lo, mask_lo));
				_mm256_store_ps(sline + 8, _mm256_blendv_ps(hi, mul_hi, mask_hi));
				for(int x = 0; x < xmax; ++x){ dline[x] = sline[x]; }
			}
		}
	}
}

template <bool ENABLE_FMA>
inline void convolute(float *dst, const float *src, const float *kernel){
	const int num_blocks = BLOCK_WIDTH * BLOCK_HEIGHT;
	const int block_size = SUBBLOCK_WIDTH * SUBBLOCK_HEIGHT;
	for(int b = 0; b < num_blocks; ++b){
		for(int i = 0; i < SUBBLOCK_HEIGHT; ++i){
			const __m256 s_re = _mm256_load_ps(src + i * 16);
			const __m256 s_im = _mm256_load_ps(src + i * 16 + 8);
			const __m256 k_re = _mm256_load_ps(kernel + i * 16);
			const __m256 k_im = _mm256_load_ps(kernel + i * 16 + 8);
			__m256 d_re = _mm256_load_ps(dst + i * 16);
			__m256 d_im = _mm256_load_ps(dst + i * 16 + 8);
			d_re = complex_madd_re<ENABLE_FMA>(s_re, s_im, k_re, k_im, d_re);
			d_im = complex_madd_im<ENABLE_FMA>(s_re, s_im, k_re, k_im, d_im);
			_mm256_store_ps(dst + i * 16,     d_re);
			_mm256_store_ps(dst + i * 16 + 8, d_im);
		}
		dst += block_size;
		src += block_size;
	}
}

template <bool ENABLE_FMA>
inline void compute_block(
	std::vector<AlignedBuffer<float>> &out_planes,
	const std::vector<AlignedBuffer<float>> &in_planes,
	const AlignedBuffer<float> &kernels, const float *biases,
	int width, int height, int pitch, int x0, int y0,
	float *in_buffer, float *out_buffer)
{
	const int num_in_planes = static_cast<int>(in_planes.size());
	const int num_out_planes = static_cast<int>(out_planes.size());
	const int num_subblocks = BLOCK_WIDTH * BLOCK_HEIGHT;
	const int subblock_size = SUBBLOCK_WIDTH * SUBBLOCK_HEIGHT;
	const int block_size =
		(SUBBLOCK_WIDTH * BLOCK_WIDTH) * (SUBBLOCK_HEIGHT * BLOCK_HEIGHT);
	for(int ip = 0; ip < num_in_planes; ++ip){
		transform_input<ENABLE_FMA>(
			in_buffer + block_size * ip, in_planes[ip], width, height, pitch, x0, y0);
	}
	const __m256 zero = _mm256_setzero_ps();
	for(int op_lo = 0; op_lo < num_out_planes; op_lo += OUT_BLOCK_SIZE){
		const int op_hi = std::min(num_out_planes, op_lo + OUT_BLOCK_SIZE);
		for(int op = op_lo; op < op_hi; ++op){
			float *bptr = out_buffer + (op - op_lo) * block_size;
			for(int i = 0; i < block_size; i += 8){ _mm256_store_ps(bptr + i, zero); }
			for(int i = 0; i < num_subblocks; ++i){
				bptr[i * subblock_size] = 240.0f * biases[op];
			}
		}
		for(int ip = 0; ip < num_in_planes; ++ip){
			const float *sptr = in_buffer + block_size * ip;
			for(int op = op_lo; op < op_hi; ++op){
				const float *kptr =
					kernels.data() + (op * num_in_planes + ip) * subblock_size;
				float *dptr = out_buffer + (op - op_lo) * block_size;
				convolute<ENABLE_FMA>(dptr, sptr, kptr);
			}
		}
		for(int op = op_lo; op < op_hi; ++op){
			float *bptr = out_buffer + (op - op_lo) * block_size;
			transform_output<ENABLE_FMA>(
				out_planes[op], bptr, width, height, pitch, x0, y0);
		}
	}
}

template <bool ENABLE_FMA>
inline std::vector<AlignedBuffer<float>> compute_step(
	const Model &model, int step, const AlignedBuffer<float> &kernels,
	const std::vector<AlignedBuffer<float>> &in_planes,
	int width, int height, int pitch, int num_threads)
{
	const int num_in_planes = static_cast<int>(model.num_in_planes(step));
	const int num_out_planes = static_cast<int>(model.num_out_planes(step));

	const int subblock_inner_width = SUBBLOCK_WIDTH - 3;
	const int subblock_inner_height = SUBBLOCK_HEIGHT - 2;
	const int block_inner_width = subblock_inner_width * BLOCK_WIDTH;
	const int block_inner_height = subblock_inner_height * BLOCK_HEIGHT;
	const int block_size =
		(SUBBLOCK_WIDTH * BLOCK_WIDTH) * (SUBBLOCK_HEIGHT * BLOCK_HEIGHT);

	const int num_x_blocks = (width + block_inner_width - 1) / block_inner_width;
	const int num_y_blocks = (height + block_inner_height - 1) / block_inner_height;

	std::vector<AlignedBuffer<float>> out_planes(num_out_planes);
	for(int i = 0; i < num_out_planes; ++i){
		out_planes[i] = std::move(AlignedBuffer<float>(pitch * height));
	}

#pragma omp parallel num_threads(num_threads)
	{
		AlignedBuffer<float> in_buffer (block_size * num_in_planes);
		AlignedBuffer<float> out_buffer(block_size * OUT_BLOCK_SIZE);
#pragma omp for
		for(int yb = 0; yb < num_y_blocks; ++yb){
			const int y0 = yb * block_inner_height;
			for(int xb = 0; xb < num_x_blocks; ++xb){
				const int x0 = xb * block_inner_width;
				compute_block<ENABLE_FMA>(
					out_planes, in_planes, kernels, model.biases(step),
					width, height, pitch, x0, y0,
					in_buffer.data(), out_buffer.data());
			}
		}
	}
	return out_planes;
}

}

template <bool ENABLE_FMA>
void DFTAVXImpl<ENABLE_FMA>::prepare(const Model &model){
	m_model = model;
	const int SUBBLOCK_SIZE = SUBBLOCK_WIDTH * SUBBLOCK_HEIGHT;
	const int bw = SUBBLOCK_WIDTH, bh = SUBBLOCK_HEIGHT;
	const int num_steps = static_cast<int>(model.num_steps());
	m_fourier_kernels.assign(num_steps, std::move(AlignedBuffer<float>()));
	for(int i = 0; i < num_steps; ++i){
		const int num_in_planes  = static_cast<int>(model.num_in_planes(i));
		const int num_out_planes = static_cast<int>(model.num_out_planes(i));
		m_fourier_kernels[i] = std::move(AlignedBuffer<float>(
			SUBBLOCK_SIZE * num_in_planes * num_out_planes));
#pragma omp parallel for
		for(int op = 0; op < num_out_planes; ++op){
			for(int ip = 0; ip < num_in_planes; ++ip){
				const float *kernel = model.coeffs(i, op, ip);
				float *p =
					m_fourier_kernels[i].data() +
					SUBBLOCK_SIZE * (op * num_in_planes + ip);
				for(int j = 0; j < SUBBLOCK_SIZE; ++j){ p[j] = 0.0f; }
				p[0] = kernel[0];
				p[(bw - 2)] = kernel[1];
				p[(bw - 3)] = kernel[2];
				p[(bh - 1) * bw] = kernel[3];
				p[(bh - 1) * bw + (bw - 2)] = kernel[4];
				p[(bh - 1) * bw + (bw - 3)] = kernel[5];
				p[(bh - 2) * bw] = kernel[6];
				p[(bh - 2) * bw + (bw - 2)] = kernel[7];
				p[(bh - 2) * bw + (bw - 3)] = kernel[8];
				block_dft2d<ENABLE_FMA>(p);
			}
		}
	}
}

template <bool ENABLE_FMA>
void DFTAVXImpl<ENABLE_FMA>::process(
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
		auto out_planes = compute_step<ENABLE_FMA>(
			m_model, i, m_fourier_kernels[i], in_planes,
			width - p, height - p, pitch, m_num_threads);
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
template class DFTAVXImpl<true>;
template class DFTAVXImpl<false>;

}
