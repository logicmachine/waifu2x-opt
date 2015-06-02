#include <immintrin.h>
#include <omp.h>
#include "avx2_impl.h"
#include "../common/x86.h"
#include "../common/aligned_buffer.h"

namespace waifu2x {

namespace {

static const int BLOCK_WIDTH = 64;
static const int BLOCK_HEIGHT = 16;

inline void convolute_add(
	float *dst, const float *src, int width, int height, int pitch,
	const KernelModel &km)
{
	assert(km.width == 3 && km.height == 3);
	const float *kernel = km.values();
	const __m256 k0 = _mm256_broadcast_ss(kernel + 0);
	const __m256 k1 = _mm256_broadcast_ss(kernel + 1);
	const __m256 k2 = _mm256_broadcast_ss(kernel + 2);
	const __m256 k3 = _mm256_broadcast_ss(kernel + 3);
	const __m256 k4 = _mm256_broadcast_ss(kernel + 4);
	const __m256 k5 = _mm256_broadcast_ss(kernel + 5);
	const __m256 k6 = _mm256_broadcast_ss(kernel + 6);
	const __m256 k7 = _mm256_broadcast_ss(kernel + 7);
	const __m256 k8 = _mm256_broadcast_ss(kernel + 8);
	for(int y = 0; y < height; y += 2){
		const float *line0 = src + y * pitch;
		const float *line1 = src + (y + 1) * pitch;
		const float *line2 = src + (y + 2) * pitch;
		const float *line3 = src + (y + 3) * pitch;
		float *dline1 = dst + y * pitch;
		float *dline2 = dst + (y + 1) * pitch;
		__m256 cur0 = _mm256_load_ps(line0);
		__m256 cur1 = _mm256_load_ps(line1);
		__m256 cur2 = _mm256_load_ps(line2);
		__m256 cur3 = _mm256_load_ps(line3);
		for(int x = 0; x < width; x += 8){
			__m256 sum1 = _mm256_load_ps(dline1 + x);
			__m256 sum2 = _mm256_load_ps(dline2 + x);
			{ // line0
				const __m256 next = _mm256_load_ps(line0 + x + 8);
				sum1 = _mm256_fmadd_ps(cur0, k0, sum1);
				const __m256 temp = _mm256_permute2f128_ps(cur0, next, 0x21);
				const __m256 shift1 = _mm256_castsi256_ps(_mm256_alignr_epi8(
					_mm256_castps_si256(temp), _mm256_castps_si256(cur0), 4));
				sum1 = _mm256_fmadd_ps(shift1, k1, sum1);
				const __m256 shift2 = _mm256_castsi256_ps(_mm256_alignr_epi8(
					_mm256_castps_si256(temp), _mm256_castps_si256(cur0), 8));
				sum1 = _mm256_fmadd_ps(shift2, k2, sum1);
				cur0 = next;
			}
			{ // line1
				const __m256 next = _mm256_load_ps(line1 + x + 8);
				sum1 = _mm256_fmadd_ps(cur1, k3, sum1);
				sum2 = _mm256_fmadd_ps(cur1, k0, sum2);
				const __m256 temp = _mm256_permute2f128_ps(cur1, next, 0x21);
				const __m256 shift1 = _mm256_castsi256_ps(_mm256_alignr_epi8(
					_mm256_castps_si256(temp), _mm256_castps_si256(cur1), 4));
				sum1 = _mm256_fmadd_ps(shift1, k4, sum1);
				sum2 = _mm256_fmadd_ps(shift1, k1, sum2);
				const __m256 shift2 = _mm256_castsi256_ps(_mm256_alignr_epi8(
					_mm256_castps_si256(temp), _mm256_castps_si256(cur1), 8));
				sum1 = _mm256_fmadd_ps(shift2, k5, sum1);
				sum2 = _mm256_fmadd_ps(shift2, k2, sum2);
				cur1 = next;
			}
			{ // line2
				const __m256 next = _mm256_load_ps(line2 + x + 8);
				sum1 = _mm256_fmadd_ps(cur2, k6, sum1);
				sum2 = _mm256_fmadd_ps(cur2, k3, sum2);
				const __m256 temp = _mm256_permute2f128_ps(cur2, next, 0x21);
				const __m256 shift1 = _mm256_castsi256_ps(_mm256_alignr_epi8(
					_mm256_castps_si256(temp), _mm256_castps_si256(cur2), 4));
				sum1 = _mm256_fmadd_ps(shift1, k7, sum1);
				sum2 = _mm256_fmadd_ps(shift1, k4, sum2);
				const __m256 shift2 = _mm256_castsi256_ps(_mm256_alignr_epi8(
					_mm256_castps_si256(temp), _mm256_castps_si256(cur2), 8));
				sum1 = _mm256_fmadd_ps(shift2, k8, sum1);
				sum2 = _mm256_fmadd_ps(shift2, k5, sum2);
				cur2 = next;
			}
			{ // line3
				const __m256 next = _mm256_load_ps(line3 + x + 8);
				sum2 = _mm256_fmadd_ps(cur3, k6, sum2);
				const __m256 temp = _mm256_permute2f128_ps(cur3, next, 0x21);
				const __m256 shift1 = _mm256_castsi256_ps(_mm256_alignr_epi8(
					_mm256_castps_si256(temp), _mm256_castps_si256(cur3), 4));
				sum2 = _mm256_fmadd_ps(shift1, k7, sum2);
				const __m256 shift2 = _mm256_castsi256_ps(_mm256_alignr_epi8(
					_mm256_castps_si256(temp), _mm256_castps_si256(cur3), 8));
				sum2 = _mm256_fmadd_ps(shift2, k8, sum2);
				cur3 = next;
			}
			_mm256_store_ps(dline1 + x, sum1);
			_mm256_store_ps(dline2 + x, sum2);
		}
	}
}

inline void compute_block(
	std::vector<AlignedBuffer<float>> &out_planes, const StepModel &sm,
	const std::vector<AlignedBuffer<float>> &in_planes, int width, int height,
	int io_pitch, int io_offset, int buf_pitch, int buf_slice,
	AlignedBuffer<float> &in_buffer, AlignedBuffer<float> &out_buffer)
{
	const int num_in_planes = static_cast<int>(sm.num_input_planes());
	const int num_out_planes = static_cast<int>(sm.num_output_planes());
	for(int op = 0; op < num_out_planes; ++op){
		const __m256 v_bias = _mm256_set1_ps(sm.biases(op));
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
			const KernelModel &km = sm.weights(op)[ip];
			convolute_add(
				out_buffer.data() + op * buf_slice,
				in_buffer.data() + ip * buf_slice,
				width, height, buf_pitch, km);
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

inline std::vector<AlignedBuffer<float>> compute_step(
	const StepModel &sm, const std::vector<AlignedBuffer<float>> &in_planes,
	int width, int height, int pitch)
{
	const int num_in_planes = static_cast<int>(sm.num_input_planes());
	const int num_out_planes = static_cast<int>(sm.num_output_planes());
	const int num_y_blocks = (height + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT;
	const int num_x_blocks = (width + BLOCK_WIDTH - 1) / BLOCK_WIDTH;

	std::vector<AlignedBuffer<float>> out_planes(num_out_planes);
	for(int i = 0; i < num_out_planes; ++i){
		out_planes[i] = std::move(AlignedBuffer<float>(pitch * (height + 2)));
	}

	const int block_pitch = BLOCK_WIDTH + 8;
	const int block_slice = block_pitch * (BLOCK_HEIGHT + 2);

#pragma omp parallel
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
				compute_block(
					out_planes, sm, in_planes, block_width, block_height,
					pitch, io_offset, block_pitch, block_slice,
					in_buffer, out_buffer);
			}
		}
	}
	return out_planes;
}


}

void AVX2Impl::prepare(const Model &model){
	m_model = model;
}

void AVX2Impl::process(
	float *dst, const float *src, int io_width, int io_height, int io_pitch,
	int x0, int y0, int block_width, int block_height, bool verbose)
{
	const int num_steps = static_cast<int>(m_model.num_steps());
	const int width = block_width + 2 * num_steps;
	const int height = block_height + 2 * num_steps;
	const int pitch = (width + 15) & ~7;
	const int image_size = pitch * (height + 2);

	std::vector<AlignedBuffer<float>> in_planes(1);
	in_planes[0] = AlignedBuffer<float>(image_size);
#pragma omp parallel for
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
		const auto &step = m_model.steps(i);
		if(verbose){
			std::cerr << "  Step " << i << "/" << num_steps << std::endl;
		}
		const int p = (i + 1) * 2;
		auto out_planes = compute_step(
			m_model.steps(i), in_planes, width - p, height - p, pitch);
		out_planes.swap(in_planes);
	}
	assert(in_planes.size() == 1);

#pragma omp parallel for
	for(int y = 0; y < height - 2 * num_steps; ++y){
		for(int x = 0; x < width - 2 * num_steps; ++x){
			dst[(y0 + y) * io_pitch + (x0 + x)] = in_planes[0][y * pitch + x];
		}
	}
}

}

