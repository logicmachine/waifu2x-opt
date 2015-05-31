#include <immintrin.h>
#include <omp.h>
#include "avx2_impl.h"
#include "../common/x86.h"
#include "../common/aligned_buffer.h"

namespace waifu2x {

namespace {

void convolute3x3(
	float *dst, const float *src, int width, int height, int dst_pitch, int src_pitch,
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
	for(int y = 0; y < height; y += 2){
		const float *line0 = src + y * src_pitch;
		const float *line1 = src + (y + 1) * src_pitch;
		const float *line2 = src + (y + 2) * src_pitch;
		const float *line3 = src + (y + 3) * src_pitch;
		float *dline1 = dst + y * dst_pitch;
		float *dline2 = dst + (y + 1) * dst_pitch;
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

void region_copy(
	AlignedBuffer &dst, const AlignedBuffer &src, int width, int height,
	int dst_x0, int dst_y0, int dst_pitch, int src_x0, int src_y0, int src_pitch)
{
	const float *src_ptr = src.data() + src_pitch * src_y0 + src_x0;
	float *dst_ptr = dst.data() + dst_pitch * dst_y0 + dst_x0;
	for(int y = 0; y < height; ++y){
		const float *src_line = src_ptr + src_pitch * y;
		float *dst_line = dst_ptr + dst_pitch * y;
		for(int x = 0; x < width; x += 8){
			_mm256_store_ps(dst_line + x, _mm256_load_ps(src_line + x));
		}
	}
}

void compute_block(
	AlignedBuffer &out_buffer, const AlignedBuffer &in_buffer, int block_width, int block_height,
	int num_in_planes, const StepModel &sm)
{
	const int num_out_planes = static_cast<int>(sm.num_output_planes());
	const int in_block_width = block_width + 8;
	const int in_block_height = block_height + 2;
	for(int i = 0; i < num_out_planes; ++i){
		float *dst_ptr = out_buffer.data() + block_width * block_height * i;
		const __m256 v_bias = _mm256_set1_ps(sm.biases(i));
		for(int y = 0; y < block_height; ++y){
			float *line = dst_ptr + y * block_width;
			for(int x = 0; x < block_width; x += 8){
				_mm256_store_ps(line + x, v_bias);
			}
		}
		for(int j = 0; j < num_in_planes; ++j){
			const float *src_ptr = in_buffer.data() + in_block_width * in_block_height * j;
			convolute3x3(
				dst_ptr, src_ptr, block_width, block_height,
				block_width, in_block_width, sm.weights(i)[j].values());
		}
		const __m256 neg_coeff = _mm256_set1_ps(0.1f);
		for(int y = 0; y < block_height; ++y){
			float *line = dst_ptr + y * block_width;
			for(int x = 0; x < block_width; x += 8){
				const __m256 v = _mm256_load_ps(line + x);
				const __m256 mask = _mm256_cmp_ps(v, _mm256_setzero_ps(), _CMP_LT_OQ);
				const __m256 mv = _mm256_mul_ps(v, neg_coeff);
				_mm256_store_ps(line + x, _mm256_blendv_ps(v, mv, mask));
			}
		}
	}
}

std::vector<AlignedBuffer> compute_step(
	const StepModel &sm, const std::vector<AlignedBuffer> &planes,
	int width, int height, int pitch)
{
	const int num_in_planes = static_cast<int>(planes.size());
	const int num_weights = static_cast<int>(sm.num_output_planes());
	std::vector<AlignedBuffer> out_planes(num_weights);
	for(int i = 0; i < num_weights; ++i){ out_planes[i] = AlignedBuffer(pitch * height); }
#pragma omp parallel
	{
		// TODO tune parameters
		const int Y_BLOCK_SIZE = 16;
		const int X_BLOCK_SIZE = 128;
		const int num_threads = omp_get_num_threads();
		const int tid = omp_get_thread_num();

		const int y_num_blocks =
			(height - 2 + Y_BLOCK_SIZE - 1) / Y_BLOCK_SIZE;
		const int y_num_blocks_per_thread =
			(y_num_blocks + num_threads - 1) / num_threads;
		const int y_block_begin = y_num_blocks_per_thread * tid;
		const int y_block_end = std::min(
			y_block_begin + y_num_blocks_per_thread, y_num_blocks);
		const int x_num_blocks =
			(width - 2 + X_BLOCK_SIZE - 1) / X_BLOCK_SIZE;

		const int in_block_size = (X_BLOCK_SIZE + 8) * (Y_BLOCK_SIZE + 2);
		const int out_block_size = X_BLOCK_SIZE * Y_BLOCK_SIZE;
		AlignedBuffer in_block_buffer(in_block_size * num_in_planes);
		AlignedBuffer out_block_buffer(out_block_size * num_weights);
		for(int yb = y_block_begin; yb < y_block_end; ++yb){
			const int y_begin = yb * Y_BLOCK_SIZE;
			const int y_end = std::min(y_begin + Y_BLOCK_SIZE, height - 2);
			for(int xb = 0; xb < x_num_blocks; ++xb){
				const int x_begin = xb * X_BLOCK_SIZE;
				const int x_end = std::min(x_begin + X_BLOCK_SIZE, width - 2);
				for(int i = 0; i < num_in_planes; ++i){
					region_copy(
						in_block_buffer, planes[i],
						x_end - x_begin + 8, y_end - y_begin + 2,
						0, i * (Y_BLOCK_SIZE + 2), (X_BLOCK_SIZE + 8),
						x_begin, y_begin, pitch);
				}
				compute_block(
					out_block_buffer, in_block_buffer, X_BLOCK_SIZE, Y_BLOCK_SIZE,
					num_in_planes, sm);
				for(int i = 0; i < num_weights; ++i){
					region_copy(
						out_planes[i], out_block_buffer,
						x_end - x_begin, y_end - y_begin,
						x_begin, y_begin, pitch,
						0, i * Y_BLOCK_SIZE, X_BLOCK_SIZE);
				}
			}
		}
	}
	return out_planes;
}

}

void process_avx2(
	float *dst, const float *src, int io_width, int io_height, int io_pitch,
	const Model &model, bool verbose)
{
	const int num_steps = static_cast<int>(model.num_steps());
	const int width = io_width + 2 * num_steps;
	const int height = io_height + 2 * num_steps;
	const int pitch = ((width + 7) & ~7) + 8;
	const int image_size = pitch * (height + 2);
	std::vector<AlignedBuffer> planes(1, AlignedBuffer(image_size));
#pragma omp parallel for
	for(int y = 0; y < height; ++y){
		const int ty = std::min(std::max(0, y - num_steps), io_height - 1);
		for(int x = 0; x < width; ++x){
			const int tx = std::min(std::max(0, x - num_steps), io_width - 1);
			planes[0][y * pitch + x] = src[ty * io_pitch + tx];
		}
	}
	for(int i = 0; i < num_steps; ++i){
		const auto &step = model.steps(i);
		if(verbose){
			std::cerr << "#" << i << ": " << step.num_input_planes()
			          << " -> " << step.num_output_planes() << std::endl;
		}
		auto out_planes = compute_step(
			step, planes, width - i * 2, height - i * 2, pitch);
		out_planes.swap(planes);
	}
	assert(planes.size() == 1);
#pragma omp parallel for
	for(int y = 0; y < io_height; ++y){
		for(int x = 0; x < io_width; ++x){
			dst[y * io_pitch + x] = planes[0][y * pitch + x];
		}
	}
}

}

