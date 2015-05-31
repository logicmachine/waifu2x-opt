#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <immintrin.h>
#include <omp.h>
#include "waifu2x.h"
#include "picojson.h"

namespace {

class ImageBuffer {
private:
	std::size_t m_size;
	float *m_pointer;
public:
	ImageBuffer()
		: m_size(0)
		, m_pointer(nullptr)
	{ }
	explicit ImageBuffer(std::size_t n)
		: m_size(n)
		, m_pointer(nullptr)
	{
		if(m_size == 0){ return; }
		m_pointer = reinterpret_cast<float *>(_mm_malloc(n * sizeof(float), 32));
	}
	ImageBuffer(const ImageBuffer &ib)
		: m_size(ib.m_size)
		, m_pointer(nullptr)
	{
		if(ib.m_pointer == nullptr){ return; }
		m_pointer = reinterpret_cast<float *>(_mm_malloc(m_size * sizeof(float), 32));
		memcpy(m_pointer, ib.m_pointer, m_size * sizeof(float));
	}
	ImageBuffer(ImageBuffer &&ib)
		: m_size(ib.m_size)
		, m_pointer(ib.m_pointer)
	{
		ib.m_pointer = nullptr;
	}
	~ImageBuffer(){
		if(m_pointer){ _mm_free(m_pointer); }
	}

	ImageBuffer &operator=(const ImageBuffer &ib){
		if(m_pointer){ _mm_free(m_pointer); }
		m_size = ib.m_size;
		if(ib.m_pointer){
			m_pointer = reinterpret_cast<float *>(_mm_malloc(m_size * sizeof(float), 32));
			memcpy(m_pointer, ib.m_pointer, m_size * sizeof(float));
		}else{
			m_pointer = nullptr;
		}
		return *this;
	}

	const float *data() const { return m_pointer; }
	float *data(){ return m_pointer; }

	float operator[](std::size_t i) const { return m_pointer[i]; }
	float &operator[](std::size_t i){ return m_pointer[i]; }
};

}

class Waifu2x::Impl {

private:
	typedef ImageBuffer ImageType;

	struct Kernel {
		int width;
		int height;
		std::vector<float> values;
	};
	struct StepModel {
		std::vector<std::vector<Kernel>> weights;
		std::vector<float> biases;
	};

	std::vector<StepModel> m_model;

	int calculate_pitch(int width){
		return ((width + 7) & ~7) + 8;
	}

	std::string load_file_content(const char *filename){
		std::ifstream ifs(filename, std::ios::in);
		std::ostringstream oss;
		std::string line;
		while(getline(ifs, line)){ oss << line << std::endl; }
		return oss.str();
	}

	std::vector<std::vector<Kernel>> parse_weights(const picojson::array &raw_weights) const {
		std::vector<std::vector<Kernel>> weights;
		for(const auto &raw_kernel_outer_value : raw_weights){
			const auto raw_kernel_outer = raw_kernel_outer_value.get<picojson::array>();
			std::vector<Kernel> w;
			for(const auto &raw_kernel_inner_value : raw_kernel_outer){
				const auto &raw_kernel_inner = raw_kernel_inner_value.get<picojson::array>();
				Kernel k;
				k.height = static_cast<int>(raw_kernel_inner.size());
				for(const auto &raw_kernel_line_value : raw_kernel_inner){
					const auto &raw_kernel_line = raw_kernel_line_value.get<picojson::array>();
					k.width = static_cast<int>(raw_kernel_line.size());
					for(const auto &raw_kernel_value : raw_kernel_line){
						k.values.push_back(static_cast<float>(atof(raw_kernel_value.to_str().c_str())));
					}
				}
				w.push_back(k);
			}
			weights.push_back(w);
		}
		return weights;
	}
	std::vector<float> parse_float_array(const picojson::array &raw_array) const {
		std::vector<float> result;
		for(const auto &raw_value : raw_array){
			result.push_back(static_cast<float>(atof(raw_value.to_str().c_str())));
		}
		return result;
	}
	StepModel parse_step(const picojson::value &raw_step) const {
		StepModel step;
		step.weights = parse_weights(raw_step.get("weight").get<picojson::array>());
		step.biases = parse_float_array(raw_step.get("bias").get<picojson::array>());
		return step;
	}

	std::vector<ImageType> compute_step(
		const StepModel &sm, const std::vector<ImageType> &planes,
		int width, int height, int pitch)
	{
		const int num_in_planes = static_cast<int>(planes.size());
		const int num_weights = static_cast<int>(sm.weights.size());
		std::vector<ImageType> out_planes(num_weights);
		for(int i = 0; i < num_weights; ++i){ out_planes[i] = ImageType(pitch * height); }
#pragma omp parallel
		{
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
			ImageType in_block_buffer(in_block_size * num_in_planes);
			ImageType out_block_buffer(out_block_size * num_weights);
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
	void region_copy(
		ImageType &dst, const ImageType &src, int width, int height,
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
		ImageType &out_buffer, const ImageType &in_buffer, int block_width, int block_height,
		int num_in_planes, const StepModel &sm)
	{
		const int num_out_planes = static_cast<int>(sm.weights.size());
		const int in_block_width = block_width + 8;
		const int in_block_height = block_height + 2;
		for(int i = 0; i < num_out_planes; ++i){
			float *dst_ptr = out_buffer.data() + block_width * block_height * i;
			const __m256 v_bias = _mm256_broadcast_ss(&sm.biases[i]);
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
					block_width, in_block_width, sm.weights[i][j].values.data());
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

public:
	Impl(){ }
	explicit Impl(const char *model_file){
		load_model(model_file);
	}

	void load_model(const char *model_file){
		std::cerr << "Load model ... ";
		picojson::value root;
		picojson::parse(root, load_file_content(model_file));
		picojson::array raw_model = root.get<picojson::array>();
		for(const auto &raw_step : raw_model){
			m_model.push_back(parse_step(raw_step));
		}
		std::cerr << "done!" << std::endl;
	}

	void process(
		float *dst, const float *src, int io_width, int io_height, int io_pitch)
	{
		const auto begin_time = std::chrono::system_clock::now();
		const int num_steps = static_cast<int>(m_model.size());
		const int width = io_width + 2 * num_steps;
		const int height = io_height + 2 * num_steps;
		const int pitch = calculate_pitch(width);
		const int image_size = pitch * height;
		std::vector<ImageType> planes(1, ImageType(image_size));
		for(int y = 0; y < height; ++y){
			const int ty = std::min(std::max(0, y - num_steps), io_height - 1);
			for(int x = 0; x < width; ++x){
				const int tx = std::min(std::max(0, x - num_steps), io_width - 1);
				planes[0][y * pitch + x] = src[ty * io_pitch + tx];
			}
		}
		int current_step = 0;
		for(const auto &step : m_model){
			std::cerr << "Step " << current_step << " (" << planes.size() << " planes)" << std::endl;
			const int cs2 = current_step * 2;
			auto out_planes = compute_step(
				step, planes, width - cs2, height - cs2, pitch);
			out_planes.swap(planes);
			++current_step;
		}
		assert(planes.size() == 1);
		for(int y = 0; y < io_height; ++y){
			for(int x = 0; x < io_width; ++x){
				dst[y * io_pitch + x] = planes[0][y * pitch + x];
			}
		}
		const auto end_time = std::chrono::system_clock::now();
		std::cerr << "Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count() << " [ms]" << std::endl;
	}

};

//-----------------------------------------------------------------------------
// pImpl
//-----------------------------------------------------------------------------
Waifu2x::Waifu2x()
	: m_impl(new Waifu2x::Impl())
{ }
Waifu2x::Waifu2x(const char *model_file)
	: m_impl(new Waifu2x::Impl(model_file))
{ }
Waifu2x::Waifu2x(const Waifu2x &obj)
	: m_impl(new Waifu2x::Impl(*obj.m_impl))
{ }
Waifu2x::~Waifu2x() = default;

Waifu2x &Waifu2x::operator=(const Waifu2x &obj){
	m_impl.reset(new Impl(*obj.m_impl));
	return *this;
}

void Waifu2x::load_model(const char *model_file){
	m_impl->load_model(model_file);
}

void Waifu2x::process(float *dst, const float *src, int width, int height, int pitch){
	m_impl->process(dst, src, width, height, pitch);
}
