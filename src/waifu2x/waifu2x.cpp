#define WAIFU2X_BUILD_SHARED_LIBRARY
#include "waifu2x.h"
#include "common/model.h"
#include "common/impl_base.h"
#include "common/aligned_buffer.h"
#include "common/x86.h"
#include "avx/avx_impl.h"
#include "dft_avx/dft_avx_impl.h"
#include <omp.h>
#include <iostream>
#include <algorithm>
#include <memory>
#include <cstdint>
#include <cassert>

// #define USE_DFT_ALGORITHM

namespace waifu2x {

inline int get_max_threads(){
#ifdef _OPENMP
	return omp_get_max_threads();
#else
	return 1;
#endif
}

class Waifu2xImpl {

private:
	static const int DEFAULT_BLOCK_WIDTH = 512;
	static const int DEFAULT_BLOCK_HEIGHT = 512;
	static const int MIN_BLOCK_WIDTH = 128;
	static const int MIN_BLOCK_HEIGHT = 64;

	std::unique_ptr<waifu2x::ImplBase> m_impl;
	int m_num_steps;
	int m_num_threads;
	int m_block_width;
	int m_block_height;

	// noncopyable
	Waifu2xImpl(const Waifu2xImpl &) = delete;
	Waifu2xImpl &operator=(const Waifu2xImpl &) = delete;

public:
	Waifu2xImpl()
		: m_impl()
		, m_num_steps(0)
		, m_num_threads(get_max_threads())
		, m_block_width(DEFAULT_BLOCK_WIDTH)
		, m_block_height(DEFAULT_BLOCK_HEIGHT)
	{ }
	explicit Waifu2xImpl(const std::string &model_json)
		: m_impl()
		, m_num_steps(0)
		, m_num_threads(get_max_threads())
		, m_block_width(DEFAULT_BLOCK_WIDTH)
		, m_block_height(DEFAULT_BLOCK_HEIGHT)
	{
		load_model(model_json);
	}

	void load_model(const std::string &model_json){
#ifndef USE_DFT_ALGORITHM
		if(test_fma()){
			if(test_avx2()){
				// FMA + AVX2
				m_impl.reset(new waifu2x::AVXImpl<true, true>());
			}else{
				// FMA + AVX
				m_impl.reset(new waifu2x::AVXImpl<true, false>());
			}
		}else if(test_avx()){
			// AVX
			m_impl.reset(new waifu2x::AVXImpl<false, false>());
		}else{
			assert(!"Unsupported CPU");
		}
#else
		if(test_fma()){
			m_impl.reset(new waifu2x::DFTAVXImpl<true>());
		}else if(test_avx()){
			m_impl.reset(new waifu2x::DFTAVXImpl<false>());
		}else{
			assert(!"Unsupported CPU");
		}
#endif
		const waifu2x::Model model(model_json);
		m_num_steps = static_cast<int>(model.num_steps());
		m_impl->prepare(model);
		m_impl->set_num_threads(m_num_threads);
		m_impl->set_block_size(m_block_width, m_block_height);
	}

	void set_num_threads(int num_threads){
		m_num_threads = num_threads;
		if(m_impl){ m_impl->set_num_threads(num_threads); }
	}
	void set_image_block_size(int width, int height){
		m_block_width =
			(width + MIN_BLOCK_WIDTH - 1) & ~(MIN_BLOCK_WIDTH - 1);
		m_block_height =
			(height + MIN_BLOCK_HEIGHT - 1) & ~(MIN_BLOCK_HEIGHT - 1);
		if(m_impl){ m_impl->set_block_size(width, height); }
	}

	int num_steps() const { return m_num_steps; }

	void process(
		float *dst, const float *src, int width, int height, int pitch,
		bool verbose)
	{
		assert(m_impl);
		const int in_block_width = m_block_width - m_num_steps * 2;
		const int in_block_height = m_block_height - m_num_steps * 2;
		const int num_x_blocks =
			(width + in_block_width - 1) / in_block_width;
		const int num_y_blocks =
			(height + in_block_height - 1) / in_block_height;
		const int num_blocks = num_x_blocks * num_y_blocks;

		for(int yb = 0, done = 0; yb < num_y_blocks; ++yb){
			const int y0 = in_block_height * yb;
			const int block_height =
				std::min(in_block_height, height - y0);
			for(int xb = 0; xb < num_x_blocks; ++xb, ++done){
				const int x0 = in_block_width * xb;
				const int block_width =
					std::min(in_block_width, width - x0);
				if(verbose){
					std::cerr << "Block " << done << "/"
					          << num_blocks << std::endl;
				}
				m_impl->process(
					dst, src, width, height, pitch,
					x0, y0, block_width, block_height,
					verbose);
			}
		}
	}

};

}

//-----------------------------------------------------------------------------
// Export functions
//-----------------------------------------------------------------------------
extern "C" {

W2X_EXPORT W2xHandle w2x_create_handle(const char *model_json){
	return reinterpret_cast<W2xHandle>(
		new waifu2x::Waifu2xImpl(std::string(model_json)));
}
W2X_EXPORT void w2x_destroy_handle(W2xHandle handle){
	delete reinterpret_cast<waifu2x::Waifu2xImpl *>(handle);
}

W2X_EXPORT int w2x_set_num_threads(W2xHandle handle, int num_threads){
	if(!handle){ return -1; }
	auto ptr = reinterpret_cast<waifu2x::Waifu2xImpl *>(handle);
	ptr->set_num_threads(num_threads);
	return 0;
}
W2X_EXPORT int w2x_set_block_size(W2xHandle handle, int width, int height){
	if(!handle){ return -1; }
	auto ptr = reinterpret_cast<waifu2x::Waifu2xImpl *>(handle);
	ptr->set_image_block_size(width, height);
	return 0;
}

W2X_EXPORT int w2x_get_num_steps(const W2xHandle handle){
	if(!handle){ return -1; }
	auto ptr = reinterpret_cast<const waifu2x::Waifu2xImpl *>(handle);
	return ptr->num_steps();
}

W2X_EXPORT int w2x_process(
	W2xHandle handle, float *dst, const float *src,
	int width, int height, int pitch, int verbose)
{
	if(!handle){ return -1; }
	auto ptr = reinterpret_cast<waifu2x::Waifu2xImpl *>(handle);
	ptr->process(dst, src, width, height, pitch, verbose != 0);
	return 0;
}

}

