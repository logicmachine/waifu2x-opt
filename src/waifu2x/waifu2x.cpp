#include "waifu2x.h"
#include "common/model.h"
#include "common/impl_base.h"
#include "common/aligned_buffer.h"
#include "avx2/avx2_impl.h"
#include <omp.h>

class Waifu2x::Impl {

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

public:
	Impl()
		: m_impl()
		, m_num_steps(0)
		, m_num_threads(omp_get_max_threads())
		, m_block_width(DEFAULT_BLOCK_WIDTH)
		, m_block_height(DEFAULT_BLOCK_HEIGHT)
	{ }
	explicit Impl(const std::string &model_json)
		: m_impl()
		, m_num_steps(0)
		, m_num_threads(omp_get_max_threads())
		, m_block_width(DEFAULT_BLOCK_WIDTH)
		, m_block_height(DEFAULT_BLOCK_HEIGHT)
	{
		load_model(model_json);
	}

	void load_model(const std::string &model_json){
		m_impl.reset(new waifu2x::AVX2Impl());
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

//-----------------------------------------------------------------------------
// pImpl
//-----------------------------------------------------------------------------
Waifu2x::Waifu2x()
	: m_impl(new Waifu2x::Impl())
{ }
Waifu2x::Waifu2x(const std::string &model_json)
	: m_impl(new Waifu2x::Impl(model_json))
{ }
Waifu2x::~Waifu2x() = default;

void Waifu2x::load_model(const std::string &model_json){
	m_impl->load_model(model_json);
}

void Waifu2x::set_num_threads(int num_threads){
	m_impl->set_num_threads(num_threads);
}
void Waifu2x::set_image_block_size(int width, int height){
	m_impl->set_image_block_size(width, height);
}

void Waifu2x::process(
	float *dst, const float *src, int width, int height, int pitch,
	bool verbose)
{
	m_impl->process(dst, src, width, height, pitch, verbose);
}
