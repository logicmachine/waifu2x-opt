#ifndef WAIFU2X_COMMON_IMPL_BASE_H
#define WAIFU2X_COMMON_IMPL_BASE_H

#include "model.h"

namespace waifu2x {

class ImplBase {

protected:
	Model m_model;
	int m_num_threads;
	int m_block_width;
	int m_block_height;

public:
	ImplBase(){ }

	virtual void prepare(const Model &model){
		m_model = model;
	}

	virtual void set_num_threads(int num_threads){
		m_num_threads = num_threads;
	}
	virtual void set_block_size(int width, int height){
		m_block_width = width;
		m_block_height = height;
	}

	virtual void process(
		float *dst, const float *src,
		int io_width, int io_height, int io_pitch,
		int x0, int y0, int block_width, int block_height,
		bool verbose = false) = 0;

};

}

#endif

