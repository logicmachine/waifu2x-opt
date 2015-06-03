#ifndef WAIFU2X_AVX_IMPL_H
#define WAIFU2X_AVX_IMPL_H

#include "../common/model.h"
#include "../common/impl_base.h"

namespace waifu2x {

template <bool ENABLE_FMA, bool ENABLE_AVX2>
class AVXImpl : public ImplBase {

public:
	virtual void prepare(const Model &model) override;

	virtual void process(
		float *dst, const float *src,
		int io_width, int io_height, int io_pitch,
		int x0, int y0, int block_width, int block_height,
		bool verbose = false) override;

};

}

#endif

