#ifndef WAIFU2X_AVX2_IMPL_H
#define WAIFU2X_AVX2_IMPL_H

#include "../common/model.h"

namespace waifu2x {

void process_avx2(
	float *dst, const float *src, int width, int height, int pitch,
	const Model &model, bool verbose);

}

#endif

