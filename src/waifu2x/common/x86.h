#ifndef WAIFU2_X86_H
#define WAIFU2_X86_H

#include <cstdlib>

#ifdef _MSC_VER
#include <malloc.h>
#endif

namespace waifu2x {

inline void *aligned_malloc(std::size_t n){
#ifdef _MSC_VER
	// Visual C++
	return _aligned_malloc(n, 32);
#else
	// GCC, Clang, ...
	void *p = nullptr;
	if(posix_memalign(&p, 32, n) != 0){ return nullptr; }
	return p;
#endif
}

inline void aligned_free(void *p){
#ifdef _MSC_VER
	_aligned_free(p);
#else
	free(p);
#endif
}

}

#endif

