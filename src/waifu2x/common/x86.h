#ifndef WAIFU2_X86_H
#define WAIFU2_X86_H

#include <cstdlib>
#include <cstdint>

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h>
#endif

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace waifu2x {

#if defined(_MSC_VER)
#	define ALIGNED(x) __declspec(align(x))
#else
#	define ALIGNED(x) __attribute__((aligned(x)))
#endif

inline void *aligned_malloc(std::size_t n){
#if defined(_MSC_VER)
	// Visual C++
	return _aligned_malloc(n, 64);
#elif defined(__MINGW32__)
	// MinGW
	return __mingw_aligned_malloc(n, 64);
#else
	// GCC, Clang, ...
	void *p = nullptr;
	if(posix_memalign(&p, 32, n) != 0){ return nullptr; }
	return p;
#endif
}

inline void aligned_free(void *p){
#if defined(_MSC_VER)
	_aligned_free(p);
#elif defined(__MINGW32__)
	// MinGW
	__mingw_aligned_free(p);
#else
	free(p);
#endif
}


struct CPUID {
	uint32_t eax, ebx, ecx, edx;
};

#ifdef _MSC_VER
inline void get_cpuid(CPUID *p, int i){
	__cpuid(reinterpret_cast<int *>(p), i);
}
#else
inline void get_cpuid(CPUID *p, int i){
	int *a = reinterpret_cast<int *>(p);
	__cpuid(i, a[0], a[1], a[2], a[3]);
}
#endif

inline bool test_fma(){
	CPUID cpuid;
	get_cpuid(&cpuid, 1);
	return (cpuid.ecx >> 12) & 1;
}
inline bool test_avx2(){
	CPUID cpuid;
	get_cpuid(&cpuid, 7);
	return (cpuid.ebx >> 5) & 1;
}
inline bool test_avx(){
	CPUID cpuid;
	get_cpuid(&cpuid, 1);
	return (cpuid.ecx >> 28) & 1;
}

}

#endif

