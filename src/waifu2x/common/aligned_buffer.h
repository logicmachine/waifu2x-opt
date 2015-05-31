#ifndef WAIFU2X_COMMON_ALIGNED_BUFFER_H
#define WAIFU2X_COMMON_ALIGNED_BUFFER_H

#include <cstring>
#include "x86.h"

namespace waifu2x {

class AlignedBuffer {
private:
	std::size_t m_size;
	float *m_pointer;

public:
	AlignedBuffer()
		: m_size(0)
		, m_pointer(nullptr)
	{ }
	explicit AlignedBuffer(std::size_t n)
		: m_size(n)
		, m_pointer(nullptr)
	{
		if(m_size == 0){ return; }
		m_pointer =
			reinterpret_cast<float *>(aligned_malloc(n * sizeof(float)));
	}
	AlignedBuffer(const AlignedBuffer &ib)
		: m_size(ib.m_size)
		, m_pointer(nullptr)
	{
		if(ib.m_pointer == nullptr){ return; }
		m_pointer =
			reinterpret_cast<float *>(aligned_malloc(m_size * sizeof(float)));
		memcpy(m_pointer, ib.m_pointer, m_size * sizeof(float));
	}
	AlignedBuffer(AlignedBuffer &&ib)
		: m_size(ib.m_size)
		, m_pointer(ib.m_pointer)
	{
		ib.m_pointer = nullptr;
	}
	~AlignedBuffer(){
		if(m_pointer){ aligned_free(m_pointer); }
	}

	AlignedBuffer &operator=(const AlignedBuffer &ib){
		if(m_pointer){ aligned_free(m_pointer); }
		m_size = ib.m_size;
		if(ib.m_pointer){
			m_pointer =
				reinterpret_cast<float *>(aligned_malloc(m_size * sizeof(float)));
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

#endif

