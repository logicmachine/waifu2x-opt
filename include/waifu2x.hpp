#ifndef WAIFU2X_HPP
#define WAIFU2X_HPP

#include "waifu2x.h"

class Waifu2x {

private:
	W2xHandle m_handle;

	// noncopyable
	Waifu2x(const Waifu2x &obj) = delete;
	Waifu2x &operator=(const Waifu2x &obj) = delete;

public:
	// Constructors and destructor
	Waifu2x()
		: m_handle(nullptr)
	{ }
	explicit Waifu2x(const std::string &model_json)
		: m_handle(nullptr)
	{
		load_model(model_json);
	}
	~Waifu2x(){
		w2x_destroy_handle(m_handle);
	}

	// Load model data
	void load_model(const std::string &model_json){
		if(m_handle){
			w2x_destroy_handle(m_handle);
			m_handle = nullptr;
		}
		m_handle = w2x_create_handle(model_json.c_str());
	}

	// Tuning parameters
	void set_num_threads(int num_threads){
		w2x_set_num_threads(m_handle, num_threads);
	}
	void set_image_block_size(int width, int height){
		w2x_set_block_size(m_handle, width, height);
	}

	// Model specification
	int num_steps() const {
		return w2x_get_num_steps(m_handle);
	}

	// Run filter
	void process(
		float *dst, const float *src, int width, int height, int pitch,
		bool verbose = false)
	{
		w2x_process(m_handle, dst, src, width, height, pitch, verbose);
	}

};

#endif
