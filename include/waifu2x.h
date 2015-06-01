#ifndef WAIFU2X_H
#define WAIFU2X_H

#include <string>
#include <memory>

class Waifu2x {

private:
	// pImpl
	class Impl;
	std::unique_ptr<Impl> m_impl;

	// noncopyable
	Waifu2x(const Waifu2x &obj) = delete;
	Waifu2x &operator=(const Waifu2x &obj) = delete;

public:
	// Constructors and destructor
	Waifu2x();
	explicit Waifu2x(const std::string &model_json);
	~Waifu2x();

	// Load model data
	void load_model(const std::string &model_json);

	// Tuning parameters
	void set_num_threads(int num_threads);
	void set_image_block_size(int width, int height);

	// Run filter
	void process(
		float *dst, const float *src, int width, int height, int pitch,
		bool verbose = false);

};

#endif
