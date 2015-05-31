#ifndef WAIFU2X_H
#define WAIFU2X_H

#include <memory>

class Waifu2x {

private:
	class Impl;
	std::unique_ptr<Impl> m_impl;

public:
	Waifu2x();
	explicit Waifu2x(const char *model_file);
	Waifu2x(const Waifu2x &obj);
	~Waifu2x();

	Waifu2x &operator=(const Waifu2x &obj);

	void load_model(const char *model_file);
	
	void process(float *dst, const float *src, int width, int height, int pitch);

};

#endif