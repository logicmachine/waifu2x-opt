#include "waifu2x.h"
#include "common/model.h"
#include "avx2/avx2_impl.h"

class Waifu2x::Impl {

private:
	waifu2x::Model m_model;

public:
	Impl(){ }
	explicit Impl(const std::string &model_json)
		: m_model(model_json)
	{ }

	void load_model(const std::string &model_json){
		m_model = waifu2x::Model(model_json);
	}

	void process(
		float *dst, const float *src, int io_width, int io_height,
		int io_pitch, bool verbose)
	{
		process_avx2(
			dst, src, io_width, io_height, io_pitch, m_model, verbose);
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
Waifu2x::Waifu2x(const Waifu2x &obj)
	: m_impl(new Waifu2x::Impl(*obj.m_impl))
{ }
Waifu2x::~Waifu2x() = default;

Waifu2x &Waifu2x::operator=(const Waifu2x &obj){
	m_impl.reset(new Impl(*obj.m_impl));
	return *this;
}

void Waifu2x::load_model(const std::string &model_json){
	m_impl->load_model(model_json);
}

void Waifu2x::process(
	float *dst, const float *src, int width, int height, int pitch,
	bool verbose)
{
	m_impl->process(dst, src, width, height, pitch, verbose);
}
