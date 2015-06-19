#include <cassert>
#include "picojson.h"
#include "model.h"

namespace waifu2x {

namespace {

std::vector<float> load_coeffs(const picojson::array &raw_kernel){
	std::vector<float> coeffs;
	assert(static_cast<int>(raw_kernel.size()) == 3);
	for(const auto &raw_kernel_line_value : raw_kernel){
		const auto &raw_kernel_line =
			raw_kernel_line_value.get<picojson::array>();
		assert(static_cast<int>(raw_kernel_line.size()) == 3);
		for(const auto &raw_kernel_value : raw_kernel_line){
			coeffs.push_back(static_cast<float>(
				atof(raw_kernel_value.to_str().c_str())));
		}
	}
	return coeffs;
}

void load_step(
	std::vector<int> &num_in_planes, std::vector<int> &num_out_planes,
	std::vector<int> &bias_offsets, std::vector<float> &biases,
	std::vector<std::vector<int>> &coeff_offsets, std::vector<float> &coeffs,
	const picojson::value &raw_step)
{
	const auto raw_weights = raw_step.get("weight").get<picojson::array>();
	int coeff_current = static_cast<int>(coeffs.size());
	coeff_offsets.push_back(std::vector<int>());
	bias_offsets.push_back(static_cast<int>(biases.size()));
	int num_outer = static_cast<int>(raw_weights.size()), num_inner = 0;
	for(const auto &raw_kernel_outer_value : raw_weights){
		const auto raw_kernel_outer =
			raw_kernel_outer_value.get<picojson::array>();
		coeff_offsets.back().push_back(coeff_current);
		num_inner = static_cast<int>(raw_kernel_outer.size());
		for(const auto &raw_kernel_inner_value : raw_kernel_outer){
			const auto raw_kernel_inner =
				raw_kernel_inner_value.get<picojson::array>();
			const auto v = load_coeffs(raw_kernel_inner);
			for(const auto &c : v){ coeffs.push_back(c); }
			coeff_current += static_cast<int>(v.size());
		}
	}
	const auto raw_biases = raw_step.get("bias").get<picojson::array>();
	for(const auto &raw_value : raw_biases){
		biases.push_back(static_cast<float>(atof(raw_value.to_str().c_str())));
	}
	num_in_planes.push_back(num_inner);
	num_out_planes.push_back(num_outer);
}

}

Model::Model()
	: m_bias_offsets()
	, m_biases()
	, m_coeff_offsets()
	, m_coeffs()
	, m_num_in_planes()
	, m_num_out_planes()
{ }

Model::Model(const std::string &model_json)
	: m_bias_offsets()
	, m_biases()
	, m_coeff_offsets()
	, m_coeffs()
{
	picojson::value root;
	picojson::parse(root, model_json);
	const auto raw_model = root.get<picojson::array>();
	for(const auto &raw_step : raw_model){
		load_step(
			m_num_in_planes, m_num_out_planes,
			m_bias_offsets, m_biases,
			m_coeff_offsets, m_coeffs, raw_step);
	}
}

std::size_t Model::num_steps() const {
	return m_num_in_planes.size();
}
std::size_t Model::num_in_planes(int step) const {
	return m_num_in_planes[step];
}
std::size_t Model::num_out_planes(int step) const {
	return m_num_out_planes[step];
}

const float *Model::biases() const {
	return m_biases.data();
}
const float *Model::biases(int step) const {
	return m_biases.data() + m_bias_offsets[step];
}
float Model::bias(int step, int out_plane) const {
	return m_biases[m_bias_offsets[step] + out_plane];
}

const float *Model::coeffs() const {
	return m_coeffs.data();
}
const float *Model::coeffs(int step) const {
	return m_coeffs.data() + m_coeff_offsets[step][0];
}
const float *Model::coeffs(int step, int out_plane) const {
	return m_coeffs.data() + m_coeff_offsets[step][out_plane];
}
const float *Model::coeffs(int step, int out_plane, int in_plane) const {
	return m_coeffs.data() + m_coeff_offsets[step][out_plane] + in_plane * 9;
}

}
