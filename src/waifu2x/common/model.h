#ifndef WAIFU2X_MODEL_H
#define WAIFU2X_MODEL_H

#include <cstdlib>
#include <cassert>
#include "picojson.h"

namespace waifu2x {

class KernelModel {
private:
	int m_width;
	int m_height;
	std::vector<float> m_values;
public:
	KernelModel()
		: m_width(0)
		, m_height(0)
		, m_values()
	{ }
	explicit KernelModel(const picojson::array &raw_kernel)
		: m_width(0)
		, m_height(0)
		, m_values()
	{
		m_height = static_cast<int>(raw_kernel.size());
		for(const auto &raw_kernel_line_value : raw_kernel){
			const auto &raw_kernel_line =
				raw_kernel_line_value.get<picojson::array>();
			m_width = static_cast<int>(raw_kernel_line.size());
			for(const auto &raw_kernel_value : raw_kernel_line){
				m_values.push_back(static_cast<float>(
					atof(raw_kernel_value.to_str().c_str())));
			}
		}
	}

	int width() const { return m_width; }
	int height() const { return m_height; }
	const float *values() const { return m_values.data(); }
};

class StepModel {
private:
	std::vector<std::vector<KernelModel>> m_weights;
	std::vector<float> m_biases;
public:
	StepModel()
		: m_weights()
		, m_biases()
	{ }
	explicit StepModel(const picojson::value &raw_step)
		: m_weights()
		, m_biases()
	{
		const auto raw_weights = raw_step.get("weight").get<picojson::array>();
		for(const auto &raw_kernel_outer_value : raw_weights){
			const auto raw_kernel_outer =
				raw_kernel_outer_value.get<picojson::array>();
			m_weights.emplace_back();
			for(const auto &raw_kernel_inner_value : raw_kernel_outer){
				const auto raw_kernel_inner =
					raw_kernel_inner_value.get<picojson::array>();
				m_weights.back().emplace_back(raw_kernel_inner);
			}
		}
		const auto raw_biases = raw_step.get("bias").get<picojson::array>();
		for(const auto &raw_value : raw_biases){
			m_biases.push_back(
				static_cast<float>(atof(raw_value.to_str().c_str())));
		}
		assert(m_weights.size() == m_biases.size());
	}

	const std::size_t num_input_planes() const { return m_weights[0].size(); }
	const std::size_t num_output_planes() const { return m_weights.size(); }

	const std::vector<std::vector<KernelModel>> &weights() const { return m_weights; }
	const std::vector<KernelModel> &weights(std::size_t i) const { return m_weights[i]; }
	const std::vector<float> &biases() const { return m_biases; }
	float biases(std::size_t i) const { return m_biases[i]; }
};

class Model {
private:
	std::vector<StepModel> m_steps;
public:
	Model()
		: m_steps()
	{ }
	explicit Model(const std::string &model_json)
		: m_steps()
	{
		picojson::value root;
		picojson::parse(root, model_json);
		const auto raw_model = root.get<picojson::array>();
		for(const auto &raw_step : raw_model){
			m_steps.emplace_back(raw_step);
		}
	}

	const std::size_t num_steps() const { return m_steps.size(); }
	const std::vector<StepModel> &steps() const { return m_steps; }
	const StepModel &steps(std::size_t i) const { return m_steps[i]; }
};

}

#endif

