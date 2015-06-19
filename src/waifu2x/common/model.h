#ifndef WAIFU2X_MODEL_H
#define WAIFU2X_MODEL_H

#include <string>
#include <vector>
#include <cstdlib>

namespace waifu2x {

class Model {

private:
	std::vector<int> m_bias_offsets;
	std::vector<float> m_biases;
	std::vector<std::vector<int>> m_coeff_offsets;
	std::vector<float> m_coeffs;
	std::vector<int> m_num_in_planes;
	std::vector<int> m_num_out_planes;

public:
	Model();
	explicit Model(const std::string &model_json);

	std::size_t num_steps() const;
	std::size_t num_in_planes(int step) const;
	std::size_t num_out_planes(int step) const;

	const float *biases() const;
	const float *biases(int step) const;
	float bias(int step, int out_plane) const;

	const float *coeffs() const;
	const float *coeffs(int step) const;
	const float *coeffs(int step, int out_plane) const;
	const float *coeffs(int step, int out_plane, int in_plane) const;

};

}

#endif

