#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <tuple>
#include <chrono>
#include <cassert>
#include "waifu2x.h"

namespace {

std::string load_file_content(const char *filename){
	std::ifstream ifs(filename, std::ios::in);
	std::ostringstream oss;
	std::string line;
	while(std::getline(ifs, line)){ oss << line << std::endl; }
	return oss.str();
}

std::tuple<int, int, std::vector<float>> load_ppm(const char *filename){
	std::ifstream ifs(filename, std::ios::in | std::ios::binary);
	std::string line;
	getline(ifs, line);
	assert(line == "P6"); // Magic (PPM)
	while(getline(ifs, line) && line[0] == '#');
	std::istringstream size_iss(line);
	int width, height;
	size_iss >> width >> height;
	while(getline(ifs, line) && line[0] == '#');
	std::istringstream depth_iss(line);
	int depth;
	depth_iss >> depth;
	std::vector<float> data(width * height * 3);
	for(int y = 0; y < height; ++y){
		for(int x = 0; x < width; ++x){
			const float r = static_cast<float>(ifs.get());
			const float g = static_cast<float>(ifs.get());
			const float b = static_cast<float>(ifs.get());
			data[(y * width + x) * 3 + 0] = r / depth;
			data[(y * width + x) * 3 + 1] = g / depth;
			data[(y * width + x) * 3 + 2] = b / depth;
		}
	}
	return std::make_tuple(width, height, data);
}
void save_ppm(const char *filename, int width, int height, const std::vector<float> &data){
	std::ofstream ofs(filename, std::ios::out | std::ios::binary);
	ofs << "P6\n";
	ofs << width << " " << height << "\n";
	ofs << 255 << "\n";
	std::vector<char> buffer(width * height * 3);
	for(int i = 0; i < width * height * 3; ++i){
		buffer[i] = std::max(0, std::min(static_cast<int>(data[i] * 255), 255));
	}
	ofs.write(buffer.data(), width * height * 3);
}

std::vector<float> rgb2yuv(const std::vector<float> &data){
	const int n = static_cast<int>(data.size());
	std::vector<float> result(n);
	for(int i = 0; i < n; i += 3){
		const float r = data[i + 0], g = data[i + 1], b = data[i + 2];
		result[i + 0] =  0.257f * r +  0.504f * g +  0.098f * b + 0.0625f; // Y
		result[i + 1] = -0.148f * r + -0.291f * g +  0.439f * b + 0.5f; // Cb
		result[i + 2] =  0.439f * r + -0.368f * g + -0.071f * b + 0.5f; // Cr
	}
	return result;
}
std::vector<float> yuv2rgb(const std::vector<float> &data){
	const int n = static_cast<int>(data.size());
	std::vector<float> result(n);
	for(int i = 0; i < n; i += 3){
		const float y = data[i + 0], cb = data[i + 1], cr = data[i + 2];
		result[i + 0] = 1.164f * (y - 0.0625f) + 1.596f * (cr - 0.5f); // R
		result[i + 1] = 1.164f * (y - 0.0625f) - 0.391f * (cb - 0.5f) - 0.813f * (cr - 0.5f); //G
		result[i + 2] = 1.164f * (y - 0.0625f) + 2.018f * (cb - 0.5f); // B
	}
	return result;
}

std::vector<float> extract_component(const std::vector<float> &data, int c){
	const int n = static_cast<int>(data.size()) / 3;
	std::vector<float> result(n);
	for(int i = 0; i < n; ++i){ result[i] = data[i * 3 + c]; }
	return result;
}
void insert_component(std::vector<float> &dst, const std::vector<float> &data, int c){
	const int n = static_cast<int>(data.size());
	for(int i = 0; i < n; ++i){ dst[i * 3 + c] = data[i]; }
}

std::vector<float> nearest_scale2x(int width, int height, const std::vector<float> &data){
	std::vector<float> result(width * height * 4);
	for(int i = 0; i < height; ++i){
		for(int j = 0; j < width; ++j){
			result[(i * 2 + 0) * (width * 2) + (j * 2 + 0)] = data[i * width + j];
			result[(i * 2 + 0) * (width * 2) + (j * 2 + 1)] = data[i * width + j];
			result[(i * 2 + 1) * (width * 2) + (j * 2 + 0)] = data[i * width + j];
			result[(i * 2 + 1) * (width * 2) + (j * 2 + 1)] = data[i * width + j];
		}
	}
	return result;
}
std::vector<float> bicubic_scale2x(int width, int height, const std::vector<float> &data){
	const float h1 = 0.625f, h2 = -0.125f;
	std::vector<float> result(width * height * 4);
	for(int i = 0; i < height; ++i){
		const int y0 = std::max(0, i - 1), y1 = i;
		const int y2 = std::min(i + 1, height - 1);
		const int y3 = std::min(i + 2, height - 1);
		for(int j = 0; j < width; ++j){
			const int x0 = std::max(0, j - 1), x1 = j;
			const int x2 = std::min(j + 1, width - 1);
			const int x3 = std::min(j + 2, width - 1);
			result[(i * 2 + 0) * (width * 2) + (j * 2 + 0)] = data[y1 * width + x1];
			result[(i * 2 + 0) * (width * 2) + (j * 2 + 1)] =
				(data[y1 * width + x0] + data[y1 * width + x3]) * h2 +
				(data[y1 * width + x1] + data[y1 * width + x2]) * h1;
			result[(i * 2 + 1) * (width * 2) + (j * 2 + 0)] =
				(data[y0 * width + x1] + data[y3 * width + x1]) * h2 +
				(data[y1 * width + x1] + data[y2 * width + x1]) * h1;
			const float t0 =
				(data[y0 * width + x0] + data[y0 * width + x3]) * h2 +
				(data[y0 * width + x1] + data[y0 * width + x2]) * h1;
			const float t1 =
				(data[y1 * width + x0] + data[y1 * width + x3]) * h2 +
				(data[y1 * width + x1] + data[y1 * width + x2]) * h1;
			const float t2 =
				(data[y2 * width + x0] + data[y2 * width + x3]) * h2 +
				(data[y2 * width + x1] + data[y2 * width + x2]) * h1;
			const float t3 =
				(data[y3 * width + x0] + data[y3 * width + x3]) * h2 +
				(data[y3 * width + x1] + data[y3 * width + x2]) * h1;
			result[(i * 2 + 1) * (width * 2) + (j * 2 + 1)] =
				(t0 + t3) * h2 + (t1 + t2) * h1;
		}
	}
	return result;
}

}

int main(int argc, char *argv[]){
	if(argc < 4){
		std::cerr << "Usage: " << argv[0] << " in_file out_file model_file" << std::endl;
		return 0;
	}
	const auto input = load_ppm(argv[1]);
	const int width = std::get<0>(input), height = std::get<1>(input);
	const auto yuv_1x = rgb2yuv(std::get<2>(input));
	const auto y_1x = extract_component(yuv_1x, 0);
	const auto u_1x = extract_component(yuv_1x, 1);
	const auto v_1x = extract_component(yuv_1x, 2);
	const auto y_2x = nearest_scale2x(width, height, y_1x);
	const auto u_2x = bicubic_scale2x(width, height, u_1x);
	const auto v_2x = bicubic_scale2x(width, height, v_1x);
	Waifu2x w2x(load_file_content(argv[3]));
	auto y_w2x = y_2x;
	const auto begin_time = std::chrono::system_clock::now();
	w2x.process(y_w2x.data(), y_2x.data(), width * 2, height * 2, width * 2, true);
	const auto end_time = std::chrono::system_clock::now();
	std::vector<float> yuv_2x(width * height * 4 * 3, 0.5f);
	insert_component(yuv_2x, y_w2x, 0);
	insert_component(yuv_2x, u_2x, 1);
	insert_component(yuv_2x, v_2x, 2);
	const auto rgb2x = yuv2rgb(yuv_2x);
	save_ppm(argv[2], width * 2, height * 2, rgb2x);
	std::cerr << "Processing time: "
	          << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count()
	          << " [ms]" << std::endl;
	return 0;
}
