#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "options.h"
#include "waifu2x.h"

namespace {

const char NOISE1_MODEL[]  = "noise1_model.json";
const char NOISE2_MODEL[]  = "noise2_model.json";
const char SCALE2X_MODEL[] = "scale2.0x_model.json";

std::string load_file_content(const char *filename){
	std::ifstream ifs(filename, std::ios::in);
	std::ostringstream oss;
	std::string line;
	while(std::getline(ifs, line)){ oss << line << std::endl; }
	return oss.str();
}

}

int main(int argc, char *argv[]){
	namespace chrono = std::chrono;
	ProgramOptions po(argc, argv);
	if(po.has_error() || po.input_file() == ""){
		po.help();
		return -1;
	}
	// Load input file
	cv::Mat input_image = cv::imread(po.input_file().c_str());
	if(!input_image.data){
		std::cerr << "Can't open " << po.input_file() << std::endl;
		return -2;
	}
	// Load models
	Waifu2x w2x_noise;
	if(po.noise_level() != 0){
		const std::string model_path =
			po.model_dir() + "/" +
			(po.noise_level() == 1 ? NOISE1_MODEL : NOISE2_MODEL);
		const std::string &model_data = load_file_content(model_path.c_str());
		w2x_noise.load_model(model_data);
	}
	Waifu2x w2x_scale;
	if(po.scale() > 1.0){
		const std::string &model_data =
			load_file_content((po.model_dir() + "/" + SCALE2X_MODEL).c_str());
		w2x_scale.load_model(model_data);
	}
	const auto begin_time = chrono::system_clock::now();
	// Convert to YUV and split to each component
	input_image.convertTo(input_image, CV_32F, 1.0 / 255.0);
	cv::cvtColor(input_image, input_image, cv::COLOR_RGB2YUV);
	std::vector<cv::Mat> components;
	cv::split(input_image, components);
	// Noise reduction
	if(po.noise_level() != 0){
		w2x_noise.set_num_threads(po.num_threads());
		if(po.block_width() == 0 && po.block_height() == 0){
			const int padding = w2x_noise.num_steps() * 2;
			w2x_noise.set_image_block_size(
				components[0].cols + padding, components[0].rows + padding);
		}else{
			w2x_noise.set_image_block_size(po.block_width(), po.block_height());
		}
		w2x_noise.process(
			reinterpret_cast<float *>(components[0].data),
			reinterpret_cast<float *>(components[0].data),
			components[0].cols, components[0].rows,
			static_cast<int>(components[0].step / sizeof(float)),
			po.is_verbose());
	}
	// Scaling
	if(po.scale() > 1.0){
		const int target_width = static_cast<int>(input_image.cols * po.scale());
		const int target_height = static_cast<int>(input_image.rows * po.scale());
		w2x_scale.set_num_threads(po.num_threads());
		if(po.block_width() == 0 && po.block_height() == 0){
			const int padding = w2x_scale.num_steps() * 2;
			w2x_scale.set_image_block_size(
				target_width + padding, target_height + padding);
		}else{
			w2x_scale.set_image_block_size(po.block_width(), po.block_height());
		}
		int current_width = input_image.cols;
		int current_height = input_image.rows;
		while(current_width < target_width || current_height < target_height){
			current_width *= 2;
			current_height *= 2;
			cv::resize(
				components[0], components[0],
				cv::Size(current_width, current_height), 0, 0, CV_INTER_NN);
			w2x_scale.process(
				reinterpret_cast<float *>(components[0].data),
				reinterpret_cast<float *>(components[0].data),
				current_width, current_height,
				static_cast<int>(components[0].step / sizeof(float)),
				po.is_verbose());
		}
		cv::resize(
			components[1], components[1],
			cv::Size(target_width, target_height), 0, 0, CV_INTER_CUBIC);
		cv::resize(
			components[2], components[2],
			cv::Size(target_width, target_height), 0, 0, CV_INTER_CUBIC);
		if(current_width != target_width || current_height != target_height){
			cv::resize(
				components[0], components[0],
				cv::Size(target_width, target_height), 0, 0, CV_INTER_CUBIC);
		}
	}
	// Merge components and convert to RGB
	cv::Mat output_image;
	cv::merge(components, output_image);
	cv::cvtColor(output_image, output_image, cv::COLOR_YUV2RGB);
	output_image.convertTo(output_image, CV_8U, 255.0);
	const auto end_time = chrono::system_clock::now();
	// Save output image
	cv::imwrite(po.output_file(), output_image);
	// Dump processing time
	if(po.is_verbose()){
		const auto duration =
			chrono::duration_cast<chrono::milliseconds>(end_time - begin_time);
		std::cerr << "Processing time: " << duration.count()
		          << " [ms]" << std::endl;
	}
	return 0;
}
