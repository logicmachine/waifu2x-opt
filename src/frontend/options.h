#ifndef WAIFU2X_OPTIONS_H
#define WAIFU2X_OPTIONS_H

#include <vector>
#include <string>
#include <cstdlib>

class ProgramOptions {

private:
	bool m_has_error;
	int m_num_threads;
	std::string m_model_dir;
	int m_noise_level;
	double m_scale;
	int m_block_width;
	int m_block_height;
	std::string m_input_file;
	std::string m_output_file;
	bool m_verbose;

	bool is_uint(const std::string &s){
		if(s.empty()){ return false; }
		for(const char c : s){
			if(!isdigit(c)){ return false; }
		}
		return true;
	}
	bool validate_positive_integer(const std::string &s){
		if(!is_uint(s)){ return false; }
		if(atoi(s.c_str()) <= 0){ return false; }
		return true;
	}
	bool validate_scale(const std::string &s){
		bool p = false;
		for(const char c : s){
			if(c == '.'){
				if(p){ return false; }
				p = true;
			}else if(!isdigit(c)){
				return false;
			}
		}
		const double f = atof(s.c_str());
		return f > 1.0;
	}

public:
	ProgramOptions(int argc, char *argv[])
		: m_has_error(false)
		, m_num_threads(0)
		, m_model_dir(".")
		, m_noise_level(0)
		, m_scale(1.0)
		, m_block_width(0)
		, m_block_height(0)
		, m_input_file()
		, m_output_file("out.png")
		, m_verbose(false)
	{
		std::vector<std::string> s(argc + 1);
		for(int i = 0; i < argc; ++i){ s[i] = std::string(argv[i]); }
		bool enable_noise = false;
		bool enable_scale = false;
		if(argc == 1){ m_has_error = true; }
		for(int i = 1; i < argc; ++i){
			if(s[i] == "-h" || s[i] == "--help"){
				m_has_error = true;
				break;
			}else if(s[i] == "-j" || s[i] == "--jobs"){
				if(validate_positive_integer(s[i + 1])){
					m_num_threads = atoi(s[i + 1].c_str());
					++i;
				}else{
					std::cerr << "Number of thread must be a positive integer" << std::endl;
					m_has_error = true;
					break;
				}
			}else if(s[i] == "--model_dir"){
				m_model_dir = s[i + 1];
				++i;
			}else if(s[i] == "--noise_level"){
				const int nl = atoi(s[i + 1].c_str());
				if(validate_positive_integer(s[i + 1]) && nl <= 2){
					m_noise_level = nl;
					++i;
				}else{
					std::cerr << "Noise level must be 1 or 2" << std::endl;
					m_has_error = true;
					break;
				}
			}else if(s[i] == "--scale_ratio"){
				if(validate_scale(s[i + 1])){
					m_scale = atof(s[i + 1].c_str());
					++i;
				}else{
					std::cerr << "Scale must be a number greater than 1.0" << std::endl;
					m_has_error = true;
					break;
				}
			}else if(s[i] == "-m" || s[i] == "--mode"){
				if(s[i + 1] == "noise"){
					enable_noise = true;
					enable_scale = false;
				}else if(s[i + 1] == "scale"){
					enable_noise = false;
					enable_scale = true;
				}else if(s[i + 1] == "noise_scale"){
					enable_noise = true;
					enable_scale = true;
				}else{
					std::cerr << "Mode must be one of <noise|scale|noise_scale>" << std::endl;
					m_has_error = true;
					break;
				}
				++i;
			}else if(s[i] == "-o" || s[i] == "--output"){
				if(s[i + 1].empty()){
					std::cerr << "Output file does not specified" << std::endl;
					m_has_error = true;
					break;
				}else{
					m_output_file = s[i + 1];
					++i;
				}
			}else if(s[i] == "-i" || s[i] == "--input"){
				if(s[i + 1].empty()){
					std::cerr << "Input file does not specified" << std::endl;
					m_has_error = true;
					break;
				}else{
					m_input_file = s[i + 1];
					++i;
				}
			}else if(s[i] == "--block_size"){
				if(i + 2 < argc && validate_positive_integer(s[i + 1]) &&
				   validate_positive_integer(s[i + 2]))
				{
					m_block_width = atoi(s[i + 1].c_str());
					m_block_height = atoi(s[i + 2].c_str());
				}else{
					std::cerr << "Block size is must be specified in two positive integers" << std::endl;
					m_has_error = true;
					break;
				}
			}else if(s[i] == "-v" || s[i] == "--verbose"){
				m_verbose = true;
			}
		}
		if(!enable_noise){ m_noise_level = 0; }
		if(!enable_scale){ m_scale = 1.0; }
	}

	void help(){
		std::cerr << "Options:" << std::endl;
		std::cerr << "  -h [--help]                           : display help message" << std::endl;
		std::cerr << "  -j [--jobs] <num>                     : set number of threads" << std::endl;
		std::cerr << "  --model_dir <dir>                     : set model directory" << std::endl;
		std::cerr << "  --scale_ratio <num>                   : set scale of output image" << std::endl;
		std::cerr << "  --noise_level <1|2>                   : set level of noise reduction" << std::endl;
		std::cerr << "  --block_size <w> <h>                  : set block size" << std::endl;
		std::cerr << "  -m [--mode] <noise|scale|noise_scale> : set processing mode" << std::endl;
		std::cerr << "  -o [--output] <file>                  : set destination file" << std::endl;
		std::cerr << "  -i [--input] <file>                   : set source file" << std::endl;
		std::cerr << "  -v [--verbose]                        : set verbose flag" << std::endl;
	}

	bool has_error() const { return m_has_error; }
	int num_threads() const { return m_num_threads; }
	const std::string &model_dir() const { return m_model_dir; }
	int noise_level() const { return m_noise_level; }
	double scale() const { return m_scale; }
	const std::string &input_file() const { return m_input_file; }
	const std::string &output_file() const { return m_output_file; }
	int block_width() const { return m_block_width; }
	int block_height() const { return m_block_height; }
	bool is_verbose() const { return m_verbose; }

};

#endif
