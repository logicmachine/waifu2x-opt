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
	std::string m_input_file;
	std::string m_output_file;
	bool m_verbose;

public:
	ProgramOptions(int argc, char *argv[])
		: m_has_error(false)
		, m_num_threads(0)
		, m_model_dir(".")
		, m_noise_level(0)
		, m_scale(1.0)
		, m_input_file()
		, m_output_file("out.png")
		, m_verbose(false)
	{
		std::vector<std::string> s(argc);
		for(int i = 0; i < argc; ++i){ s[i] = std::string(argv[i]); }
		bool enable_noise = false;
		bool enable_scale = false;
		if(argc == 1){ m_has_error = true; }
		for(int i = 1; i < argc; ++i){
			if(s[i] == "-h" || s[i] == "--help"){
				m_has_error = true;
			}else if(i + 1 < argc && (s[i] == "-j" || s[i] == "--jobs")){
				m_num_threads = atoi(s[i + 1].c_str());
				++i;
			}else if(i + 1 < argc && s[i] == "--model_dir"){
				m_model_dir = s[i + 1];
				++i;
			}else if(i + 1 < argc && s[i] == "--noise_level"){
				const int nl = atoi(s[i + 1].c_str());
				if(nl != 1 && nl != 2){
					std::cerr << "Noise level must be 1 or 2" << std::endl;
					m_has_error = true;
				}else{
					m_noise_level = nl;
				}
				++i;
			}else if(i + 1 < argc && s[i] == "--scale_ratio"){
				m_scale = atof(s[i + 1].c_str());
				++i;
			}else if(i + 1 < argc && (s[i] == "-m" || s[i] == "--mode")){
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
				}
				++i;
			}else if(i + 1 < argc && (s[i] == "-o" || s[i] == "--output")){
				m_output_file = s[i + 1];
				++i;
			}else if(i + 1 < argc && (s[i] == "-i" || s[i] == "--input")){
				m_input_file = s[i + 1];
				++i;
			}else if(s[i] == "-v" || s[i] == "--verbose"){
				m_verbose = true;
			}else{
				m_input_file = s[i];
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
	bool is_verbose() const { return m_verbose; }

};

#endif
