// #include <cstdlib>
// #include <cmath>
// #include <ctime>
// #include <fstream>
#include <iostream>
// #include <map>
// #include <sstream>
#include <string>
// #include <vector>
#include "getopt.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

// function definitions
void about();
cv::Mat openImage(const std::string& file);
void printHelp();

int main(int argc, char* argv[])
{
	std::string inputFilename;
	std::string outputFilename;
	std::string weightsFilename;
	bool overwrite = false;

    // command line options
	int c;
	while (1)
	{
		static struct option long_options[] =
		{
			{"file", required_argument, 0, 'f'},
			{"output", required_argument, 0, 'o'},
			{"help", no_argument, 0, 'h'},
			{"color", no_argument, 0, 'c'},
			{"grayscale", no_argument, 0, 'g'},
			{"replace", no_argument, 0, 'r'},
			{"version", no_argument, 0, 'v'},
			{"weights", required_argument, 0, 'w'},
			{0, 0, 0, 0}
		};

		int option_index = 0; // stores option index
		c = getopt_long(argc, argv, "f:o:hcgrvw:", long_options, &option_index);

		if (c == -1) break; // end of options
        
        switch (c)
		{
			case 0:
				break; // option set a flag
				
			case 'f':
				inputFilename = optarg;
				break;
				
			case 'o':
				outputFilename = optarg;
				break;
				
			case 'h':
				printHelp();
				return 0;
			
			case 'r':
				overwrite = true;
				break;

			case 'v':
				about();
				return 0;
				
			case 'w':
				weightsFilename = optarg;
				break;
				
			case '?': // getopt_long already printed an error message
				printHelp();
				return 1;
		}
    }

    // 1. Load the input file
	cv::Mat inputImage;
    if(inputFilename.length() > 0)
    {
		std::cout << "Input filename: " << inputFilename << std::endl;
		std::cout << "Loading input image...   ";
		inputImage = openImage(inputFilename);
        if(inputImage.empty())
		{
			std::cout << "failed" << std::endl;
			return 1;
		}
		else std::cout << "done" << std::endl;
		
    }
    else
    {
        std::cerr << "Error: No input file specified." << std::endl << "  Run 'nox --help' for usage guide." << std::endl;
        return 1;
    }
	
	// 2. Output filename
	if(outputFilename.length() < 1)
	{
		outputFilename = inputFilename;
		size_t lastindex = outputFilename.find_last_of(".");
		outputFilename.insert(lastindex, "_nox");
	}
	std::cout << "Output filename: " << outputFilename << std::endl;
	if(cv::utils::fs::exists(outputFilename))
	{
		if (overwrite) std::cout << "Warning: Will overwrite: " + outputFilename << std::endl;
		else std::cerr << 
			"Error: Output file already exists."  << std::endl <<
			"  Run with '-r' switch activated (run 'nox --help' for usage guide) or change\n"
			"  output filename." << std::endl;
	}
	
	// 3. Load weights
	cv::dnn::Net model;
	bool color = (inputImage.channels() == 3);
	if(weightsFilename.length() < 1)
	{
		if (color) weightsFilename = "noxGeneratorColor.pb";
		else weightsFilename = "noxGeneratorGrayscale.pb";
	}
	std::cout << "Weights in: " << weightsFilename << std::endl;
	if(cv::utils::fs::exists(weightsFilename))
	{
		try
		{
			std::cout << "Loading weights...   ";
			model = cv::dnn::readNet(weightsFilename);
		}
		catch (cv::Exception& e)
		{
			std::cerr << e.msg << std::endl;
			return 1;
		}
		
	}
	std::cout << "done" << std::endl;

    return 0;
}

void about()
{
	std::cout << "nox" << " by Christopher M. Harvey" << std::endl;
	std::cout << "(c) by Christopher M. Harvey 2023" << std::endl << std::endl;
}

cv::Mat openImage(const std::string& file) // returns 32 bit float
{
	cv::Mat image;
	if(cv::utils::fs::exists(file)) image = cv::imread(file, cv::IMREAD_UNCHANGED); // read the file
	if(!image.empty()) // check for invalid input
	{
		if(image.channels() == 4) cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
		if(image.depth() == CV_8U) image.convertTo(image, CV_32F, 1./255.); // convert 8 bit integer to 32 bit float scaling to range 0 to 1
		else if(image.depth() == CV_16U) image.convertTo(image, CV_32F, 1./65535.); // convert 16 bit integer to 32 bit float scaling to range 0 to 1
		else if(image.depth() == CV_16F) image.convertTo(image, CV_32F); // convert 16 bit integer to 32 bit float without rescaling
		else if(image.depth() == CV_32F) return image; // already 32 bit float
		else return cv::Mat(); // if not 8, 16 or 32 bit, return empty
	}
	return image; // image will be empty if file does not exist
}

void printHelp()
{
	printf("nox usage:\n");
	printf(
		"  -f,  --file      filename of input starry image\n");
	printf(
		"  -o,  --output    filename of output starless image (otherwise '_nox' will be\n"
		"                   appended\n\n");
	printf(
		"  -c,  --color     use color weights (grayscale input will be duplicated as RGB\n"
		"                   channels; RGB output will be converted to grayscale)\n");
	printf(
		"  -g,  --grayscale   use grayscale weights (color input will be converted to\n"
		"                   grayscale; output will be grayscale\n");
	printf(
		"  -h,  --help      print this help information\n");
	printf(
		"  -r,  --replace   overwrite if output file already exists (default: false)\n");
	printf(
		"  -v,  --version   about nox\n");
	printf(
		"  -w,  --weights   filename of model weights (.pb) (otherwise\n"
		"                   'noxGeneratorColor.pb' or 'noxGeneratorGrayscale.pb' will be\n"
		"                   used.");
	printf("\n");	
}