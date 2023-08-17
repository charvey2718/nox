#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include "getopt.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <cppflow/cppflow.h>

// function definitions
void about();
void modelInference(int& done_num, int& done_den, bool& complete, const cv::Mat& src, cv::Mat& dst, std::string weights, int dim, int crop, unsigned int batch_size = 1);
cv::Mat openImage(const std::string& file);
void printHelp();
void saveImage(const std::string& file, const cv::Mat image);

int main(int argc, char* argv[])
{
	// POSIX command line parameters
	std::string inputFilename;
	std::string outputFilename;
	std::string weightsFilename;
	bool overwrite = false;
	bool color = false;
	bool grayscale = false;
	int batch_size = 1;
	int patch_size = 512;
	int stride = 128;

    // command line options
	int c;
	while (1)
	{
		static struct option long_options[] =
		{
			{"batch", required_argument, 0, 'b'},
			{"color", no_argument, 0, 'c'},
			{"file", required_argument, 0, 'f'},
			{"grayscale", no_argument, 0, 'g'},
			{"help", no_argument, 0, 'h'},
			{"output", required_argument, 0, 'o'},
			{"patch", required_argument, 0, 'p'},
			{"replace", no_argument, 0, 'r'},
			{"stride", required_argument, 0, 's'},
			{"version", no_argument, 0, 'v'},
			{"weights", required_argument, 0, 'w'},
			{0, 0, 0, 0}
		};

		int option_index = 0; // stores option index
		c = getopt_long(argc, argv, "b:cf:gho:p:rs:vw:", long_options, &option_index);

		if (c == -1) break; // end of options
        
        switch (c)
		{
			case 0:
				break; // option set a flag
			
			case 'b':
				batch_size = atoi(optarg);
				break;
				
			case 'c':
				color = true;
				break;
			
			case 'f':
				inputFilename = optarg;
				break;
			
			case 'g':
				grayscale = true;
				break;
			
			case 'h':
				printHelp();
				return 0;
			
			case 'o':
				outputFilename = optarg;
				break;
			
			case 'p':
				patch_size = atoi(optarg);
				break;
			
			case 'r':
				overwrite = true;
				break;
			
			case 's':
				stride = atoi(optarg);
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
		std::cout << "Loading input image... ";
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
		if (overwrite) std::cout << "Warning: Will overwrite " + outputFilename << std::endl;
		else
		{
			std::cerr << 
			"Error: Output file already exists."  << std::endl <<
			"  Run with '-r' switch activated (run 'nox --help' for usage guide) or change\n"
			"  output filename." << std::endl;
			return 1;
		}
	}
	
	// 3. Load weights
	if(color && grayscale)
	{
		std::cerr << "Error: Cannot use both grayscale and color weights."  << std::endl << "  Run 'nox --help' for usage guide." << std::endl;
		return 1;
	}
	if(weightsFilename.length() < 1)
	{
		if(grayscale || (!color && inputImage.channels() == 1)) weightsFilename = "noxGeneratorGrayscale.pb";
		else if(color || (!grayscale && inputImage.channels() == 3)) weightsFilename = "noxGeneratorColor.pb";
	}
	if(cv::utils::fs::exists(weightsFilename)) std::cout << "Weights in: " << weightsFilename << std::endl;
	else
	{
		std::cerr << "Error: Model weights not found." << std::endl;
		return 1;
	}
	
	// 4. Prepare input
	std::cout << "Input image dimensions: " << inputImage.cols << " * " << inputImage.rows << " * " << inputImage.channels() << std::endl;
	if(inputImage.channels() == 1 && color)
	{
		std::cout << "Converting to color... ";
		cvtColor(inputImage,inputImage,cv::COLOR_GRAY2BGR);
		std::cout << "done" << std::endl;
	}
	else if(inputImage.channels() == 3 && grayscale)
	{
		std::cout << "Converting to grayscale... ";
		cvtColor(inputImage,inputImage,cv::COLOR_BGR2GRAY);
		std::cout << "done" << std::endl;
	}
	cv::Scalar iden(1., 1., 1.);
	if(inputImage.channels() == 1) iden = cv::Scalar(1.);
	inputImage = inputImage*2. - iden;
	
	// 5. Process input
	cv::Mat out = cv::Mat::zeros(inputImage.size(), inputImage.type()); // to hold output
	int crop = int((patch_size - stride)/2);
	std::cout << "Patch size: " << patch_size << std::endl;
	std::cout << "Stride: " << stride << std::endl;
	if (crop >= 0)
	{
		int done_num = 0;
		int done_den;
		bool complete = false;
		std::cout << "Batch size: " << batch_size << std::endl;
		std::thread run([&](){modelInference(done_num, done_den, complete, inputImage, out, weightsFilename, patch_size, crop, batch_size);});
		while(!complete)
		{
			if(done_den > 0) std::cout << "\rProcessing... " << int(ceil(100.*done_num/done_den)) << "% (press CTRL+C to stop)" << std::flush;
			std::fflush(stdout);
			
			std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		}
		if(done_den > 0) std::cout << "\rProcessing... 100%                       " << std::endl;
		run.join();
		out = (out + iden)/2;
		out = cv::min(cv::max(out, 0.), 1.);
	}
	else
	{
		std::cerr << "Error: stride may not exceed patch size."  << std::endl << "  Run 'nox --help' for usage guide." << std::endl;
		return 1;
	}
	
	// 6. Save image
	out.convertTo(out, CV_16U, 65536.0); // Convert 32-bit to 16-bit before saving
	saveImage(outputFilename, out);
	
    return 0;
}

void about()
{
	std::cout << "nox version 1.0" << std::endl;
	std::cout << "(c) by Christopher M. Harvey 2023" << std::endl << std::endl;
}

void modelInference(int& done_num, int& done_den, bool& complete, const cv::Mat& src, cv::Mat& dst, std::string weights, int dim, int crop, unsigned int batch_size)
{
    // Add border around noisy image for patch borderCropping
	cv::Mat nsyPlusBrdr;
    cv::copyMakeBorder(src, nsyPlusBrdr, crop, crop, crop, crop, cv::BORDER_REFLECT);
    int sizeX = nsyPlusBrdr.cols; // image width
    int sizeY = nsyPlusBrdr.rows; // image height
	
	// Split into patches
	std::vector<cv::Mat> patches;	
	int fromRow = 0; // row pixel index
	int fromCol = 0; // col pixel index
	int nRows = 0;
	int nCols = 0;
	bool finalRow = false;
	while(true) // loop rows
	{
		nRows++;
		if(fromRow + dim >= sizeY)
		{
			fromRow = sizeY - dim;
			finalRow = true;
		}
		
		bool finalCol = false;
		while(true) // loop cols
		{
			if(finalRow) nCols++; // count columns on final row only
            
			if(fromCol + dim >= sizeX)
			{
				fromCol = sizeX - dim;
				finalCol = true;
			}
            
			fromCol = std::max(0, fromCol);
			fromRow = std::max(0, fromRow);
			patches.push_back(nsyPlusBrdr(cv::Rect(fromCol, fromRow, fromCol + dim < sizeX ? dim : sizeX - fromCol, fromRow + dim < sizeY ? dim : sizeY - fromRow)));
            
			if(finalCol)
			{
				fromCol = 0;
				break;
			}
			else fromCol += (dim - 2*crop);
		}
		
		if(finalRow) break;
		else fromRow += (dim - 2*crop);
	}
	done_den = patches.size();
	
	// Inference
	std::vector<cv::Mat> inferred_patches(patches.size());
	auto f = [&](unsigned int from, unsigned int num) // lambda expression to process batch in thread
	{
		cppflow::model* model = NULL; // Tensorflow version
		// cv::dnn::Net model; // OpenCV version
		if(cv::utils::fs::exists(weights))
		{
			try
			{
				// Tensorflow version - requires Tensorflow C API DLL
				model = new cppflow::model(weights, cppflow::model::FROZEN_GRAPH);
				
				// OpenCV version - not used as OpenCV cannot read the model
				// model = cv::dnn::readNet(weights);
			}
			catch (cv::Exception& e)
			{
				std::cerr << e.msg << std::endl;
				return;
			}
		}
		// model.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
		// model.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
		
		// Infer a batch
		std::vector<float> batch; // Tensorflow version - Build batch as std::vector of floats and convert to tensor of appropriate shape
		// std::vector<cv::Mat> batch; // OpenCV version - Batch is a vector of cv::Mat
		for(unsigned int i = from; i < patches.size() && i < from + num; ++i) // i is patch index
		{
			// build batch
			cv::Mat flat = patches[i].clone().reshape(1, patches[i].total()*patches[i].channels());
			std::vector<float> flatPatch = patches[i].isContinuous()? flat : flat.clone();
			batch.insert(batch.end(), flatPatch.begin(), flatPatch.end());

			unsigned int current_batch_size = int(batch.size() / patches[i].rows / patches[i].cols / patches[i].channels());
			if(current_batch_size == batch_size || i == patches.size() - 1) // when batch is filled (or have run out of patches)...
			{
				// Tensorflow version - requires Tensorflow C API DLL
				auto batchTensor = cppflow::tensor(batch, {current_batch_size, patches[i].rows, patches[i].cols, patches[i].channels()}); // ...put images in tensor
				auto outputTensor = ((*model)({{"inputs:0", batchTensor}}, {{"Identity:0"}}))[0]; // and infer batch
				std::vector<float> outputVector = outputTensor.get_data<float>();
				
				// OpenCV version - not used as OpenCV cannot read the model
				// cv::Mat blob = cv::dnn::blobFromImages(batch);
				// model.setInput(blob);
				// cv::Mat outblob = model.forward(); // ...infer batch
				// std::vector<cv::Mat> outarray;
				// cv::dnn::imagesFromBlob(outblob, outarray);
				
				// store inferred patches in corresponding place in vector
				size_t count = 0; // Tensorflow version - 'count' is number of stored floats
				// size_t count = 0; // OpenCV version - 'count' is number of stored patches
				for(size_t j = i + 1 - current_batch_size; j < i + 1; ++j) // j is patch index
				{
					// Tensorflow version - copy relevant patch floats from output to std::vector, convert to cv::Mat and reshape
					size_t num = patches[j].rows*patches[j].cols*patches[j].channels();
					std::vector patch = std::vector<float>(outputVector.begin() + count, outputVector.begin() + count + num);
					inferred_patches[j] = cv::Mat(patch, true);
					inferred_patches[j] = inferred_patches[j].reshape(patches[i].channels(), patches[i].rows);
					count += num;
					
					// OpenCV version
					// inferred_patches[j] = outarray[count];
					// count++;
				}
				done_num += current_batch_size;
				
				batch.clear(); // clear batch ready to be filled with next set of patches
			}
		}
		if(model != NULL) delete model;
	};
	// Tensorflow version - Tensorflow C API handles multithreading automatically
	f(0, patches.size());
	
	// OpenCV version - distribute the processing to the available number of threads
	// unsigned int num_threads = std::thread::hardware_concurrency();
	// if(num_threads == 0) num_threads = 1;
	// int patches_per_thread = patches.size()/num_threads;
	// if(patches_per_thread == 0)
	// {
		// patches_per_thread = 1;
		// num_threads = patches.size();
	// }
	// std::vector<std::thread> threads;
	// for(unsigned int i = 0; i < num_threads; i++)
	// {
		// std::thread th(f, patches_per_thread*i, (i == num_threads - 1 ? patches.size() - patches_per_thread*i : patches_per_thread));
		// threads.push_back(std::move(th));
	// }
	// for(auto& i:threads) i.join();
	
	
	// Reassemble image
	fromRow = 0; // row pixel index
	fromCol = 0; // col pixel index
	int placeDim = dim - 2*crop; // dimension of each patch without crop to be placed in assembled image
	int count = 0;
	for(int i = 0; i < nRows; ++i)
	{
		if(i == nRows - 1) fromRow = (sizeY - 2*crop) - placeDim;
		
		for(int j = 0; j < nCols; ++j)
		{
			if(j == nCols - 1) fromCol = (sizeX - 2*crop) - placeDim;
            
			fromCol = std::max(0, fromCol);
			fromRow = std::max(0, fromRow);
			int w = fromCol + placeDim < sizeX - 2*crop ? placeDim:sizeX - 2*crop - fromCol;
			int h = fromRow + placeDim < sizeY - 2*crop ? placeDim:sizeY - 2*crop - fromRow;
			inferred_patches[count](cv::Rect(crop, crop, w, h)).copyTo(
				dst(cv::Rect(fromCol, fromRow, w, h)));
			count = count + 1;
            
			if(j == nCols - 1)
			{
				fromCol = 0;
				break;
			}
			else fromCol += placeDim;
		}
        
		fromRow += placeDim;
	}
	complete = true;
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
		"  -b,  --batch     number of patches to infer simultaneously (default: 1)\n");
	printf(
		"  -c,  --color     use color weights (grayscale input will be duplicated as RGB\n"
		"                   channels; RGB output will be converted to grayscale)\n");
	printf(
		"  -f,  --file      filename of input starry image\n");
	printf(
		"  -g,  --grayscale   use grayscale weights (color input will be converted to\n"
		"                   grayscale; output will be grayscale\n");
	printf(
		"  -h,  --help      print this help information\n");
	printf(
		"  -o,  --output    filename of output starless image (otherwise '_nox' will be\n"
		"                   appended\n\n");
	printf(
		"  -p,  --patch     width and height of square patches that image is decomposed\n"
		"                   into for inference (default: 512)\n");
	printf(
		"  -r,  --replace   overwrite if output file already exists (default: false)\n");
	printf(
		"  -s,  --stride    distance in pixels between adjacent patches (default: 128)\n");
	printf(
		"  -v,  --version   about nox\n");
	printf(
		"  -w,  --weights   filename of model weights (.pb) (otherwise either\n"
		"                   'noxGeneratorColor.pb' or 'noxGeneratorGrayscale.pb' will be\n"
		"                   selected).");
	printf("\n\n");
}

void saveImage(const std::string& file, const cv::Mat image)
{
	cv::imwrite(file, image);
}
