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
void TFmodelInference(int& done_num, int& done_den, bool& complete, const cv::Mat& src, cv::Mat& dst, std::string weights, int dim, int crop, unsigned int batch_size = 1);
void CVmodelInference(int& done_num, int& done_den, bool& complete, const cv::Mat& src, cv::Mat& dst, std::string weights, int dim, int crop, unsigned int batch_size = 1);
cv::Mat openImage(const std::string& file);
void printHelp();
void saveImage(const std::string& file, const cv::Mat image);

// function definitions for temporary stretching of linear data
void AutoStretch(const cv::Mat input, double autoShadows, double autoBackground, bool linked, std::vector<double>& shadows, std::vector<double>& midtones);
void TransformHistogram(cv::Mat& image, double redShadows, double redMidtones, double redHighlights, double redLower, double redUpper, double greenShadows, double greenMidtones, double greenHighlights, double greenLower, double greenUpper, double blueShadows, double blueMidtones, double blueHighlights, double blueLower, double blueUpper, cv::Mat mask = cv::Mat());
double MTF(double x, double s, double m, double h, double lb, double ub);
double MTFInverse(double y, double x0, double s, double m, double h, double lb, double ub);
void UntransformHistogram(cv::Mat& image, const cv::Mat& image0, double redShadows, double redMidtones, double redHighlights, double redLower, double redUpper, double greenShadows, double greenMidtones, double greenHighlights, double greenLower, double greenUpper, double blueShadows, double blueMidtones, double blueHighlights, double blueLower, double blueUpper);

int main(int argc, char* argv[])
{
	// POSIX command line parameters
	std::string inputFilename;
	std::string outputFilename;
	std::string weightsFilename;
	bool overwrite = false;
	bool color = false;
	bool grayscale = false;
	bool linearData = false;
	int batch_size = 1;
	int bit_depth = 16;
	int patch_size = 512;
	int stride = 128;
	double autoShadows = -0.75;
	double autoBackground = 0.1875;

    // command line options
	int c;
	while (1)
	{
		static struct option long_options[] =
		{
			{"batch", required_argument, 0, 'b'},
			{"bitdepth", required_argument, 0, 'B'},
			{"color", no_argument, 0, 'c'},
			{"shadows", required_argument, 0, 'd'},
			{"file", required_argument, 0, 'f'},
			{"grayscale", no_argument, 0, 'g'},
			{"help", no_argument, 0, 'h'},
			{"linear", no_argument, 0, 'l'},
			{"output", required_argument, 0, 'o'},
			{"patch", required_argument, 0, 'p'},
			{"replace", no_argument, 0, 'r'},
			{"stride", required_argument, 0, 's'},
			{"target", required_argument, 0, 't'},
			{"version", no_argument, 0, 'v'},
			{"weights", required_argument, 0, 'w'},
			{0, 0, 0, 0}
		};

		int option_index = 0; // stores option index
		c = getopt_long(argc, argv, "b:B:cd:f:ghlo:p:rs:t:vw:", long_options, &option_index);

		if (c == -1) break; // end of options
        
        switch (c)
		{
			case 0:
				break; // option set a flag
			
			case 'b':
				batch_size = atoi(optarg);
				break;

			case 'B':
				bit_depth = atoi(optarg);

				if(bit_depth != 16 && bit_depth != 32 && bit_depth != 64)
				{
					std::cerr << "Error: Invalid bit depth '" << bit_depth << "'. Allowed values are 16, 32, or 64." << std::endl;
					printHelp();
					return 1;
				}
				break;
				
			case 'c':
				color = true;
				break;
				
			case 'd':
				autoShadows = atof(optarg);
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
			
			case 'l':
				linearData = true;
				break;
			
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
				
			case 't':
				autoBackground = atof(optarg);
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
		std::cout << "Loading input image... " << std::flush;
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
	
	// Color conversions
	if(inputImage.channels() == 1 && color)
	{
		std::cout << "Converting to color... " << std::flush;
		cvtColor(inputImage,inputImage,cv::COLOR_GRAY2BGR);
		std::cout << "done" << std::endl;
	}
	else if(inputImage.channels() == 3 && grayscale)
	{
		std::cout << "Converting to grayscale... " << std::flush;
		cvtColor(inputImage,inputImage,cv::COLOR_BGR2GRAY);
		std::cout << "done" << std::endl;
	}
	cv::Mat noxInputImage = inputImage.clone();

	// Apply MTF stretch to linear data
	std::vector<double> shadows, midtones;
	if(linearData)
	{
		std::cout << "Auto stretching using d = " << autoShadows << " and t = " << autoBackground << "... " << std::flush;;
		std::cout << "done" << std::endl;
		AutoStretch(noxInputImage, autoShadows, autoBackground, false, shadows, midtones);
		std::cout << "Shadows clipping = " << shadows[0] << " (R), " << shadows[1] << " (G), " << shadows[2] << " (B)" << std::endl;
		std::cout << "Midtones = " << midtones[0] << " (R), " << midtones[1] << " (G), " << midtones[2] << " (B)" << std::endl;
		TransformHistogram(noxInputImage, shadows[0], midtones[0], 1., 0., 1., shadows[1], midtones[1], 1., 0., 1., shadows[2], midtones[2], 1., 0., 1.);
	}

	// Transform to -1 to +1 range
	cv::Scalar iden(1., 1., 1.);
	if(noxInputImage.channels() == 1) iden = cv::Scalar(1.);
	noxInputImage = noxInputImage*2. - iden;
	
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
		std::thread run([&](){TFmodelInference(done_num, done_den, complete, noxInputImage, out, weightsFilename, patch_size, crop, batch_size);});
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
	
	// 6. Reverse MTF stretch
	if(linearData)
	{
		UntransformHistogram(out, inputImage, shadows[0], midtones[0], 1., 0., 1., shadows[1], midtones[1], 1., 0., 1., shadows[2], midtones[2], 1., 0., 1.);
	}
	
	// 7. Save image
	switch (bit_depth)
	{
		case 16:
			// 16-bit unsigned integer (default)
			out.convertTo(out, CV_16U, 65535.0);
			break;

		case 32:
			// 32-bit floating point
			out.convertTo(out, CV_32F);
			break;

		case 64:
			// 64-bit floating point
			out.convertTo(out, CV_64F);
			break;
	}

	saveImage(outputFilename, out);
	
    return 0;
}

void about()
{
	std::cout << "nox version 1.1" << std::endl;
	std::cout << "(c) by Christopher M. Harvey 2025" << std::endl << std::endl;
}

void TFmodelInference(int& done_num, int& done_den, bool& complete, const cv::Mat& src, cv::Mat& dst, std::string weights, int dim, int crop, unsigned int batch_size)
{
	// Tensorflow version - requires Tensorflow C API DLL
	
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
	
	// Inference
	cppflow::model model(weights, cppflow::model::FROZEN_GRAPH);
	std::vector<cv::Mat> inferred_patches(patches.size());
	auto f = [&](unsigned int from, unsigned int num) // lambda expression to process a group of patches in thread
	{
		// Infer a batch
		std::vector<float> batch; // Build batch as std::vector of floats and convert to tensor of appropriate shape
		for(unsigned int i = from; i < patches.size() && i < from + num; ++i) // i is patch index
		{
			// build batch
			cv::Mat flat = patches[i].clone().reshape(1, patches[i].total()*patches[i].channels());
			std::vector<float> flatPatch = patches[i].isContinuous()? flat : flat.clone();
			batch.insert(batch.end(), flatPatch.begin(), flatPatch.end());

			unsigned int current_batch_size = int(batch.size() / patches[i].rows / patches[i].cols / patches[i].channels());
			if(current_batch_size == batch_size || i == patches.size() - 1 || i == from + num - 1) // when batch is filled (or have run out of patches)...
			{
				auto batchTensor = cppflow::tensor(batch, {current_batch_size, patches[i].rows, patches[i].cols, patches[i].channels()}); // ...put images in tensor
				auto outputTensor = (model({{"inputs:0", batchTensor}}, {{"Identity:0"}}))[0]; // and infer batch
				std::vector<float> outputVector = outputTensor.get_data<float>();
				
				// store inferred patches in corresponding place in vector
				size_t count = 0; // number of stored floats
				for(size_t j = i + 1 - current_batch_size; j < i + 1; ++j) // j is patch index
				{
					// Tensorflow version - copy relevant patch floats from output to std::vector, convert to cv::Mat and reshape
					size_t num2 = patches[j].rows*patches[j].cols*patches[j].channels();
					std::vector patch = std::vector<float>(outputVector.begin() + count, outputVector.begin() + count + num2);
					inferred_patches[j] = cv::Mat(patch, true);
					inferred_patches[j] = inferred_patches[j].reshape(patches[i].channels(), patches[i].rows);
					count += num2;
				}
				done_num += current_batch_size;
				
				batch.clear(); // clear batch ready to be filled with next set of patches
			}
		}
	};
	std::cout << "Number of patches: " << patches.size() << std::endl;
	unsigned int num_batches = ceil(1.*patches.size()/batch_size);
	std::cout << "Number of batches: " << num_batches << std::endl;
	unsigned int num_threads = std::thread::hardware_concurrency();
	if(num_threads == 0) num_threads = 1;
	if(num_batches < num_threads) num_threads = num_batches;
	std::cout << "Threads used: " << num_threads << std::endl;
	unsigned int batches_per_thread = ceil(1.*num_batches/num_threads);
	std::cout << "Batches per thread: " << batches_per_thread << std::endl;
	unsigned int patches_per_thread = batches_per_thread*batch_size;
	done_den = patches_per_thread*num_threads;
	
	std::vector<std::thread> threads;
	for(unsigned int i = 0; i < num_threads; i++)
	{
		if(i != num_threads - 1)
		{
			std::thread th(f, patches_per_thread*i, patches_per_thread);
			threads.push_back(std::move(th));
		}
		else // final thread may need to recalculate some patches to maintain batch size
		{
			unsigned int overshoot = patches_per_thread*i + patches_per_thread - patches.size();
			std::thread th(f, patches_per_thread*i - overshoot, patches_per_thread);
			threads.push_back(std::move(th));
		}
	}
	for(auto& i:threads) i.join();
	
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

void CVmodelInference(int& done_num, int& done_den, bool& complete, const cv::Mat& src, cv::Mat& dst, std::string weights, int dim, int crop, unsigned int batch_size)
{
	// OpenCV version - not used as OpenCV cannot read the model yet
	
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
	
	// Inference
	std::vector<cv::Mat> inferred_patches(patches.size());
	auto f = [&](unsigned int from, unsigned int num) // lambda expression to process a group of patches in thread
	{
		cv::dnn::Net model;
		if(cv::utils::fs::exists(weights))
		{
			try
			{
				model = cv::dnn::readNet(weights);
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
		std::vector<cv::Mat> batch;
		for(unsigned int i = from; i < patches.size() && i < from + num; ++i) // i is patch index
		{
			// build batch
			batch.push_back(patches[i]);

			unsigned int current_batch_size = batch.size();
			if(current_batch_size == batch_size || i == patches.size() - 1 || i == from + num - 1) // when batch is filled (or have run out of patches)...
			{
				cv::Mat blob = cv::dnn::blobFromImages(batch);
				model.setInput(blob);
				cv::Mat outblob = model.forward(); // ...infer batch
				std::vector<cv::Mat> outarray;
				cv::dnn::imagesFromBlob(outblob, outarray);
				
				// store inferred patches in corresponding place in vector
				size_t count = 0; // number of stored patches
				for(size_t j = i + 1 - current_batch_size; j < i + 1; ++j) // j is patch index
				{
					inferred_patches[j] = outarray[count];
					count++;
				}
				done_num += current_batch_size;
				
				batch.clear(); // clear batch ready to be filled with next set of patches
			}
		}
	};
	std::cout << "Number of patches: " << patches.size() << std::endl;
	unsigned int num_batches = ceil(1.*patches.size()/batch_size);
	std::cout << "Number of batches: " << num_batches << std::endl;
	unsigned int num_threads = std::thread::hardware_concurrency();
	if(num_threads == 0) num_threads = 1;
	if(num_batches < num_threads) num_threads = num_batches;
	std::cout << "Threads used: " << num_threads << std::endl;
	unsigned int batches_per_thread = ceil(1.*num_batches/num_threads);
	std::cout << "Batches per thread: " << batches_per_thread << std::endl;
	unsigned int patches_per_thread = batches_per_thread*batch_size;
	done_den = patches_per_thread*num_threads;
	
	std::vector<std::thread> threads;
	for(unsigned int i = 0; i < num_threads; i++)
	{
		if(i != num_threads - 1)
		{
			std::thread th(f, patches_per_thread*i, patches_per_thread);
			threads.push_back(std::move(th));
		}
		else // final thread may need to recalculate some patches to maintain batch size
		{
			unsigned int overshoot = patches_per_thread*i + patches_per_thread - patches.size();
			std::thread th(f, patches_per_thread*i - overshoot, patches_per_thread);
			threads.push_back(std::move(th));
		}
	}
	for(auto& i:threads) i.join();
	
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
		"  -B,  --bitdepth  output bit depth: 16 | 32 | 64\n"
        "                   16 = uint16 (default), 32 = float32, 64 = float64\n");
	printf(
		"  -c,  --color     use color weights (grayscale input will be duplicated as RGB\n"
		"                   channels; RGB output will be converted to grayscale)\n");
	printf(
		"  -d,  --shadows   number of standard deviations relative to the median pixel\n"
		"                   value for shadows clipping in temporary midtones transfer\n"
		"                   function stretch (default: -0.75)\n");
	printf(
		"  -f,  --file      filename of input starry image\n");
	printf(
		"  -g,  --grayscale   use grayscale weights (color input will be converted to\n"
		"                   grayscale; output will be grayscale\n");
	printf(
		"  -h,  --help      print this help information\n");
	printf(
		"  -l,  --linear    automatically apply temporary midtones transfer function\n"
		"                   stretch to linear data (default: off)\n");
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
		"  -t,  --target    target mean pixel value in temporary midtones transfer\n"
		"                   function stretch (default: 0.1875)\n");
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

void AutoStretch(const cv::Mat input, double autoShadows, double autoBackground, bool linked, std::vector<double>& shadows, std::vector<double>& midtones)
{
	std::vector<cv::Mat> bgr_planes;
	cv::split(input, bgr_planes);
	
	cv::Mat image; // to trial transformations on
	if(linked)
	{
		bgr_planes.erase(std::remove_if( // remove planes that are completely full of zeros
			bgr_planes.begin(), bgr_planes.end(),
			[](const cv::Mat& plane) {
				return cv::countNonZero(plane) < 1;
			}), bgr_planes.end());
		cv::hconcat(bgr_planes, image); // BGR planes need joining together into one channel
	}
	else image = input;

	double m0, m1, s; // m1 and s will hold the solved shadows and midtones values after iterating
	const double eps = 1e-7; // convergence tolerance
	for(int i = image.channels() - 1; i >= 0; --i)
	{
		if(!linked)
		{
			image = bgr_planes[i];
			if(cv::countNonZero(image) < 1) // don't attempt to auto-stretch planes that are completely full of zeros
			{
				shadows.push_back(0);
				midtones.push_back(0.5);
				continue;
			}
		}
		cv::Scalar mean, stdev;
		cv::meanStdDev(image, mean, stdev);
		std::vector<float> vecFromMat;
		image.reshape(0, 1).copyTo(vecFromMat);
		std::nth_element(vecFromMat.begin(), vecFromMat.begin() + vecFromMat.size()/2, vecFromMat.end());
		double median = vecFromMat[vecFromMat.size()/2];
		s = median + autoShadows*stdev[0];
		if(s <= 0) s = 0;
		m0 = s + 1e-7; m1 = 1.0 - 1e-7; // range of values

		// Use Dekker's method to find midtones value for the given autoBackground
		// https://en.wikipedia.org/wiki/Brent%27s_method#Dekker's_method
		cv::Mat im = image.clone();
		TransformHistogram(im, s, m0, 1.0, 0.0, 1.0, s, m0, 1.0, 0.0, 1.0, s, m0, 1.0, 0.0, 1.0);
		double fx0 = cv::mean(im)[0] - autoBackground;
		im = image.clone();
		TransformHistogram(im, s, m1, 1.0, 0.0, 1.0, s, m1, 1.0, 0.0, 1.0, s, m1, 1.0, 0.0, 1.0);
		double fx1 = cv::mean(im)[0] - autoBackground;
		double m2 = m0; // m0 is the previous value of m1 and [m1, m2] always contains 0
		double fx2 = fx0;
		while(true)
		{
			if((fx1 < 0) == (fx2 < 0))
			{
				m2 = m0;
				fx2 = fx0;
			}
			if(abs(fx2) < abs(fx1)) // ensure fx1 is the smallest value so far
			{
				m0 = m1; fx0 = fx1;
				m1 = m2; fx1 = fx2;
				m2 = m0; fx2 = fx0;
			}
			double m = (m1 + m2)/2; // midpoint
			if(abs(m - m1) <= eps) break; // stop iterating
			double p = (m1 - m0)*fx1; // p/q is the secant step
			double q;
			if(p >= 0)
			{
				q = fx0 - fx1;
			}
			else
			{
				q = fx1 - fx0;
				p = -p;
			}
			m0 = m1;
			fx0 = fx1;
			if(p <= (m - m1)*q) // secant
			{
				m1 = m1 + p/q;
				im = image.clone();
				TransformHistogram(im, s, m1, 1.0, 0.0, 1.0, s, m1, 1.0, 0.0, 1.0, s, m1, 1.0, 0.0, 1.0);
				fx1 = cv::mean(im)[0] - autoBackground;
				// std::cout << "secant: " << std::to_string(m1) << " " << std::to_string(fx1) << std::endl;
			}
			else // bisection
			{
				m1 = m;
				im = image.clone();
				TransformHistogram(im, s, m1, 1.0, 0.0, 1.0, s, m1, 1.0, 0.0, 1.0, s, m1, 1.0, 0.0, 1.0);
				fx1 = cv::mean(im)[0] - autoBackground;
				// std::cout << "bisection: " << std::to_string(m1) << " " << std::to_string(fx1) << std::endl;
			}
		}
		if(m1<s || m1 > 1.0) m1 = 0.5;
		if(!linked)
		{
			shadows.push_back(s);
			midtones.push_back(m1);
		}
	}
	if(linked)
	{
		for(int i = 0; i < input.channels(); ++i)
		{
			shadows.push_back(s);
			midtones.push_back(m1);
		}
	}
}

void TransformHistogram(cv::Mat& image, double redShadows, double redMidtones, double redHighlights, double redLower, double redUpper, double greenShadows, double greenMidtones, double greenHighlights, double greenLower, double greenUpper, double blueShadows, double blueMidtones, double blueHighlights, double blueLower, double blueUpper, cv::Mat mask)
{
	int nRows = image.rows;
	auto f = [&](int i0, int nRows) // define a lambda expression to loop through nRows starting at row i
	{
		int nCols = image.cols*image.channels();
		for(int i = i0; i < i0 + nRows; ++i)
		{
			float* p = image.ptr<float>(i);
			float* pm = NULL;
			if(!mask.empty()) pm = mask.ptr<float>(i);
			for(int j = 0; j < nCols; j += image.channels())
			{
				if(image.channels() == 1)
				{
					if(pm != NULL) p[j] = p[j] + pm[j]*(MTF(p[j], redShadows, redMidtones, redHighlights, redLower, redUpper) - p[j]);
					else p[j] = MTF(p[j], redShadows, redMidtones, redHighlights, redLower, redUpper);
				}
				else
				{
					if(pm != NULL)
					{
						float maskval = pm[j/3];
						if(maskval < 0.) maskval = 0.;
						else if(maskval > 1.) maskval = 1.;
						p[j] = p[j] + maskval*(MTF(p[j], blueShadows, blueMidtones, blueHighlights, blueLower, blueUpper) - p[j]);
						p[j + 1] = p[j + 1] + maskval*(MTF(p[j + 1], greenShadows, greenMidtones, greenHighlights, greenLower, greenUpper) - p[j + 1]);
						p[j + 2] = p[j + 2] + maskval*(MTF(p[j + 2], redShadows, redMidtones, redHighlights, redLower, redUpper) - p[j + 2]);
					}
					else
					{
						p[j] = MTF(p[j], blueShadows, blueMidtones, blueHighlights, blueLower, blueUpper);
						p[j + 1] = MTF(p[j + 1], greenShadows, greenMidtones, greenHighlights, greenLower, greenUpper);
						p[j + 2] = MTF(p[j + 2], redShadows, redMidtones, redHighlights, redLower, redUpper);
					}
				}
			}
		}
	};
	unsigned int num_threads = std::thread::hardware_concurrency();
	if(num_threads == 0) num_threads = 1;
	unsigned int rows_per_thread = nRows/num_threads;
	std::vector<std::thread> threads;
	for(unsigned int i = 0; i < num_threads; i++)
	{
		std::thread th(f, rows_per_thread*i, (i == num_threads - 1 ? nRows - rows_per_thread*i : rows_per_thread));
		threads.push_back(std::move(th));
	}
	for(auto& i : threads) i.join();
	image = cv::min(cv::max(image, 0.), 1.);
}

double MTF(double x, double s, double m, double h, double lb, double ub)
{
	double y;
	if(x < s) y = 0.0; // shadows clipping
	else if(x > h) y = 1.0; // highlights clipping
	else
	{
		x = (x - s)/(h - s); // rescale x coordinate after clipping
		m = (m - s)/(h - s); // rescale m after clipping
		y = (m - 1.0)*x/((2.0*m - 1.0)*x - m); // MTF
	}
	y = (y - lb)/(ub - lb); // bounds expansion
	
	return y;
}

double MTFInverse(double y, double x0, double s, double m, double h, double lb, double ub)
{
    double x;
    y = y*(ub - lb) + lb; // undo bounds expansion
    if(y <= 0.0) x = x0; // shadows clipped
    else if(y >= 1.0) x = x0; // highlights clipped
	else
	{
		m = (m - s)/(h - s);
		x = m*y/(2.0*m*y - m - y + 1.0);
		x = x*(h - s) + s; // rescale x after clipping
	}

    return x;
}

void UntransformHistogram(cv::Mat& image, const cv::Mat& image0, double redShadows, double redMidtones, double redHighlights, double redLower, double redUpper, double greenShadows, double greenMidtones, double greenHighlights, double greenLower, double greenUpper, double blueShadows, double blueMidtones, double blueHighlights, double blueLower, double blueUpper)
{
	int nRows = image.rows;
	auto f = [&](int i0, int nRows) // define a lambda expression to loop through nRows starting at row i
	{
		int nCols = image.cols*image.channels();
		for(int i = i0; i < i0 + nRows; ++i)
		{
			float* p = image.ptr<float>(i);
			const float* p0 = image0.ptr<float>(i); // same pixel in original image (used in MTFInverse to restore original pixel value in case of clipping)
			for(int j = 0; j < nCols; j += image.channels())
			{
				if(image.channels() == 1)
				{
					p[j] = MTFInverse(p[j], p0[j], redShadows, redMidtones, redHighlights, redLower, redUpper);
				}
				else
				{
					p[j] = MTFInverse(p[j], p0[j], blueShadows, blueMidtones, blueHighlights, blueLower, blueUpper);
					p[j + 1] = MTFInverse(p[j + 1], p0[j + 1], greenShadows, greenMidtones, greenHighlights, greenLower, greenUpper);
					p[j + 2] = MTFInverse(p[j + 2], p0[j + 2], redShadows, redMidtones, redHighlights, redLower, redUpper);
				}
			}
		}
	};
	unsigned int num_threads = std::thread::hardware_concurrency();
	if(num_threads == 0) num_threads = 1;
	unsigned int rows_per_thread = nRows/num_threads;
	std::vector<std::thread> threads;
	for(unsigned int i = 0; i < num_threads; i++)
	{
		std::thread th(f, rows_per_thread*i, (i == num_threads - 1 ? nRows - rows_per_thread*i : rows_per_thread));
		threads.push_back(std::move(th));
	}
	for(auto& i : threads) i.join();
	image = cv::min(cv::max(image, 0.), 1.);
}
