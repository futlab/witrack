#include "rectifier.h"
#include <fstream>
#include <sstream>
#include "rectifier.h"
#include <vector>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

void parseLine(std::vector<double> &dst, const std::string &line)
{
	using namespace std;
	istringstream iss(line);
	string s;
	while (getline(iss, s, ' ')) 
		dst.push_back(stod(s));
}

Rectifier::Rectifier(const std::string & fileName, double alpha)
{
	using namespace std;
	ifstream fs(fileName);
	string line;
	int width, height;
	std::vector<double> distortion, intrinsics, projection, rectification;
	while (getline(fs, line)) {
		if (line == "width") {
			if (getline(fs, line)) 
				width = stoi(line);
		}
		else if (line == "height") {
			if (getline(fs, line)) 
				height = stoi(line);
		}
		else if (line == "distortion") {
			if (getline(fs, line)) 
				parseLine(distortion, line);
		}
		else if (line == "camera matrix") {
			for (int x = 3; x && getline(fs, line); x--) 
				parseLine(intrinsics, line);
		}
		else if (line == "projection") {
			for (int x = 3; x && getline(fs, line); x--) 
				parseLine(projection, line);
		}
		else if (line == "rectification") {
			for (int x = 3; x && getline(fs, line); x--) 
				parseLine(rectification, line);
		}
	}
	fs.close();
	size = cv::Size(width, height);
	cv::Mat intr(3, 3, CV_64F, intrinsics.data());
	cv::Mat ncm = cv::getOptimalNewCameraMatrix(intr, distortion, size, alpha);
	cv::Mat r(3, 3, CV_64F, rectification.data());
	cv::initUndistortRectifyMap(intr, distortion, r, ncm, size, CV_32F, mapX, mapY);
}

void Rectifier::rectify(const cv::Mat & src, cv::Mat & dst)
{
	cv::remap(src, dst, mapX, mapY, cv::INTER_LINEAR);
}
