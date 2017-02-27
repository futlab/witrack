#pragma once

#include <string>
#include <opencv2/core.hpp>

class Rectifier
{
	cv::Mat mapX, mapY;
public:
	cv::Size size;
	Rectifier(const std::string &fileName, double alpha = 0.0);
	void rectify(const cv::Mat &src, cv::Mat &dst);
};
