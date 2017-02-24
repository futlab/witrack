#include <vector>
#include <unordered_map>
#include <map>
#include <memory>
#include "features.h"
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

//template<int width, int height, typename Pixel = uint8_t>
struct FeatureDescriptor
{
	enum {
		height = 16,
		width = 16
	};
	typedef uint8_t Pixel;
	uint32_t data[height];
	int x, y, response;
	int bestMatch, bestX, bestY;
	void scan(const Pixel *center, size_t line)
	{
		Pixel c = *center;
		const Pixel * src = center - width / 2 - (height / 2) * line;
		uint32_t *dst = data;
		for (int y = height; --y >= 0;) {
			uint32_t d = 0;
			for (int x = width; --x >= 0;) {
				d <<= 1;
				if (*(src++) > c) d |= 1;
			}
			*(dst++) = d;
			src += line - width;
		}
		assert(dst == data + height);
	}
	int match(const Pixel *center, size_t line)
	{
		int r = 0;
		Pixel c = *center;
		const Pixel * src = center - width / 2 - (height / 2) * line;
		uint32_t *dst = data;
		const int mask = 1 << (width - 1);
		for (int y = height; --y >= 0;) {
			uint32_t d = *(dst++);
			for (int x = width; --x >= 0; d <<= 1)
				if ((*(src++) > c) == ((d & mask) != 0)) r++;
			src += line - width;
		}
		assert(dst == data + height);
		return r;
	}
	FeatureDescriptor(const cv::KeyPoint &kp) : x((int) kp.pt.x), y((int) kp.pt.y), response((int) kp.response), bestMatch(0) { }
};

class Keyframe
{
public:
	enum {
		cellSize = 32
	};
	std::map<int, FeatureDescriptor> map;
	Keyframe(const cv::Mat &src) {
		auto ffd = cv::FastFeatureDetector::create(20);
		std::vector<cv::KeyPoint> kps;
		ffd->detect(src, kps);
		int mx = src.cols - 9, my = src.rows - 9;
		for (const auto &kp : kps) {
			if (kp.pt.x <= 8 || kp.pt.y <= 8) continue;
			if (kp.pt.x >= mx || kp.pt.y >= my) continue;
			int key = (int(kp.pt.x) / cellSize) | ((int(kp.pt.y) / cellSize) << 16);
			auto it = map.find(key);
			if (it == map.end())
				map.emplace(key, kp);
			else if (it->second.response < kp.response)
				it->second = FeatureDescriptor(kp);
		}
		const int width = src.cols;
		for (auto &ifd : map) {
			auto &fd = ifd.second;
			fd.scan(src.data + fd.y * width + fd.x, width);
		}
	}
	void match(const cv::Mat &src) {
		const int width = src.cols;
		auto ffd = cv::FastFeatureDetector::create(20);
		std::vector<cv::KeyPoint> kps;
		ffd->detect(src, kps);
		int mx = src.cols - 9, my = src.rows - 9;
		for (const auto &kp : kps) {
			int x = (int) kp.pt.x, y = (int) kp.pt.y;
			if (kp.pt.x <= 8 || kp.pt.y <= 8) continue;
			if (kp.pt.x >= mx || kp.pt.y >= my) continue;
			int kx = int(kp.pt.x) / cellSize;
			int ky = int(kp.pt.y) / cellSize;
			int key = kx | (ky << 16);
			auto it = map.find(key);
			if (it == map.end()) continue;
			auto &fd = it->second;
			int m = fd.match(src.data + y * width + x, width);
			if (m > fd.bestMatch) {
				fd.bestMatch = m;
				fd.bestX = x;
				fd.bestY = y;
			}
		}

	}

};

void getFeatures(const cv::Mat &src)
{
	static std::unique_ptr<Keyframe> kf;

	if (kf) {
		kf->match(src);

		cv::Mat view;
		cv::cvtColor(src, view, CV_GRAY2RGB);
		for (const auto &ifd : kf->map) {
			const auto &fd = ifd.second;
			if (fd.bestMatch > 0)
			{
				cv::drawMarker(view, cv::Point(fd.x, fd.y), cv::Scalar(255, 0, 0));
				cv::drawMarker(view, cv::Point(fd.bestX, fd.bestY), cv::Scalar(0, 0, 255));
			}
		}
		cv::imshow("Features", view);
	}
	kf = std::make_unique<Keyframe>(src);
}
