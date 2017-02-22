// witrack.cpp: определяет точку входа для консольного приложения.
//

#include "stdafx.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "jacobix.h"
#include <vector>
#include "jacobi.h"
#include <time.h>
#include "conveyor.h"
#include "visualizer.h"
#include "features.h"
//#include "trackFeatures.h"

double calcD(const uint8_t * data, size_t d, size_t count)
{
	double sum = 0, sum2 = 0;
	for (const uint8_t * e = data + d * count; data < e; data += d)
	{
		uint8_t v = *data;
		sum += v;
		sum2 += v * v;
	}
	sum /= count;
	sum2 /= count;
	return sum2 - sum * sum;
}

double calcD1(const uint8_t * data, size_t d, size_t count)
{
	uint8_t o = *data;
	double sum = 0;
	for (const uint8_t * e = data + d * count; data < e; data += d)
	{
		uint8_t v = *data;
		sum += (o - v) * (o - v);
		o = v;
	}
	sum /= count - 1;
	return sum;
}




int px = 141, py = 420;

void cbf(int event, int x, int y, int flags, void* userdata)
{
	if (event == cv::EVENT_LBUTTONDOWN)
	{
		px = x;
		py = y;
	}
}

void calcError(const cv::Mat &f, cv::Mat em)
{
	using namespace cv;
	int16_t *emp = (int16_t *)em.data;
	const uint8_t *p = f.data;
	Size size = f.size();
	for (int y = 0; y < size.height; y++)
		for (int x = 0; x < size.width; x++, emp++, p++)
			if (y)
				*emp = (int)*(p - size.width) - (int)*p;
			else
				*emp = 0;
}

cv::Mat visError(const cv::Mat &em)
{
	cv::Mat ve(em.size(), CV_8UC3);
	assert(em.type() == CV_16S);
	int16_t *emp = (int16_t *)em.data;
	uint8_t *p = ve.data;
	while (emp < (int16_t *)em.dataend)
	{
		int e = *(emp++);
		*(p++) = (e > 0) ? e : 0;
		*(p++) = 0;
		*(p++) = (e < 0) ? -e : 0;
	}
	return ve;
}

cv::Mat zoom(const cv::Mat &in, int z)
{
	cv::Mat out;
	cv::resize(in, out, in.size() * z, 0, 0, cv::INTER_NEAREST);
	return out;
}

cv::Mat solveGN(cv::Mat src, double shift, int count)
{
	using namespace cv;
	Mat du(2, 3, CV_64F);
	GradientX<int16_t> g;
	g.set(src);
	JacobiX<int16_t, 2> j(&g);
	typedef JacobiX<uint8_t, 2>::Vector State;
	int w = 3, h = 10;
	j.xMin = px - w;
	j.xMax = px + w;
	j.yCenter = py;
	j.yMin = py - h;
	j.yMax = py + h;
	//auto jj = flyflow::Jacobi::create(flyflow::Jacobi::jtSkewX);
	Mat t = cv::Mat::eye(2, 3, CV_64F);
	//Mat whiteBox(src.size(), CV_8U, cv::Scalar(100)), mask;
	j.state[0] = shift;
	//j.state[1] = 0.1;
	imshow("GX", zoom(j.gx2debug(), 5));
	Mat srct;

	for (int x = 0; x < exp(count * 0.3); x++)
	{
		j.A.reset();
		j.B.reset();
		j.E.reset();
		//Mat skew = j.skew();

		if (j.yMin < 0) j.yMin = 0;
		if (j.yMax >= src.size().height) j.yMax = src.size().height - 1;

		j.calcA();
		State ds;
		j.solve(ds);
		
		//calcError(srct, em);
		//imshow("Error", visError(em));
		//return visError(em);
		//j->solve(em, du);
		//std::cout << "gx:" << j->gx();
		//std::cout << "du: " << du;
		j.state += ds;
		j.yMin--;
		j.yMax++;
	}
	int scale = 1 + 500 / (j.yMax - j.yMin + 1);
	imshow("A", j.A.get(scale));
	imshow("B", j.B.get(scale));
	imshow("E", j.E.get(scale));
	//Mat skew = j.skew();
	//Mat em(skew.size(), CV_16S);
	//calcError(skew, em);
	//imshow("EM", zoom(visError(em), 5));
	//imshow("Skew", zoom(skew, 5));

	Mat color;
	cvtColor(src, color, CV_GRAY2RGB);
	j.drawArea<3, 2>(color);
	return color;
}

void testHough(const cv::Mat src)
{
	cv::Mat cn, cnc;
	int th = 16;
	cv::Canny(src, cn, th, th * 3, 3);
	//cn = src;

	cv::cvtColor(cn, cnc, CV_GRAY2BGR);

	std::vector<cv::Vec4i> lines;
	/*cv::HoughLinesP(cn, lines, 1, CV_PI / 180, 50, 100, 10);
	for (size_t i = 0; i < lines.size(); i++)
	{
		cv::Vec4i l = lines[i];
		cv::line(cnc, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
	}*/

	cv::imshow("Hough", cnc);
}

int main()
{
	using namespace cv;

	flyflow::Visualizer vis("vis");
	flyflow::Conveyor conv(&vis);
	VideoCapture vc(0);


	//Mat im = imread("C:/src/witrack/wi5.png");
	//VideoCapture cap("D:/Видео/DJI_0006.avi");

	Mat transform = Mat::eye(2, 3, CV_64F);

	namedWindow("Control");
	namedWindow("Wires");
	cv::setMouseCallback("Wires", cbf, NULL);
	int shift = 50, br = 25, it = 1;
	cv::createTrackbar("Shift", "Control", &shift, 100);
	cv::createTrackbar("Bright", "Control", &br, 50);
	cv::createTrackbar("Iterations", "Control", &it, 20);
	int idx = 1;
	int64_t tmo = getTickCount();
	double dt = 0;
	bool pause = false;
	bool reverse = false;

	std::vector<Point2f> points, points_r;
	std::vector<int> upvec;

	upvec.resize(50);
	std::fill(upvec.begin(), upvec.end(), 1);

	do {
		char buf[64];
		sprintf_s(buf, "d:/Видео/wi1/dji%04d.png", idx);
		//sprintf_s(buf, "d:/wires/png1/dji6_%04d.png", idx);
		if (!pause) idx += reverse ? -1 : 1;

		Mat im = imread(buf);
		//vc >> im;
		if (im.empty()) return 0;


		/*Mat imw;
		transform.at<double>(0, 1) = (shift + 100) * 0.001;
		transform.at<double>(0, 2) = (shift + 100) * -0.001 * im.size().width;
		warpAffine(im, imw, transform, im.size());
		Mat disp;// (im.size(), CV_8U, Scalar(0, 0, 0));
		Mat dispd(im.size(), CV_64F, Scalar(0, 0, 0));
		Size size = im.size();
		int hg = 150;
		for (int x = 0; x < size.width; x++)
			for (int y = 0; y < size.height - hg; y++)
			{
				double d = 
					calcD(imw.data + (y * size.width + x) * 3, 3 * size.width, hg) +
					calcD(imw.data + (y * size.width + x) * 3 + 1, 3 * size.width, hg) +
					calcD(imw.data + (y * size.width + x) * 3 + 2, 3 * size.width, hg);
				double d1 =
					calcD1(imw.data + (y * size.width + x) * 3, 3 * size.width, hg) +
					calcD1(imw.data + (y * size.width + x) * 3 + 1, 3 * size.width, hg) +
					calcD1(imw.data + (y * size.width + x) * 3 + 2, 3 * size.width, hg);
				dispd.at<double>(y, x) = d + d1;
			}
		dispd /= 4 * exp((br - 25) * 0.05);
		dispd.convertTo(disp, CV_8U);
		//imshow("Wires", disp);
		//imshow("Disp", disp);
		Mat gns = im(Rect(px, py, 10, 30)), gnsg;
		//cv::flip(gns, gns, 1);*/
		cv::Mat gnsg;
		cvtColor(im, gnsg, CV_BGR2GRAY);
		blur(gnsg, gnsg, Size(3, 3));
		//cv::Mat gnsg;
		//cvtColor(im, gnsg, CV_BGR2GRAY);
		//testHough(gnsg);
		//Mat gnr;
		//resize(gn, gnr, gnsg.size() * 2, INTER_NEAREST);
		conv.pushHistory = !pause;
		getFeatures(gnsg);
		cv::Mat cr = conv.onImage(gnsg);
		if (!cr.empty())
			imshow("Conv result", cr);

		//trackFeatures(gnsg, cv::Mat(), points, points_r, upvec);

		for (int x = 0; x < points.size(); x++)
		{
			//cv::drawMarker(im, points[x], upvec[x] ? Scalar(0, 255, 0) : Scalar(0, 0, 0));
			cv::rectangle(im, cv::Rect(points[x], points[x] + Point2f(2, 2)), Scalar(0, 255, 0));
		}

		//conv.
		vis.show();

		//Mat gn = solveGN(gnsg, (shift - 50) * 0.01, it);
		//imshow("Wires", gn);

		int64_t tm = getTickCount();
		dt += (double(tm - tmo) / getTickFrequency() - dt) * 0.1;
		//if(dt) fps += (dt - fps) * 0.1;
		tmo = tm;
		getTickCount();
		//sprintf_s(buf, "dt: %f", dt);
		//cv::putText(im, buf, cv::Point(10, 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, cv::Scalar(0, 255, 0));
		imshow("Input", im);
		switch (waitKey(1))
		{
		case 27: return 0;
		case 'p': pause = !pause; break;
		case 'r': reverse = !reverse; break;
		case 'n': idx++; break;
		case 'm': idx--; break;
		default:
			break;
		}
	} while (true);
    return 0;
}

