#pragma once
#include <stdint.h>
#include <opencv2/core.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

template<class T> constexpr int cv_type();
template<> constexpr int cv_type<uint8_t>() { return CV_8U; }
template<> constexpr int cv_type<uint16_t>() { return CV_16U; }
template<> constexpr int cv_type<int16_t>() { return CV_16S; }
template<> constexpr int cv_type<int32_t>() { return CV_32S; }

template<size_t cs = 1, typename T = uint8_t>
void shift(const T *src, T *dst, size_t w, double delta)
{
	double d = floor(delta);
	delta -= d;
	src += (size_t) d;
	T prev = *(src++);
	unsigned int d2 = (int)(delta * 65536), d1 = 65536 - d2;
	for (T * e = dst + w; dst < e; dst++)
	{
		T next = *(src++);
		*dst = (d1 * prev + d2 * next) >> 16;
		prev = next;
	}
}

template<class T>
class GradientX
{
private:
	int kernel_;
	double scale_;
	cv::Mat gx_, img_;
public:
	inline const cv::Mat &gx() { return gx_; }
	inline const cv::Mat &img() { return img_; }
	inline double scale2() { return scale_ * scale_; }
	inline double scale() { return scale_; }
	GradientX(int kernel = CV_SCHARR) : kernel_(kernel) {}
	void set(const cv::Mat &f, double scale = 1.0)
	{
		switch (kernel_)
		{
		case CV_SCHARR:
			scale_ = scale / 32;
			break;
		case 1:
			scale_ = scale / 2;
			break;
		case 3:
			scale_ = scale / 8;
			break;
		case 5:
			scale_ = scale / 58;
			break;
		}
		cv::Mat gx;
		if (f.type() == CV_8U && cv_type<T>() == CV_32S)
		{
			cv::Sobel(f, gx, CV_16S, 1, 0, kernel_);
			gx.convertTo(gx_, cv_type<T>());
		}
		else
		{
			f.copyTo(img_);
			cv::Sobel(f, gx_, cv_type<T>(), 1, 0, kernel_);
		}
		//calcA();
	}
};

void e2s(int e, uint8_t *p)
{
	p[0] = (e < 0) ? -e : 0;
	p[1] = 0;
	p[2] = (e > 0) ? e : 0;
}

class MatDumper
{
private:
	cv::Mat data, mask;
	cv::Size size;
	int xMin, xMax, yMin, yMax;
	double max, min;
public:
	void reset()
	{
		xMax = 0;
		yMax = 0;
		xMin = INT_MAX;
		yMin = INT_MAX;
		data = cv::Mat::zeros(size, data.type());
		mask = cv::Mat::zeros(size, mask.type());
		max = 0;
		min = 0;
	}
	MatDumper(const cv::Size &size) : size(size), data(size, CV_64F, cv::Scalar(0)), mask(size, CV_8U, cv::Scalar(0)) { reset();  }
	void dump(double value, size_t offset)
	{
		int y = int(offset / size.width);
		int x = int(offset % size.width);
		assert(y < size.height);
		if (x < xMin) xMin = x;
		if (x > xMax) xMax = x;
		if (y < yMin) yMin = y;
		if (y > yMax) yMax = y;
		mask.data[offset] = 255;
		((double *)data.data)[offset] = value;
		if (!isnan(value)) {
			if (value < min) min = value;
			if (value > max) max = value;
		}
	}
	cv::Mat get(int zoom)
	{
		if (xMin == INT_MAX) return cv::Mat();
		cv::Mat d, m;
		int xi = xMin ? xMin - 1 : 0;
		int yi = yMin ? yMin - 1 : 0;
		int xa = (xMax < size.width  - 1) ? xMax + 1 : size.width  - 1;
		int ya = (yMax < size.height - 1) ? yMax + 1 : size.height - 1;

		data(cv::Rect(xi, yi, xa - xi + 1, ya - yi + 1)).copyTo(d);
		mask(cv::Rect(xi, yi, xa - xi + 1, ya - yi + 1)).copyTo(m);
		const double *pd = (const double *)d.data;
		const uint8_t *pm = (const uint8_t *)m.data;
		cv::Mat res(m.size(), CV_8UC3, cv::Scalar(255, 255, 255));
		uint8_t *pr = res.data;
		if (max < -min) max = -min;
		double mul = max ? 255.0 / max : 0.0;
		for (size_t count = m.dataend - m.datastart; count; count--, pm++, pd++, pr += 3)
		{
			if (!*pm) continue;
			double value = *pd;
			if (isnan(value))
			{
				pr[0] = 0;
				pr[1] = 255;
				pr[2] = 255;
			}
			else {
				pr[0] = (value < 0) ? (uint8_t)lround(-value * mul) : 0;
				pr[1] = 0;
				pr[2] = (value > 0) ? (uint8_t)lround(value * mul) : 0;
			}
		}
		cv::Mat out;
		cv::resize(res, out, res.size() * zoom, 0, 0, cv::INTER_NEAREST);
		char buf[64];
		sprintf_s(buf, "max = %f", max);
		cv::putText(out, buf, cv::Point(10, 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, cv::Scalar(0, 255, 0));
		return out;
	}
};

template<class T, int order = 1>
class JacobiX
{
public:
	typedef Eigen::Matrix<double, order, order> Matrix;
	typedef Eigen::Matrix<double, order, 1> Vector;
private:
	GradientX<T> *gradient;
	Matrix invA_;
public:
	MatDumper A, B, E;
	int yCenter, xMin, xMax, yMin, yMax;
	cv::Mat debug;
	Vector state;
	JacobiX(GradientX<T> *gradient) : gradient(gradient), A(gradient->img().size()), B(gradient->img().size()), E(gradient->img().size()) { state.setZero(); }
	void dump(cv::Mat &mat, int val, size_t o)
	{
		if (mat.empty())
			mat = cv::Mat(gradient->gx().size(), CV_8UC3, cv::Scalar(255, 255, 255));// cv::Mat(yMax - yMin + 1, (xMax - xMin) * 3, CV_8UC3, cv::Scalar(255, 255, 255));
		e2s(val, mat.data + o * 3);
	}
	cv::Mat matPOI(cv::Mat A)
	{
		int xm = xMin - (xMax - xMin);
		cv::Mat t;
		A(cv::Rect(xm, yMin, (xMax - xMin) * 3, yMax - yMin + 1)).copyTo(t);
		return t;
	}
	cv::Mat gx2debug()
	{
		int xm = xMin - (xMax - xMin);
		cv::Mat gx;
		gradient->gx()(cv::Rect(xm, yMin, (xMax - xMin) * 3, yMax - yMin + 1)).copyTo(gx);
		return visError(gx / 4);
	}
	inline double calcPoly(double x) const
	{
		double sum = state[0] * x;
		if (order > 1) sum += state[1] * x * x;
		if (order > 2) sum += state[2] * x * x * x;
		return sum;
	}
	template<int cc = 1, int o = 0>
	void drawArea(cv::Mat & out)
	{
		int w = out.size().width;
		for (int y = yMin; y <= yMax; y++)
		{
			double y1 = y - yCenter, y2 = y1 * y1, y3 = y2 * y1;
			uint8_t * pOut = (uint8_t *)out.data + (y * w + xMin + (int) calcPoly(y1)) * cc + o;
			*pOut = 255;
			pOut += (xMax - xMin + 1) * cc;
			*pOut = 255;
		}
	}
	cv::Mat skew()
	{
		using namespace cv;
		const Mat &src = gradient->img();
		int w = xMax - xMin + 1, sw = src.size().width;
		Mat res(yMax - yMin + 1, w, CV_8U);
		uint8_t *p = res.data;
		for (int y = yMin; y <= yMax; y++)
		{
			//memcpy(res.data + (y - yMin) * w, src.data + y * sw + xMin, w);
			shift<1, uint8_t>(src.data + y * sw + xMin, res.data + (y - yMin) * w, w, calcPoly(y - yCenter));
		}
		return res;
	}

	void calcA()
	{
		Matrix a = Matrix::Zero();
		Vector v;
		const cv::Mat &gx_ = gradient->gx();
		int w = gx_.cols;

		for (int y = yMin; y <= yMax; y++)
		{
			double y1 = y - yCenter, y2 = y1 * y1, y3 = y2 * y1;
			const T * pGx = (const T *)gx_.data + y * w + xMin + (int) calcPoly(y1);
			for (int x = xMax - xMin + 2; --x;)
			{
				double gx = *(pGx++);
				v[0] = gx * y1;
				if (order > 1) v[1] = gx * y2;
				if (order > 2) v[2] = gx * y3;
				a += v * v.transpose();
				A.dump(v[0] * v[0], pGx - 1 - (const T *)gx_.data);
			}
		}
		a *= gradient->scale2();
		invA_ = a.inverse();
	}
	inline double calcBforLine(Vector & b, const T *gx, const uint8_t *line, const uint8_t *linePrev, int width, double delta, double y1)
	{
		double sumError = 0;
		double y2 = y1 * y1, y3 = y2 * y1;
		double ka, kb;
		uint8_t pb;
		if (delta < 0) {
			linePrev--;
			ka = 1.0 + delta;
			kb = -delta;
		}
		else {
			kb = 1.0 - delta;
			ka = 1.0 - kb;
		}
		pb = *(linePrev++);

		Vector v;
		while (width--) {
			double e = *(line++);
			uint8_t pa = *(linePrev++);
			e -= pa * ka + pb * kb;
			pb = pa;
			sumError += e * e;

			double ge = *(gx++) * e;
			v[0] = ge * y1;
			if (order > 1) v[1] = ge * y2;
			if (order > 2) v[2] = ge * y3;
			E.dump(e,		 gx - 1 - (const T *)gradient->gx().data);
			B.dump(v[0], gx - 1 - (const T *)gradient->gx().data);
			b += v;
		}
		return sumError;
	}

	double calcB(Vector & b)
	{
		b.setZero();
		const cv::Mat &gx_ = gradient->gx(), &img = gradient->img();
		int w = gx_.cols;
		int wa = xMax - xMin + 1;
		double sumError = 0;

		double oldPoly = 0;
		for (int y = yCenter + 1; y <= yMax; y++)
		{
			double y1 = y - yCenter;
			double poly = calcPoly(y1);
			size_t offset = y * w + xMin + (int)poly;
			size_t oo = 1 + (y - yCenter) / 8;
			sumError += calcBforLine(b, ((const T *)gx_.data) + offset, ((const uint8_t*)img.data) + offset, ((const uint8_t*)img.data) + offset + oo * w, wa, (poly - oldPoly) * oo, y1);
			oldPoly = poly;
		}
		oldPoly = 0;
		for (int y = yCenter - 1; y >= yMin; y--)
		{
			double y1 = y - yCenter;
			double poly = calcPoly(y1);
			size_t offset = y * w + xMin + (int)poly;
			size_t oo = 1 + (yCenter - y) / 8;
			sumError += calcBforLine(b, ((const T *)gx_.data) + offset, ((const uint8_t*)img.data) + offset, ((const uint8_t*)img.data) + offset - oo * w, wa, (poly - oldPoly) * oo, y1);
			oldPoly = poly;
		}		
		b *= gradient->scale();
		return sumError;
	}
	inline double solve(Vector & r)
	{
		Vector b;
		double sumError = calcB(b);
		r = (invA_ * b);
		return sumError;
	}
};

