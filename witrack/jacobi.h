#ifndef FRAME_H
#define FRAME_H

#include <vector>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <limits>
#include <memory>

namespace flyflow
{

class Jacobi
{
protected:
    Jacobi(){};
    cv::Mat gx_, gy_; // Gradients
    double scale_;
public:
    inline const cv::Mat & gx() const { return gx_;}
    inline const cv::Mat & gy() const { return gy_;}
    inline double scale() const {return scale_;}
    virtual void set(const cv::Mat &f, double scale = 1.0) = 0;
    virtual void shrink(const Jacobi * p) = 0;
    virtual void solve(const cv::Mat & e, cv::Mat & dt) const = 0;
    virtual ~Jacobi(){};
    enum Type
    {
        jtShift, jtStereoX, jtAffine, jtSkewX
    };
    typedef std::unique_ptr<Jacobi> Ptr;
    static Ptr create(Type jType, int cvType = CV_16S);
};



//typedef Jacobi<int16_t> JacobiShift;
//typedef Jacobi<int16_t, true, true, true, false, false, false> JacobiStereoX;
//typedef Jacobi<int16_t, true, true, true, true, true, true> JacobiAffine;

class GaussNewton
{
public:
    int stepCount_;
    double thres_;
    GaussNewton() :
        thres_(5) {}
    template<class T, bool out = false> double calcError(cv::Mat & me)
    {
        assert(me.type() == cv_type<T>());
        double e = 0;
        //T thres = T (std::numeric_limits<T>::max() * thres_);
        T thres = (T) thres_;
        for(T * p = (T *) me.data, * end = (T *) me.dataend; p < end; p++)
        {
            if(abs(*p) > thres) {e++; if(out) *p = 255;}
            else *p = 0;
        }
        return e / (me.cols * me.rows);
    }

	inline static void rectify(cv::Mat & pose)
	{
		//pose.at<double>(0, 0) = 
		Eigen::Vector2d sum;
		double
			p00 = pose.at<double>(0, 0),
			p01 = pose.at<double>(0, 1),
			p10 = pose.at<double>(1, 0),
			p11 = pose.at<double>(1, 1);

		sum << pose.at<double>(0, 0) + pose.at<double>(1, 0), pose.at<double>(0, 1) + pose.at<double>(1, 1);
		sum /= sum.norm() * sqrt(2.0);
		pose.at<double>(1, 0) = sum[0] - sum[1];
		pose.at<double>(1, 1) = sum[1] + sum[0];
		pose.at<double>(0, 0) = sum[0] + sum[1];
		pose.at<double>(0, 1) = sum[1] - sum[0];
		double
			rp00 = pose.at<double>(0, 0),
			rp01 = pose.at<double>(0, 1),
			rp10 = pose.at<double>(1, 0),
			rp11 = pose.at<double>(1, 1);
	}

    template<int writeOut = 0> double solve(const cv::Mat & f0, const cv::Mat & f1, const Jacobi * j, cv::Mat & pose, std::vector<cv::Mat> * out = 0, double maxError = 0.1, int maxStepCount = 20, bool rectify = false)
    {
        //std::vector<cv::Mat> v = {f0, f1, f1};
        //cv::merge(v, out);

        cv::Mat du(2, 3, CV_64F);
        int h = f0.rows, w = f0.cols;
        cv::Mat whiteBox(f0.rows, f0.cols, CV_8U, cv::Scalar(100)), mask;
        double bestE = 1E10;
        cv::Mat p = pose.clone();
        double step = 0.3;

        for(int x = 0; x < maxStepCount; x++)
        {
            cv::Mat f0t;
            cv::warpAffine(f0,        f0t, p, cv::Size(w, h), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT, cv::Scalar(255));
            cv::warpAffine(whiteBox, mask, p, cv::Size(w, h), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT, cv::Scalar(0));

            cv::Mat em;
            cv::subtract(f1, f0t, em, mask, CV_16S);
            j->solve(em, du);
            double te = calcError<int16_t, true>(em);//cv::norm(em);


            if(writeOut & 2)
            {
                std::cout << "--- it " << x + 1 << std::endl;
                std::cout << "u = " << std::endl << p << std::endl << std::endl;
                std::cout << "e = " << te << std::endl;
                std::cout << "du = " << std::endl << du << std::endl << std::endl;
            }

            if((writeOut & 1) && out)
            {
                out->push_back(f0t.clone());
                out->push_back(em.clone());
            }
			
            //if(te < bestE)
            {
                p.copyTo(pose);
                if(te < maxError && x > 0)
                {
                    stepCount_ = x;
                    return te;
                }
                bestE = te;
                //step = 1.0;
            }
            p += du * step;
			//if (rectify) this->rectify(p);
        }
        stepCount_ = maxStepCount;
        return bestE;
    }
};


}
#endif // FRAME_H
