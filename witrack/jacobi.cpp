#include "jacobi.h"
#include <iostream>
#include <random>

namespace flyflow
{

template<class T, bool usegxx = false, bool usegxy = false, bool usegx = true,
                  bool usegyx = false, bool usegyy = false, bool usegy = true>
class JacobiT: public Jacobi
{
public:
	static constexpr int size() { return (usegx?1:0) + (usegxx?1:0) + (usegxy?1:0) + (usegy?1:0) + (usegyx?1:0) + (usegyy?1:0);}
    typedef Eigen::Matrix<double, (usegx?1:0)+(usegxx?1:0)+(usegxy?1:0)+(usegy?1:0)+(usegyx?1:0)+(usegyy?1:0), (usegx?1:0)+(usegxx?1:0)+(usegxy?1:0)+(usegy?1:0)+(usegyx?1:0)+(usegyy?1:0)> Matrix;
    typedef Eigen::Matrix<double, (usegx?1:0)+(usegxx?1:0)+(usegxy?1:0)+(usegy?1:0)+(usegyx?1:0)+(usegyy?1:0), 1> Vector;
private:
    int kernel_;
    Eigen::Matrix<double, (usegx?1:0)+(usegxx?1:0)+(usegxy?1:0)+(usegy?1:0)+(usegyx?1:0)+(usegyy?1:0), (usegx?1:0)+(usegxx?1:0)+(usegxy?1:0)+(usegy?1:0)+(usegyx?1:0)+(usegyy?1:0)> invA_;
    void calcA()
    {
        Matrix a = Matrix::Zero();
        const T * p1, * p2;
        double gx, gy;
        Vector v;
        if(usegx || usegxx || usegxy) p1 = (const T *) gx_.data;
        if(usegy) p2 = (const T *) gy_.data;
        int w = (usegx || usegxx || usegxy) ? gx_.cols : gy_.cols;
        int h = (usegx || usegxx || usegxy) ? gx_.rows : gy_.rows;
        for(int y = 0; y < h; y++)
            for(int x = 0; x < w; x++)
            {
                double * r = v.data();
                if(usegx || usegxx || usegxy) gx = *(p1++);
                if(usegy) gy = *(p2++);
                if(usegxx) *(r++) = gx * x;
                if(usegxy) *(r++) = gx * y;
                if(usegx) *(r++) = gx;

                if(usegyx) *(r++) = gy * x;
                if(usegyy) *(r++) = gy * y;
                if(usegy) *(r++) = gy;
                assert(r - v.data() == size());
                a += v * v.transpose();
            }
        a *= scale_ * scale_;
        invA_ = a.inverse();
    }
    template<class Te> void calcB(Vector & b, const cv::Mat & e) const
    {
        assert(!usegx || (e.rows == gx_.rows && e.cols == gx_.cols));
        assert(!usegy || (e.rows == gy_.rows && e.cols == gy_.cols));
        //std::cout << "cv_type<Te>() = " << cv_type<Te>() << "; e.type() = " << e.type() << std::endl;
        assert(cv_type<Te>() == e.type());
        b.setZero();
        const T * p1, * p2;
        double gx, gy;
        Vector v;
        if(usegx || usegxx || usegxy) p1 = (const T *) gx_.data;
        if(usegy) p2 = (const T *) gy_.data;
        const Te * pe = (const Te *) e.data;
        int w = e.cols;
        int h = e.rows;
        for(int y = 0; y < h; y++)
            for(int x = 0; x < w; x++)
            {
                double * r = v.data();
				double e = *(pe++);
                if(usegx || usegxx || usegxy) gx = *(p1++) * e;
                if(usegy) gy = *(p2++) * e;
                if(usegxx) *(r++) = gx * x;
                if(usegxy) *(r++) = gx * y;
                if(usegx) *(r++) = gx;

                if(usegyx) *(r++) = gy * x;
                if(usegyy) *(r++) = gy * y;
                if(usegy) *(r++) = gy;
                assert(r - v.data() == size());
                b += v;
            }
        b *= scale_;
    }

public:
    JacobiT(int kernel = CV_SCHARR): kernel_(kernel) {}
    void set(const cv::Mat &f, double scale = 1.0)
    {
        switch(kernel_)
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
        //case 7: scale /=
        }
        cv::Mat gx, gy;
        if(usegx || usegxx || usegxy)
        {
            if(f.type() == CV_8U && cv_type<T>() == CV_32S)
            {
                cv::Sobel(f, gx, CV_16S, 1, 0, kernel_);
                gx.convertTo(gx_, cv_type<T>());
            }
            else
                cv::Sobel(f, gx_, cv_type<T>(), 1, 0, kernel_);
        }
        if(usegy)
        {
            if(f.type() == CV_8U && cv_type<T>() == CV_32S)
            {
                cv::Sobel(f, gy, CV_16S, 0, 1, kernel_);
                gy.convertTo(gy_, cv_type<T>());
            }
            else
                cv::Sobel(f, gy_, cv_type<T>(), 0, 1, kernel_);
        }
        calcA();
    }
    void shrink(const Jacobi * p)
    {
        if(usegx)
        {
            if(gx_.empty()) gx_ = cv::Mat(p->gx().rows / 2, p->gx().cols / 2, cv_type<T>());
            flyflow::shrink<T>(p->gx(), gx_);
        }
        if(usegy)
        {
            if(gy_.empty()) gy_ = cv::Mat(p->gy().rows / 2, p->gy().cols / 2, cv_type<T>());
            flyflow::shrink<T>(p->gy(), gy_);
        }
        scale_ = p->scale() * 2;
        calcA();
    }
    inline void solve(const cv::Mat & e, Vector & r) const
    {
        Vector b;
        switch(e.type())
        {
        case CV_16S:
            calcB<int16_t>(b, e);
            break;
        case CV_16U:
            calcB<uint16_t>(b, e);
            break;
        case CV_32S:
            calcB<int32_t>(b, e);
            break;
        default:
            assert(!"Unknown error type");
        }
        r = (invA_ * b);
    }
    void solve(const cv::Mat & e, cv::Mat & dt) const
    {
        assert(dt.rows == 2 && dt.cols == 3 && dt.type() == CV_64F);
        Vector v;
        solve(e, v);
        const double * pv = v.data();
        double * pd = (double *) dt.data;
        *(pd++) = usegxx ? *(pv++) : 0;
        *(pd++) = usegxy ? *(pv++) : 0;
        *(pd++) = usegx ? *(pv++) : 0;
        *(pd++) = usegyx ? *(pv++) : 0;
        *(pd++) = usegyy ? *(pv++) : 0;
        *(pd++) = usegy ? *(pv++) : 0;
    }
};

template<class T> Jacobi::Ptr createT(Jacobi::Type jType)
{
    switch(jType)
    {
    case Jacobi::jtShift:
        return Jacobi::Ptr(new JacobiT<T, false, false, true, false, false, true>());
    case Jacobi::jtAffine:
        return Jacobi::Ptr(new JacobiT<T, true, true, true, true, true, true>());
    case Jacobi::jtStereoX:
        return Jacobi::Ptr(new JacobiT<T, true, true, true, false, false, false>());
	case Jacobi::jtSkewX:
		return Jacobi::Ptr(new JacobiT<T, false, true, false, false, false, false>());
    default:
        assert(!"Unknown type");
		return nullptr;
    }
}

Jacobi::Ptr Jacobi::create(Jacobi::Type jType, int cvType)
{
    switch(cvType)
    {
    case CV_16S:
        return createT<int16_t>(jType);
    case CV_32S:
        return createT<int32_t>(jType);
    default:
        assert(!"Unknown type");
		return nullptr;
    }
}

#ifdef USE_GTEST

#include <gtest/gtest.h>


TEST(JacobiTest, size)
{
    JacobiT<int32_t, false, false, true, false, false, true> j1;
    EXPECT_EQ(j1.size(), 2);
    JacobiT<int32_t, true, true, true, false, false, false> j2;
    EXPECT_EQ(j2.size(), 3);
    JacobiT<int32_t, true, true, true, true, true, true> j3;
    EXPECT_EQ(j3.size(), 6);
}

TEST(JacobiTest, gradient)
{
    typedef uint8_t Tf;
    JacobiT<int32_t, false, false, true, false, false, true> j32;
    JacobiT<int16_t, false, false, true, false, false, true> j16;
    int w = 3, h = 3;
    cv::Mat a(h, w, cv_type<Tf>());
    for(int t = 0; t < 100; t++)
    {
        uint8_t * p = a.data;
        int mx = std::rand() % 64;
        int my = std::rand() % 64;
        for(int y = 0; y < h; y++) for(int x = 0; x < w; x++)
            *(p++) = x * mx + y * my;
        j32.set(a);
        j16.set(a);
        //std::cout << "gx = " << std::endl << j16.gx() << std::endl << std::endl;
        EXPECT_NEAR(j32.gx().at<int32_t>(1, 1) * j32.scale(), mx, 1E-6);
        EXPECT_NEAR(j32.gy().at<int32_t>(1, 1) * j32.scale(), my, 1E-6);
        EXPECT_NEAR(j16.gx().at<int16_t>(1, 1) * j16.scale(), mx, 1E-6);
        EXPECT_NEAR(j16.gy().at<int16_t>(1, 1) * j16.scale(), my, 1E-6);
    }
}

TEST(JacobiTest, shrink)
{
    typedef uint8_t Tf;
    for(int t = 0; t < 100; t++)
    {
        int w = std::rand() % 64 + 1, h = std::rand() % 64 + 1;
        cv::Mat a(h, w, cv_type<Tf>(), cv::Scalar(std::rand() & 255));
        JacobiT<int32_t, false, false, true, false, false, true> j1, j2, j3;
        j1.set(a);
        EXPECT_EQ(j1.gx().cols, w);
        EXPECT_EQ(j1.gx().rows, h);
        EXPECT_EQ(j1.gy().cols, w);
        EXPECT_EQ(j1.gy().rows, h);
        j2.shrink(&j1);
        EXPECT_EQ(j2.gx().cols, w / 2);
        EXPECT_EQ(j2.gx().rows, h / 2);
        EXPECT_EQ(j2.gy().cols, w / 2);
        EXPECT_EQ(j2.gy().rows, h / 2);
        j3.shrink(&j2);
        EXPECT_EQ(j3.gx().cols, w / 4);
        EXPECT_EQ(j3.gx().rows, h / 4);
        EXPECT_EQ(j3.gy().cols, w / 4);
        EXPECT_EQ(j3.gy().rows, h / 4);
    }
}

TEST(JacobiTest, shrinked_gradient)
{
    typedef int16_t Tj;
    JacobiT<int16_t, false, false, true, false, false, true> j0, j1, j2, j2t;
    int w = 12, h = 12;
    typedef uint8_t Tf;
    cv::Mat a0(h, w, cv_type<Tf>()), a1(h / 2, w / 2, cv_type<Tf>()), a2(h / 4, w / 4, cv_type<Tf>());
    for(int t = 0; t < 10; t++)
    {
        Tf * p = (Tf *)a0.data;
        int mx = std::rand() % 10;
        int my = std::rand() % 10;
        for(int y = 0; y < h; y++) for(int x = 0; x < w; x++)
            *(p++) = x * mx + y * my;
        shrink<Tf>(a0, a1);
        shrink<Tf>(a1, a2);
        //std::cout << "a0 = " << std::endl << a0 << std::endl << std::endl;
        //std::cout << "a1 = " << std::endl << a1 << std::endl << std::endl;
        //std::cout << "a2 = " << std::endl << a2 << std::endl << std::endl;
        j0.set(a0);
        j1.shrink(&j0);
        j2.shrink(&j1);
        //std::cout << "j0.gx = " << std::endl << a0 << std::endl << std::endl;
        //std::cout << "a1.gx = " << std::endl << a1 << std::endl << std::endl;
        //std::cout << "j2.gx = " << std::endl << j2.gx() * j2.scale() << std::endl << std::endl;

        j2t.set(a2);
        //std::cout << "j2t.gx = " << std::endl << j2t.gx() * j2t.scale() << std::endl << std::endl;
        EXPECT_EQ(h / 4, j2.gx().rows);
        EXPECT_EQ(w / 4, j2.gx().cols);
        //std::cout << "gx = " << std::endl << j.gx() << std::endl << std::endl;
        EXPECT_NEAR(j2.gx().at<Tj>(1, 1) * j2.scale(), j2t.gx().at<Tj>(1, 1) * j2t.scale(), 1E-6);
        EXPECT_NEAR(j2.gy().at<Tj>(1, 1) * j2.scale(), j2t.gy().at<Tj>(1, 1) * j2t.scale(), 1E-6);
    }
}

TEST(JacobiTest, solve_shift)
{
    typedef uint8_t Tf;
    JacobiT<int32_t, false, false, true, false, false, true> sj;
    int w = 20, h = 20;
    cv::Mat a(h, w, cv_type<Tf>()), b;
    uint8_t * p = a.data;
    for(int y = 0; y < h; y++) for(int x = 0; x < w; x++)
        *(p++) = x * (w - x) + y * (h - y);
    sj.set(a);
    cv::Mat t = cv::Mat::eye(2, 3, CV_64F);
    cv::Mat dt = cv::Mat(2, 3, CV_64F);
    cv::Mat wb(h, w, CV_8U, cv::Scalar(255)), m;

    for(int x = 0; x < 40; x++)
    {
        t.at<double>(0, 2) = (rand() % w) - w / 2;
        t.at<double>(1, 2) = (rand() % h) - h / 2;

        cv::warpAffine(wb, m, t, cv::Size(h, w));
        cv::warpAffine(a, b, t, cv::Size(h, w));
        cv::Mat e;
        cv::subtract(b, a, e, m, CV_16S);
        sj.solve(e, dt);

        EXPECT_EQ(0, dt.at<double>(0, 0));
        EXPECT_EQ(0, dt.at<double>(1, 0));
        EXPECT_EQ(0, dt.at<double>(0, 1));
        EXPECT_EQ(0, dt.at<double>(1, 1));
        if(t.at<double>(0, 2) == 0)
            EXPECT_NEAR(0, dt.at<double>(0, 2), 0.1);
        else
        {
            EXPECT_GT(-dt.at<double>(0, 2) / t.at<double>(0, 2), 0.04);
            EXPECT_LT(-dt.at<double>(0, 2) / t.at<double>(0, 2), 1);
        }
        if(t.at<double>(1, 2) == 0)
            EXPECT_NEAR(0, dt.at<double>(1, 2), 0.1);
        else
        {
            EXPECT_GT(-dt.at<double>(1, 2) / t.at<double>(1, 2), 0.05);
            EXPECT_LT(-dt.at<double>(1, 2) / t.at<double>(1, 2), 1);
        }
        //std::cout << "t = " << std::endl << t << std::endl << std::endl;
        //std::cout << "dt = " << std::endl << dt << std::endl << std::endl;
    }
}

TEST(GaussNewtonTest, solve_shift)
{
    typedef uint8_t Tf;
    typedef JacobiT<int32_t, false, false, true, false, false, true> J;
    J sj;
    int w = 20, h = 20;
    cv::Mat a(h, w, cv_type<Tf>());
    uint8_t * p = a.data;
    for(int y = 0; y < h; y++) for(int x = 0; x < w; x++)
        *(p++) = x * (w - x) + y * (h - y);
    sj.set(a);
    GaussNewton gn(0.01, 20);
    cv::Mat t = cv::Mat::eye(2, 3, CV_64F);
    for(int x = 0; x < 50; x++)
    {
        t.at<double>(0, 2) = (rand() % w) - w / 2;
        t.at<double>(1, 2) = (rand() % h) - h / 2;

        //std::cout << "t1 = " << std::endl << t << std::endl << std::endl;
        gn.solve(a, a, &sj, t);
        EXPECT_NEAR(0, t.at<double>(0, 2), 0.1);
        EXPECT_NEAR(0, t.at<double>(1, 2), 0.1);
        //std::cout << "t2 = " << std::endl << t << std::endl << std::endl;

    }
}

TEST(GaussNewtonTest, solve_shift2)
{
    typedef uint8_t Tf;
    typedef JacobiT<int32_t, false, false, true, false, false, true> J;
    J sj;
    int w = 20, h = 20;
    cv::Mat b(h, w, CV_8U);
    uint8_t * p = b.data;
    for(int y = 0; y < h; y++) for(int x = 0; x < w; x++)
        *(p++) = x * (w - x) + y * (h - y);
    sj.set(b);
    GaussNewton gn(0);
    cv::Mat t = cv::Mat::eye(2, 3, CV_64F);
    for(int x = 0; x < 50; x++)
    {
        cv::Mat a(h, w, CV_8U);
        p = a.data;
        int sx = (rand() % 10) - 5;
        int sy = (rand() % 10) - 5;

        for(int y = 0; y < h; y++) for(int x = 0; x < w; x++)
            *(p++) = (x - sx) * (w - x + sx) + (y - sy) * (h - y + sy);

        double e = gn.solve(a, b, &sj, t);
        /*std::cout << "sx = " << sx << "; sy = " << sy << std::endl;
        std::cout << "t = " << std::endl << t << std::endl << std::endl;
        std::cout << "e = " << e << std::endl;*/
        EXPECT_NEAR(sx, t.at<double>(0, 2), 1.5);
        EXPECT_NEAR(sy, t.at<double>(1, 2), 1.5);
    }
}



/*TEST(GaussNewtonTest, solve_affine)
{
    typedef Jacobi<int32_t, true, true, true, true, true, true> J;
    J sj;
    int w = 20, h = 20;
    cv::Mat a(h, w, CV_8U), b;
    uint8_t * p = a.data;
    for(int y = 0; y < h; y++) for(int x = 0; x < w; x++)
        *(p++) = x * (w - x) + 128 - y * (h - y);
    sj.set(a);
    GaussNewton gn;
    cv::Mat t = cv::Mat::eye(2, 3, CV_64F);
    for(int x = 0; x < 50; x++)
    {
        t.at<double>(0, 2) = (rand() % w) - w / 2;
        t.at<double>(1, 2) = (rand() % h) - h / 2;

        //std::cout << "t1 = " << std::endl << t << std::endl << std::endl;
        gn.solve<J>(a, a, sj, t);
        EXPECT_NEAR(0, t.at<double>(0, 2), 0.1);
        EXPECT_NEAR(0, t.at<double>(1, 2), 0.1);
        //std::cout << "t2 = " << std::endl << t << std::endl << std::endl;

    }
}*/

#endif

}
