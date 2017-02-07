#include "visualizer.h"

namespace flyflow
{

Visualizer::Visualizer(std::string name): name_(name)
{
    type_ = CV_8UC3;
}

void Visualizer::newColumn(const cv::Mat & ref, double scale)
{
    double min, max;
    cv::minMaxLoc(ref, &min, &max);
    a_ = 255 / (max - min);
    b_ = -min * a_;
    //scale_ = max ? (255 / max) : 1.0;
    ref.convertTo(ref_, CV_8U, a_, b_);
    cv::Mat col = column_;
    //scale_ = scale;
    if(result_.empty()) result_ = col;
    else
    {
        if(col.rows > result_.rows) cv::vconcat(result_, cv::Mat(col.rows - result_.rows, result_.cols, type_, cv::Scalar(0, 0, 0)), result_);
        else if(result_.rows > col.rows) cv::vconcat(col, cv::Mat(result_.rows - col.rows, col.cols, type_, cv::Scalar(0, 0, 0)), col);
        cv::hconcat(result_, col, result_);
    }
    column_ = cv::Mat();
}

void Visualizer::show()
{
    if(result_.empty()) result_ = column_;

    if(!result_.empty()) cv::imshow(name_, result_);
    result_ = cv::Mat();
    column_ = cv::Mat();
}

void Visualizer::add(const cv::Mat &image)
{
    cv::Mat t;//(ref_.rows, ref_.cols, CV_8U, cv::Scalar(0, 0, 0));
    cv::Mat scaled;
    image.convertTo(scaled, CV_8U, a_, b_);
    //cv::bitwise_and(ref_, scaled, t);
    cv::addWeighted(ref_, 0.5, scaled, 0.5, 0, t);
    std::vector<cv::Mat> v = {ref_, t, scaled};
    cv::Mat im;
    cv::merge(v, im);
    if(column_.empty()) column_ = im;
    else
    {
        assert(column_.cols == im.cols);
        cv::vconcat(column_, im, column_);
    }
}

void Visualizer::add(const std::string &text, char c)
{
    cv::Scalar s(0, 0, 0);
    switch(c)
    {
    case 'r': s = cv::Scalar(0, 0, 255); break;
    case 'g': s = cv::Scalar(0, 255, 0); break;
    case 'b': s = cv::Scalar(255, 0, 0); break;
    case 'w': s = cv::Scalar(255, 255, 255); break;
    }

    int cols = column_.empty() ? ref_.cols : column_.cols;
    cv::Mat t(20, cols, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::putText(t, text, cv::Point(1,10), cv::FONT_HERSHEY_SIMPLEX, 0.4, s);
    if(column_.empty()) column_ = t;
    else
        cv::vconcat(column_, t, column_);
}

}
