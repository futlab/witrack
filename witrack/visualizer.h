#ifndef VISUALIZER_H
#define VISUALIZER_H
#include <opencv2/opencv.hpp>
#include <string>

namespace flyflow
{

class Visualizer
{
private:
    cv::Mat column_, result_, ref_;
    std::string name_;
    int type_;
    double a_, b_;
public:
    Visualizer(std::string name);
    void add(const cv::Mat & image);
    void add(const std::string & text, char c = 0);
    void newColumn(const cv::Mat &ref, double scale);
    void show();
};

}
#endif // VISUALIZER_H
