#pragma once
#ifndef KNEARESTANDSVM_COMMON_HPP_
#define KNEARESTANDSVM_COMMON_HPP_

#define _CRT_SECURE_NO_WARNINGS 1

#ifdef WIN
#pragma warning(disable : 4819)
#pragma warning(disable : 4305)
#endif

#include <iostream>
#include <string>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <utility>
#include <algorithm>

#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>

#include <opencv2/opencv.hpp>
// This above line include below headders
//#include "opencv2/core.hpp"
//#include "opencv2/imgproc.hpp"
//#include "opencv2/photo.hpp"
//#include "opencv2/video.hpp"
//#include "opencv2/features2d.hpp"
//#include "opencv2/objdetect.hpp"
//#include "opencv2/calib3d.hpp"
//#include "opencv2/imgcodecs.hpp"
//#include "opencv2/videoio.hpp"
//#include "opencv2/highgui.hpp"
//#include "opencv2/ml.hpp"

// OpenCL
#include <opencv2/core/ocl.hpp>

// eigen
#ifdef EIGEN
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <opencv2/core/eigen.hpp>
#endif

//cont_lib
#include <opencv2/text.hpp>
#include <opencv2/datasets/or_mnist.hpp>

#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

//tesseract
#ifdef TESSERACT
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#endif

#define LOG(message) std::cout << message << std::endl
#define FORMATOUT std::cout << boost::format
#define FORMATOUTF(arg, arg2) writing_file << boost::format(arg) arg2

#define TYPE1

extern cv::String data_path;

struct sParams
{
    sParams()
    {
        USEVIDEO = false;
        USEIMAGE = false;
        VIDEONAME = "movie.avi";
        CAMERANAME = "camera.xml";
        ARMAKERNAME = "maker.jpg";
        IMAGENAME = "maker.jpg";
        CHANGESIZE = -1;
    };
    void loadParams(const std::string &name)
    {
        cv::FileStorage fs;
        if (!fs.open(name, cv::FileStorage::READ))
        {
            fs.open(name, cv::FileStorage::WRITE);

            cv::write(fs, "USEVIDEO", int(USEVIDEO));
            cv::write(fs, "USEIMAGE", int(USEVIDEO));
            cv::write(fs, "VIDEONAME", VIDEONAME);
            cv::write(fs, "CAMERANAME", CAMERANAME);
            cv::write(fs, "ARMAKERNAME", ARMAKERNAME);
            cv::write(fs, "IMAGENAME", IMAGENAME);
            cv::write(fs, "CHANGESIZE", CHANGESIZE);
        }
        else
        {
            cv::FileNode node(fs.fs, NULL);

            USEVIDEO = int(node["USEVIDEO"]) == 1 ? true : false;
            USEIMAGE = int(node["USEIMAGE"]) == 1 ? true : false;
            VIDEONAME = std::string(node["VIDEONAME"]);
            CAMERANAME = std::string(node["CAMERANAME"]);
            ARMAKERNAME = std::string(node["ARMAKERNAME"]);
            IMAGENAME = std::string(node["IMAGENAME"]);
            CHANGESIZE = node["CHANGESIZE"];
        }
    };
    //
    bool USEVIDEO;
    bool USEIMAGE;
    std::string VIDEONAME;
    std::string CAMERANAME;
    std::string ARMAKERNAME;
    std::string IMAGENAME;
    int CHANGESIZE;
};

// toString template
template <typename T>
std::string inline ToString(T val)
{
    std::stringstream stream;
    stream << val;
    return stream.str();
}

cv::Ptr<cv::UMat> inline static GetReturnMatrix(cv::Mat return_image)
{

    cv::UMat tmpMat, temp_umat;
    return_image.copyTo(temp_umat);

    if (temp_umat.channels() == 1)
    {
        FORMATOUT("Convert color \n");
        cv::cvtColor(temp_umat, tmpMat, cv::COLOR_GRAY2BGR);
        cv::Ptr<cv::UMat> outMat(new cv::UMat(tmpMat));
        return outMat;
    }
    else
    {
        FORMATOUT("No convert color \n");
        cv::Ptr<cv::UMat> outMat(new cv::UMat(temp_umat));
        return outMat;
    }
}

cv::Ptr<cv::UMat> inline static GetReturnMatrix(cv::UMat return_image)
{

    cv::UMat tmpMat;
    if (return_image.channels() == 1)
    {
        FORMATOUT("Convert color \n");
        cv::cvtColor(return_image, tmpMat, cv::COLOR_GRAY2BGR);
        cv::Ptr<cv::UMat> outMat(new cv::UMat(tmpMat));
        return outMat;
    }
    else
    {
        FORMATOUT("No convert color \n");
        cv::Ptr<cv::UMat> outMat(new cv::UMat(return_image));
        return outMat;
    }
}

void inline static MatrixCheck(cv::Mat inMat)
{

    FORMATOUT("----------------------- \n");
    FORMATOUT("cols & rows    : %1% x %2% \n") % inMat.cols % inMat.rows;
    FORMATOUT("dimention      : %1% \n") % inMat.dims;
    // サイズ（2次元の場合）
    FORMATOUT("size           : %1% \n") % inMat.size();
    FORMATOUT("channels       : %1% \n") % inMat.channels();
    switch (inMat.depth())
    {
    case CV_8U:
        FORMATOUT("bit depth      : CV_8U \n");
        break;
    case CV_8S:
        FORMATOUT("bit depth      : CV_8S \n");
        break;
    case CV_16U:
        FORMATOUT("bit depth      : CV_16U \n");
        break;
    case CV_16S:
        FORMATOUT("bit depth      : CV_16S \n");
        break;
    case CV_32S:
        FORMATOUT("bit depth      : CV_32S \n");
        break;
    case CV_32F:
        FORMATOUT("bit depth      : CV_32F \n");
        break;
    case CV_64F:
        FORMATOUT("bit depth      : CV_64F \n");
        break;
    }
    //複数チャンネルから成る）要素のサイズ
    FORMATOUT("elem size      : %1% \n") % inMat.elemSize();
    //要素の総数 total
    FORMATOUT("total elem     : %1% \n") % inMat.total();
    // 1ステップ内のチャンネル総数
    FORMATOUT("step/elemSize1 : %1% \n") % inMat.step1();
    //データは連続か？
    FORMATOUT("isContinuous   : %1% \n") % inMat.isContinuous();
    //部分行列か？
    FORMATOUT("isSubmatrix    : %1% \n") % inMat.isSubmatrix();
    //データは空か？
    FORMATOUT("empty          : %1% \n") % inMat.empty();
    FORMATOUT("----------------------- \n");
}

void inline static MatrixCheck(cv::UMat inMat)
{
    MatrixCheck(inMat.getMat(cv::ACCESS_FAST));
}

void inline static MatrixValueDisplay(cv::Mat inMat1)
{

    std::string filename = "log.txt";
    std::ofstream writing_file;

    writing_file.open(filename.c_str(), std::ios::out);

    std::vector<cv::Mat> mat_channels;
    cv::split(inMat1, mat_channels);

    int value1;
    int value2;
    int value3;

    int cols = inMat1.cols;
    int rows = inMat1.rows;
    for (int j = 0; j < rows; j++)
    {
        FORMATOUT("\n-------------------------------------\n");
        FORMATOUTF("\n-------------------------------------\n", );
        for (int i = 0; i < cols; i++)
        {
            value1 = static_cast<int>(mat_channels[0].at<uchar>(j, i));
            value2 = static_cast<int>(mat_channels[1].at<uchar>(j, i));
            value3 = static_cast<int>(mat_channels[2].at<uchar>(j, i));
            FORMATOUT("(B%1%:G%2%:R%3%) ") % (int)value1 % (int)value2 % (int)value3;
            FORMATOUTF("(B%1%:G%2%:R%3%) ", % (int)value1 % (int)value2 % (int)value3);
        }
    }
    FORMATOUT("\n-------------------------------------\n");
}

void inline static MatrixValueDisplay(cv::UMat inMat1)
{
    MatrixValueDisplay(inMat1.getMat(cv::ACCESS_FAST));
}

void inline static MatrixSingleValueDisplay(cv::Mat inMat1)
{

    MatrixCheck(inMat1);

    cv::Vec3b value1;

    int cols = inMat1.cols;
    int rows = inMat1.rows;
    for (int j = 0; j < rows; j++)
    {
        FORMATOUT("\n-------------------------------------\n");
        for (int i = 0; i < cols; i++)
        {
            // j行目の先頭画素のポインタを取得
            cv::Vec3b *src = inMat1.ptr<cv::Vec3b>(j);
            cv::Vec3b vec3b = src[i]; // i番目にアクセス
            FORMATOUT("(%1% ") % static_cast<int>(vec3b[0]);
            FORMATOUT("%1% ") % static_cast<int>(vec3b[1]);
            FORMATOUT("%1%) ") % static_cast<int>(vec3b[2]);
        }
    }
}

void inline static MatrixSingleValueDisplay(cv::UMat inMat1)
{

    MatrixSingleValueDisplay(inMat1.getMat(cv::ACCESS_FAST));
}

void inline static DrawCircle(cv::Mat &img, cv::Point center, int color_id)
{
    const static cv::Scalar colors[] = {CV_RGB(0, 0, 255),
                                        CV_RGB(0, 128, 255),
                                        CV_RGB(0, 255, 255),
                                        CV_RGB(0, 255, 0),
                                        CV_RGB(255, 128, 0),
                                        CV_RGB(255, 255, 0),
                                        CV_RGB(255, 0, 0),
                                        CV_RGB(255, 0, 255)};

    int radius;
    radius = 10;
    cv::circle(img, center, radius, colors[color_id], 1, 8, 0);
}

void inline static detectAndDraw(cv::Mat &img, cv::CascadeClassifier &cascade, double scale, bool tryflip, cv::Scalar color, int minNeighbors)
{
    int i = 0;
    double t = 0;
    std::vector<cv::Rect> faces, faces2;
    const static cv::Scalar colors[] = {CV_RGB(0, 0, 255),
                                        CV_RGB(0, 128, 255),
                                        CV_RGB(0, 255, 255),
                                        CV_RGB(0, 255, 0),
                                        CV_RGB(255, 128, 0),
                                        CV_RGB(255, 255, 0),
                                        CV_RGB(255, 0, 0),
                                        CV_RGB(255, 0, 255)};
    cv::Mat gray, smallImg(cvRound(img.rows / scale), cvRound(img.cols / scale), CV_8UC1);

    if (img.channels() != 1)
    {
        cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    }
    else
    {
        gray = img;
    }

    resize(gray, smallImg, smallImg.size(), 0, 0, cv::INTER_LINEAR);
    equalizeHist(smallImg, smallImg);

    t = (double)cv::getTickCount();
    cascade.detectMultiScale(
        smallImg,
        faces,
        1.1,          //  scaleFactor    画像スケールにおける縮小量を表します。
        minNeighbors, // minNeighbors    物体候補となる矩形は、最低でもこの数だけの近傍矩形を含む必要があります。
        0
            //|CV_HAAR_FIND_BIGGEST_OBJECT
            //|CV_HAAR_DO_ROUGH_SEARCH
            | cv::CASCADE_SCALE_IMAGE,
        cv::Size(10, 10) // minSize   物体が取り得る最小サイズ。これよりも小さい物体は無視されます。
        //,
        // cv::Size(100, 100) //maxSize   物体が取り得る最大サイズ。
    );
    if (tryflip)
    {
        cv::flip(smallImg, smallImg, 1);
        cascade.detectMultiScale(
            smallImg,
            faces2,
            1.1, //  scaleFactor    画像スケールにおける縮小量を表します。
            2,   // minNeighbors    物体候補となる矩形は、最低でもこの数だけの近傍矩形を含む必要があります。
            0
                //|CV_HAAR_FIND_BIGGEST_OBJECT
                //|CV_HAAR_DO_ROUGH_SEARCH
                | cv::CASCADE_SCALE_IMAGE,
            cv::Size(30, 30));
        for (std::vector<cv::Rect>::const_iterator r = faces2.begin(); r != faces2.end(); r++)
        {
            faces.push_back(cv::Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
        }
    }
    t = (double)cv::getTickCount() - t;
    printf("detection time = %g ms\n", t / ((double)cv::getTickFrequency() * 1000.));

    FORMATOUT("count = %1% \n") % faces.size();

    for (std::vector<cv::Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++)
    {
        std::vector<cv::Rect> nestedObjects;
        cv::Point center;
        // cv::Scalar color = colors[i % 8];
        // cv::Scalar color = 1;
        int radius;

        double aspect_ratio = (double)r->width / r->height;
        // if (0.75 < aspect_ratio && aspect_ratio < 1.3)
        //{
        center.x = cvRound((r->x + r->width * 0.5) * scale);
        center.y = cvRound((r->y + r->height * 0.5) * scale);
        radius = cvRound((r->width + r->height) * 0.25 * scale);
        circle(img, center, radius, color, 3, 8, 0);
        FORMATOUT("center = %1% %2% \n") % center.x % center.y;
        //}
    }
    // cv::imshow("result", img);
}

cv::Ptr<std::vector<cv::Point>> inline static detectCenter(cv::Mat &img, cv::CascadeClassifier &cascade, double scale, int minNeighbors)
{
    int i = 0;
    double t = 0;
    cv::Ptr<std::vector<cv::Point>> Points(new std::vector<cv::Point>);
    std::vector<cv::Rect> tagets;
    cv::Mat gray, smallImg(cvRound(img.rows / scale), cvRound(img.cols / scale), CV_8UC1);

    if (img.channels() != 1)
    {
        cv::cvtColor(img, gray, cv::COLOR_BGR2RGB);
    }
    else
    {
        gray = img;
    }

    cv::resize(gray, smallImg, smallImg.size(), 0, 0, cv::INTER_LINEAR);
    cv::equalizeHist(smallImg, smallImg);

    t = (double)cv::getTickCount();
    cascade.detectMultiScale(
        smallImg,
        tagets,
        1.1,          //  scaleFactor    画像スケールにおける縮小量を表します。
        minNeighbors, // minNeighbors    物体候補となる矩形は、最低でもこの数だけの近傍矩形を含む必要があります。
        0
            //|CV_HAAR_FIND_BIGGEST_OBJECT
            //|CV_HAAR_DO_ROUGH_SEARCH
            //| CV_HAAR_DO_CANNY_PRUNING
            | cv::CASCADE_SCALE_IMAGE,
        cv::Size(20, 20) // minSize   物体が取り得る最小サイズ。これよりも小さい物体は無視されます。
                         //,
                         // cv::Size(100, 100) //maxSize   物体が取り得る最大サイズ。
    );
    t = (double)cv::getTickCount() - t;
    printf("detection time = %g ms\n", t / ((double)cv::getTickFrequency() * 1000.));

    FORMATOUT("count = %1% \n") % tagets.size();

    for (std::vector<cv::Rect>::const_iterator r = tagets.begin(); r != tagets.end(); r++, i++)
    {
        cv::Mat smallImgROI;
        std::vector<cv::Rect> nestedObjects;
        cv::Point center;
        double aspect_ratio = (double)r->width / r->height;
        // if (0.75 < aspect_ratio && aspect_ratio < 1.3)
        //{
        center.x = cvRound((r->x + r->width * 0.5) * scale);
        center.y = cvRound((r->y + r->height * 0.5) * scale);
        Points->push_back(center);
        FORMATOUT("center = %1% %2% \n") % center.x % center.y;
        //}
    }

    return Points;
}

cv::Ptr<std::vector<cv::Mat>> inline static GetFindContourMats(cv::Mat inMat1, int minimum_width, cv::Mat &retMat)
{

    cv::Mat taget_image, image_for_contours, dst_img, draw_img;

    //オリジナルコピー
    dst_img = inMat1.clone();

    cv::imwrite("image_original.png", dst_img);

    // 前処理
    // グレースケールに変換する
    cv::cvtColor(dst_img, image_for_contours, cv::COLOR_BGR2GRAY);
    cv::imwrite("image_gray.png", image_for_contours);
    //正規化
    cv::normalize(image_for_contours, image_for_contours, 0, 255, cv::NORM_MINMAX);
    // 2値化
    // cv::threshold(image_for_contours, image_for_contours, 60, 255, cv::THRESH_OTSU);
    // cv::threshold(image_for_contours, image_for_contours, 60, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::threshold(image_for_contours, image_for_contours, 70, 255, cv::THRESH_BINARY);

    draw_img = image_for_contours.clone();
    cv::imwrite("image_threshold.png", image_for_contours);

    std::vector<cv::Mat> contours;
    std::vector<cv::Vec4i> hierarchy;
    //注意： 元の image は，この関数によって書き換えられます．
    cv::findContours(image_for_contours, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    //
    cv::Ptr<std::vector<cv::Mat>> ret_mat_vector(new std::vector<cv::Mat>);
    std::vector<cv::Rect> rects;
    for (size_t i = 0; i < contours.size(); i++)
    {
        rects.push_back(cv::boundingRect(contours[i]));
    }

    std::sort(rects.begin(), rects.end(),
              [](const cv::Rect &a, const cv::Rect &b)
              { return (a.x < b.x); });

    int rect_number = 0;
    for (cv::Rect rect : rects)
    {

        if (rect.height < 20)
        {
            continue;
        }
        if ((rect.width * rect.height) > 200 * 200)
        {
            continue;
        }

        //最初の枠を除く
        if (rect.x != 1)
        {
            cv::Mat temp;
            draw_img(rect).copyTo(temp);
            cv::imwrite("clip" + ToString(rect_number) + ".png", temp);
            ret_mat_vector->push_back(temp);
        }

        //枠作成
        cv::line(dst_img, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y), cv::Scalar(0, 128, 0), 2, 0);
        cv::line(dst_img, cv::Point(rect.x + rect.width, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), cv::Scalar(0, 128, 0), 2, 0);
        cv::line(dst_img, cv::Point(rect.x + rect.width, rect.y + rect.height), cv::Point(rect.x, rect.y + rect.height), cv::Scalar(0, 128, 0), 2, 0);
        cv::line(dst_img, cv::Point(rect.x, rect.y + rect.height), cv::Point(rect.x, rect.y), cv::Scalar(0, 128, 0), 2, 0);

        rect_number++;
    }

    retMat = dst_img.clone();

    cv::imwrite("image_contours.png", dst_img);

    return ret_mat_vector;
}

void inline static ConvertToVectorOrMat(int picIndx, cv::String filepath, cv::Mat tg_mat, cv::Mat_<float> &vector)
{

    cv::Mat train, tmp_image, convert_mat;

    if (filepath != "")
    {
        train = cv::imread(filepath);
    }
    else
    {
        train = tg_mat;
    }

    ////サイズ変換
    cv::resize(
        train,
        convert_mat,
        cv::Size(28, 28), 0, 0, cv::INTER_CUBIC);

    //ベクトル化
    int idx = 0;
    int cols = convert_mat.cols;
    int rows = convert_mat.rows;
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            cv::Vec3b bgr = convert_mat.at<unsigned char>(row, col);
            vector(picIndx, idx) = bgr[0];
            idx++;
        }
    }
}

cv::Mat inline static LevelingPicture(cv::Mat input_image)
{
    cv::Mat gray_img, retMat, tmp_img, dst_img;

    // グレースケールに変換する
    cv::cvtColor(input_image, gray_img, cv::COLOR_BGR2GRAY);
    // 2値化
    // cv::threshold(gray_img, tmp_img, 100, 255, cv::THRESH_BINARY);
    cv::threshold(gray_img, tmp_img, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // cv::imwrite("./Tmp/t002.png", tmp_img);

    //コーナ検出
    std::vector<cv::Point2f> corners;
    std::vector<cv::Point2f>::iterator it_corner;
    cv::goodFeaturesToTrack(tmp_img, corners, 50, 0.10, 10);

    // 4コーナ検索
    std::vector<cv::Point2f> corners4;
    std::sort(corners.begin(), corners.end(),
              [](cv::Point2f a, cv::Point2f b) -> int
              { return (a.x < b.x); });
    corners4.push_back(corners[0]);
    FORMATOUT("Point1 %10.1f, %10.1f \n") % corners[0].x % corners[0].y;

    std::sort(corners.begin(), corners.end(),
              [](cv::Point2f a, cv::Point2f b) -> int
              { return (a.y < b.y); });
    corners4.push_back(corners[0]);
    FORMATOUT("Point2 %10.1f, %10.1f \n") % corners[0].x % corners[0].y;

    std::sort(corners.begin(), corners.end(),
              [](cv::Point2f a, cv::Point2f b) -> int
              { return (a.x > b.x); });
    corners4.push_back(corners[0]);
    FORMATOUT("Point2 %10.1f, %10.1f \n") % corners[0].x % corners[0].y;

    std::sort(corners.begin(), corners.end(),
              [](cv::Point2f a, cv::Point2f b) -> int
              { return (a.y > b.y); });
    corners4.push_back(corners[0]);
    FORMATOUT("Point2 %10.1f, %10.1f \n") % corners[0].x % corners[0].y;

    cv::UMat point_img(gray_img.rows, gray_img.cols, CV_8UC1, cv::Scalar(0));

    //コーナ描画
    if (false)
    {
        cv::Point point;
        for (auto it_corner = corners.begin(); it_corner != corners.end(); ++it_corner)
        {
            point.x = static_cast<int>(it_corner->x);
            point.y = static_cast<int>(it_corner->y);
            cv::circle(point_img, point, 1, cv::Scalar(200), -1);
            cv::circle(point_img, point, 8, cv::Scalar(200));
        }
        for (auto it_corner = corners4.begin(); it_corner != corners4.end(); ++it_corner)
        {
            FORMATOUT("%10.1f, %10.1f \n") % it_corner->x % it_corner->y;
            point.x = static_cast<int>(it_corner->x);
            point.y = static_cast<int>(it_corner->y);
            cv::circle(point_img, point, 8, cv::Scalar(150), -1);
        }
        cv::floodFill(tmp_img, cv::Point(1, 1), cv::Scalar(255));
        // cv::imwrite("./Tmp/t004.png", tmp_img);
    }

    //回転 Rotate the image by warpAffine taking the affine matrix
    auto angle = cv::fastAtan2((corners4[1].x - corners4[0].x), (corners4[0].y - corners4[1].y));
    std::cout << (boost::format("Angle %10.1f \n") % angle).str();

    cv::Point2f centerPt((float)point_img.cols / 2, (float)point_img.rows / 2);
    auto affine_matrix = cv::getRotationMatrix2D(centerPt, angle, 1);
    cv::warpAffine(tmp_img, dst_img, affine_matrix, tmp_img.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    return dst_img;
}

void inline static ResizeKeepRetio(cv::Mat &taget_mat, int width)
{

    if (width < 0)
    {
        //オリジナル
        return;
    }

    cv::Mat convert_mat, baseMat;

    if (taget_mat.channels() == 1)
    {
        baseMat = cv::Mat::zeros(cv::Size(width, width), CV_8UC1);
    }
    else
    {
        baseMat = cv::Mat::zeros(cv::Size(width, width), CV_8UC3);
    }

    std::cout << taget_mat.cols << " A" << std::endl;

    //縦横どっちか長い方
    int big_width = taget_mat.cols > taget_mat.rows ? taget_mat.cols : taget_mat.rows;
    // double ratio = ((double)width / (double)big_width) * 0.65;
    double ratio = ((double)width / (double)big_width);

    //サイズ変換
    // std::cout << taget_mat.cols << " B" << std::endl;
    // cv::resize(tmp_image, convert_mat, cv::Size(), wariai, wariai, cv::INTER_CUBIC);
    cv::resize(taget_mat, convert_mat, cv::Size(), ratio, ratio, cv::INTER_NEAREST);

    //中心をアンカーにして配置
    cv::Mat Roi1(baseMat, cv::Rect((width - convert_mat.cols) / 2, (width - convert_mat.rows) / 2, convert_mat.cols, convert_mat.rows));
    convert_mat.copyTo(Roi1);

    taget_mat = baseMat.clone();
}

void inline Convert_to_ml_Vector(const std::vector<cv::Mat> &train_samples, cv::Mat &trainData)
{
    //--Convert data
    const int rows = (int)train_samples.size();
    const int cols = std::max<int>(train_samples[0].cols, train_samples[0].rows);

    cv::Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
    trainData = cv::Mat(rows, cols, CV_32FC1);
    std::vector<cv::Mat>::const_iterator itr = train_samples.begin();
    std::vector<cv::Mat>::const_iterator end = train_samples.end();
    for (int i = 0; itr != end; ++itr, ++i)
    {
        CV_Assert(itr->cols == 1 || itr->rows == 1);
        if (itr->cols == 1)
        {
            transpose(*(itr), tmp);
            tmp.copyTo(trainData.row(i));
        }
        else if (itr->rows == 1)
        {
            itr->copyTo(trainData.row(i));
        }
    }
}

void inline Convert_to_HogVec(int picIndx, cv::Mat &original_train_mat, std::vector<cv::Mat> &gradients_lst, cv::Mat_<int> &vector_Label, int label)
{

    cv::Mat taget_image2, mat_for_vector;
    cv::HOGDescriptor hog;

    hog.winSize = cv::Size(32, 32);
    std::vector<cv::Point> location;
    std::vector<float> descriptors;

    hog.compute(original_train_mat, descriptors, cv::Size(8, 8), cv::Size(0, 0), location);

    gradients_lst.push_back(cv::Mat(descriptors).clone());

    //教師データをセット
    vector_Label(0, picIndx) = label;

#ifdef _DEBUG
    // imshow( "gradient", get_hogdescriptor_visu2(mat_for_vector.clone(), descriptors, hog.winSize) );
    // waitKey( 5 );
#endif
}
