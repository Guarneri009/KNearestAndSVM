/*!
@file		Cam.h
@brief		header for CCam
*/
#pragma once
#ifndef KNEARESTANDSVM_IMAGE_SOURCE_HPP
#define KNEARESTANDSVM_IMAGE_SOURCE_HPP


#include <opencv2/opencv.hpp>

/*!
@class		ImageSource
@brief		ImageSource interface
*/
class ImageSource
{
public:
    ImageSource();
    ~ImageSource();

    bool OpenCamera(int &w, int &h, int &c);
    bool OpenVideo(const std::string &name, int &w, int &h, int &c);
    bool OpenPicture(const std::string &name, int &w, int &h, int &c);
    void Close(void);
    bool Get(cv::Mat &img);
    bool GetOriginal(cv::Mat &img);
    void SaveImage(void);
    bool LoadParameters(const std::string &name, const bool undist = true);
    bool SaveParameters(const std::string &name) const;
    bool SaveVideo(const std::string &name, const float fps = 30.f);

    cv::Mat camera_matrix_;
    cv::Mat distortion_coefficients_;
    int display_size_ = -1;

private:

    cv::Mat image_;

    int device_id_; //!< devide ID
    cv::VideoCapture video_capture_;

    bool is_image_;    //!< video or not
    bool is_video_;    //!< video or not
    bool is_picture_ = false;
    bool is_undist_;            //!< use undistorted image or not

    cv::Mat mMapx, mMapy;	//!< map for undistortion

    int mWidth;		//!< image width
    int mHeight;	//!< image height
    int mChannel;	//!< number of channels
    int mSize;		//!< image data size (width*height*channels)

    cv::VideoWriter mWriter;	//!< opencv video class
    std::string mName;			//!< file name
    std::string mVidepName;			//!< file name
    bool mSave;		//!< save or not
};


#endif //KNEARESTANDSVM_IMAGE_SOURCE_HPP
