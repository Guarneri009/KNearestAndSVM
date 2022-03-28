/*!
@file		Cam.cpp
@brief		functions in CCam
*/

#include "common.hpp"
#include "image_source.hpp"

/*!
@brief		constructor
*/
ImageSource::ImageSource()
{
    device_id_ = 0;
    mSave = false;
    is_video_ = false;
    is_undist_ = false;
    mSize = false;
    mChannel = 1;
    mWidth=0;     //!< image width
    mHeight=0;    //!< image height
    mChannel=0;   //!< number of channels
}

/*!
@brief		destructor
*/
ImageSource::~ImageSource()
{
    Close();
}

/*!
@brief		open video
@param[in]	name	file name
@param[out]	w		image width
@param[out]	h		image height
@param[out]	c		num of channels
@retval		successed or not
*/
bool ImageSource::OpenVideo(
    const std::string &name,
    int &w,
    int &h,
    int &c
)
{
    is_video_ = true;
    mName = name;

    bool isOpen = video_capture_.open(mName);

    if (isOpen) {
        video_capture_ >> image_;

        //ResizeKeepRetio(mImg, displaySize);

        mWidth = w = image_.cols;
        mHeight = h = image_.rows;
        mChannel = c = image_.channels();
        mSize = mHeight*(int)image_.step;

    }

    return isOpen;
}

/*!
@brief		open Image
@param[in]	name	file name
@param[out]	w		image width
@param[out]	h		image height
@param[out]	c		num of channels
@retval		successed or not
*/
bool ImageSource::OpenPicture(
    const std::string &name,
    int &w,
    int &h,
    int &c
)
{

    image_ = cv::imread(name);

    //if (mImg.data != NULL){
    if (!image_.empty()) {

        mWidth = w = image_.cols;
        mHeight = h = image_.rows;
        mChannel = c = image_.channels();

        mSize = mHeight*(int)image_.step;
    }
    else {
        return false;
    }
    is_image_ = true;
    return true;
}

/*!
@brief		open camera
@param[out]	w	image width
@param[out]	h	image height
@param[out]	c	num of channels
@retval		successed or not
*/
bool ImageSource::OpenCamera(
    int &w,
    int &h,
    int &c
)
{
    if (!video_capture_.isOpened()) {

        if (!video_capture_.open(device_id_)) {		// if opening failed
            printf("Cam ID %d not found\n", device_id_);
            return false;
        }

        // VGA
        video_capture_.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        video_capture_.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

        // check image
        int count = 0;
        while (image_.data == NULL) {

            video_capture_ >> image_;

            //ResizeKeepRetio(mImg, displaySize);

            ++count;

            if (count > 10) {		// if retrieval failed
                printf("Cannot retrieve images\n");
                return false;
            }
        }

        mWidth = w = image_.cols;
        mHeight = h = image_.rows;
        mChannel = c = image_.channels();

        mSize = mHeight*(int)image_.step;


        //SaveVideo("testvideo.avi");


        return true;
    }
    else {
        return false;
    }
}

/*!
@brief		close camera
*/
void ImageSource::Close(void)
{
    if (video_capture_.isOpened()) {
        video_capture_.release();
    }
}

/*!
@brief		get image
@param[out]	img		image
@retval		successed or not
*/
bool ImageSource::Get(cv::Mat &img)
{
    //picture sequence
    if (is_picture_) {
        if (is_undist_) {
            cv::remap(image_.clone(), image_, mMapx, mMapy, cv::INTER_LINEAR);
        }
        img = image_.clone();
        return true;
    }

    //Video
    if (is_video_ & image_.empty()) {
        video_capture_.open(mName);
        video_capture_ >> image_;
        return true;
    }


    if (video_capture_.isOpened()) {
        video_capture_ >> image_;
        return true;
    }
    else {
        fprintf(stderr, "Failed to Camera\n");
        return false;
    }

    //picture sequence
//    if (video_capture_.isOpened()) {
//        video_capture_ >> image_;
//
//        if (is_video_ & image_.empty()) {
//            video_capture_.open(mName);
//            video_capture_ >> image_;
//            SaveImage();
//        }
//
//        //std::cout << mImg.cols << " C" << std::endl;
//        //ResizeKeepRetio(mImg, displaySize);
//    }
//    else {
//        fprintf(stderr, "Failed to capture\n");
//        return false;
//    }

//    if (is_undist_) {
//        cv::remap(image_.clone(), image_, mMapx, mMapy, cv::INTER_LINEAR);
//    }
//
//    img = image_.clone();
    //ResizeKeepRetio(img, displaySize);

    return true;
}

/*!
@brief		get image
@param[out]	img		image
@retval		successed or not
*/
bool ImageSource::GetOriginal(cv::Mat &img)
{
    if (is_image_) {
        img = image_.clone();
        //ResizeKeepRetio(img, displaySize);
        return true;
    }

    if (video_capture_.isOpened()) {
        video_capture_ >> image_;
        SaveImage();

        //video�̏ꍇ�čĐ�
        if (is_video_ & image_.empty()) {
            video_capture_.open(mName);
            video_capture_ >> image_;
            SaveImage();
        }

        //std::cout << mImg.cols << " C" << std::endl;
        //ResizeKeepRetio(mImg, displaySize);
    }
    else {
        fprintf(stderr, "Failed to capture\n");
        return false;
    }

    img = image_.clone();
    //ResizeKeepRetio(img, displaySize);
    return true;
}

/*!
@brief      save video
@param[in]  name        file name
@param[in]  fps         fps for recording
@retval     succeeded or not
*/
bool ImageSource::SaveVideo(
    const std::string &name,
    const float fps
)
{
    bool isOpen;
    isOpen = video_capture_.isOpened();

    if (isOpen) {   // if camera opened

        mSave = true;
        mVidepName = name;

        bool isColor;

        if (mChannel == 1) {
            isColor = false;
        }
        else {
            isColor = true;
        }

#ifdef _MSC_VER
        bool ret = mWriter.open(mVidepName, CV_FOURCC_PROMPT, fps, cv::Size(mWidth, mHeight), isColor);
#else
        bool ret = mWriter.open(mVidepName, CV_FOURCC_DEFAULT, fps, cv::Size(mWidth, mHeight), isColor);
#endif
        return ret;
    }
    else {
        return false;
    }
}

/*!
@brief		save image to video
*/
void ImageSource::SaveImage(void)
{
    if (mSave) {
        mWriter << image_;
    }
}

/*!
@brief		load camera parameters (call after Open)
@param[in]	name		file name
@param[in]	undist		undistort image or not
@retval		successed or not
*/
bool ImageSource::LoadParameters(
    const std::string &name,
    const bool undist
)
{
    cv::FileStorage fs;
    if (!fs.open(name, cv::FileStorage::READ)) {
        fprintf(stderr, "Cannot load camera parameters\n");
        return false;
    }

    cv::FileNode node(fs.fs, NULL);
    cv::read(node["camera_matrix"], camera_matrix_);
    cv::read(node["distortion_coefficients"], distortion_coefficients_);

    is_undist_ = undist;

    if (is_undist_) {
        cv::initUndistortRectifyMap(camera_matrix_, distortion_coefficients_, cv::Mat(), camera_matrix_, image_.size(), CV_32FC1, mMapx, mMapy);
        distortion_coefficients_ = cv::Mat_<double>::zeros(5, 1);
    }

    return true;
}

/*!
@brief		save camera parameters
@param[in]	name		file name
@retval		successed or not
*/
bool ImageSource::SaveParameters(const std::string &name) const
{
    if (camera_matrix_.empty() || distortion_coefficients_.empty()) {
        fprintf(stderr, "Empty parameters\n");
        return false;
    }

    cv::FileStorage fs;
    fs.open(name + ".xml", cv::FileStorage::WRITE);
    cv::write(fs, "camera_matrix", camera_matrix_);
    cv::write(fs, "distortion_coefficients", distortion_coefficients_);
    fs.release();

    return true;
}
