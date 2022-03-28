#include "common.hpp"

cv::String ml_path = "../data/";
cv::String KNearest = ml_path + "KNearestDigit.xml"; // file name
cv::String filenameSvm = ml_path + "SVMDigit.xml";   // file name
cv::Ptr<cv::ml::KNearest> knn = cv::ml::StatModel::load<cv::ml::KNearest>(
    KNearest);
cv::Ptr<cv::ml::SVM> svm = cv::ml::StatModel::load<cv::ml::SVM>(filenameSvm);

int K = 5;
cv::Mat response, dist;
std::vector<cv::Mat> gr_lst;

cv::String knn_first = "";
cv::String knn_second = "";
cv::String svm_first = "";
cv::String svm_second = "";

cv::Mat number_calclate(cv::Mat image_)
{

    cv::Mat retMat;
    // Sprit number conture matrix
    cv::Ptr<std::vector<cv::Mat>> train_mats = GetFindContourMats(image_, 100, retMat);

    int flag = 0;
    int flag2 = 0;

    cv::Mat_<int> train_labels_dummy(1, 4 * 30);

    // KNearest
    for (cv::Mat train_mat : *train_mats)
    {

        std::vector<cv::Mat> gradient_list;
        cv::Mat train_vector;

        ResizeKeepRetio(train_mat, 32);

        Convert_to_HogVec(0, train_mat, gradient_list, train_labels_dummy, -1);
        Convert_to_ml_Vector(gradient_list, train_vector);
        knn->findNearest(train_vector, K, cv::noArray(), response, dist);

        // Result
        std::cerr << "KNearest:" << response << std::endl;
        // std::cerr << "KNearest:" << dist << std::endl;
        int ans = response.at<float>(0, 0);
        if (ans == 0)
        {
            flag = 1;
            continue;
        }

        flag == 0 ? knn_first += ToString(ans) : knn_second += ToString(ans);
    }

    // SVM
    for (cv::Mat train_mat : *train_mats)
    {

        std::vector<cv::Mat> gradient_list;
        cv::Mat train_vector;

        ResizeKeepRetio(train_mat, 32);

        Convert_to_HogVec(0, train_mat, gradient_list, train_labels_dummy, -1);
        Convert_to_ml_Vector(gradient_list, train_vector);
        knn->findNearest(train_vector, K, cv::noArray(), response, dist);

        // Result
        int response_Svm = static_cast<int>(svm->predict(train_vector));
        std::cerr << "SVM:" << response_Svm << std::endl;
        int ans2 = response_Svm;
        if (ans2 == 0)
        {
            flag2 = 1;
            continue;
        }

        flag2 == 0 ? svm_first += ToString(ans2) : svm_second += ToString(ans2);
    }

    int first = std::atoi(knn_first.c_str());
    int second = std::atoi(knn_second.c_str());
    int ans = first + second;
    std::cerr << "ANS:" << ans << std::endl;

    int first2 = std::atoi(svm_first.c_str());
    int second2 = std::atoi(svm_second.c_str());
    int ans2 = first2 + second2;
    std::cerr << "ANS:" << ans2 << std::endl;

    // cv::Mat dst_img = cv::Mat(200, 300, CV_8UC1, cv::Scalar(255, 255, 255));
    cv::Mat dst_img = retMat.clone();
    // dst_img = image_.clone();

    if (second != 0)
    {
        cv::putText(dst_img, "KNN:" + ToString(ans), cv::Point(30, 200),
                    cv::FONT_HERSHEY_PLAIN, 3.0, cv::Scalar(0, 0, 255), 3);
    }
    else
    {
        cv::putText(dst_img, "KNN:X", cv::Point(30, 200), cv::FONT_HERSHEY_PLAIN,
                    3.0, cv::Scalar(0, 0, 255), 3);
    }

    if (second2 != 0)
    {
        cv::putText(dst_img, "SVM:" + ToString(ans2), cv::Point(30, 250),
                    cv::FONT_HERSHEY_PLAIN, 3.0, cv::Scalar(0, 0, 255), 3);
    }
    else
    {
        cv::putText(dst_img, "SVM:X", cv::Point(30, 250), cv::FONT_HERSHEY_PLAIN,
                    3.0, cv::Scalar(0, 0, 255), 3);
    }

    // system("pause");
    // return GetReturnMatrix(dst_img);
    return dst_img;
}

int main()
{
#ifndef _WIN32
    char dir[255];
    getcwd(dir, 255);
    std::cout << "Current Directory : " << dir << std::endl;
#endif

    std::cout << "start.." << std::endl;
    if (cv::ocl::haveOpenCL())
    {
        LOG("OpenCL is avaiable...");
    }

    cv::ocl::setUseOpenCL(false);
    cv::ocl::setUseOpenCL(true);
    if (cv::ocl::useOpenCL())
    {
        LOG("OpenCL is Use...");
    }
    else
    {
        LOG("OpenCL is Not Use...");
    }

    int device_id_ = 0; //!< devide ID
    cv::VideoCapture video_capture_;

    if (!video_capture_.open(device_id_))
    { // if opening failed
        printf("Camera ID = %d not found\n", device_id_);
        return false;
    }

    // VGA
    video_capture_.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    video_capture_.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    cv::Mat image_;
    cv::Mat image_out;

    // check image
    int count = 0;
    for (;;)
    {

        video_capture_ >> image_;

        ResizeKeepRetio(image_, 640);

        image_out = number_calclate(image_);

        cv::imshow("Image View", image_out);
        char key = (char)cv::waitKey(video_capture_.isOpened() ? 50 : 50);
        if (key == 'e')
            break;
    }

    return 0;
}
