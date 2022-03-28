#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>


using namespace cv;
using namespace cv::ml;
using namespace std;

#define LOG(message) std::cout << message << std::endl
#define FORMATOUT std::cout << boost::format
#define FORMATOUTF(arg,arg2) writing_file << boost::format(arg) arg2

//toString template
template <typename T>
std::string ToString(T val)
{
    std::stringstream stream;
    stream << val;
    return stream.str();
}

/*
* Convert training/testing set to be used by OpenCV Machine Learning algorithms.
* TrainData is a matrix of size (#samples x max(#cols,#rows) per samples), in 32FC1.
* Transposition of samples are made if needed.
*/
void convert_to_ml2(const std::vector< cv::Mat > & train_samples, cv::Mat& trainData)
{
    //--Convert data
    const int rows = (int)train_samples.size();
    const int cols = std::max<int>(train_samples[0].cols, train_samples[0].rows);
    cv::Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
    trainData = cv::Mat(rows, cols, CV_32FC1);
    vector< Mat >::const_iterator itr = train_samples.begin();
    vector< Mat >::const_iterator end = train_samples.end();
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

// From http://www.juergenwiki.de/work/wiki/doku.php?id=public:hog_descriptor_computation_and_visualization
cv::Mat get_hogdescriptor_visu2(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size)
{
    const int DIMX = size.width;
    const int DIMY = size.height;
    float zoomFac = 3;
    Mat visu;
    resize(color_origImg, visu, Size((int)(color_origImg.cols*zoomFac), (int)(color_origImg.rows*zoomFac)));

    int cellSize = 8;
    int gradientBinSize = 9;
    float radRangeForOneBin = (float)(CV_PI / (float)gradientBinSize); // dividing 180 into 9 bins, how large (in rad) is one bin?
                                                                       // prepare data structure: 9 orientation / gradient strenghts for each cell
    int cells_in_x_dir = DIMX / cellSize;
    int cells_in_y_dir = DIMY / cellSize;
    float*** gradientStrengths = new float**[cells_in_y_dir];
    int** cellUpdateCounter = new int*[cells_in_y_dir];
    for (int y = 0; y < cells_in_y_dir; y++)
    {
        gradientStrengths[y] = new float*[cells_in_x_dir];
        cellUpdateCounter[y] = new int[cells_in_x_dir];
        for (int x = 0; x < cells_in_x_dir; x++)
        {
            gradientStrengths[y][x] = new float[gradientBinSize];
            cellUpdateCounter[y][x] = 0;

            for (int bin = 0; bin < gradientBinSize; bin++)
                gradientStrengths[y][x][bin] = 0.0;
        }
    }

    // nr of blocks = nr of cells - 1
    // since there is a new block on each cell (overlapping blocks!) but the last one
    int blocks_in_x_dir = cells_in_x_dir - 1;
    int blocks_in_y_dir = cells_in_y_dir - 1;

    // compute gradient strengths per cell
    int descriptorDataIdx = 0;
    int cellx = 0;
    int celly = 0;

    for (int blockx = 0; blockx < blocks_in_x_dir; blockx++)
    {
        for (int blocky = 0; blocky < blocks_in_y_dir; blocky++)
        {
            // 4 cells per block ...
            for (int cellNr = 0; cellNr < 4; cellNr++)
            {
                // compute corresponding cell nr
                cellx = blockx;
                celly = blocky;
                if (cellNr == 1) celly++;
                if (cellNr == 2) cellx++;
                if (cellNr == 3)
                {
                    cellx++;
                    celly++;
                }

                for (int bin = 0; bin < gradientBinSize; bin++)
                {
                    float gradientStrength = descriptorValues[descriptorDataIdx];
                    descriptorDataIdx++;

                    gradientStrengths[celly][cellx][bin] += gradientStrength;

                } // for (all bins)


                  // note: overlapping blocks lead to multiple updates of this sum!
                  // we therefore keep track how often a cell was updated,
                  // to compute average gradient strengths
                cellUpdateCounter[celly][cellx]++;

            } // for (all cells)
        } // for (all block x pos)
    } // for (all block y pos)


      // compute average gradient strengths
    for (celly = 0; celly < cells_in_y_dir; celly++)
    {
        for (cellx = 0; cellx < cells_in_x_dir; cellx++)
        {

            float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];

            // compute average gradient strenghts for each gradient bin direction
            for (int bin = 0; bin < gradientBinSize; bin++)
            {
                gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
            }
        }
    }

    // draw cells
    for (celly = 0; celly < cells_in_y_dir; celly++)
    {
        for (cellx = 0; cellx < cells_in_x_dir; cellx++)
        {
            int drawX = cellx * cellSize;
            int drawY = celly * cellSize;

            int mx = drawX + cellSize / 2;
            int my = drawY + cellSize / 2;

            rectangle(visu, Point((int)(drawX*zoomFac), (int)(drawY*zoomFac)), Point((int)((drawX + cellSize)*zoomFac), (int)((drawY + cellSize)*zoomFac)), Scalar(100, 100, 100), 1);

            // draw in each cell all 9 gradient strengths
            for (int bin = 0; bin < gradientBinSize; bin++)
            {
                float currentGradStrength = gradientStrengths[celly][cellx][bin];

                // no line to draw?
                if (currentGradStrength == 0)
                    continue;

                float currRad = bin * radRangeForOneBin + radRangeForOneBin / 2;

                float dirVecX = cos(currRad);
                float dirVecY = sin(currRad);
                float maxVecLen = (float)(cellSize / 2.f);
                float scale = 2.5; // just a visualization scale, to see the lines better

                                   // compute line coordinates
                float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
                float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
                float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
                float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

                // draw gradient visualization
                line(visu, Point((int)(x1*zoomFac), (int)(y1*zoomFac)), Point((int)(x2*zoomFac), (int)(y2*zoomFac)), Scalar(0, 255, 0), 1);

            } // for (all bins)

        } // for (cellx)
    } // for (celly)


      // don't forget to free memory allocated by helper data structures!
    for (int y = 0; y < cells_in_y_dir; y++)
    {
        for (int x = 0; x < cells_in_x_dir; x++)
        {
            delete[] gradientStrengths[y][x];
        }
        delete[] gradientStrengths[y];
        delete[] cellUpdateCounter[y];
    }
    delete[] gradientStrengths;
    delete[] cellUpdateCounter;

    return visu;

} // get_hogdescriptor_visu

void ConvertToVector(int picIndx, cv::String filepath, cv::Mat_<float>& vector, cv::Mat_<int>& vector_Label, int label) {

    cv::Mat original_train_mat = cv::imread(filepath);
    cv::Mat mat_for_vector;

    // 前処理
    // グレースケール、正規化、2値化
    cv::cvtColor(original_train_mat, mat_for_vector, cv::COLOR_BGR2GRAY);
    cv::normalize(mat_for_vector, mat_for_vector, 0, 255, cv::NORM_MINMAX);
    cv::threshold(mat_for_vector, mat_for_vector, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    int idx = 0;
    int rows = mat_for_vector.rows;
    int cols = mat_for_vector.cols;
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            cv::Vec3b bgr = mat_for_vector.at<unsigned char>(row, col);
            vector(picIndx, idx) = bgr[0];
            idx++;
        }
    }
    //教師データをセット
    vector_Label(0, picIndx) = label;
}

void ConvertToHogVevtor(int picIndx, cv::String filepath, std::vector< Mat >& gr_lst, cv::Mat_<int>& vector_Label, int label) {

    cv::Mat original_train_mat = cv::imread(filepath);
    cv::Mat taget_image2, mat_for_vector;

    ////サイズ変換
    cv::resize(
        original_train_mat,
        taget_image2,
        cv::Size(8 * 4, 8 * 4), 0, 0, cv::INTER_CUBIC);

    // 前処理
    // グレースケール、正規化、2値化
    cv::cvtColor(taget_image2, mat_for_vector, cv::COLOR_BGR2GRAY);
    //cv::normalize(mat_for_vector, mat_for_vector, 0, 255, cv::NORM_MINMAX);
    cv::threshold(mat_for_vector, mat_for_vector, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    cv::HOGDescriptor hog;
    hog.winSize = cv::Size(32, 32);
    std::vector<cv::Point> location;
    std::vector< float > descriptors;

    hog.compute(mat_for_vector, descriptors, Size(8, 8), Size(0, 0), location);

    gr_lst.push_back(Mat(descriptors).clone());

    //教師データをセット
    vector_Label(0, picIndx) = label;

#ifdef _DEBUG
    //if (label == 4) {
    //    imshow("gradient", get_hogdescriptor_visu2(mat_for_vector.clone(), descriptors, hog.winSize));
    //    imwrite("image_HOG.png", get_hogdescriptor_visu2(mat_for_vector.clone(), descriptors, hog.winSize));
    //    waitKey(3000);
    //}
#endif

}

//int main_KNearest(int argc, char** argv)
int main_KNearest()
{
    int count = 30;

#ifdef WIN
    cv::String filePath = "../opencvsorce/";
#else
    cv::String filePath = "./opencvsorce/";
#endif

    cv::String filenameKNearest = filePath + "ML/KNearestDigit.xml";	// file name
    cv::String filenameSVM = filePath + "ML/SVMDigit.xml";	// file name
    cv::String picture_path = filePath + "train/KNearestDisit/";
    cv::String picture = "";

    cv::Mat_<int> train_labels(1, 30 * 4);
    std::vector< Mat > gradient_lst;
    std::vector< Mat > gradient_test_lst;

    //KNearest
    cv::Ptr<cv::ml::KNearest>  knn(cv::ml::KNearest::create());
    // Set up SVM's parameters
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    //SVM training..
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setKernel(cv::ml::SVM::LINEAR);
    //svm->setKernel(cv::ml::SVM::RBF);
    svm->setGamma(3);
    //停止基準
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));

    //Train
    Mat train_data;
    Mat train_data2;
    int idx = 0;
    if (true) {
        std::cerr << "KNearest train.." << std::endl;
        std::cerr << "SVM train.." << std::endl;
        std::cerr << "----------------------" << std::endl;

        //ベクトル化 1-3
        int idx = 0;
        for (int num = 1; num <= 3; num++) {
            for (int type = 0; type < count; type++) {
                picture = picture_path + "img_" + ToString(num) + "_" + ToString(type) + ".bmp";
                ConvertToHogVevtor(idx++, picture, gradient_lst, train_labels, num);
            }
        }

        //P
        for (int type = 0; type < count; type++) {
            picture = picture_path + "img_P_" + ToString(type) + ".bmp";
            ConvertToHogVevtor(idx++, picture, gradient_lst, train_labels, 0);
        }

        convert_to_ml2(gradient_lst, train_data);
        // Train the SVKNearestM
        knn->train(train_data, cv::ml::ROW_SAMPLE, train_labels);
        // Train the SVM
        svm->train(train_data, cv::ml::ROW_SAMPLE, train_labels);
        
        knn->save(filenameKNearest);
        svm->save(filenameSVM);
    }


    // ============ TEST ============
    picture = picture_path + "img_2_21.bmp";

    ConvertToHogVevtor(idx++, picture, gradient_test_lst, train_labels, -1);
    convert_to_ml2(gradient_test_lst, train_data2);

    int K = 1;
    cv::Mat response, dist;

    knn->findNearest(train_data2, K, cv::noArray(), response, dist);


    std::cerr << "KNearest:" << response << std::endl;
    std::cerr << "KNearest:" << dist << std::endl;

    int response_Svm = static_cast<int>(svm->predict(train_data2));
    std::cerr << "SVM:" << response_Svm << std::endl;

    //system("pause");

    return 0;

}

