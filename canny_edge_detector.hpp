#include <iostream>
#include <string>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void CannyEdgeDetector() {
	Mat image = imread(INSERT IMAGE PATH);

	namedWindow("Original image", WINDOW_AUTOSIZE);
	imshow("Original image", image);

    Mat greyscaleImage = Mat::zeros(image.rows, image.cols, CV_8U);

    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            cv::Vec3b rgb = image.at<cv::Vec3b>(y, x);

            uchar grayscale = static_cast<uchar>((rgb[0] + rgb[1] + rgb[2]) / 3);

            greyscaleImage.at<uchar>(y, x) = grayscale;
        }
    }

    namedWindow("Greyscale image", WINDOW_AUTOSIZE);
    imshow("Greyscale image", greyscaleImage);

	Mat imageWithGaussianFilterApplied = Mat::zeros(image.rows, image.cols, CV_8U);

	int kernelSize = 7;
	double sigma = 3.3;

    int halfSize = kernelSize / 2;
    double kernelSum = 0.0;

    Mat kernel = Mat::zeros(kernelSize, kernelSize, CV_64F);

    for (int i = -halfSize; i <= halfSize; ++i) {
        for (int j = -halfSize; j <= halfSize; ++j) {
            double value = exp(-(i * i + j * j) / (2.0 * sigma * sigma));
            kernel.at<double>(i + halfSize, j + halfSize) = value;
            kernelSum += value;
        }
    }

    kernel /= kernelSum;

    for (int i = 0; i < greyscaleImage.rows; i++) {
        for (int j = 0; j < greyscaleImage.cols; j++) {
            double sum = 0.0;
            for (int k = -halfSize; k <= halfSize; k++) {
                for (int l = -halfSize; l <= halfSize; l++) {
                    int imgX = j + l;
                    int imgY = i + k;
                    imgX = max(0, min(imgX, image.cols - 1));
                    imgY = max(0, min(imgY, image.rows - 1));

                    sum += greyscaleImage.at<uchar>(imgY, imgX) * kernel.at<double>(k + halfSize, l + halfSize);
                }
            }
            imageWithGaussianFilterApplied.at<uchar>(i, j) = static_cast<uchar>(sum);
        }
    }

	namedWindow("Image after applying Gaussian filter", WINDOW_AUTOSIZE);
	imshow("Image after applying Gaussian filter", imageWithGaussianFilterApplied);


    Mat intensityGradient;
    Mat imageWithIntensityGradientApplied;

    Sobel(imageWithGaussianFilterApplied, intensityGradient, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
    convertScaleAbs(intensityGradient, imageWithIntensityGradientApplied);

    namedWindow("Image with intensity gradient applied", WINDOW_AUTOSIZE);
    imshow("Image with intensity gradient applied", imageWithIntensityGradientApplied);

    Mat edges;
    Canny(imageWithIntensityGradientApplied, edges, 50, 150);

    Mat edges_bool;
    edges.convertTo(edges_bool, CV_8U);
    edges_bool = edges > 0;

    Mat imageWithSuppressedEdges = Mat::zeros(edges.size(), CV_8U);;

    for (int y = 0; y < edges.rows; ++y) {
        for (int x = 0; x < edges.cols; ++x) {
            if (edges_bool.at<uchar>(y, x)) {
                bool isLocalMaximum = true;

                for (int i = -kernelSize / 2; i <= kernelSize / 2; ++i) {
                    for (int j = -kernelSize / 2; j <= kernelSize / 2; ++j) {
                        if (y + i >= 0 && y + i < edges.rows && x + j >= 0 && x + j < edges.cols) {
                            if (edges_bool.at<uchar>(y + i, x + j) > edges_bool.at<uchar>(y, x)) {
                                isLocalMaximum = false;
                                break;
                            }
                        }
                    }
                }

                if (isLocalMaximum) {
                    imageWithSuppressedEdges.at<uchar>(y, x) = 255;
                }
            }
        }
    }

    namedWindow("Image with suppressed edges", WINDOW_AUTOSIZE);
    imshow("Image with suppressed edges", imageWithSuppressedEdges);

    int thresholds = 100;

    Mat imageWithGradientThresholdingApplied;

    threshold(imageWithSuppressedEdges, imageWithGradientThresholdingApplied, thresholds, 255, THRESH_BINARY);

    namedWindow("Image with gradient thresholding applied", WINDOW_AUTOSIZE);
    imshow("Image with gradient thresholding applied", imageWithGradientThresholdingApplied);

	waitKey(0);
}
