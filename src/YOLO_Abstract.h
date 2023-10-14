#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <utility>
#include <filesystem>
#include "DetectedFeature.h"
#include <optional>


class YOLO_Abstract {
public:
   //Use a YOLO Model to make inferences on an image. Returns a vector of pairs, where each pair is {feature index, feature location (OpenCV screen space)}
   //Optionally, pass in a cv::Mat to use as a labeled output image and a vector of class names to use as labels
   virtual std::vector<DetectedFeature> getFeatureLocationsInImage(  const cv::Mat& inputImage, 
                                                                     std::optional<std::reference_wrapper<cv::Mat>> outputImage= std::nullopt, 
                                                                     std::optional<std::vector<std::string>> classNames= std::nullopt ) = 0;
   virtual void loadModel( std::filesystem::path pathToModel );
   float SCORE_THRESHOLD = 0.5;
   float NMS_THRESHOLD = 0.5;
protected:
   std::unique_ptr<cv::dnn::Net> net;
   int INPUT_WIDTH = 640;
   int INPUT_HEIGHT = 640;
};