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
                                                                     std::optional<std::vector<std::string>> classNames= std::nullopt );
   virtual void loadModel( std::filesystem::path pathToModel );
   float SCORE_THRESHOLD = 0.5;
   float NMS_THRESHOLD = 0.5;

   enum YOLO_TYPES { V5_OBJDETECTION, V8_OBJDETECTION, V8_POSE, YOLO_NULL };
protected:
   std::unique_ptr<cv::dnn::Net> net;
   virtual std::vector<cv::Mat> pre_process( const cv::Mat& inputImage ) = 0;
   virtual std::vector<DetectedFeature> post_process( std::vector < cv::Mat>& netOutputs, float x_scale_factor, float y_scale_factor ) = 0;
   virtual void drawLabeledImage( const cv::Mat& inputImage, cv::Mat& outputImage, std::vector<DetectedFeature> detections, std::optional<std::vector<std::string>> classNames = std::nullopt ) = 0;
   int INPUT_WIDTH = 640;
   int INPUT_HEIGHT = 640;

};
