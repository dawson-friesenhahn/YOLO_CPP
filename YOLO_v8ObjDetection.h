#pragma once
#include "YOLO_Abstract.h"

class YOLO_v8ObjDetection : public YOLO_Abstract {
public:
   virtual std::vector<DetectedFeature> getFeatureLocationsInImage(  const cv::Mat& inputImage, 
                                                                     std::optional<std::reference_wrapper<cv::Mat>> outputImage = std::nullopt, 
                                                                     std::optional<std::vector<std::string>> classNames = std::nullopt ) override;
private:
   std::vector<cv::Mat> pre_process( const cv::Mat& inputImage );
   std::vector<DetectedFeature> post_process( std::vector < cv::Mat>& netOutputs, float x_scale_factor, float y_scale_factor );
   void drawLabeledImage( const cv::Mat& inputImage, cv::Mat& outputImage, std::vector<DetectedFeature> detections, std::optional<std::vector<std::string>> classNames= std::nullopt );
   std::vector<cv::Rect> boundingBoxes; //vector of retained bounding boxes after post_processing is done. Retained so we can draw pretty pictures.
};