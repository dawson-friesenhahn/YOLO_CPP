#pragma once
#include "YOLO_Abstract.h"

class YOLO_v8Pose : public YOLO_Abstract {
public:
   //should probably just use the Abstract version, but signature is here for convenience
   /*virtual std::vector<DetectedFeature> getFeatureLocationsInImage(const cv::Mat& inputImage,
                                                                     std::optional<std::reference_wrapper<cv::Mat>> outputImage = std::nullopt,
                                                                     std::optional<std::vector<std::string>> classNames = std::nullopt ) override; */
   void setNumClasses( int numClasses );
private:
   std::vector<cv::Mat> pre_process( const cv::Mat& inputImage ) override;
   std::vector<DetectedFeature> post_process( std::vector < cv::Mat>& netOutputs, float x_scale_factor, float y_scale_factor ) override;
   void drawLabeledImage( const cv::Mat& inputImage, cv::Mat& outputImage, std::vector<DetectedFeature> detections, std::optional<std::vector<std::string>> classNames = std::nullopt ) override;
   std::vector<cv::Rect> boundingBoxes; //vector of retained bounding boxes after post_processing is done. Retained so we can draw pretty pictures.
   int numKeypoints = -1;
   int numClasses = 1; 
   std::vector<DetectedFeature> allKeypoints;
};
