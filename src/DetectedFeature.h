#pragma once
#include <opencv2/opencv.hpp>

struct DetectedFeature {
   cv::Point2f location = cv::Point2f( 0, 0 );
   float confidence = -1;
   int classIndex = -1;
};