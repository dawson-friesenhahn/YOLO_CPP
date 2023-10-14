#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>

struct DetectedFeature {
   cv::Point2f location = cv::Point2f( 0, 0 );
   float confidence = -1;
   int classIndex = -1;
   std::string toString() const;

   
};

std::ostream& operator<<( std::ostream& os, const DetectedFeature& feat );