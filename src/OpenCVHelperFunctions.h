#pragma once
#include <opencv2/opencv.hpp>

namespace OpenCVHelperFunctions {
   cv::Rect makeRectFromCXCYWidthHeight( const float& center_x, const float& center_y, const float& width, const float& height );
}