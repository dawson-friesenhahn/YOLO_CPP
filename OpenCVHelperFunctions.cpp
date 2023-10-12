#include "OpenCVHelperFunctions.h"

cv::Rect OpenCVHelperFunctions::makeRectFromCXCYWidthHeight( const float& center_x, const float& center_y, const float& width, const float& height )
{
   float topLeftX = center_x - (width / 2);
   float topLeftY = center_y - (height / 2);

   return cv::Rect( static_cast<int>(topLeftX), static_cast<int>(topLeftY), static_cast<int>(width), static_cast<int>(height) );
}
