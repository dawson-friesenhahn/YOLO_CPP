#include <opencv2/opencv.hpp>
#include "YOLO_Abstract.h"
#include "YOLO_v8ObjDetection.h"


int main() {
   std::unique_ptr<YOLO_Abstract> yolo = std::make_unique<YOLO_v8ObjDetection>(); //goal is to be able to easily sub in a new model architecture here
   yolo->loadModel( "../models/v8_objDetection.onnx" );
   
   cv::Mat inputImg = cv::imread( "../images/f16_1.png" );
   cv::Mat outputImg = cv::Mat();

   yolo->getFeatureLocationsInImage( inputImg, outputImg );

   cv::imshow( "Input:", inputImg );
   cv::waitKey( 0 );

   cv::imshow( "Output:", outputImg );
   cv::waitKey( 0 );

}