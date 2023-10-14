#include <opencv2/opencv.hpp>
#include "YOLO_Abstract.h"
#include "YOLO_v8ObjDetection.h"
#include "YOLO_v8Pose.h"
#include "YOLO_v5ObjDetection.h"
#include "DetectedFeature.h"


int main() {
   //std::unique_ptr<YOLO_Abstract> yolo = std::make_unique<YOLO_v8ObjDetection>(); //goal is to be able to easily sub in a new model architecture here
   //yolo->loadModel( "../models/v8_objDetection.onnx" );
   std::unique_ptr<YOLO_Abstract> yolo = std::make_unique<YOLO_v5ObjDetection>();
   yolo->loadModel( "../models/v5_objDetection.onnx" );
   
   cv::Mat inputImg = cv::imread( "../images/f16_2.png" );
   cv::Mat outputImg = cv::Mat();

   auto detections= yolo->getFeatureLocationsInImage( inputImg, outputImg );

   for( const auto& detection : detections ) {
      std::cout << detection << "\n";
   }


   cv::imshow( "Input:", inputImg );
   cv::waitKey( 0 );

   cv::imshow( "Output:", outputImg );
   cv::waitKey( 0 );

}