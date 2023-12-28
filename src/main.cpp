#include <opencv2/opencv.hpp>
#include "YOLO_Abstract.h"
#include "YOLO_v8ObjDetection.h"
#include "YOLO_v8Pose.h"
#include "YOLO_v5ObjDetection.h"
#include "DetectedFeature.h"


int main() {
   
   std::unique_ptr<YOLO_Abstract> yolo = std::make_unique<YOLO_v8ObjDetection>(); 
   yolo->loadModel( "../models/v8_objDetection.onnx" );
  
   
   cv::Mat inputImg = cv::imread( "../images/f16_1.png" );
   
   cv::Mat outputImg = cv::Mat();

   auto detections= yolo->getFeatureLocationsInImage( inputImg, outputImg );

   for( const auto& detection : detections ) {
      std::cout << detection << "\n";
   }
   
   cv::imshow( "Output (CUDA Backend):", outputImg );
   cv::waitKey( 0 );


   yolo->loadModel( "../models/v8_objDetection.onnx", false );

   detections = yolo->getFeatureLocationsInImage( inputImg, outputImg );

   for( const auto& detection : detections ) {
      std::cout << detection << "\n";
   }

   cv::imshow( "Output (CPU Backend):", outputImg );
   cv::waitKey( 0 );

}