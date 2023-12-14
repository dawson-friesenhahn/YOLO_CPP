#include <opencv2/opencv.hpp>
#include "YOLO_Abstract.h"
#include "YOLO_v8ObjDetection.h"
#include "YOLO_v8Pose.h"
#include "YOLO_v5ObjDetection.h"
#include "DetectedFeature.h"


int main() {
   
   
   //Uncomment for v8ObjDetection
   //std::unique_ptr<YOLO_Abstract> yolo = std::make_unique<YOLO_v8ObjDetection>(); //goal is to be able to easily sub in a new model architecture here
   //yolo->loadModel( "../models/v8_objDetection.onnx" );
  
   //Uncomment for v5ObjDetection
   //std::unique_ptr<YOLO_Abstract> yolo = std::make_unique<YOLO_v5ObjDetection>();
   //yolo->loadModel( "../models/v5_objDetection.onnx" );

   //Uncomment for v8Pose
   std::unique_ptr<YOLO_Abstract> yolo = std::make_unique<YOLO_v8Pose>();
   yolo->loadModel( "../models/v8_Pose.onnx" );
   
   
   cv::Mat inputImg = cv::imread( "../images/f16_1.png" );


   //Uncomment for v8Pose (multi-model). Same class is being used (YOLOv_v8Pose), but I left these 3 lines here to show how I used it.
   //std::unique_ptr<YOLO_Abstract> yolo = std::make_unique<YOLO_v8Pose>();
   //yolo->loadModel( "../models/v8_pose_multiClass.onnx" );
   //dynamic_cast<YOLO_v8Pose*>(yolo.get())->setNumClasses( 3 ); //I don't like this line, but I couldn't think of a concise way to determine the number of classes just by parsing the onnx file. I'm sure with some thought, we could figure it out.
   //cv::Mat inputImg = cv::imread( "../images/capture0.jpeg" );
   
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