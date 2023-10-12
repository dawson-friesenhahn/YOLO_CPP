#include <opencv2/opencv.hpp>
#include "YOLO_Abstract.h"
#include "YOLO_v8ObjDetection.h"

//float SCORE_THRESHOLD = 0.3;
//float NMS_THRESHOLD = 0.5;
//
//float INPUT_WIDTH = 640.0;
//float INPUT_HEIGHT = 640.0;
//
//
//cv::Rect makeRectFromCXCYWidthHeight( const float& center_x, const float& center_y, const float& width, const float& height ) {
//   float topLeftX = center_x - (width / 2);
//   float topLeftY = center_y - (height / 2); 
//
//   return cv::Rect( static_cast<int>( topLeftX ), static_cast<int>( topLeftY ), static_cast<int>( width ), static_cast<int>( height ) );
//}
//
//
////class DetectedFeature {
////public:
////   DetectedFeature( cv::Rect bbox, float score, int classIndex ) : bbox( bbox ), score( score ), classIndex( classIndex ) {};
////   cv::Rect bbox;
////   float score;
////   int classIndex;
////   //std::string className;
////
////};
//
//
//int main() {
//   std::cout << "hello, world" << std::endl;
//
//   cv::Mat input= cv::imread( "C:/repos/aburn/usr/hub/GeometryCorrector/yolov8_models/f16_1.png" );
//   auto net= std::make_shared<cv::dnn::Net>( cv::dnn::readNetFromONNX("C:/repos/aburn/usr/hub/GeometryCorrector/yolov8_models/v8_objDetection.onnx"));
//
//   auto layerNames = net->getLayerNames();
//   
//   
//   
//   cv::Mat blob;
//
//   cv::dnn::blobFromImage( input, blob, 1. / 255., cv::Size( 640, 640 ), cv::Scalar(), true, false );
//   float x_scaling_factor = input.cols / INPUT_WIDTH;
//   float y_scaling_factor = input.rows / INPUT_HEIGHT;
//
//
//
//
//   std::vector<cv::Mat> outputs;
//   //pre process
//   try
//   {
//      net->setInput( blob );
//      
//      net->forward( outputs, net->getUnconnectedOutLayersNames() );
//      //std::cout << std::format(" *******Forward Pass Took {}\n", std::chrono::duration< double, std::milli>(std::chrono::system_clock::now() - timeMark));
//      //timeMark = std::chrono::system_clock::now();
//
//   }
//   catch( cv::Exception& e )
//   {
//      std::cout << e.msg << std::endl; // output exception message
//   }
//
//   //post process
//   const int anchors = outputs[0].size[2];//8400 for obj detection
//   const int channels = outputs[0].size[1];//30 for obj detection
//   outputs[0] = outputs[0].reshape( 1, channels );
//   cv::Mat output = outputs[0].t();
//
//   std::vector<cv::Rect> bboxList;
//   std::vector<float> scoreList;
//   std::vector<int> classList;
//
//   for( int i = 0; i < anchors; i++ ) {
//      auto row = output.row(i).ptr<float>(); //glenn says order is cx,cy, w, h, class confidences here: https://github.com/ultralytics/ultralytics/issues/1563
//      auto scores = row + 4;
//      float max_score = *(scores);
//      int max_score_idx = 0;
//      for( int class_score_idx = 1; class_score_idx < channels - 4; class_score_idx++ ) {
//         float score = *(scores + class_score_idx);
//         if( score > max_score ) {
//            max_score = score;
//            max_score_idx = class_score_idx;
//         }
//      }
//
//      if( max_score > SCORE_THRESHOLD ) {
//         //std::cout << "hey, good detection? " << std::endl;
//         float center_x = *row * x_scaling_factor;
//         float center_y = *(row + 1) * y_scaling_factor;
//         float width = *(row + 2) * x_scaling_factor;
//         float height = *(row + 3) * y_scaling_factor;
//
//         bboxList.push_back( makeRectFromCXCYWidthHeight( center_x, center_y, width, height ) );
//         scoreList.push_back( max_score );
//         classList.push_back( max_score_idx );
//      }
//   }
//
//   //NMS
//   std::vector<int> indices;
//   cv::dnn::NMSBoxes( bboxList, scoreList, SCORE_THRESHOLD, NMS_THRESHOLD, indices );
//
//   ////Wrap up detections in a coherent data structure
//   //std::vector<DetectedFeature> detections;
//   //for( const auto& index : indices ) {
//   //   detections.push_back( DetectedFeature( bboxList.at( index ), scoreList.at( index ), classList.at( index ) ));
//   //}
//
//   ////draw a pretty picture
//   //for( const auto& detection : detections ) {
//   //   cv::rectangle( input, detection.bbox, cv::Scalar( 255, 0, 0 ) );
//   //}
//
//   cv::imshow( "Detections:", input );
//   cv::waitKey( 0 );
//   //int trash = 0;
//   //std::cin >> trash;
//}


int main() {
   std::unique_ptr<YOLO_Abstract> yolo = std::make_unique<YOLO_v8ObjDetection>();
   yolo->loadModel( "C:/repos/aburn/usr/hub/GeometryCorrector/yolov8_models/v8_objDetection.onnx" );
   
   cv::Mat inputImg = cv::imread( "C:/repos/aburn/usr/hub/GeometryCorrector/yolov8_models/f16_1.png" );
   cv::Mat outputImg = cv::Mat();

   yolo->getFeatureLocationsInImage( inputImg, outputImg );

   cv::imshow( "Input:", inputImg );
   cv::waitKey( 0 );

   cv::imshow( "Output:", outputImg );
   cv::waitKey( 0 );

}