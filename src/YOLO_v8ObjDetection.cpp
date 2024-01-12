#include "YOLO_v8ObjDetection.h"
#include "OpenCVHelperFunctions.h"

std::vector<cv::Mat> YOLO_v8ObjDetection::pre_process( const cv::Mat& inputImage )
{
   cv::Mat blob;
   cv::dnn::blobFromImage( inputImage, blob, 1. / 255., cv::Size( INPUT_WIDTH, INPUT_HEIGHT ), cv::Scalar(), true, false ); //TODO hard coded 640x640 here
   
   std::vector<cv::Mat> outputs;
   //pre process
   try
   {
      net->setInput( blob );

      net->forward( outputs, net->getUnconnectedOutLayersNames() );
      //std::cout << std::format(" *******Forward Pass Took {}\n", std::chrono::duration< double, std::milli>(std::chrono::system_clock::now() - timeMark));
      //timeMark = std::chrono::system_clock::now();

   }
   catch( cv::Exception& e )
   {
      std::cout << e.msg << std::endl; // output exception message
   }

   return outputs;
}

std::vector<DetectedFeature> YOLO_v8ObjDetection::post_process( std::vector<cv::Mat>& netOutputs, float x_scale_factor, float y_scale_factor, std::optional<std::pair<int, int>> tb_lr_borders_forSquare )
{
   const int anchors = netOutputs[0].size[2];//8400 for obj detection
   const int channels = netOutputs[0].size[1];//30 for obj detection
   netOutputs[0] = netOutputs[0].reshape( 1, channels );
   cv::Mat output = netOutputs[0].t();

   std::vector<cv::Rect> bboxList;
   std::vector<float> scoreList;
   std::vector<int> classList;
   std::vector<cv::Point2f> locations;

   for( int i = 0; i < anchors; i++ ) {
      auto row = output.row( i ).ptr<float>(); //glenn says order is cx,cy, w, h, class confidences here: https://github.com/ultralytics/ultralytics/issues/1563
      auto scores = row + 4;
      float max_score = *(scores);
      int max_score_idx = 0;
      for( int class_score_idx = 1; class_score_idx < channels - 4; class_score_idx++ ) { //TODO change to minMaxLoc
         float score = *(scores + class_score_idx);
         if( score > max_score ) {
            max_score = score;
            max_score_idx = class_score_idx;
         }
      }

      if( max_score > SCORE_THRESHOLD ) {
         //std::cout << "hey, good detection? " << std::endl;
         float center_x = *row * x_scale_factor;
         float center_y = *(row + 1) * y_scale_factor;
         float width = *(row + 2) * x_scale_factor;
         float height = *(row + 3) * y_scale_factor;

         if( tb_lr_borders_forSquare ) {
            center_y = center_y - tb_lr_borders_forSquare.value().first;
            center_x = center_x - tb_lr_borders_forSquare.value().second;
         }

         bboxList.push_back( OpenCVHelperFunctions::makeRectFromCXCYWidthHeight( center_x, center_y, width, height ) );
         locations.push_back( { center_x, center_y } );
         scoreList.push_back( max_score );
         classList.push_back( max_score_idx );
      }
   }

   //NMS
   std::vector<int> indices;
   cv::dnn::NMSBoxes( bboxList, scoreList, SCORE_THRESHOLD, NMS_THRESHOLD, indices );

   std::vector<DetectedFeature> detections;
   this->boundingBoxes.clear();
   for( const auto& index : indices ) {
      DetectedFeature feat;

      feat.classIndex = classList.at( index );
      feat.confidence = scoreList.at( index );

      auto it = std::find_if( detections.begin(), detections.end(), [classIndex = feat.classIndex]( const DetectedFeature& feat ) {return feat.classIndex == classIndex; } );

      if( it != detections.end() && (*it).confidence > feat.confidence ) { //we've already found this feature before, and the other detection has a higher confidence.
         continue;
      }

      feat.location = locations.at( index );

      boundingBoxes.push_back( bboxList.at( index ) );
      detections.push_back( feat );
   }

   return detections;

}

void YOLO_v8ObjDetection::drawLabeledImage( const cv::Mat& inputImage, cv::Mat& outputImage, const std::vector<DetectedFeature>& detections, std::optional<std::vector<std::string>> classNames )
{
   outputImage = inputImage.clone();
   assert( detections.size() == this->boundingBoxes.size() );
   for( int i = 0; i < detections.size(); i++ ) {
      cv::rectangle( outputImage, boundingBoxes.at( i ), { 255,0,0 } );
      cv::drawMarker( outputImage, detections.at( i ).location, { 255,0,0 }, cv::MARKER_CROSS, 5 );

      std::string label;
      if( classNames ) {
         label = classNames->at( detections.at( i ).classIndex );
      }
      else {
         label = std::to_string( detections.at( i ).classIndex );
      }

      label += ": " + std::to_string( detections.at( i ).confidence );
      cv::putText( outputImage, label, detections.at( i ).location, cv::FONT_HERSHEY_SIMPLEX, 0.25, { 255,0,0 } );

   }
}
