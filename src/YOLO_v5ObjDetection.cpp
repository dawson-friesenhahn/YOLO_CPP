#include "YOLO_v5ObjDetection.h"
#include "OpenCVHelperFunctions.h"

std::vector<cv::Mat> YOLO_v5ObjDetection::pre_process( const cv::Mat& inputImage )
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

std::vector<DetectedFeature> YOLO_v5ObjDetection::post_process( std::vector<cv::Mat>& netOutputs, float x_scale_factor, float y_scale_factor, std::optional<std::pair<int, int>> tb_lr_borders_forSquare )
{
   const int anchors = netOutputs[0].size[1];
   const int channels = netOutputs[0].size[2];
   

   netOutputs[0] = netOutputs[0].reshape( 0, anchors );
   cv::Mat output = netOutputs[0];

   std::vector<cv::Rect> bboxList;
   std::vector<float> scoreList;
   std::vector<int> classList;
   std::vector<cv::Point2f> locations;

   for( int i = 0; i < anchors; i++ ) {
      auto row = output.row( i ).ptr<float>(); 
      auto score = row + 4; //overall objectness score         
    

      if( *score > SCORE_THRESHOLD ) { //this is the overall class score

         float center_x = *row * x_scale_factor;
         float center_y = *(row + 1) * y_scale_factor;
         float width = *(row + 2) * x_scale_factor;
         float height = *(row + 3) * y_scale_factor;

         if( tb_lr_borders_forSquare ) {
            center_y = center_y - tb_lr_borders_forSquare.value().first;
            center_x = center_x - tb_lr_borders_forSquare.value().second;
         }


         cv::Mat class_scores( 1, channels-5, CV_32FC1, row+5 );//make a new Mat of just the individual class scores
         double max_class_score = -1;
         cv::Point max_class_index;

         cv::minMaxLoc( class_scores, 0, &max_class_score, 0, &max_class_index );

         bboxList.push_back( OpenCVHelperFunctions::makeRectFromCXCYWidthHeight( center_x, center_y, width, height ) );
         scoreList.push_back( max_class_score );
         classList.push_back( max_class_index.x ); 
         locations.push_back( { center_x, center_y } );

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

      auto it = std::find_if( detections.begin(), detections.end(), [classIndex= feat.classIndex]( const DetectedFeature& feat ) {return feat.classIndex == classIndex; } );

      if( it != detections.end() && (*it).confidence > feat.confidence ) { //we've already found this feature before, and the other detection has a higher confidence.
         continue;
      }

      feat.location = locations.at( index );
      
      boundingBoxes.push_back( bboxList.at( index ) );
      detections.push_back( feat );
   }

   return detections;

}

void YOLO_v5ObjDetection::drawLabeledImage( const cv::Mat& inputImage, cv::Mat& outputImage, const std::vector<DetectedFeature>& detections, std::optional<std::vector<std::string>> classNames )
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
