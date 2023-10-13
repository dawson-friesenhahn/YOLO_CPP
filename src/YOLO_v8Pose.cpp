#include "YOLO_v8Pose.h"
#include "OpenCVHelperFunctions.h"

std::vector<DetectedFeature>  YOLO_v8Pose::getFeatureLocationsInImage( const cv::Mat& inputImage, std::optional<std::reference_wrapper<cv::Mat>> outputImage, std::optional<std::vector<std::string>> classNames ) {
   auto forwardPassResults = this->pre_process( inputImage );


   float x_scaling_factor = static_cast<float>(inputImage.cols) / INPUT_WIDTH;
   float y_scaling_factor = static_cast<float>(inputImage.rows) / INPUT_HEIGHT;

   auto results = post_process( forwardPassResults, x_scaling_factor, y_scaling_factor );

   if( outputImage ) {
      this->drawLabeledImage( inputImage, outputImage.value(), results, classNames );
   }
   return results;
}

std::vector<cv::Mat> YOLO_v8Pose::pre_process( const cv::Mat& inputImage )
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

std::vector<DetectedFeature> YOLO_v8Pose::post_process( std::vector<cv::Mat>& netOutputs, float x_scale_factor, float y_scale_factor )
{
   const int anchors = netOutputs[0].size[2];//8400 for obj detection
   const int channels = netOutputs[0].size[1];
   if( numKeypoints != (channels - 4) / 3 ) {
      if( numKeypoints != -1 ) {
         std::cout << "Hey, the number of keypoints changed. That's weird.\n";
      }
      numKeypoints = (channels - 4) / 3;
   }



   netOutputs[0] = netOutputs[0].reshape( 1, channels );
   cv::Mat output = netOutputs[0].t();

   std::vector<cv::Rect> bboxList;
   std::vector<float> scoreList;
   std::vector<int> classList;

   std::vector<std::vector<DetectedFeature>> listOfListOfKeypoints;

   for( int i = 0; i < anchors; i++ ) {
      auto row = output.row( i ).ptr<float>(); 
      auto score = row + 4;
      auto kp_ptr = row + 5;
     

      if( *score > SCORE_THRESHOLD ) { //this is the overall class score
         //std::cout << "hey, good detection? " << std::endl;
         float center_x = *row * x_scale_factor;
         float center_y = *(row + 1) * y_scale_factor;
         float width = *(row + 2) * x_scale_factor;
         float height = *(row + 3) * y_scale_factor;

         bboxList.push_back( OpenCVHelperFunctions::makeRectFromCXCYWidthHeight( center_x, center_y, width, height ) );
         scoreList.push_back( *score );
         classList.push_back( 0 ); //TODO where is the class index actually scored?


         std::vector<DetectedFeature> kps;
         for( int k = 0; k < numKeypoints; k++ ) { //get this from dimension of output
            DetectedFeature feat;
            float kps_x = (*(kp_ptr + 3 * k)) * x_scale_factor;
            float kps_y = (*(kp_ptr + 3 * k + 1)) * y_scale_factor;
            float kps_s = *(kp_ptr + 3 * k + 2);

            feat.location = { kps_x, kps_y };
            feat.confidence = kps_s;
            feat.classIndex = k; //just assumes that keypoints are in the same order always
            kps.push_back( feat );
         }
         listOfListOfKeypoints.push_back( kps );

      }
   }

   //NMS
   std::vector<int> indices;
   cv::dnn::NMSBoxes( bboxList, scoreList, SCORE_THRESHOLD, NMS_THRESHOLD, indices );

   std::vector<DetectedFeature> detections;
   this->boundingBoxes.clear(); 
   for( const auto& index : indices ) {
      boundingBoxes.push_back( bboxList.at( index ) );
      auto keypoints = listOfListOfKeypoints.at( index );
      detections.insert( detections.end(), keypoints.begin(), keypoints.end() );
   }

   return detections;

}

void YOLO_v8Pose::drawLabeledImage( const cv::Mat& inputImage, cv::Mat& outputImage, std::vector<DetectedFeature> detections, std::optional<std::vector<std::string>> classNames )
{
   outputImage = inputImage.clone();
   
   assert( boundingBoxes.size() == detections.size() / numKeypoints );

   for( int i = 0; i < boundingBoxes.size(); i++ ) {
      cv::rectangle( outputImage, boundingBoxes.at( i ), { 255,0,0 } ); //TODO maybe make this so that different instances get different colors

      for( int kpt_index = 0; kpt_index < numKeypoints; kpt_index++ ) {
         std::string label;
         if( classNames ) {
            label = classNames->at( detections.at( i*numKeypoints + kpt_index ).classIndex );
         }
         else {
            label = std::to_string( detections.at( i*numKeypoints + kpt_index ).classIndex );
         }
         cv::drawMarker( outputImage, detections.at( i * numKeypoints + kpt_index ).location, { 255,0,0 }, cv::MARKER_CROSS, 5 );



         label += ": " + std::to_string( detections.at( i * numKeypoints + kpt_index ).confidence );
         cv::putText( outputImage, label, detections.at( i * numKeypoints + kpt_index ).location, cv::FONT_HERSHEY_SIMPLEX, 0.25, { 255,0,0 } );
      }

   }
}
