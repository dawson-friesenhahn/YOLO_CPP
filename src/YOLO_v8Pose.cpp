#include "YOLO_v8Pose.h"
#include "OpenCVHelperFunctions.h"

void YOLO_v8Pose::setNumClasses( int numClasses )
{
   this->numClasses = numClasses;
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

std::vector<DetectedFeature> YOLO_v8Pose::post_process( std::vector<cv::Mat>& netOutputs, float x_scale_factor, float y_scale_factor, std::optional<std::pair<int, int>> tb_lr_borders_forSquare )
{
   const int anchors = netOutputs[0].size[2];//8400 for obj detection
   const int channels = netOutputs[0].size[1]; //bbox cx, cy, w, h, class0Score, class1Score, ... classNScore, then kp x,y, score. This number should be 4+ num_classes + num_keypoints * 3. If not, maybe you forgot to train with keypoint visibility in your labels so you don't have keypoint confidence in the output.
   if( numKeypoints != (channels - 4- this->numClasses) / 3 ) {
      if( numKeypoints != -1 ) {
         std::cout << "Hey, the number of keypoints changed. That's weird.\n";
      }
      numKeypoints = (channels - 4- this->numClasses) / 3;
   }



   netOutputs[0] = netOutputs[0].reshape( 1, channels );
   cv::Mat output = netOutputs[0].t();

   std::vector<cv::Rect> bboxList;
   std::vector<float> scoreList;
   std::vector<int> classList;

   std::vector<std::vector<DetectedFeature>> listOfListOfKeypoints;

   for( int i = 0; i < anchors; i++ ) {
      auto row = output.row( i ).ptr<float>(); 
      auto classScore_ptr = row + 4;
      auto kp_ptr = row + 4 + this->numClasses;
      
      int bestClassIdx = 0;
      float bestClassScore = *classScore_ptr;

      for( int j = 1; j < this->numClasses; j++ ) {
         if( *(classScore_ptr + j) > bestClassScore ) {
            bestClassIdx = j;
            bestClassScore = *(classScore_ptr + j);
         }
      }
     

      if( bestClassScore > SCORE_THRESHOLD ) { //this is the overall class score
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
         scoreList.push_back( bestClassScore );
         classList.push_back( bestClassIdx ); 


         std::vector<DetectedFeature> kps;
         for( int k = 0; k < numKeypoints; k++ ) { //get this from dimension of output
            DetectedFeature feat;
            float kps_x = (*(kp_ptr + 3 * k)) * x_scale_factor;
            float kps_y = (*(kp_ptr + 3 * k + 1)) * y_scale_factor;
            float kps_s = *(kp_ptr + 3 * k + 2);

            if( tb_lr_borders_forSquare ) {
               kps_y = kps_y - tb_lr_borders_forSquare.value().first;
               kps_x = kps_x - tb_lr_borders_forSquare.value().second;
            }

            feat.location = { kps_x, kps_y };
            feat.confidence = kps_s;
            feat.classIndex = (numKeypoints * bestClassIdx) + k; //this attempts to make a classIndex here match what an identical feature would produce in a v5ObjectDetection
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
   allKeypoints.clear();
   for( const auto& index : indices ) {
      boundingBoxes.push_back( bboxList.at( index ) );
      auto keypoints = listOfListOfKeypoints.at( index );
      allKeypoints.insert( allKeypoints.end(), keypoints.begin(), keypoints.end() ); //keep all the keypoints in this vector, for drawing the labels
      for( auto& keypoint : keypoints ) {
         if( keypoint.confidence > this->SCORE_THRESHOLD ) {
            detections.push_back( keypoint ); //only the good keypoints get returned.
         }
      }
   }

   return detections;

}

void YOLO_v8Pose::drawLabeledImage( const cv::Mat& inputImage, cv::Mat& outputImage, const std::vector<DetectedFeature>& detections, std::optional<std::vector<std::string>> classNames )
{
   static const cv::Scalar red( { 0, 0, 255 } );
   static const cv::Scalar green( { 0, 255, 0 } );

   outputImage = inputImage.clone();
   
   assert( boundingBoxes.size() == allKeypoints.size() / numKeypoints );

   for( int i = 0; i < boundingBoxes.size(); i++ ) {
      cv::rectangle( outputImage, boundingBoxes.at( i ), { 255,0,0 } ); //TODO maybe make this so that different instances get different colors
   }

   for( const auto& keypt: allKeypoints) {
      std::string label;
      if( classNames ) {
         label = classNames->at( keypt.classIndex );
      }
      else {
         label = std::to_string( keypt.classIndex );
      }

      cv::Scalar color;
      if( keypt.confidence > this->SCORE_THRESHOLD ) {
         color = green;
      }
      else {
         color = red;
      }

      cv::drawMarker( outputImage, keypt.location, color, cv::MARKER_CROSS, 5 );
      label += ": " + std::to_string( keypt.confidence );
      cv::putText( outputImage, label, keypt.location, cv::FONT_HERSHEY_SIMPLEX, 0.25, color );
   }

   
}
