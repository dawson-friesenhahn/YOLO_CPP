#include "YOLO_Abstract.h"

std::vector<DetectedFeature> YOLO_Abstract::getFeatureLocationsInImage( const cv::Mat& inputImage, std::optional<std::reference_wrapper<cv::Mat>> outputImage, std::optional<std::vector<std::string>> classNames )
{
   std::vector<DetectedFeature> results;

   if( this->net == nullptr ) {
      if( outputImage && outputImage.value().get().empty() ) { //prevents us from doing something silly and trying to display an empty output
         outputImage.value().get() = inputImage.clone();
      }
      return results;
   }

   auto forwardPassResults = this->pre_process( inputImage );


   float x_scaling_factor = static_cast<float>(inputImage.cols) / INPUT_WIDTH;
   float y_scaling_factor = static_cast<float>(inputImage.rows) / INPUT_HEIGHT;

   results = post_process( forwardPassResults, x_scaling_factor, y_scaling_factor );

   if( outputImage ) {
      this->drawLabeledImage( inputImage, outputImage.value(), results, classNames );
   }
   return results;
}

void YOLO_Abstract::loadModel( std::filesystem::path pathToModel )
{
   try {
      this->net = std::make_unique<cv::dnn::Net>( cv::dnn::readNetFromONNX( pathToModel.string() ) );
   }
   catch( std::exception e ) {
      std::cout << e.what();
      std::abort();
   }
   this->net->setPreferableBackend( cv::dnn::DNN_BACKEND_CUDA ); //these should make things faster, but for some reason v8 obj Detection doesn't work with CUDA? v5 and v8 Pose seem to be working fine. 
   this->net->setPreferableTarget( cv::dnn::DNN_TARGET_CUDA );
}
