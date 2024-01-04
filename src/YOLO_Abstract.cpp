#include "YOLO_Abstract.h"

std::vector<DetectedFeature> YOLO_Abstract::getFeatureLocationsInImage( const cv::Mat& inputImage, std::optional<std::reference_wrapper<cv::Mat>> outputImage, std::optional<std::vector<std::string>> classNames, bool forceSquareImage )
{
   std::vector<DetectedFeature> results;

   if( this->net == nullptr ) {
      if( outputImage && outputImage.value().get().empty() ) { //prevents us from doing something silly and trying to display an empty output
         outputImage.value().get() = inputImage.clone();
      }
      return results;
   }

   std::optional<cv::Mat> squareImage;
   std::optional<std::pair<int, int>> tb_lr_borders;

   if( forceSquareImage ) {
      int border_top_bottom = 0;
      int border_left_right = 0;
      int delta = std::abs( inputImage.rows - inputImage.cols ) / 2;//used to make the image square while keeping its aspect ration, which the blob function will not necessarily do
      inputImage.rows < inputImage.cols ? border_top_bottom = delta : border_left_right = delta;
      //NEED TO DO THIS TO ENSURE IT IS DOING inference without resizing the image and stretching it in some way

      tb_lr_borders = std::pair<int, int>( border_top_bottom, border_left_right );

      squareImage = cv::Mat();

      //AftrTimer tempTimer("--------CopyMakeBorder");
      cv::copyMakeBorder( inputImage, squareImage.value(), border_top_bottom, border_top_bottom, border_left_right, border_left_right, cv::BORDER_CONSTANT, cv::Scalar( 0, 0, 0 ) );
   }


   auto forwardPassResults = this->pre_process( squareImage.value_or(inputImage) );


   float x_scaling_factor = static_cast<float>(squareImage.value_or(inputImage).cols) / INPUT_WIDTH;
   float y_scaling_factor = static_cast<float>(squareImage.value_or(inputImage).rows) / INPUT_HEIGHT;

   results = post_process( forwardPassResults, x_scaling_factor, y_scaling_factor, tb_lr_borders );

   //if( tb_lr_borders ) {
   //   for( auto& result : results ) { //this works for now, but means bounding box labels are wrong. the "correct" thing to do would be to tell drawLabeledImage what the size of the borders are so that it can correct there (fixes bad bounding boxes)
   //      result.location.x = result.location.x - (*tb_lr_borders).second;
   //      result.location.y = result.location.y - (*tb_lr_borders).first;
   //   }
   //}

   if( outputImage ) {
      this->drawLabeledImage( inputImage, outputImage.value(), results, classNames );
   }
   return results;
}

void YOLO_Abstract::loadModel( std::filesystem::path pathToModel, bool useCUDABackend )
{
   try {
      this->net = std::make_unique<cv::dnn::Net>( cv::dnn::readNetFromONNX( pathToModel.string() ) );
   }
   catch( std::exception e ) {
      std::cout << e.what();
      std::abort();
   }
   if( useCUDABackend ) {
      this->net->setPreferableBackend( cv::dnn::DNN_BACKEND_CUDA ); //these should make things faster, but for some reason v8 obj Detection doesn't work with CUDA? v5 and v8 Pose seem to be working fine. 
      this->net->setPreferableTarget( cv::dnn::DNN_TARGET_CUDA );
   }
   else {
      this->net->setPreferableBackend( cv::dnn::DNN_BACKEND_OPENCV );
      this->net->setPreferableTarget( cv::dnn::DNN_TARGET_CPU );
   }
}
