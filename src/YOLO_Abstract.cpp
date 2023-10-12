#include "YOLO_Abstract.h"

void YOLO_Abstract::loadModel( std::filesystem::path pathToModel )
{
   this->net= std::make_unique<cv::dnn::Net>( cv::dnn::readNetFromONNX( pathToModel.string() ) );
}
