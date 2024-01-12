#pragma once
#include "YOLO_Abstract.h"


namespace YOLO_CPP {

   std::unique_ptr<YOLO_Abstract> makeYOLOAbstractUniquePtr( const YOLO_Abstract::YOLO_TYPES& yoloType );



}



