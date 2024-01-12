#include "YOLO_AbstractFactory.h"
#include "YOLO_v5ObjDetection.h"
#include "YOLO_v8ObjDetection.h"
#include "YOLO_v8Pose.h"

std::unique_ptr<YOLO_Abstract> YOLO_CPP::makeYOLOAbstractUniquePtr( const YOLO_Abstract::YOLO_TYPES& yoloType )
{
   switch( yoloType ) {
   case YOLO_Abstract::V5_OBJDETECTION:
      return std::make_unique<YOLO_v5ObjDetection>();
   case YOLO_Abstract::V8_OBJDETECTION:
      return std::make_unique<YOLO_v8ObjDetection>();
   case YOLO_Abstract::V8_POSE:
      return std::make_unique<YOLO_v8Pose>();
   default:
      std::cout << "Why are you trying to load a YOLO_NULL?\n";
      std::abort();
   }
}
