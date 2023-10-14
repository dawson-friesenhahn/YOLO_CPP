#include "DetectedFeature.h"

std::string DetectedFeature::toString() const
{
   std::string ret = "Detected Feature:\n ";
   ret += "\tClass Index: " + std::to_string( classIndex ) + "\n";
   ret += "\tConfidence: " + std::to_string( confidence ) + "\n";
   ret += "\tLocation: (" + std::to_string( location.x ) + ", " + std::to_string( location.y ) + ")";
   return ret;
}

std::ostream& operator<<( std::ostream& os, const DetectedFeature& feat )
{
   os << feat.toString();
   return os;
}
