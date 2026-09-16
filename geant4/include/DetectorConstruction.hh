#ifndef DetectorConstruction_h
#define DetectorConstruction_h 1

#include "G4VUserDetectorConstruction.hh"
#include "globals.hh"

class G4VPhysicalVolume;

// Cylindrical molybdenum target: radius 125 mm, length 60 cm, matching the
// SHiP proton target dimensions used for the HNL production-flux estimate.
class DetectorConstruction : public G4VUserDetectorConstruction
{
  public:
    DetectorConstruction() = default;
    ~DetectorConstruction() override = default;

    G4VPhysicalVolume* Construct() override;

    static constexpr G4double kTargetRadius = 125.0; // mm
    static constexpr G4double kTargetLength = 600.0; // mm (60 cm)
};

#endif
