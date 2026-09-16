#include "DetectorConstruction.hh"

#include "G4NistManager.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"
#include "G4VisAttributes.hh"

G4VPhysicalVolume* DetectorConstruction::Construct()
{
  G4NistManager* nist = G4NistManager::Instance();

  // World: vacuum box, generously larger than the target.
  G4Material* vacuum = nist->FindOrBuildMaterial("G4_Galactic");
  const G4double worldSize = 4.0 * kTargetLength * mm;
  G4Box* solidWorld = new G4Box("World", worldSize, worldSize, worldSize);
  G4LogicalVolume* logicWorld = new G4LogicalVolume(solidWorld, vacuum, "World");
  G4VPhysicalVolume* physWorld = new G4PVPlacement(
      nullptr, G4ThreeVector(), logicWorld, "World", nullptr, false, 0, true);

  // Target: solid cylindrical molybdenum block.
  G4Material* molybdenum = nist->FindOrBuildMaterial("G4_Mo");
  G4Tubs* solidTarget = new G4Tubs("Target",
                                    0.0,
                                    kTargetRadius * mm,
                                    0.5 * kTargetLength * mm,
                                    0.0, 360.0 * deg);
  G4LogicalVolume* logicTarget = new G4LogicalVolume(solidTarget, molybdenum, "Target");
  new G4PVPlacement(nullptr, G4ThreeVector(), logicTarget, "Target",
                     logicWorld, false, 0, true);

  G4VisAttributes* targetVis = new G4VisAttributes(G4Colour(0.6, 0.6, 0.7));
  targetVis->SetForceSolid(true);
  logicTarget->SetVisAttributes(targetVis);

  return physWorld;
}
