#include "PrimaryGeneratorAction.hh"
#include "DetectorConstruction.hh"

#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"
#include "G4Event.hh"

PrimaryGeneratorAction::PrimaryGeneratorAction()
  : fParticleGun(new G4ParticleGun(1))
{
  G4ParticleDefinition* proton =
      G4ParticleTable::GetParticleTable()->FindParticle("proton");

  fParticleGun->SetParticleDefinition(proton);
  fParticleGun->SetParticleMomentumDirection(G4ThreeVector(0.0, 0.0, 1.0));
  fParticleGun->SetParticleEnergy(400.0 * GeV);
  fParticleGun->SetParticlePosition(
      G4ThreeVector(0.0, 0.0, -0.5 * DetectorConstruction::kTargetLength * mm - 1.0 * mm));
}

PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
  delete fParticleGun;
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
  fParticleGun->GeneratePrimaryVertex(event);
}
