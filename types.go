package main

import (
	mrand "math/rand"
)

// Environment holds world-level parameters.
type Environment struct {
	Gravity      float64
	GroundHeight int
	PadX         float64
	PadWidth     float64
}

type Lander struct {
	x, y   float64
	vx, vy float64
	angle  float64
}

// Policy decides actions for a lander each frame.
type Policy interface {
	Decide(l *Lander, env Environment, distGreen, distRed float64) (thrust bool, rotate float64)
}

// RandomPolicy issues random small rotations and intermittent thrust.
type RandomPolicy struct {
	rng *mrand.Rand
}

func (p *RandomPolicy) Decide(l *Lander, env Environment, distGreen, distRed float64) (bool, float64) {
	rotate := (p.rng.Float64() - 0.5) * 0.12
	thrust := p.rng.Float64() < 0.08
	return thrust, rotate
}

type Agent struct {
	Lander
	thrusting  bool
	landed     bool
	crashed    bool
	killed     bool // true if removed for going off-camera (score = 0)
	policy     Policy
	greenCoins int  // count of green coins collected
	redCoins   int  // count of red coins collected
	isChampion bool // true if this agent is from hall of fame
}

// Coin represents a collectible in the world
type Coin struct {
	x, y      float64
	isGreen   bool
	collected bool // true if collected in current generation
}
