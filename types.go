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

// Rule represents a single rulebook rule: IF(inputIndex < someValue) THEN modify actionIndex
type Rule struct {
	InputIndex       int     // which input feature to check (0-9)
	LessThan         bool    // true for <, false for >=
	SomeValue        float64 // threshold value to compare against
	ActionIndex      int     // which action to modify (0-3)
	RelativeModifier float64 // modifier to add to action probability (-1 to 1)
}

// Rulebook is a collection of 32 rules
type Rulebook struct {
	Rules [32]Rule
}

type Agent struct {
	Lander
	thrusting  bool
	landed     bool
	crashed    bool
	killed     bool // true if removed for going off-camera (score = 0)
	policy     Policy
	greenCoins int    // count of green coins collected
	redCoins   int    // count of red coins collected
	isChampion bool   // true if this agent is from hall of fame
	agentType  string // "nn" or "rulebook"
}

// Coin represents a collectible in the world
type Coin struct {
	x, y      float64
	isGreen   bool
	collected bool // true if collected in current generation
}
