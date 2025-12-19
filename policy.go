package main

import (
	"math"

	"lunar-lander-coach/nn"
)

// NNPolicy maps an 8-element state to a 4-way action distribution.
type NNPolicy struct {
	Nets [4]*nn.NNModule // 4 heads producing logits for actions: nop, main, right, left
}

// Decide maps the 10-element state into a 4-way action distribution and returns
// a selected discrete action mapped to (thrust, rotate).
// Inputs: dx, dy, vx, vy, angle, sin(angle), cos(angle), speed, distToGreenCoin, distToRedCoin
func (p *NNPolicy) Decide(l *Lander, env Environment, distGreen, distRed float64) (bool, float64) {
	groundY := float64(screenHeight - env.GroundHeight)
	dx := (env.PadX - l.x) / float64(screenWidth)
	dy := (groundY - l.y) / float64(screenHeight)
	vx := l.vx / 10.0
	vy := l.vy / 10.0
	ang := l.angle / math.Pi
	sag := math.Sin(l.angle)
	cag := math.Cos(l.angle)
	speed := math.Hypot(l.vx, l.vy) / 10.0

	features := make([]float32, 10)
	features[0] = float32(dx)
	features[1] = float32(dy)
	features[2] = float32(vx)
	features[3] = float32(vy)
	features[4] = float32(ang)
	features[5] = float32(sag)
	features[6] = float32(cag)
	features[7] = float32(speed)
	features[8] = float32(distGreen / float64(screenWidth))
	features[9] = float32(distRed / float64(screenWidth))

	logits := make([]float64, 4)
	maxLogit := math.Inf(-1)
	for i := 0; i < 4; i++ {
		if p.Nets[i] == nil {
			logits[i] = 0
		} else {
			logits[i] = float64(p.Nets[i].Forward(features))
		}
		if logits[i] > maxLogit {
			maxLogit = logits[i]
		}
	}

	sum := 0.0
	probs := make([]float64, 4)
	for i := 0; i < 4; i++ {
		e := math.Exp(logits[i] - maxLogit)
		probs[i] = e
		sum += e
	}
	for i := 0; i < 4; i++ {
		probs[i] /= sum
	}

	action := 0
	best := probs[0]
	for i := 1; i < 4; i++ {
		if probs[i] > best {
			best = probs[i]
			action = i
		}
	}

	var thrust bool
	var rotate float64
	switch action {
	case 0:
		thrust = false
		rotate = 0
	case 1:
		thrust = true
		rotate = 0
	case 2:
		thrust = true
		rotate = 0.08
	case 3:
		thrust = true
		rotate = -0.08
	}
	return thrust, rotate
}
