package main

import "math"

func normalizeAngle(a float64) float64 {
	for a > math.Pi {
		a -= 2 * math.Pi
	}
	for a < -math.Pi {
		a += 2 * math.Pi
	}
	return a
}

func agentScore(a *Agent, env Environment) float64 {
	if a.killed {
		return 0.0
	}

	// Calculate distance from ideal landing position
	posErr := math.Abs(a.x - env.PadX)

	// For vertical position, penalize being far from ground
	groundY := float64(screenHeight - env.GroundHeight)
	heightAboveGround := groundY - a.y

	// Velocity error (magnitude)
	velErr := math.Hypot(a.vx, a.vy)

	// Angle error (radians, normalized)
	angErr := math.Abs(normalizeAngle(a.angle))

	// Scale parameters (tunable)
	posSigma := env.PadWidth / 6.0 // horizontal position tolerance
	if posSigma <= 0 {
		posSigma = 10.0
	}
	heightSigma := 50.0 // vertical position tolerance
	velSigma := 0.8     // velocity tolerance
	angSigma := 0.25    // angle tolerance

	// Gaussian-like score based on proximity to ideal state
	p := math.Exp(-0.5 * ((posErr/posSigma)*(posErr/posSigma) +
		(heightAboveGround/heightSigma)*(heightAboveGround/heightSigma) +
		(velErr/velSigma)*(velErr/velSigma) +
		(angErr/angSigma)*(angErr/angSigma)))

	// Landed agents get a significant bonus
	baseScore := 100.0 * p
	if a.landed {
		baseScore += 50.0 // bonus for successful landing
	}

	// Add coin bonuses: +1 for green, -1 for red
	coinBonus := float64(a.greenCoins - a.redCoins)
	return baseScore + coinBonus
}

func computeStats(xs []float64) (mean, max, min, std, variance float64) {
	n := float64(len(xs))
	if n == 0 {
		return 0, 0, 0, 0, 0
	}
	sum := 0.0
	max = xs[0]
	min = xs[0]
	for _, v := range xs {
		sum += v
		if v > max {
			max = v
		}
		if v < min {
			min = v
		}
	}
	mean = sum / n
	varSum := 0.0
	for _, v := range xs {
		d := v - mean
		varSum += d * d
	}
	variance = varSum / n
	std = math.Sqrt(variance)
	return
}
