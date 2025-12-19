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
	posErr := math.Abs(a.x - env.PadX)
	velErr := math.Hypot(a.vx, a.vy)
	angErr := math.Abs(normalizeAngle(a.angle))

	posSigma := env.PadWidth / 6.0
	if posSigma <= 0 {
		posSigma = 10.0
	}
	velSigma := 0.8
	angSigma := 0.25

	p := math.Exp(-0.5 * ((posErr/posSigma)*(posErr/posSigma) + (velErr/velSigma)*(velErr/velSigma) + (angErr/angSigma)*(angErr/angSigma)))
	if !a.landed {
		p *= 0.01
	}
	// Add coin bonuses: +1 for green, -1 for red
	coinBonus := float64(a.greenCoins - a.redCoins)
	return 100.0*p + coinBonus
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
