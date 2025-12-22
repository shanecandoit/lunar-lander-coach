package main

import (
	mrand "math/rand"
)

// NewRandomRulebook creates a rulebook with 32 random rules
func NewRandomRulebook() Rulebook {
	var rb Rulebook
	for i := 0; i < 32; i++ {
		rb.Rules[i] = Rule{
			InputIndex:       mrand.Intn(10),        // 0-9 for the 10 input features
			LessThan:         mrand.Float64() < 0.5, // 50% chance of < vs >=
			SomeValue:        mrand.Float64()*2 - 1, // random value in [-1, 1]
			ActionIndex:      mrand.Intn(4),         // 0-3 for the 4 actions
			RelativeModifier: mrand.Float64()*2 - 1, // random modifier in [-1, 1]
		}
	}
	return rb
}

// CloneRulebook creates a deep copy of a rulebook
func CloneRulebook(src Rulebook) Rulebook {
	var dst Rulebook
	dst.Rules = src.Rules
	return dst
}

// CrossoverRulebook creates a child rulebook from two parents using uniform crossover
func CrossoverRulebook(a, b Rulebook) Rulebook {
	var child Rulebook
	for i := 0; i < 32; i++ {
		if mrand.Float64() < 0.5 {
			child.Rules[i] = a.Rules[i]
		} else {
			child.Rules[i] = b.Rules[i]
		}
	}
	return child
}

// MutateRulebook applies mutations to a rulebook
// mutationRate: probability of mutating each rule
// mutationScale: how much to change values (0-1)
func MutateRulebook(rb *Rulebook, mutationRate, mutationScale float32) {
	for i := 0; i < 32; i++ {
		if mrand.Float64() < float64(mutationRate) {
			// Decide what aspect of the rule to mutate
			switch mrand.Intn(5) {
			case 0: // mutate input index
				rb.Rules[i].InputIndex = mrand.Intn(10)
			case 1: // flip lessThan
				rb.Rules[i].LessThan = !rb.Rules[i].LessThan
			case 2: // mutate threshold value
				delta := (mrand.Float64()*2 - 1) * float64(mutationScale)
				rb.Rules[i].SomeValue += delta
				// Clamp to reasonable range
				if rb.Rules[i].SomeValue > 2 {
					rb.Rules[i].SomeValue = 2
				}
				if rb.Rules[i].SomeValue < -2 {
					rb.Rules[i].SomeValue = -2
				}
			case 3: // mutate action index
				rb.Rules[i].ActionIndex = mrand.Intn(4)
			case 4: // mutate relative modifier
				delta := (mrand.Float64()*2 - 1) * float64(mutationScale)
				rb.Rules[i].RelativeModifier += delta
				// Clamp to [-1, 1]
				if rb.Rules[i].RelativeModifier > 1 {
					rb.Rules[i].RelativeModifier = 1
				}
				if rb.Rules[i].RelativeModifier < -1 {
					rb.Rules[i].RelativeModifier = -1
				}
			}
		}
	}
}
