package nn

import (
	"encoding/json"
	"math"
	mrand "math/rand"
	"os"
	"time"
)

// NNModule is a tiny two-layer network with a single scalar input,
// 8 hidden units and a single scalar output. The implementation uses
// 16 weights and two scalar biases (one per layer) as requested.
//
// Forward pass:
//
//	hidden_j = input * w1[j] + b1   for j in 0..7  (w1 = Weights[0:8])
//	out = sum_j hidden_j * w2[j] + b2            (w2 = Weights[8:16])
//	return ReLU(out)
type NNModule struct {
	Weights [16]float32 `json:"weights"`
	Bias1   float32     `json:"bias1"`
	Bias2   float32     `json:"bias2"`
}

// NewRandomNN constructs a network with random weights in [-1,1]
// and biases initialized to 0.
func NewRandomNN() *NNModule {
	// r := mrand.New(mrand.NewSource(42))
	n := &NNModule{}
	for i := 0; i < 16; i++ {
		n.Weights[i] = (mrand.Float32()*2 - 1)
	}
	n.Bias1 = 0
	n.Bias2 = 0
	return n
}

// Forward runs the tiny network on an 8-element input vector and returns
// a single scalar output (ReLU applied). The input slice must have length 8.
func (n *NNModule) Forward(input []float32) float32 {
	if len(input) != 8 {
		// if length mismatches, be robust: treat missing entries as 0 or ignore extras
		in := make([]float32, 8)
		copy(in, input)
		input = in
	}

	// first stage: compute a scalar from the 8 input weights + bias1
	var u float32
	for i := 0; i < 8; i++ {
		u += input[i] * n.Weights[i]
	}
	u += n.Bias1

	// second stage: combine u with the second 8 weights to produce output
	out := float32(0)
	for j := 0; j < 8; j++ {
		out += u * n.Weights[8+j]
	}
	out += n.Bias2

	// ReLU
	if out < 0 {
		out = 0
	}
	if math.IsNaN(float64(out)) || math.IsInf(float64(out), 0) {
		out = 0
	}
	return out
}

// ToJSON returns the JSON encoding of the network.
func (n *NNModule) ToJSON() ([]byte, error) {
	return json.MarshalIndent(n, "", "  ")
}

// Save writes the network JSON to a file path.
func (n *NNModule) Save(path string) error {
	b, err := n.ToJSON()
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

// LoadFromJSON decodes a network from JSON bytes.
func LoadFromJSON(b []byte) (*NNModule, error) {
	var n NNModule
	if err := json.Unmarshal(b, &n); err != nil {
		return nil, err
	}
	return &n, nil
}

// Load loads a network from a file path.
func Load(path string) (*NNModule, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return LoadFromJSON(b)
}

// Mutate applies random perturbations to the weights and biases.
// mutationRate is the probability each parameter is mutated.
// mutationScale controls the amplitude of uniform perturbation.
func (n *NNModule) Mutate(mutationRate, mutationScale float32) {
	r := mrand.New(mrand.NewSource(time.Now().UnixNano()))
	for i := 0; i < 16; i++ {
		if r.Float32() < mutationRate {
			// uniform perturbation in [-scale, +scale]
			n.Weights[i] += (r.Float32()*2 - 1) * mutationScale
		}
	}
	if r.Float32() < mutationRate {
		n.Bias1 += (r.Float32()*2 - 1) * mutationScale
	}
	if r.Float32() < mutationRate {
		n.Bias2 += (r.Float32()*2 - 1) * mutationScale
	}
}

// Crossover creates a child by uniform crossover between two parents.
func Crossover(a, b *NNModule) *NNModule {
	r := mrand.New(mrand.NewSource(time.Now().UnixNano()))
	child := &NNModule{}
	for i := 0; i < 16; i++ {
		if r.Float32() < 0.5 {
			child.Weights[i] = a.Weights[i]
		} else {
			child.Weights[i] = b.Weights[i]
		}
	}
	if r.Float32() < 0.5 {
		child.Bias1 = a.Bias1
	} else {
		child.Bias1 = b.Bias1
	}
	if r.Float32() < 0.5 {
		child.Bias2 = a.Bias2
	} else {
		child.Bias2 = b.Bias2
	}
	return child
}
