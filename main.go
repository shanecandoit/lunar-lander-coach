package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"image/color"
	"log"
	"math"
	mrand "math/rand"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"github.com/hajimehoshi/ebiten/v2/inpututil"

	"lunar-lander-coach/nn"
)

const (
	screenWidth     = 640
	screenHeight    = 480
	numAgents       = 100
	episodeMaxSteps = 200
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
	Decide(l *Lander, env Environment) (thrust bool, rotate float64)
}

// RandomPolicy issues random small rotations and intermittent thrust.
type RandomPolicy struct {
	rng *mrand.Rand
}

func (p *RandomPolicy) Decide(l *Lander, env Environment) (bool, float64) {
	// small random rotation [-0.06, 0.06]
	rotate := (p.rng.Float64() - 0.5) * 0.12
	// modest chance to thrust each frame
	thrust := p.rng.Float64() < 0.08
	return thrust, rotate
}

// NNPolicy uses two small NNModules to decide thrust and rotation.
type NNPolicy struct {
	Nets [4]*nn.NNModule // 4 heads producing logits for actions: nop, main, right, left
}

// Decide maps the 8-element state into a 4-way action distribution and returns
// a selected discrete action mapped to (thrust, rotate).
func (p *NNPolicy) Decide(l *Lander, env Environment) (bool, float64) {
	// compute an 8-element state feature vector
	groundY := float64(screenHeight - env.GroundHeight)
	dx := (env.PadX - l.x) / float64(screenWidth) // horizontal error
	dy := (groundY - l.y) / float64(screenHeight) // vertical error (positive when above ground)
	vx := l.vx / 10.0                             // normalize velocities
	vy := l.vy / 10.0
	ang := l.angle / math.Pi // normalized angle
	sag := math.Sin(l.angle)
	cag := math.Cos(l.angle)
	speed := math.Hypot(l.vx, l.vy) / 10.0

	features := make([]float32, 8)
	features[0] = float32(dx)
	features[1] = float32(dy)
	features[2] = float32(vx)
	features[3] = float32(vy)
	features[4] = float32(ang)
	features[5] = float32(sag)
	features[6] = float32(cag)
	features[7] = float32(speed)

	// produce logits from 4 heads
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

	// softmax (stable)
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

	// pick action deterministically (argmax). Could sample for stochasticity.
	action := 0
	best := probs[0]
	for i := 1; i < 4; i++ {
		if probs[i] > best {
			best = probs[i]
			action = i
		}
	}

	// map action to thrust/rotate
	// 0: nop, 1: main thrust, 2: right thrust, 3: left thrust
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

type Agent struct {
	Lander
	thrusting bool
	landed    bool
	crashed   bool
	killed    bool // true if removed for going off-camera (score = 0)
	policy    Policy
}

type Game struct {
	env      Environment
	agents   []*Agent
	paused   bool
	bodyImg  *ebiten.Image
	flameImg *ebiten.Image
	// summary display
	summaryShown bool
	summaryLines []string
	step         int
	saved        bool
	generation   int
}

func NewGame(n int) *Game {
	seed := int64(42)
	mrand.New(mrand.NewSource(seed))
	body := ebiten.NewImage(12, 18)
	body.Fill(color.RGBA{220, 220, 220, 255})

	flame := ebiten.NewImage(6, 10)
	flame.Fill(color.RGBA{255, 140, 0, 255})

	env := Environment{
		Gravity:      0.06,
		GroundHeight: 24,
		PadX:         screenWidth / 2,
		PadWidth:     120,
	}

	g := &Game{
		env:        env,
		bodyImg:    body,
		flameImg:   flame,
		step:       0,
		generation: 1,
	}

	// spawn agents with varied starting positions and random policies
	seedBase := time.Now().UnixNano()
	for i := 0; i < n; i++ {
		rng := mrand.New(mrand.NewSource(seedBase + int64(i)*7919))
		// create an agent with a pair of small random neural modules
		pos := Lander{x: rng.Float64()*float64(screenWidth-40) + 20, y: rng.Float64()*100 + 20}
		a := &Agent{
			Lander: pos,
			policy: &NNPolicy{Nets: [4]*nn.NNModule{nn.NewRandomNN(), nn.NewRandomNN(), nn.NewRandomNN(), nn.NewRandomNN()}},
		}
		g.agents = append(g.agents, a)
	}

	return g
}

func (g *Game) Update() error {
	if inpututil.IsKeyJustPressed(ebiten.KeyP) {
		g.paused = !g.paused
	}
	if g.paused {
		return nil
	}

	// advance episode step and end episode if we reached max steps
	g.step++

	for _, a := range g.agents {
		if a.landed || a.crashed {
			continue
		}

		thrust, rotate := a.policy.Decide(&a.Lander, g.env)
		a.Lander.angle += rotate
		a.thrusting = thrust
		if thrust {
			t := 0.13
			a.vx += math.Sin(a.angle) * t
			a.vy -= math.Cos(a.angle) * t
		}

		// gravity
		a.vy += g.env.Gravity

		// integrate
		a.x += a.vx
		a.y += a.vy

		// off-camera detection: if the lander leaves a reasonable area, kill it
		if a.x < -50 || a.x > float64(screenWidth)+50 || a.y < -50 || a.y > float64(screenHeight)+50 {
			a.killed = true
			a.crashed = true
			a.vx = 0
			a.vy = 0
			continue
		}

		// bounds (keep lander within visible area)
		if a.x < 0 {
			a.x = 0
			a.vx = 0
		}
		if a.x > screenWidth {
			a.x = screenWidth
			a.vx = 0
		}

		// collision with ground
		groundY := float64(screenHeight - g.env.GroundHeight)
		if a.y >= groundY {
			if math.Abs(a.vy) < 2.5 && math.Abs(a.vx) < 2.0 && math.Abs(normalizeAngle(a.angle)) < 0.5 {
				a.landed = true
				a.y = groundY
				a.vx = 0
				a.vy = 0
				a.angle = 0
			} else {
				a.crashed = true
				a.y = groundY
				a.vx = 0
				a.vy = 0
			}
		}
	}
	return nil
}

func normalizeAngle(a float64) float64 {
	for a > math.Pi {
		a -= 2 * math.Pi
	}
	for a < -math.Pi {
		a += 2 * math.Pi
	}
	return a
}

// agentScore computes a fitness score in [0,100].
// It returns 100.0 when the lander is exactly at the pad center with
// zero velocities and zero angle. Scores decay quickly as errors grow.
func agentScore(a *Agent, env Environment) float64 {
	// killed agents receive 0 score
	if a.killed {
		return 0.0
	}
	// position error (distance from pad center)
	posErr := math.Abs(a.x - env.PadX)
	// velocity error (magnitude)
	velErr := math.Hypot(a.vx, a.vy)
	// angle error (radians, normalized)
	angErr := math.Abs(normalizeAngle(a.angle))

	// scale parameters (tunable)
	posSigma := env.PadWidth / 6.0 // a few pixels gives sharp drop-off
	if posSigma <= 0 {
		posSigma = 10.0
	}
	velSigma := 0.8
	angSigma := 0.25

	// Gaussian-like score (1.0 at zero error)
	p := math.Exp(-0.5 * ((posErr/posSigma)*(posErr/posSigma) + (velErr/velSigma)*(velErr/velSigma) + (angErr/angSigma)*(angErr/angSigma)))

	// if not landed, strongly downweight score (we only want landed ships to be competitive)
	if !a.landed {
		p *= 0.01
	}

	return 100.0 * p
}

// computeStats returns mean, max, min, std, var for the slice.
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

// buildSummaryLines creates printable lines containing stats and top/bottom 10.
func buildSummaryLines(scores []float64, agents []*Agent) []string {
	lines := []string{}
	mean, maxv, minv, std, varv := computeStats(scores)
	lines = append(lines, fmt.Sprintf("Summary: mean=%.3f  max=%.3f  min=%.3f  std=%.3f  var=%.3f", mean, maxv, minv, std, varv))

	// prepare index+score pairs for sorting
	type entry struct {
		idx int
		val float64
	}
	es := make([]entry, 0, len(scores))
	for i, v := range scores {
		es = append(es, entry{idx: i, val: v})
	}

	sort.Slice(es, func(i, j int) bool { return es[i].val > es[j].val })
	lines = append(lines, "\nBest 10:")
	for i := 0; i < 10 && i < len(es); i++ {
		e := es[i]
		state := ""
		if agents[e.idx].landed {
			state = "landed"
		} else if agents[e.idx].crashed {
			state = "crashed"
		}
		lines = append(lines, fmt.Sprintf("%2d: score=%.3f  %s", e.idx, e.val, state))
	}

	// worst 10
	sort.Slice(es, func(i, j int) bool { return es[i].val < es[j].val })
	lines = append(lines, "\nWorst 10:")
	for i := 0; i < 10 && i < len(es); i++ {
		e := es[i]
		state := ""
		if agents[e.idx].landed {
			state = "landed"
		} else if agents[e.idx].crashed {
			state = "crashed"
		}
		lines = append(lines, fmt.Sprintf("%2d: score=%.3f  %s", e.idx, e.val, state))
	}
	return lines
}

func (g *Game) Draw(screen *ebiten.Image) {
	screen.Fill(color.RGBA{12, 12, 32, 255})

	// ground
	ground := ebiten.NewImage(screenWidth, g.env.GroundHeight)
	ground.Fill(color.RGBA{80, 180, 90, 255})
	op := &ebiten.DrawImageOptions{}
	op.GeoM.Translate(0, float64(screenHeight-g.env.GroundHeight))
	screen.DrawImage(ground, op)

	// pad
	pad := ebiten.NewImage(int(g.env.PadWidth), 8)
	pad.Fill(color.RGBA{200, 200, 80, 255})
	pOp := &ebiten.DrawImageOptions{}
	pOp.GeoM.Translate(g.env.PadX-g.env.PadWidth/2, float64(screenHeight-g.env.GroundHeight-8))
	screen.DrawImage(pad, pOp)

	// draw agents
	landedCount := 0
	crashedCount := 0
	killedCount := 0
	for _, a := range g.agents {
		if a.landed {
			landedCount++
		}
		if a.crashed {
			crashedCount++
		}
		if a.killed {
			killedCount++
		}

		if a.thrusting && !a.landed && !a.crashed {
			w, h := g.flameImg.Size()
			fOp := &ebiten.DrawImageOptions{}
			fOp.GeoM.Translate(-float64(w)/2, -float64(h)/2)
			fOp.GeoM.Rotate(a.angle)
			// position behind the lander
			offset := 12.0
			dx := math.Sin(a.angle)
			dy := -math.Cos(a.angle)
			fx := a.x - dx*offset
			fy := a.y - dy*offset
			fOp.GeoM.Translate(fx, fy)
			screen.DrawImage(g.flameImg, fOp)
		}

		// color by state
		var cm ebiten.ColorM
		if a.landed {
			cm.Scale(0.6, 1.0, 0.6, 1)
		} else if a.crashed {
			cm.Scale(1.0, 0.5, 0.5, 1)
		} else {
			cm.Scale(1, 1, 1, 1)
		}

		w, h := g.bodyImg.Size()
		drawOp := &ebiten.DrawImageOptions{}
		drawOp.GeoM.Translate(-float64(w)/2, -float64(h)/2)
		drawOp.GeoM.Rotate(a.angle)
		drawOp.GeoM.Translate(a.x, a.y)
		drawOp.ColorM = cm
		screen.DrawImage(g.bodyImg, drawOp)
	}

	// compute best score for display
	best := 0.0
	allFinished := true
	scores := make([]float64, 0, len(g.agents))
	for _, a := range g.agents {
		s := agentScore(a, g.env)
		scores = append(scores, s)
		if s > best {
			best = s
		}
		if !a.landed && !a.crashed {
			allFinished = false
		}
	}

	// if we've reached the max episode length, mark unfinished agents as crashed
	if g.step >= episodeMaxSteps {
		for _, a := range g.agents {
			if !a.landed && !a.crashed {
				a.crashed = true
			}
		}
		// recompute scores and allFinished
		best = 0.0
		allFinished = true
		scores = scores[:0]
		for _, a := range g.agents {
			s := agentScore(a, g.env)
			scores = append(scores, s)
			if s > best {
				best = s
			}
			if !a.landed && !a.crashed {
				allFinished = false
			}
		}
		// save generation data (CSV + weights) once
		if !g.saved {
			if err := g.saveGenerationData(scores); err != nil {
				log.Printf("error saving generation data: %v", err)
			} else {
				g.saved = true
				// evolve population and start next generation
				// preserve ~10% elites; use mutation rate 8% and smaller mutation scale 0.05
				g.evolvePopulation(0.08, 0.05)
			}
		}
	}

	// compute ratios
	total := float64(len(g.agents))
	crashedRatio := 0.0
	killedRatio := 0.0
	if total > 0 {
		crashedRatio = float64(crashedCount) / total
		killedRatio = float64(killedCount) / total
	}
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Gen:%d  Agents:%d  Landed:%d  Crashed:%d(%.0f%%)  Killed:%d(%.0f%%)  BestScore:%.2f  P: pause", g.generation, len(g.agents), landedCount, crashedCount, crashedRatio*100.0, killedCount, killedRatio*100.0, best), 8, 8)

	// when all agents have finished, prepare and show a summary table
	if allFinished && !g.summaryShown {
		g.summaryLines = buildSummaryLines(scores, g.agents)
		g.summaryShown = true
	}

	if g.summaryShown {
		sx := 8
		sy := 28
		for i, line := range g.summaryLines {
			ebitenutil.DebugPrintAt(screen, line, sx, sy+i*14)
		}
	}
}

func (g *Game) Layout(outsideWidth, outsideHeight int) (int, int) {
	return screenWidth, screenHeight
}

// saveGenerationData writes a CSV with per-agent state and a JSON file
// containing all agents' network weights. Filenames are timestamped.
func (g *Game) saveGenerationData(scores []float64) error {
	if err := os.MkdirAll("data", 0o755); err != nil {
		return err
	}
	ts := time.Now().Format("2006-01-02_15-04-05")

	// CSV
	csvPath := filepath.Join("data", ts+".csv")
	cf, err := os.Create(csvPath)
	if err != nil {
		return err
	}
	defer cf.Close()
	w := csv.NewWriter(cf)
	defer w.Flush()
	// header
	if err := w.Write([]string{"idx", "x", "y", "vx", "vy", "angle", "landed", "crashed", "killed", "score"}); err != nil {
		return err
	}
	for i, a := range g.agents {
		s := "0"
		if i < len(scores) {
			s = strconv.FormatFloat(scores[i], 'f', 6, 64)
		}
		row := []string{
			strconv.Itoa(i),
			strconv.FormatFloat(a.x, 'f', 6, 64),
			strconv.FormatFloat(a.y, 'f', 6, 64),
			strconv.FormatFloat(a.vx, 'f', 6, 64),
			strconv.FormatFloat(a.vy, 'f', 6, 64),
			strconv.FormatFloat(a.angle, 'f', 6, 64),
			strconv.FormatBool(a.landed),
			strconv.FormatBool(a.crashed),
			strconv.FormatBool(a.killed),
			s,
		}
		if err := w.Write(row); err != nil {
			return err
		}
	}

	// Weights JSON
	best := 0.0
	for _, v := range scores {
		if v > best {
			best = v
		}
	}
	scoreStr := fmt.Sprintf("%.3f", best)
	weightsPath := filepath.Join("data", fmt.Sprintf("weights_%s_%s.json", ts, scoreStr))

	type savedAgent struct {
		Index   int             `json:"index"`
		Nets    [4]*nn.NNModule `json:"nets"`
		Score   float64         `json:"score"`
		Landed  bool            `json:"landed"`
		Crashed bool            `json:"crashed"`
		Killed  bool            `json:"killed"`
	}
	sa := make([]savedAgent, 0, len(g.agents))
	for i, a := range g.agents {
		sc := 0.0
		if i < len(scores) {
			sc = scores[i]
		}
		sa = append(sa, savedAgent{Index: i, Nets: aPolicyNets(a.policy), Score: sc, Landed: a.landed, Crashed: a.crashed, Killed: a.killed})
	}

	payload := struct {
		Timestamp string       `json:"timestamp"`
		BestScore float64      `json:"best_score"`
		Agents    []savedAgent `json:"agents"`
	}{
		Timestamp: ts,
		BestScore: best,
		Agents:    sa,
	}
	jb, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(weightsPath, jb, 0o644); err != nil {
		return err
	}
	return nil
}

// evolvePopulation creates a new generation from the current agents.
// It preserves ~10% elites (exact clones) and fills the rest via crossover+mutation.
// mutationRate: per-parameter mutation probability.
// mutationScale: mutation amplitude.
func (g *Game) evolvePopulation(mutationRate, mutationScale float32) {
	// compute scores and indices
	n := len(g.agents)
	scores := make([]float64, n)
	for i, a := range g.agents {
		scores[i] = agentScore(a, g.env)
	}
	type entry struct {
		idx   int
		score float64
	}
	es := make([]entry, 0, n)
	for i, s := range scores {
		es = append(es, entry{i, s})
	}
	sort.Slice(es, func(i, j int) bool { return es[i].score > es[j].score })

	// determine elite count as ~10% of population
	eliteCount := n / 10
	if eliteCount < 1 {
		eliteCount = 1
	}

	// collect elites (exact clones, no mutation)
	elites := make([]*NNPolicy, 0, eliteCount)
	for i := 0; i < eliteCount; i++ {
		idx := es[i].idx
		if np, ok := g.agents[idx].policy.(*NNPolicy); ok {
			var nets [4]*nn.NNModule
			for j := 0; j < 4; j++ {
				if np.Nets[j] != nil {
					nets[j] = cloneNN(np.Nets[j])
				}
			}
			elites = append(elites, &NNPolicy{Nets: nets})
		}
	}

	r := mrand.New(mrand.NewSource(time.Now().UnixNano()))

	// tournament selection from top half (to keep selection pressure but preserve diversity)
	topK := n / 2
	if topK < 2 {
		topK = 2
	}
	tournamentSize := 3
	pickParent := func() *NNPolicy {
		// pick tournamentSize random candidates from topK and return best
		bestIdx := -1
		bestScore := math.Inf(-1)
		for t := 0; t < tournamentSize; t++ {
			ri := r.Intn(topK)
			cand := es[ri]
			if cand.score > bestScore {
				bestScore = cand.score
				bestIdx = cand.idx
			}
		}
		if bestIdx >= 0 {
			if np, ok := g.agents[bestIdx].policy.(*NNPolicy); ok {
				// return a clone to avoid aliasing
				var nets [4]*nn.NNModule
				for j := 0; j < 4; j++ {
					if np.Nets[j] != nil {
						nets[j] = cloneNN(np.Nets[j])
					}
				}
				return &NNPolicy{Nets: nets}
			}
		}
		// fallback: random new nets
		return &NNPolicy{Nets: [4]*nn.NNModule{nn.NewRandomNN(), nn.NewRandomNN(), nn.NewRandomNN(), nn.NewRandomNN()}}
	}

	// build new population
	newAgents := make([]*Agent, 0, n)
	// copy elites (exact)
	for i := 0; i < len(elites); i++ {
		pos := Lander{x: r.Float64()*float64(screenWidth-40) + 20, y: r.Float64()*100 + 20}
		newAgents = append(newAgents, &Agent{Lander: pos, policy: elites[i]})
	}

	// fill rest with children
	for len(newAgents) < n {
		p1 := pickParent()
		p2 := pickParent()
		var childNets [4]*nn.NNModule
		for j := 0; j < 4; j++ {
			aNet := p1.Nets[j]
			bNet := p2.Nets[j]
			if aNet == nil && bNet == nil {
				childNets[j] = nn.NewRandomNN()
			} else if aNet == nil {
				childNets[j] = cloneNN(bNet)
			} else if bNet == nil {
				childNets[j] = cloneNN(aNet)
			} else {
				childNets[j] = nn.Crossover(aNet, bNet)
			}
			// mutate child
			if childNets[j] != nil {
				childNets[j].Mutate(mutationRate, mutationScale)
			}
		}
		pos := Lander{x: r.Float64()*float64(screenWidth-40) + 20, y: r.Float64()*100 + 20}
		newAgents = append(newAgents, &Agent{Lander: pos, policy: &NNPolicy{Nets: childNets}})
	}

	// replace population
	g.agents = newAgents
	// reset episode
	g.step = 0
	g.summaryShown = false
	g.summaryLines = nil
	// mark as not saved for new generation
	g.saved = false
	// advance generation counter
	g.generation++
}

// cloneNN performs a deep copy of an NNModule.
func cloneNN(src *nn.NNModule) *nn.NNModule {
	if src == nil {
		return nil
	}
	dst := &nn.NNModule{}
	dst.Bias1 = src.Bias1
	dst.Bias2 = src.Bias2
	dst.Weights = src.Weights
	return dst
}

// aPolicyNets attempts to extract the 4 nets from a Policy if it's an NNPolicy.
func aPolicyNets(p Policy) [4]*nn.NNModule {
	var out [4]*nn.NNModule
	if np, ok := p.(*NNPolicy); ok {
		out = np.Nets
	}
	return out
}

func main() {
	ebiten.SetWindowSize(screenWidth, screenHeight)
	ebiten.SetWindowTitle("Lunar Lander — 100 Agents")
	g := NewGame(numAgents)
	if err := ebiten.RunGame(g); err != nil {
		log.Fatal(err)
	}
}
