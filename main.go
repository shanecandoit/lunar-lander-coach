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

	"lunar-lander-coach/nn"
)

const (
	screenWidth     = 640
	screenHeight    = 480
	numAgents       = 100
	episodeMaxSteps = 400
)

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

	// draw spawn point marker (S)
	spawnMarker := ebiten.NewImage(16, 16)
	spawnMarker.Fill(color.RGBA{255, 255, 0, 180}) // Yellow translucent
	sOp := &ebiten.DrawImageOptions{}
	sOp.GeoM.Translate(g.spawnPoint.x-8, g.spawnPoint.y-8)
	screen.DrawImage(spawnMarker, sOp)
	ebitenutil.DebugPrintAt(screen, "S", int(g.spawnPoint.x)-3, int(g.spawnPoint.y)-7)

	// draw coins
	for _, c := range g.coins {
		coinImg := ebiten.NewImage(8, 8)
		alpha := uint8(255)
		if c.collected {
			alpha = 80 // translucent when collected
		}
		if c.isGreen {
			coinImg.Fill(color.RGBA{50, 220, 50, alpha})
		} else {
			coinImg.Fill(color.RGBA{220, 50, 50, alpha})
		}
		cOp := &ebiten.DrawImageOptions{}
		cOp.GeoM.Translate(c.x-4, c.y-4)
		screen.DrawImage(coinImg, cOp)
	}

	// draw agents
	landedCount := 0
	crashedCount := 0
	killedCount := 0
	totalGreenCoins := 0
	totalRedCoins := 0
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
		totalGreenCoins += a.greenCoins
		totalRedCoins += a.redCoins

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

		// color by state and agent type
		var cm ebiten.ColorM
		if a.landed {
			cm.Scale(0.6, 1.0, 0.6, 1)
		} else if a.crashed {
			cm.Scale(1.0, 0.5, 0.5, 1)
		} else if a.agentType == "rulebook" {
			// Orange color for rulebook agents
			cm.Scale(1.0, 0.65, 0.2, 1)
		} else {
			// Default white for NN agents
			cm.Scale(1, 1, 1, 1)
		}

		w, h := g.bodyImg.Size()
		drawOp := &ebiten.DrawImageOptions{}
		drawOp.GeoM.Translate(-float64(w)/2, -float64(h)/2)
		drawOp.GeoM.Rotate(a.angle)
		drawOp.GeoM.Translate(a.x, a.y)
		drawOp.ColorM = cm
		screen.DrawImage(g.bodyImg, drawOp)

		// Draw blue border around champions
		if a.isChampion {
			borderImg := ebiten.NewImage(w+4, h+4)
			borderImg.Fill(color.RGBA{50, 100, 255, 200})
			borderOp := &ebiten.DrawImageOptions{}
			borderOp.GeoM.Translate(-float64(w+4)/2, -float64(h+4)/2)
			borderOp.GeoM.Rotate(a.angle)
			borderOp.GeoM.Translate(a.x, a.y)
			screen.DrawImage(borderImg, borderOp)
			// Redraw the body on top with appropriate color
			screen.DrawImage(g.bodyImg, drawOp)
		}
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
			// Update running total of landed agents
			g.totalLanded += landedCount

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
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Gen:%d  Agents:%d  Landed:%d (%d)  Crashed:%d(%.0f%%)  Killed:%d(%.0f%%)  BestScore:%.2f  P: pause  Middle-click: move spawn", g.generation, len(g.agents), landedCount, g.totalLanded, crashedCount, crashedRatio*100.0, killedCount, killedRatio*100.0, best), 8, 8)

	// Calculate total landed percentage
	landedPercent := 0.0
	if g.totalAgents > 0 {
		landedPercent = float64(g.totalLanded) / float64(g.totalAgents) * 100.0
	}
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Total Landed: %d / %d (%.2f%%)  Coins: +%d  -%d  (L-click: green, R-click: red)", g.totalLanded, g.totalAgents, landedPercent, totalGreenCoins, totalRedCoins), 8, 460)

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

	// Weights JSON for landed agents only
	best := 0.0
	for _, v := range scores {
		if v > best {
			best = v
		}
	}

	type savedAgent struct {
		Index      int             `json:"index"`
		Nets       [4]*nn.NNModule `json:"nets"`
		Score      float64         `json:"score"`
		Landed     bool            `json:"landed"`
		Crashed    bool            `json:"crashed"`
		Killed     bool            `json:"killed"`
		GreenCoins int             `json:"green_coins"`
		RedCoins   int             `json:"red_coins"`
	}

	// Collect only landed agents
	landedAgents := make([]savedAgent, 0)
	for i, a := range g.agents {
		if a.landed {
			sc := 0.0
			if i < len(scores) {
				sc = scores[i]
			}
			landedAgents = append(landedAgents, savedAgent{
				Index:      i,
				Nets:       aPolicyNets(a.policy),
				Score:      sc,
				Landed:     a.landed,
				Crashed:    a.crashed,
				Killed:     a.killed,
				GreenCoins: a.greenCoins,
				RedCoins:   a.redCoins,
			})
		}
	}

	// Only save if we have landed agents
	if len(landedAgents) > 0 {
		scoreStr := fmt.Sprintf("%.3f", best)
		weightsPath := filepath.Join("data", fmt.Sprintf("landed_weights_%s_%s.json", ts, scoreStr))

		payload := struct {
			Timestamp    string       `json:"timestamp"`
			Generation   int          `json:"generation"`
			BestScore    float64      `json:"best_score"`
			LandedCount  int          `json:"landed_count"`
			LandedAgents []savedAgent `json:"landed_agents"`
		}{
			Timestamp:    ts,
			Generation:   g.generation,
			BestScore:    best,
			LandedCount:  len(landedAgents),
			LandedAgents: landedAgents,
		}
		jb, err := json.MarshalIndent(payload, "", "  ")
		if err != nil {
			return err
		}
		if err := os.WriteFile(weightsPath, jb, 0o644); err != nil {
			return err
		}
	}

	return nil
}

// evolvePopulation creates a new generation from the current agents.
// It preserves ~10% elites (exact clones) and fills the rest via crossover+mutation.
// Also maintains a hall of fame of top 10 champions across all generations.
// Handles both NN and Rulebook agents separately.
// mutationRate: per-parameter mutation probability.
// mutationScale: mutation amplitude.
func (g *Game) evolvePopulation(mutationRate, mutationScale float32) {
	// Separate NN and Rulebook agents
	var nnAgents, rulebookAgents []*Agent
	for _, a := range g.agents {
		if a.agentType == "nn" {
			nnAgents = append(nnAgents, a)
		} else if a.agentType == "rulebook" {
			rulebookAgents = append(rulebookAgents, a)
		}
	}

	r := mrand.New(mrand.NewSource(time.Now().UnixNano()))
	newAgents := make([]*Agent, 0, len(g.agents))

	// Evolve NN agents
	if len(nnAgents) > 0 {
		newNNAgents := evolveNNAgents(g, nnAgents, r, mutationRate, mutationScale)
		newAgents = append(newAgents, newNNAgents...)
	}

	// Evolve Rulebook agents
	if len(rulebookAgents) > 0 {
		newRulebookAgents := evolveRulebookAgents(g, rulebookAgents, r, mutationRate, mutationScale)
		newAgents = append(newAgents, newRulebookAgents...)
	}

	// replace population
	g.agents = newAgents
	// reset episode
	g.step = 0
	g.summaryShown = false
	g.summaryLines = nil
	// mark as not saved for new generation
	g.saved = false
	// reset all coins for new generation
	for i := range g.coins {
		g.coins[i].collected = false
	}
	// advance generation counter
	g.generation++
	// update total agents counter
	g.totalAgents += len(newAgents)
}

// evolveNNAgents handles evolution for NN agents
func evolveNNAgents(g *Game, nnAgents []*Agent, r *mrand.Rand, mutationRate, mutationScale float32) []*Agent {
	n := len(nnAgents)
	scores := make([]float64, n)
	for i, a := range nnAgents {
		scores[i] = agentScore(a, g.env)
	}

	type entry struct {
		idx    int
		score  float64
		policy *NNPolicy
	}
	es := make([]entry, 0, n)
	for i, s := range scores {
		var pol *NNPolicy
		if np, ok := nnAgents[i].policy.(*NNPolicy); ok {
			pol = np
		}
		es = append(es, entry{i, s, pol})
	}
	sort.Slice(es, func(i, j int) bool { return es[i].score > es[j].score })

	// Update hall of fame with best performers from this generation
	for i := 0; i < 5 && i < len(es); i++ {
		if es[i].policy != nil {
			// Deep clone for hall of fame
			var nets [4]*nn.NNModule
			for j := 0; j < 4; j++ {
				if es[i].policy.Nets[j] != nil {
					nets[j] = cloneNN(es[i].policy.Nets[j])
				}
			}
			champion := &NNPolicy{Nets: nets}
			g.hallOfFame = append(g.hallOfFame, champion)
		}
	}
	// Keep only top 10 in hall of fame
	if len(g.hallOfFame) > 10 {
		g.hallOfFame = g.hallOfFame[len(g.hallOfFame)-10:]
	}

	// determine elite count as ~10% of population
	eliteCount := n / 10
	if eliteCount < 1 {
		eliteCount = 1
	}

	// collect elites (exact clones, no mutation)
	elites := make([]*NNPolicy, 0, eliteCount)
	for i := 0; i < eliteCount; i++ {
		if es[i].policy != nil {
			var nets [4]*nn.NNModule
			for j := 0; j < 4; j++ {
				if es[i].policy.Nets[j] != nil {
					nets[j] = cloneNN(es[i].policy.Nets[j])
				}
			}
			elites = append(elites, &NNPolicy{Nets: nets})
		}
	}

	// tournament selection from top half
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
			if np, ok := nnAgents[bestIdx].policy.(*NNPolicy); ok {
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

	// build new NN population
	newAgents := make([]*Agent, 0, n)

	// First, add hall of fame champions (guaranteed to compete, never mutated)
	for _, champ := range g.hallOfFame {
		if len(newAgents) >= n {
			break
		}
		var nets [4]*nn.NNModule
		for j := 0; j < 4; j++ {
			if champ.Nets[j] != nil {
				nets[j] = cloneNN(champ.Nets[j])
			}
		}
		newAgents = append(newAgents, &Agent{Lander: g.spawnPoint, policy: &NNPolicy{Nets: nets}, isChampion: true, agentType: "nn"})
	}

	// Then add current generation elites (exact clones, no mutation)
	for i := 0; i < len(elites) && len(newAgents) < n; i++ {
		newAgents = append(newAgents, &Agent{Lander: g.spawnPoint, policy: elites[i], agentType: "nn"})
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
		newAgents = append(newAgents, &Agent{Lander: g.spawnPoint, policy: &NNPolicy{Nets: childNets}, agentType: "nn"})
	}

	return newAgents
}

// evolveRulebookAgents handles evolution for Rulebook agents
func evolveRulebookAgents(g *Game, rulebookAgents []*Agent, r *mrand.Rand, mutationRate, mutationScale float32) []*Agent {
	n := len(rulebookAgents)
	scores := make([]float64, n)
	for i, a := range rulebookAgents {
		scores[i] = agentScore(a, g.env)
	}

	type entry struct {
		idx    int
		score  float64
		policy *RulebookPolicy
	}
	es := make([]entry, 0, n)
	for i, s := range scores {
		var pol *RulebookPolicy
		if rp, ok := rulebookAgents[i].policy.(*RulebookPolicy); ok {
			pol = rp
		}
		es = append(es, entry{i, s, pol})
	}
	sort.Slice(es, func(i, j int) bool { return es[i].score > es[j].score })

	// Update rulebook hall of fame with best performers from this generation
	for i := 0; i < 5 && i < len(es); i++ {
		if es[i].policy != nil {
			rb := CloneRulebook(es[i].policy.Rulebook)
			champion := &RulebookPolicy{Rulebook: rb}
			g.rulebookHallOfFame = append(g.rulebookHallOfFame, champion)
		}
	}
	// Keep only top 10 in hall of fame
	if len(g.rulebookHallOfFame) > 10 {
		g.rulebookHallOfFame = g.rulebookHallOfFame[len(g.rulebookHallOfFame)-10:]
	}

	// determine elite count as ~10% of population
	eliteCount := n / 10
	if eliteCount < 1 {
		eliteCount = 1
	}

	// collect elites (exact clones, no mutation)
	elites := make([]*RulebookPolicy, 0, eliteCount)
	for i := 0; i < eliteCount; i++ {
		if es[i].policy != nil {
			rb := CloneRulebook(es[i].policy.Rulebook)
			elites = append(elites, &RulebookPolicy{Rulebook: rb})
		}
	}

	// tournament selection from top half
	topK := n / 2
	if topK < 2 {
		topK = 2
	}
	tournamentSize := 3
	pickParent := func() *RulebookPolicy {
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
			if rp, ok := rulebookAgents[bestIdx].policy.(*RulebookPolicy); ok {
				rb := CloneRulebook(rp.Rulebook)
				return &RulebookPolicy{Rulebook: rb}
			}
		}
		// fallback: random new rulebook
		return &RulebookPolicy{Rulebook: NewRandomRulebook()}
	}

	// build new rulebook population
	newAgents := make([]*Agent, 0, n)

	// First, add hall of fame champions (guaranteed to compete, never mutated)
	for _, champ := range g.rulebookHallOfFame {
		if len(newAgents) >= n {
			break
		}
		rb := CloneRulebook(champ.Rulebook)
		newAgents = append(newAgents, &Agent{Lander: g.spawnPoint, policy: &RulebookPolicy{Rulebook: rb}, isChampion: true, agentType: "rulebook"})
	}

	// Then add current generation elites (exact clones, no mutation)
	for i := 0; i < len(elites) && len(newAgents) < n; i++ {
		newAgents = append(newAgents, &Agent{Lander: g.spawnPoint, policy: elites[i], agentType: "rulebook"})
	}

	// fill rest with children
	for len(newAgents) < n {
		p1 := pickParent()
		p2 := pickParent()
		childRulebook := CrossoverRulebook(p1.Rulebook, p2.Rulebook)
		MutateRulebook(&childRulebook, mutationRate, mutationScale)
		newAgents = append(newAgents, &Agent{Lander: g.spawnPoint, policy: &RulebookPolicy{Rulebook: childRulebook}, agentType: "rulebook"})
	}

	return newAgents
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
