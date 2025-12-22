package main

import (
	"fmt"
	"image/color"
	"math"

	"lunar-lander-coach/nn"
	mrand "math/rand"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"github.com/hajimehoshi/ebiten/v2/inpututil"
)

// Game holds world and rendered assets.
type Game struct {
	env                Environment
	agents             []*Agent
	coins              []Coin
	hallOfFame         []*NNPolicy       // top 10 NN champions across all generations, never mutated
	rulebookHallOfFame []*RulebookPolicy // top 10 rulebook champions across all generations, never mutated
	paused             bool
	spawnPoint         Lander // Starting position for all agents (S marker)
	bodyImg            *ebiten.Image
	flameImg           *ebiten.Image
	// summary display
	summaryShown bool
	summaryLines []string
	step         int
	saved        bool
	generation   int
	// running totals across all generations
	totalLanded int
	totalAgents int
	// curriculum learning parameters
	startGen      int     // generation to start curriculum (0)
	endGen        int     // generation to end curriculum (100)
	startPadWidth float64 // starting pad width (1/3 of screen)
	endPadWidth   float64 // ending pad width (1/10 of screen)
	startGravity  float64 // starting gravity (easier)
	endGravity    float64 // ending gravity (realistic)
	// UI state for sliders
	draggingSlider string  // "" or "padWidth-start", "padWidth-end", "gravity-start", "gravity-end"
	dragOffsetX    float64 // offset from slider control point to mouse
	dragOffsetY    float64
}

func NewGame(n int) *Game {
	seed := int64(42)
	mrand.New(mrand.NewSource(seed))
	body := ebiten.NewImage(12, 18)
	body.Fill(color.RGBA{220, 220, 220, 255})

	flame := ebiten.NewImage(6, 10)
	flame.Fill(color.RGBA{255, 140, 0, 255})

	// Curriculum learning setup
	startPadWidth := float64(screenWidth) / 3.0 // 1/3 of screen
	endPadWidth := float64(screenWidth) / 10.0  // 1/10 of screen
	startGravity := 0.02                        // Easy gravity
	endGravity := 0.06                          // Realistic gravity

	env := Environment{
		Gravity:      startGravity, // Will progress to endGravity
		GroundHeight: 24,
		PadX:         screenWidth / 2,
		PadWidth:     startPadWidth, // Will progress to endPadWidth
	}

	g := &Game{
		env:                env,
		bodyImg:            body,
		flameImg:           flame,
		step:               0,
		generation:         1,
		hallOfFame:         make([]*NNPolicy, 0, 10),
		rulebookHallOfFame: make([]*RulebookPolicy, 0, 10),
		totalLanded:        0,
		totalAgents:        n,
		spawnPoint:         Lander{x: float64(screenWidth / 2), y: 50}, // Top center
		// Curriculum learning parameters
		startGen:      0,
		endGen:        100,
		startPadWidth: startPadWidth,
		endPadWidth:   endPadWidth,
		startGravity:  startGravity,
		endGravity:    endGravity,
	}

	// Create 100 NN agents
	for i := 0; i < n; i++ {
		a := &Agent{
			Lander:    g.spawnPoint,
			policy:    &NNPolicy{Nets: [4]*nn.NNModule{nn.NewRandomNN(), nn.NewRandomNN(), nn.NewRandomNN(), nn.NewRandomNN()}},
			agentType: "nn",
		}
		g.agents = append(g.agents, a)
	}

	// Create 100 rulebook agents
	for i := 0; i < n; i++ {
		rb := NewRandomRulebook()
		a := &Agent{
			Lander:    g.spawnPoint,
			policy:    &RulebookPolicy{Rulebook: rb},
			agentType: "rulebook",
		}
		g.agents = append(g.agents, a)
	}

	g.totalAgents = len(g.agents)
	return g
}

func (g *Game) Update() error {
	if inpututil.IsKeyJustPressed(ebiten.KeyP) {
		g.paused = !g.paused
	}

	// Handle mouse clicks when paused
	if g.paused {
		// Handle slider dragging
		g.handleSliderDragging()

		// Handle middle mouse button to move spawn point
		if inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonMiddle) {
			mx, my := ebiten.CursorPosition()
			g.spawnPoint.x = float64(mx)
			g.spawnPoint.y = float64(my)
		}

		// Handle left/right clicks to add/remove coins
		if inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonLeft) || inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonRight) {
			mx, my := ebiten.CursorPosition()
			isGreen := inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonLeft)

			// Check if clicking on existing coin to remove it
			removed := false
			for i := len(g.coins) - 1; i >= 0; i-- {
				c := g.coins[i]
				dist := math.Hypot(c.x-float64(mx), c.y-float64(my))
				if dist < 10 { // 10 pixel radius for clicking
					// Remove this coin
					g.coins = append(g.coins[:i], g.coins[i+1:]...)
					removed = true
					break
				}
			}

			// If didn't remove a coin, add a new one
			if !removed {
				g.coins = append(g.coins, Coin{x: float64(mx), y: float64(my), isGreen: isGreen})
			}
		}
		return nil
	}

	g.step++

	for _, a := range g.agents {
		if a.landed || a.crashed {
			continue
		}

		// Find closest green and red coins
		distGreen, distRed := g.closestCoinDistances(&a.Lander)

		thrust, rotate := a.policy.Decide(&a.Lander, g.env, distGreen, distRed)
		a.Lander.angle += rotate
		a.thrusting = thrust
		if thrust {
			t := 0.13
			a.vx += math.Sin(a.angle) * t
			a.vy -= math.Cos(a.angle) * t
		}

		a.vy += g.env.Gravity
		a.x += a.vx
		a.y += a.vy

		if a.x < -50 || a.x > float64(screenWidth)+50 || a.y < -50 || a.y > float64(screenHeight)+50 {
			a.killed = true
			a.crashed = true
			a.vx = 0
			a.vy = 0
			continue
		}

		if a.x < 0 {
			a.x = 0
			a.vx = 0
		}
		if a.x > screenWidth {
			a.x = screenWidth
			a.vx = 0
		}

		groundY := float64(screenHeight - g.env.GroundHeight)
		if a.y >= groundY {
			// Check if within landing pad bounds
			padLeft := g.env.PadX - g.env.PadWidth/2
			padRight := g.env.PadX + g.env.PadWidth/2
			onPad := a.x >= padLeft && a.x <= padRight

			if math.Abs(a.vy) < 2.5 && math.Abs(a.vx) < 2.0 && math.Abs(normalizeAngle(a.angle)) < 0.5 && onPad {
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

		// Check coin collection
		for i := range g.coins {
			if g.coins[i].collected {
				continue // already collected this generation
			}
			dist := math.Hypot(a.x-g.coins[i].x, a.y-g.coins[i].y)
			if dist < 12 { // collision radius
				if g.coins[i].isGreen {
					a.greenCoins++
				} else {
					a.redCoins++
				}
				// Mark coin as collected (not removed)
				g.coins[i].collected = true
			}
		}
	}
	return nil
}

// closestCoinDistances returns the distance to the closest green and red coins
func (g *Game) closestCoinDistances(l *Lander) (distGreen, distRed float64) {
	distGreen = 1e9 // large default
	distRed = 1e9
	for _, c := range g.coins {
		if c.collected {
			continue // skip collected coins
		}
		dist := math.Hypot(l.x-c.x, l.y-c.y)
		if c.isGreen && dist < distGreen {
			distGreen = dist
		} else if !c.isGreen && dist < distRed {
			distRed = dist
		}
	}
	return
}

// updateCurriculumDifficulty calculates and applies the current difficulty based on generation
func (g *Game) updateCurriculumDifficulty() {
	gen := float64(g.generation)
	startGen := float64(g.startGen)
	endGen := float64(g.endGen)

	// Calculate progress ratio (0.0 to 1.0)
	progress := 0.0
	if gen <= startGen {
		progress = 0.0
	} else if gen >= endGen {
		progress = 1.0
	} else {
		progress = (gen - startGen) / (endGen - startGen)
	}

	// Interpolate pad width (starts large, gets smaller)
	g.env.PadWidth = g.startPadWidth + (g.endPadWidth-g.startPadWidth)*progress

	// Interpolate gravity (starts low, gets higher)
	g.env.Gravity = g.startGravity + (g.endGravity-g.startGravity)*progress
}

// handleSliderDragging handles mouse interactions with curriculum sliders
func (g *Game) handleSliderDragging() {
	mx, my := ebiten.CursorPosition()
	mouseX := float64(mx)
	mouseY := float64(my)

	// Check if starting a drag
	if inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonLeft) {
		// Check all control points
		sliderX := 420.0
		sliderY := 60.0
		sliderHeight := 120.0
		margin := 20.0
		axisMargin := 25.0

		// PadWidth slider control points
		padStartX := sliderX + axisMargin
		padStartY := sliderY + 25 + sliderHeight - 35
		padEndX := sliderX + 200 - axisMargin
		padEndY := sliderY + 25

		if g.isNearPoint(mouseX, mouseY, padStartX, padStartY, 10) {
			g.draggingSlider = "padWidth-start"
			g.dragOffsetX = mouseX - padStartX
			g.dragOffsetY = mouseY - padStartY
		} else if g.isNearPoint(mouseX, mouseY, padEndX, padEndY, 10) {
			g.draggingSlider = "padWidth-end"
			g.dragOffsetX = mouseX - padEndX
			g.dragOffsetY = mouseY - padEndY
		}

		// Gravity slider control points
		gravStartX := sliderX + axisMargin
		gravStartY := sliderY + sliderHeight + margin + 25 + sliderHeight - 35
		gravEndX := sliderX + 200 - axisMargin
		gravEndY := sliderY + sliderHeight + margin + 25

		if g.isNearPoint(mouseX, mouseY, gravStartX, gravStartY, 10) {
			g.draggingSlider = "gravity-start"
			g.dragOffsetX = mouseX - gravStartX
			g.dragOffsetY = mouseY - gravStartY
		} else if g.isNearPoint(mouseX, mouseY, gravEndX, gravEndY, 10) {
			g.draggingSlider = "gravity-end"
			g.dragOffsetX = mouseX - gravEndX
			g.dragOffsetY = mouseY - gravEndY
		}
	}

	// Handle ongoing drag
	if g.draggingSlider != "" && ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) {
		// Update the values based on drag position
		sliderX := 420.0
		sliderWidth := 200.0
		axisMargin := 25.0
		axisWidth := sliderWidth - 2*axisMargin

		switch g.draggingSlider {
		case "padWidth-start":
			// Calculate new start value based on X position
			relX := (mouseX - g.dragOffsetX - (sliderX + axisMargin)) / axisWidth
			relX = math.Max(0, math.Min(1, relX))
			minPad := 30.0
			maxPad := float64(screenWidth) / 2.0
			g.startPadWidth = minPad + relX*(maxPad-minPad)

		case "padWidth-end":
			// Calculate new end value based on X position
			relX := (mouseX - g.dragOffsetX - (sliderX + axisMargin)) / axisWidth
			relX = math.Max(0, math.Min(1, relX))
			minPad := 30.0
			maxPad := float64(screenWidth) / 2.0
			g.endPadWidth = minPad + relX*(maxPad-minPad)

		case "gravity-start":
			// Calculate new start value based on X position
			relX := (mouseX - g.dragOffsetX - (sliderX + axisMargin)) / axisWidth
			relX = math.Max(0, math.Min(1, relX))
			minGrav := 0.01
			maxGrav := 0.15
			g.startGravity = minGrav + relX*(maxGrav-minGrav)

		case "gravity-end":
			// Calculate new end value based on X position
			relX := (mouseX - g.dragOffsetX - (sliderX + axisMargin)) / axisWidth
			relX = math.Max(0, math.Min(1, relX))
			minGrav := 0.01
			maxGrav := 0.15
			g.endGravity = minGrav + relX*(maxGrav-minGrav)
		}

		// Recalculate current difficulty
		g.updateCurriculumDifficulty()
	}

	// Check if ending a drag
	if !ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) {
		g.draggingSlider = ""
	}
}

// isNearPoint checks if a point is within a certain distance of another point
func (g *Game) isNearPoint(x1, y1, x2, y2, threshold float64) bool {
	return math.Hypot(x1-x2, y1-y2) < threshold
}

// drawCurriculumSliders draws the curriculum learning progression visualizer
func (g *Game) drawCurriculumSliders(screen *ebiten.Image) {
	// Slider dimensions
	sliderX := 420.0
	sliderY := 60.0
	sliderWidth := 200.0
	sliderHeight := 120.0
	margin := 20.0

	// Draw two sliders: one for PadWidth, one for Gravity
	g.drawSlider(screen, "Pad Width", sliderX, sliderY, sliderWidth, sliderHeight,
		g.startGen, g.endGen, g.startPadWidth, g.endPadWidth, g.env.PadWidth, "padWidth")

	g.drawSlider(screen, "Gravity", sliderX, sliderY+sliderHeight+margin, sliderWidth, sliderHeight,
		g.startGen, g.endGen, g.startGravity, g.endGravity, g.env.Gravity, "gravity")
}

// drawSlider draws a single slider with generation on Y-axis and value on X-axis
func (g *Game) drawSlider(screen *ebiten.Image, title string, x, y, width, height float64,
	startGen, endGen int, startVal, endVal, currentVal float64, sliderID string) {

	// Background box
	bg := ebiten.NewImage(int(width), int(height))
	bg.Fill(color.RGBA{40, 40, 60, 220})
	bgOp := &ebiten.DrawImageOptions{}
	bgOp.GeoM.Translate(x, y)
	screen.DrawImage(bg, bgOp)

	// Title
	ebitenutil.DebugPrintAt(screen, title, int(x)+5, int(y)+5)

	// Axes
	axisColor := color.RGBA{150, 150, 150, 255}
	axisMargin := 25.0
	axisX := x + axisMargin
	axisY := y + 25
	axisWidth := width - 2*axisMargin
	axisHeight := height - 35

	// Draw Y-axis (generation)
	yAxis := ebiten.NewImage(2, int(axisHeight))
	yAxis.Fill(axisColor)
	yAxisOp := &ebiten.DrawImageOptions{}
	yAxisOp.GeoM.Translate(axisX, axisY)
	screen.DrawImage(yAxis, yAxisOp)

	// Draw X-axis (value)
	xAxis := ebiten.NewImage(int(axisWidth), 2)
	xAxis.Fill(axisColor)
	xAxisOp := &ebiten.DrawImageOptions{}
	xAxisOp.GeoM.Translate(axisX, axisY+axisHeight)
	screen.DrawImage(xAxis, xAxisOp)

	// Labels for axes
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("G%d", startGen), int(axisX)-15, int(axisY+axisHeight)-7)
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("G%d", endGen), int(axisX)-15, int(axisY)-7)
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("%.0f", startVal), int(axisX), int(axisY+axisHeight)+5)
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("%.0f", endVal), int(axisX+axisWidth)-25, int(axisY+axisHeight)+5)

	// Calculate control point positions
	// Start point: bottom-left
	startX := axisX
	startY := axisY + axisHeight

	// End point: top-right (or based on end values)
	endX := axisX + axisWidth
	endY := axisY

	// Draw progression line
	numSteps := 50
	for i := 0; i < numSteps; i++ {
		progress := float64(i) / float64(numSteps-1)
		gen := float64(startGen) + (float64(endGen)-float64(startGen))*progress

		// Calculate Y position (generation axis)
		genY := axisY + axisHeight - (gen-float64(startGen))/(float64(endGen)-float64(startGen))*axisHeight

		// Calculate value at this generation
		val := startVal + (endVal-startVal)*progress

		// Calculate X position (value axis)
		valX := axisX + (val-startVal)/(endVal-startVal)*axisWidth

		// Draw point
		if i > 0 {
			dot := ebiten.NewImage(3, 3)
			dot.Fill(color.RGBA{100, 200, 255, 255})
			dotOp := &ebiten.DrawImageOptions{}
			dotOp.GeoM.Translate(valX-1, genY-1)
			screen.DrawImage(dot, dotOp)
		}
	}

	// Draw control points (start and end)
	g.drawControlPoint(screen, startX, startY, "start", sliderID+"-start")
	g.drawControlPoint(screen, endX, endY, "end", sliderID+"-end")

	// Draw current generation marker
	if g.generation >= startGen && g.generation <= endGen {
		genProgress := float64(g.generation-startGen) / float64(endGen-startGen)
		curY := axisY + axisHeight - genProgress*axisHeight
		curX := axisX + (currentVal-startVal)/(endVal-startVal)*axisWidth

		marker := ebiten.NewImage(6, 6)
		marker.Fill(color.RGBA{255, 255, 0, 255})
		markerOp := &ebiten.DrawImageOptions{}
		markerOp.GeoM.Translate(curX-3, curY-3)
		screen.DrawImage(marker, markerOp)
	}

	// Show current values
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Current: %.2f (Gen %d)", currentVal, g.generation),
		int(x)+5, int(y+height)-15)
}

// drawControlPoint draws a draggable control point
func (g *Game) drawControlPoint(screen *ebiten.Image, x, y float64, label, id string) {
	size := 8.0
	point := ebiten.NewImage(int(size), int(size))

	// Highlight if being dragged
	if g.draggingSlider == id {
		point.Fill(color.RGBA{255, 200, 100, 255})
	} else {
		point.Fill(color.RGBA{255, 100, 100, 255})
	}

	pointOp := &ebiten.DrawImageOptions{}
	pointOp.GeoM.Translate(x-size/2, y-size/2)
	screen.DrawImage(point, pointOp)

	// Label
	ebitenutil.DebugPrintAt(screen, label, int(x)+6, int(y)-4)
}
