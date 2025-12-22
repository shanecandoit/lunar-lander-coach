package main

import (
	"image/color"
	"math"

	"lunar-lander-coach/nn"
	mrand "math/rand"

	"github.com/hajimehoshi/ebiten/v2"
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
