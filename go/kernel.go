package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
)

// Semantic axes for deity representation
var AXES = []string{
	"authority", "transcendence", "care", "justice", "wisdom", "power",
	"fertility", "war", "death", "creation", "nature", "order",
}

// Historical deity semantic priors
func getDeityPriors() map[string][]float64 {
	priors := map[string][]float64{
		"zeus":    {0.9, 0.8, 0.3, 0.7, 0.6, 0.9, 0.2, 0.8, 0.1, 0.4, 0.3, 0.8},
		"odin":    {0.8, 0.7, 0.4, 0.6, 0.9, 0.7, 0.1, 0.9, 0.8, 0.3, 0.2, 0.5},
		"amun":    {0.9, 0.9, 0.6, 0.8, 0.8, 0.8, 0.3, 0.2, 0.1, 0.9, 0.1, 0.9},
		"marduk":  {0.9, 0.6, 0.5, 0.9, 0.7, 0.9, 0.1, 0.8, 0.3, 0.7, 0.1, 0.9},
		"indra":   {0.8, 0.5, 0.4, 0.7, 0.6, 0.9, 0.2, 0.9, 0.2, 0.3, 0.4, 0.6},
		"shango":  {0.7, 0.4, 0.3, 0.8, 0.5, 0.8, 0.1, 0.7, 0.2, 0.2, 0.6, 0.5},
		"kami":    {0.3, 0.8, 0.8, 0.4, 0.7, 0.4, 0.6, 0.1, 0.1, 0.5, 0.9, 0.8},
		"manitou": {0.2, 0.9, 0.9, 0.3, 0.8, 0.3, 0.7, 0.1, 0.2, 0.6, 0.9, 0.4},
	}

	// Normalize all vectors
	for name, vector := range priors {
		norm := 0.0
		for _, v := range vector {
			norm += v * v
		}
		norm = math.Sqrt(norm)
		if norm > 0 {
			for i := range vector {
				vector[i] /= norm
			}
		}
		priors[name] = vector
	}

	return priors
}

func getTheonyms() []string {
	return []string{
		"zeus", "odin", "amun", "marduk", "indra", "shango", "kami", "manitou",
		"apollo", "freya", "ptah", "ishtar", "perun", "teotl", "nut", "hades",
		"thor", "isis", "ra", "quetzal", "tyr", "bast", "lugh", "brigid",
		"taranis", "nana", "enar", "yah", "baal",
	}
}

type Config struct {
	N                   int     `json:"n"`
	StepsPerGeneration  int     `json:"steps_per_generation"`
	MaxMessageLen       int     `json:"max_message_len"`
	LearningRate        float64 `json:"learning_rate"`
	PenaltyRate         float64 `json:"penalty_rate"`
	PrestigeAlpha       float64 `json:"prestige_alpha"`
	RitualPeriod        int     `json:"ritual_period"`
	RitualBonus         float64 `json:"ritual_bonus"`
	BaseSuccessThresh   float64 `json:"base_success_thresh"`
	MutationRate        float64 `json:"mutation_rate"`
	ExplorationEps      float64 `json:"exploration_eps"`
	GenerationMixK      int     `json:"generation_mix_k"`
	Seed                int64   `json:"seed"`
	TopoWindow          int     `json:"topo_window"`
	UseDeityPriors      bool    `json:"use_deity_priors"`
	BeliefInfluence     float64 `json:"belief_influence"`
	Coercion            float64 `json:"coercion"`
	SocialNetwork       string  `json:"social_network"`
	SocialK             int     `json:"social_k"`
	SocialP             float64 `json:"social_p"`
	ClusterUpdateFreq   int     `json:"cluster_update_freq"`
	ClusterThreshold    float64 `json:"cluster_threshold"`
}

type Context struct {
	Task string
	Role string
	Place string
	Tod  string
	Vec  []float64
}

func NewContext(rng *rand.Rand) *Context {
	tasks := []string{"hunt", "gather", "build", "trade", "ritual", "teach"}
	roles := []string{"leader", "shaman", "warrior", "elder", "child", "healer"}
	places := []string{"forest", "river", "mountain", "cave", "village", "field"}
	tods := []string{"dawn", "morning", "noon", "evening", "dusk", "night"}

	ctx := &Context{
		Task:  tasks[rng.Intn(len(tasks))],
		Role:  roles[rng.Intn(len(roles))],
		Place: places[rng.Intn(len(places))],
		Tod:   tods[rng.Intn(len(tods))],
		Vec:   make([]float64, len(AXES)),
	}

	// Generate semantic vector
	for i := range ctx.Vec {
		ctx.Vec[i] = rng.Float64()*2.0 - 1.0
	}
	normalizeVector(ctx.Vec)

	return ctx
}

type Agent struct {
	ID     int
	Belief []float64
	W      float64
	Assoc  map[string][]float64
	Freq   map[string]int
	mu     sync.RWMutex
}

func NewAgent(id int, belief []float64) *Agent {
	return &Agent{
		ID:     id,
		Belief: belief,
		W:      1.0,
		Assoc:  make(map[string][]float64),
		Freq:   make(map[string]int),
	}
}

type Metrics struct {
	ZipfSlope      float64
	HeapsK         float64
	CondEntropy    float64
	TopoSimilarity float64
	Churn          float64
}

type SwarmKernel struct {
	cfg         Config
	rng         *rand.Rand
	agents      []*Agent
	t           int
	gen         int
	tokens      []string
	types       map[string]int
	bigrams     map[[2]string]int
	lastToken   string
	prefForm    map[int]string
	metrics     Metrics
	socialGraph map[int][]int
	clusters    [][]int
	centroids   [][]float64
	mu          sync.RWMutex
}

func NewSwarmKernel(cfg Config) *SwarmKernel {
	rng := rand.New(rand.NewSource(cfg.Seed))
	
	k := &SwarmKernel{
		cfg:         cfg,
		rng:         rng,
		agents:      make([]*Agent, 0, cfg.N),
		tokens:      make([]string, 0),
		types:       make(map[string]int),
		bigrams:     make(map[[2]string]int),
		prefForm:    make(map[int]string),
		socialGraph: make(map[int][]int),
		clusters:    make([][]int, 0),
		centroids:   make([][]float64, 0),
	}

	k.initAgents()
	k.buildSocialNetwork()
	return k
}

func (k *SwarmKernel) initAgents() {
	deityPriors := getDeityPriors()
	theonyms := getTheonyms()

	for i := 0; i < k.cfg.N; i++ {
		var belief []float64

		if k.cfg.UseDeityPriors {
			// Choose 1-2 deities for belief foundation
			chosenCount := k.rng.Intn(2) + 1
			deityNames := make([]string, 0, len(deityPriors))
			for name := range deityPriors {
				deityNames = append(deityNames, name)
			}

			belief = make([]float64, len(AXES))
			for j := 0; j < chosenCount; j++ {
				deityName := deityNames[k.rng.Intn(len(deityNames))]
				if deityVec, exists := deityPriors[deityName]; exists {
					for idx, val := range deityVec {
						belief[idx] += val
					}
				}
			}

			// Add jitter
			for j := range belief {
				belief[j] += (k.rng.Float64()*2.0 - 1.0) * 0.1
			}
			normalizeVector(belief)
		} else {
			belief = randomUnitVector(k.rng, len(AXES))
		}

		agent := NewAgent(i, belief)

		// Seed with theonyms
		for _, name := range theonyms {
			var vec []float64
			if k.cfg.UseDeityPriors {
				if deityVec, exists := deityPriors[name]; exists {
					vec = jitterVector(deityVec, k.rng, 0.1)
				} else {
					vec = randomUnitVector(k.rng, len(AXES))
				}
			} else {
				vec = randomUnitVector(k.rng, len(AXES))
			}
			agent.Assoc[name] = vec
		}

		k.agents = append(k.agents, agent)
		k.prefForm[i] = ""
	}
}

func (k *SwarmKernel) buildSocialNetwork() {
	// Watts-Strogatz small-world network
	for i := 0; i < k.cfg.N; i++ {
		k.socialGraph[i] = make([]int, 0)
	}

	ringK := k.cfg.SocialK / 2

	// Create ring lattice
	for i := 0; i < k.cfg.N; i++ {
		for j := 1; j <= ringK; j++ {
			neighbor := (i + j) % k.cfg.N
			k.socialGraph[i] = append(k.socialGraph[i], neighbor)
			k.socialGraph[neighbor] = append(k.socialGraph[neighbor], i)
		}
	}

	// Rewire with probability p
	for i := 0; i < k.cfg.N; i++ {
		neighbors := make([]int, len(k.socialGraph[i]))
		copy(neighbors, k.socialGraph[i])

		for _, oldNeighbor := range neighbors {
			if k.rng.Float64() < k.cfg.SocialP {
				// Remove old edge
				k.removeEdge(i, oldNeighbor)

				// Add new edge
				candidates := make([]int, 0)
				for j := 0; j < k.cfg.N; j++ {
					if j != i && !k.hasEdge(i, j) {
						candidates = append(candidates, j)
					}
				}

				if len(candidates) > 0 {
					newNeighbor := candidates[k.rng.Intn(len(candidates))]
					k.socialGraph[i] = append(k.socialGraph[i], newNeighbor)
					k.socialGraph[newNeighbor] = append(k.socialGraph[newNeighbor], i)
				}
			}
		}
	}
}

func (k *SwarmKernel) removeEdge(a, b int) {
	k.socialGraph[a] = removeInt(k.socialGraph[a], b)
	k.socialGraph[b] = removeInt(k.socialGraph[b], a)
}

func (k *SwarmKernel) hasEdge(a, b int) bool {
	for _, neighbor := range k.socialGraph[a] {
		if neighbor == b {
			return true
		}
	}
	return false
}

func (k *SwarmKernel) Run(steps int) {
	for step := 0; step < steps; step++ {
		k.step()

		if step%500 == 0 {
			k.updateMetrics()
			fmt.Printf("t=%6d gen=%d | zipf=%+.2f heaps=%.3f entropy=%.2f topo=%+.2f churn=%.2f\n",
				k.t, k.gen, k.metrics.ZipfSlope, k.metrics.HeapsK,
				k.metrics.CondEntropy, k.metrics.TopoSimilarity, k.metrics.Churn)
		}

		if k.t%k.cfg.ClusterUpdateFreq == 0 {
			k.updateClusters()
		}
	}
}

func (k *SwarmKernel) step() {
	k.t++

	// Select speaker weighted by prestige
	weights := make([]float64, len(k.agents))
	for i, agent := range k.agents {
		weights[i] = agent.W
	}
	speakerIdx := weightedChoice(k.rng, weights)

	// Select hearers from social network
	hearers := k.selectHearers(speakerIdx)

	// Generate context
	ctx := NewContext(k.rng)

	// Produce message
	msg := k.produce(speakerIdx, ctx)

	// Update token statistics
	k.mu.Lock()
	for _, token := range msg {
		k.tokens = append(k.tokens, token)
		k.types[token]++

		if k.lastToken != "" {
			bigram := [2]string{k.lastToken, token}
			k.bigrams[bigram]++
		}
		k.lastToken = token
	}
	k.mu.Unlock()

	// Interaction and learning
	success := k.interact(speakerIdx, hearers, ctx, msg)
	k.learnFrom(speakerIdx, msg, ctx, success)
	for _, hearerIdx := range hearers {
		k.learnFrom(hearerIdx, msg, ctx, success)
	}

	// Update prestige
	k.updatePrestige([]int{speakerIdx}, success)
	k.updatePrestige(hearers, success)

	// Mutations
	if k.rng.Float64() < k.cfg.MutationRate {
		agentIdx := k.rng.Intn(len(k.agents))
		k.mutateAgent(agentIdx)
	}
}

func (k *SwarmKernel) selectHearers(speakerIdx int) []int {
	neighbors, exists := k.socialGraph[speakerIdx]
	if !exists || len(neighbors) == 0 {
		// Fallback to random selection
		candidates := make([]int, 0)
		for i := 0; i < k.cfg.N; i++ {
			if i != speakerIdx {
				candidates = append(candidates, i)
			}
		}
		return randomSample(k.rng, candidates, min(2, len(candidates)))
	}

	if k.cfg.Coercion > 0.0 {
		// Weight by belief similarity
		speakerBelief := k.agents[speakerIdx].Belief
		weights := make([]float64, len(neighbors))

		for i, neighborIdx := range neighbors {
			neighborBelief := k.agents[neighborIdx].Belief
			sim := cosineSimilarity(speakerBelief, neighborBelief)
			weight := math.Exp(sim * (1.0 + 9.0*k.cfg.Coercion))
			weights[i] = weight
		}

		chosenIndices := weightedSample(k.rng, len(neighbors), weights, min(2, len(neighbors)))
		result := make([]int, len(chosenIndices))
		for i, idx := range chosenIndices {
			result[i] = neighbors[idx]
		}
		return result
	}

	return randomSample(k.rng, neighbors, min(2, len(neighbors)))
}

func (k *SwarmKernel) produce(agentIdx int, ctx *Context) []string {
	agent := k.agents[agentIdx]
	agent.mu.RLock()
	defer agent.mu.RUnlock()

	scored := make([]struct {
		form  string
		score float64
	}, 0, len(agent.Assoc))

	totalFreq := 0
	for _, freq := range agent.Freq {
		totalFreq += freq
	}

	for form, vec := range agent.Assoc {
		ctxScore := dotProduct(vec, ctx.Vec)
		beliefScore := dotProduct(vec, agent.Belief) * k.cfg.BeliefInfluence
		freqScore := 0.1 * math.Log(float64(agent.Freq[form]+1)/float64(totalFreq+1))
		score := ctxScore + beliefScore + freqScore
		scored = append(scored, struct {
			form  string
			score float64
		}{form, score})
	}

	msg := make([]string, k.cfg.MaxMessageLen)
	for i := 0; i < k.cfg.MaxMessageLen; i++ {
		choice := k.softmaxChoice(scored)
		msg[i] = choice
	}
	return msg
}

func (k *SwarmKernel) softmaxChoice(items []struct {
	form  string
	score float64
}) string {
	if k.rng.Float64() < k.cfg.ExplorationEps {
		return items[k.rng.Intn(len(items))].form
	}

	maxScore := math.Inf(-1)
	for _, item := range items {
		if item.score > maxScore {
			maxScore = item.score
		}
	}

	expScores := make([]float64, len(items))
	sum := 0.0
	for i, item := range items {
		expScores[i] = math.Exp(item.score - maxScore)
		sum += expScores[i]
	}

	r := k.rng.Float64() * sum
	acc := 0.0
	for i, expScore := range expScores {
		acc += expScore
		if acc >= r {
			return items[i].form
		}
	}
	return items[len(items)-1].form
}

func (k *SwarmKernel) interact(speakerIdx int, hearers []int, ctx *Context, msg []string) bool {
	// Simplified success model
	return k.rng.Float64() > k.cfg.BaseSuccessThresh
}

func (k *SwarmKernel) learnFrom(agentIdx int, msg []string, ctx *Context, success bool) {
	agent := k.agents[agentIdx]
	agent.mu.Lock()
	defer agent.mu.Unlock()

	lr := k.cfg.LearningRate
	if !success {
		lr = -k.cfg.PenaltyRate
	}

	for _, token := range msg {
		if _, exists := agent.Assoc[token]; !exists {
			agent.Assoc[token] = randomUnitVector(k.rng, len(AXES))
		}

		assocVec := agent.Assoc[token]
		for i, ctxVal := range ctx.Vec {
			assocVec[i] += lr * ctxVal
		}
		normalizeVector(assocVec)
		agent.Assoc[token] = assocVec

		agent.Freq[token]++
	}
}

func (k *SwarmKernel) updatePrestige(agentIndices []int, success bool) {
	delta := k.cfg.PrestigeAlpha
	if !success {
		delta = -k.cfg.PrestigeAlpha * 0.3
	}

	for _, idx := range agentIndices {
		agent := k.agents[idx]
		agent.mu.Lock()
		agent.W = clamp(agent.W*(1.0+delta), 0.1, 10.0)
		agent.mu.Unlock()
	}
}

func (k *SwarmKernel) mutateAgent(agentIdx int) {
	// Simplified mutation
}

func (k *SwarmKernel) updateClusters() {
	k.mu.Lock()
	defer k.mu.Unlock()

	k.centroids = k.centroids[:0]
	k.clusters = k.clusters[:0]

	for _, agent := range k.agents {
		if len(k.centroids) == 0 {
			k.centroids = append(k.centroids, copyVector(agent.Belief))
			k.clusters = append(k.clusters, []int{agent.ID})
			continue
		}

		distances := make([]float64, len(k.centroids))
		for i, centroid := range k.centroids {
			distances[i] = 1.0 - cosineSimilarity(agent.Belief, centroid)
		}

		minDist := distances[0]
		bestIdx := 0
		for i, dist := range distances {
			if dist < minDist {
				minDist = dist
				bestIdx = i
			}
		}

		if minDist < k.cfg.ClusterThreshold {
			k.clusters[bestIdx] = append(k.clusters[bestIdx], agent.ID)
		} else {
			k.centroids = append(k.centroids, copyVector(agent.Belief))
			k.clusters = append(k.clusters, []int{agent.ID})
		}
	}

	// Recalculate centroids
	newCentroids := make([][]float64, 0, len(k.clusters))
	for _, cluster := range k.clusters {
		if len(cluster) == 0 {
			continue
		}

		centroid := make([]float64, len(AXES))
		for _, agentID := range cluster {
			for i, val := range k.agents[agentID].Belief {
				centroid[i] += val
			}
		}

		for i := range centroid {
			centroid[i] /= float64(len(cluster))
		}
		normalizeVector(centroid)
		newCentroids = append(newCentroids, centroid)
	}
	k.centroids = newCentroids
}

func (k *SwarmKernel) updateMetrics() {
	k.mu.RLock()
	defer k.mu.RUnlock()

	k.metrics.ZipfSlope = k.calculateZipfSlope()
	k.metrics.HeapsK = k.calculateHeapsK()
	k.metrics.CondEntropy = k.calculateConditionalEntropy()
	k.metrics.TopoSimilarity = 0.0 // Placeholder
	k.metrics.Churn = 0.0          // Placeholder
}

func (k *SwarmKernel) calculateZipfSlope() float64 {
	if len(k.types) < 2 {
		return 0.0
	}

	counts := make([]int, 0, len(k.types))
	for _, count := range k.types {
		counts = append(counts, count)
	}
	sort.Sort(sort.Reverse(sort.IntSlice(counts)))

	logRanks := make([]float64, 0)
	logFreqs := make([]float64, 0)

	for i, count := range counts {
		if count > 0 {
			logRanks = append(logRanks, math.Log(float64(i+1)))
			logFreqs = append(logFreqs, math.Log(float64(count)))
		}
	}

	if len(logRanks) < 2 {
		return 0.0
	}

	// Simple linear regression
	n := float64(len(logRanks))
	sumX := sum(logRanks)
	sumY := sum(logFreqs)
	sumXY := 0.0
	sumX2 := 0.0

	for i, x := range logRanks {
		y := logFreqs[i]
		sumXY += x * y
		sumX2 += x * x
	}

	slope := (n*sumXY - sumX*sumY) / (n*sumX2 - sumX*sumX)
	return slope
}

func (k *SwarmKernel) calculateHeapsK() float64 {
	if len(k.tokens) == 0 {
		return 0.0
	}

	types := float64(len(k.types))
	tokens := float64(len(k.tokens))

	if tokens <= 1.0 {
		return 0.0
	}

	return types / math.Pow(tokens, 0.5)
}

func (k *SwarmKernel) calculateConditionalEntropy() float64 {
	if len(k.bigrams) == 0 {
		return 0.0
	}

	prevCounts := make(map[string]int)
	for bigram, count := range k.bigrams {
		prevCounts[bigram[0]] += count
	}

	entropy := 0.0
	totalBigrams := 0
	for _, count := range k.bigrams {
		totalBigrams += count
	}

	for prev, prevCount := range prevCounts {
		pPrev := float64(prevCount) / float64(totalBigrams)
		conditionalEntropy := 0.0

		for bigram, count := range k.bigrams {
			if bigram[0] == prev {
				pNextGivenPrev := float64(count) / float64(prevCount)
				if pNextGivenPrev > 0.0 {
					conditionalEntropy -= pNextGivenPrev * math.Log2(pNextGivenPrev)
				}
			}
		}

		entropy += pPrev * conditionalEntropy
	}

	return entropy
}

// Utility functions
func normalizeVector(vec []float64) {
	norm := 0.0
	for _, v := range vec {
		norm += v * v
	}
	norm = math.Sqrt(norm)
	if norm > 0 {
		for i := range vec {
			vec[i] /= norm
		}
	}
}

func randomUnitVector(rng *rand.Rand, dim int) []float64 {
	vec := make([]float64, dim)
	for i := range vec {
		vec[i] = rng.Float64()*2.0 - 1.0
	}
	normalizeVector(vec)
	return vec
}

func jitterVector(base []float64, rng *rand.Rand, jitter float64) []float64 {
	vec := make([]float64, len(base))
	for i, v := range base {
		vec[i] = v + (rng.Float64()*2.0-1.0)*jitter
	}
	normalizeVector(vec)
	return vec
}

func copyVector(vec []float64) []float64 {
	result := make([]float64, len(vec))
	copy(result, vec)
	return result
}

func dotProduct(a, b []float64) float64 {
	sum := 0.0
	for i, v := range a {
		sum += v * b[i]
	}
	return sum
}

func cosineSimilarity(a, b []float64) float64 {
	dot := dotProduct(a, b)
	normA := 0.0
	normB := 0.0
	for i, v := range a {
		normA += v * v
		normB += b[i] * b[i]
	}
	normA = math.Sqrt(normA)
	normB = math.Sqrt(normB)

	if normA > 0 && normB > 0 {
		return dot / (normA * normB)
	}
	return 0.0
}

func weightedChoice(rng *rand.Rand, weights []float64) int {
	total := sum(weights)
	r := rng.Float64() * total

	for i, weight := range weights {
		r -= weight
		if r <= 0 {
			return i
		}
	}
	return len(weights) - 1
}

func weightedSample(rng *rand.Rand, n int, weights []float64, k int) []int {
	result := make([]int, 0, k)
	total := sum(weights)

	for i := 0; i < k && len(result) < n; i++ {
		r := rng.Float64() * total
		for j, weight := range weights {
			r -= weight
			if r <= 0 {
				result = append(result, j)
				break
			}
		}
	}
	return result
}

func randomSample(rng *rand.Rand, items []int, k int) []int {
	if k >= len(items) {
		result := make([]int, len(items))
		copy(result, items)
		return result
	}

	result := make([]int, k)
	indices := make([]int, len(items))
	copy(indices, items)

	for i := 0; i < k; i++ {
		j := rng.Intn(len(indices))
		result[i] = indices[j]
		indices[j] = indices[len(indices)-1]
		indices = indices[:len(indices)-1]
	}
	return result
}

func removeInt(slice []int, item int) []int {
	for i, v := range slice {
		if v == item {
			return append(slice[:i], slice[i+1:]...)
		}
	}
	return slice
}

func sum(slice []float64) float64 {
	total := 0.0
	for _, v := range slice {
		total += v
	}
	return total
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func clamp(value, min, max float64) float64 {
	if value < min {
		return min
	}
	if value > max {
		return max
	}
	return value
}
