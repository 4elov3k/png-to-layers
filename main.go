package main

import (
	"bytes"
	"encoding/csv"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"io"
	"io/ioutil"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// -------------------- CLI --------------------

func main() {
	inPath := flag.String("in", "", "input image path (png/jpg)")
	outDir := flag.String("out", "out", "output directory")
	layers := flag.Int("layers", 4, "number of layers (ignored if -thresholds provided)")
	mode := flag.String("mode", "equal", "layering mode: equal|percentile")
	thresholds := flag.String("thresholds", "", "custom thresholds, comma-separated (0..255). Ex: 60,120,180")
	invert := flag.Bool("invert", false, "invert grayscale (useful if your source is 'black=light')")

	// image adjustment
	gamma := flag.Float64("gamma", 1.0, "gamma correction (1.0 = none). <1 brightens, >1 darkens")
	contrast := flag.Float64("contrast", 1.0, "contrast multiplier (1.0 = none). >1 more contrast")
	brightness := flag.Float64("brightness", 0.0, "brightness add [-1..+1] (0 = none)")
	normalize := flag.Bool("normalize", true, "normalize image to use full 0..255 range before layering")

	// outputs
	writeSVG := flag.Bool("svg", false, "also output svg contours per layer")
	pixelSize := flag.Float64("pxmm", 0.1, "SVG: size of one pixel in mm for viewBox->mm mapping (0.1mm/px is common-ish)")
	dither := flag.Bool("dither", false, "apply Floyd–Steinberg dithering per layer (raster masks)")

	// optional: export a single CSV manifest
	manifest := flag.Bool("manifest", true, "write manifest.csv with layer ranges and files")

	flag.Parse()

	if *inPath == "" {
		fmt.Println("ERROR: -in is required")
		os.Exit(1)
	}

	img, err := loadImage(*inPath)
	if err != nil {
		fmt.Println("ERROR loading:", err)
		os.Exit(1)
	}

	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fmt.Println("ERROR creating out dir:", err)
		os.Exit(1)
	}

	gray := toGray(img)

	// Normalize range if requested
	if *normalize {
		gray = normalizeGray(gray)
	}

	// Apply brightness/contrast/gamma
	if *contrast != 1.0 || *brightness != 0.0 || *gamma != 1.0 {
		gray = adjustGray(gray, *contrast, *brightness, *gamma)
	}

	if *invert {
		gray = invertGray(gray)
	}

	// Build thresholds
	var th []int
	if strings.TrimSpace(*thresholds) != "" {
		th, err = parseThresholds(*thresholds)
		if err != nil {
			fmt.Println("ERROR thresholds:", err)
			os.Exit(1)
		}
	} else {
		switch strings.ToLower(*mode) {
		case "equal":
			th = thresholdsEqual(*layers)
		case "percentile":
			th = thresholdsPercentile(gray, *layers)
		default:
			fmt.Println("ERROR: unknown -mode. Use equal|percentile")
			os.Exit(1)
		}
	}

	// Layer ranges:
	// We define slices by [0..th0], (th0+1..th1], ... (last+1..255]
	// "Darker = lower value" (0=black), so "strongest layer" is first range.
	ranges := buildRanges(th)

	// Create outputs per range: mask where pixels in range are white (255) else black.
	var rows [][]string
	rows = append(rows, []string{"layer", "min", "max", "png", "svg"})

	for i, r := range ranges {
		mask := maskForRange(gray, r.Min, r.Max)
		if *dither {
			// Dither the mask edge as a halftone-ish: use original gray but clamp to the band
			// and then dither to binary. This can soften transitions on wood.
			mask = ditherBandToBinary(gray, r.Min, r.Max)
		}

		base := fmt.Sprintf("layer_%02d_%03d-%03d", i+1, r.Min, r.Max)
		pngPath := filepath.Join(*outDir, base+".png")

		if err := savePNG(pngPath, mask); err != nil {
			fmt.Println("ERROR writing png:", err)
			os.Exit(1)
		}

		svgPath := ""
		if *writeSVG {
			// Extract contours from binary mask
			contours := marchingSquares(mask)
			svgPath = filepath.Join(*outDir, base+".svg")
			if err := saveSVG(svgPath, contours, mask.Bounds().Dx(), mask.Bounds().Dy(), *pixelSize); err != nil {
				fmt.Println("ERROR writing svg:", err)
				os.Exit(1)
			}
		}

		fmt.Printf("Wrote %s\n", pngPath)
		if svgPath != "" {
			fmt.Printf("Wrote %s\n", svgPath)
		}

		rows = append(rows, []string{
			strconv.Itoa(i + 1),
			strconv.Itoa(r.Min),
			strconv.Itoa(r.Max),
			filepath.Base(pngPath),
			filepath.Base(svgPath),
		})
	}

	if *manifest {
		mPath := filepath.Join(*outDir, "manifest.csv")
		if err := writeCSV(mPath, rows); err != nil {
			fmt.Println("ERROR writing manifest:", err)
			os.Exit(1)
		}
		fmt.Printf("Wrote %s\n", mPath)
	}

	fmt.Println("Done.")
}

// -------------------- Image IO --------------------

func loadImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".png":
		return png.Decode(f)
	case ".jpg", ".jpeg":
		return jpeg.Decode(f)
	default:
		// Try auto-detect
		img, _, err := image.Decode(f)
		return img, err
	}
}

func savePNG(path string, img image.Image) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := png.Encoder{CompressionLevel: png.BestCompression}
	return enc.Encode(f, img)
}

// -------------------- Grayscale processing --------------------

func toGray(img image.Image) *image.Gray {
	b := img.Bounds()
	g := image.NewGray(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			c := color.GrayModel.Convert(img.At(x, y)).(color.Gray)
			g.SetGray(x, y, c)
		}
	}
	return g
}

func normalizeGray(g *image.Gray) *image.Gray {
	b := g.Bounds()
	minV := uint8(255)
	maxV := uint8(0)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		i := g.PixOffset(b.Min.X, y)
		row := g.Pix[i : i+(b.Dx())]
		for _, v := range row {
			if v < minV {
				minV = v
			}
			if v > maxV {
				maxV = v
			}
		}
	}
	if maxV == minV {
		return g
	}
	scale := 255.0 / float64(maxV-minV)
	out := image.NewGray(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			v := g.GrayAt(x, y).Y
			n := (float64(v) - float64(minV)) * scale
			if n < 0 {
				n = 0
			}
			if n > 255 {
				n = 255
			}
			out.SetGray(x, y, color.Gray{Y: uint8(n + 0.5)})
		}
	}
	return out
}

// contrast: multiply around mid(0.5). brightness: add [-1..+1] in normalized space.
// gamma: pow(value, 1/gamma) style.
func adjustGray(g *image.Gray, contrast, brightness, gamma float64) *image.Gray {
	b := g.Bounds()
	out := image.NewGray(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			v := float64(g.GrayAt(x, y).Y) / 255.0
			// brightness/contrast around 0.5
			v = (v-0.5)*contrast + 0.5 + brightness
			if v < 0 {
				v = 0
			}
			if v > 1 {
				v = 1
			}
			// gamma
			if gamma != 1.0 && v > 0 {
				v = math.Pow(v, 1.0/gamma)
			}
			out.SetGray(x, y, color.Gray{Y: uint8(v*255 + 0.5)})
		}
	}
	return out
}

func invertGray(g *image.Gray) *image.Gray {
	b := g.Bounds()
	out := image.NewGray(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			v := g.GrayAt(x, y).Y
			out.SetGray(x, y, color.Gray{Y: 255 - v})
		}
	}
	return out
}

// -------------------- Thresholding / ranges --------------------

func parseThresholds(s string) ([]int, error) {
	parts := strings.Split(s, ",")
	var out []int
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		v, err := strconv.Atoi(p)
		if err != nil {
			return nil, err
		}
		if v < 0 || v > 255 {
			return nil, fmt.Errorf("threshold %d out of range 0..255", v)
		}
		out = append(out, v)
	}
	sort.Ints(out)
	// Remove duplicates
	out = uniqueInts(out)
	return out, nil
}

func uniqueInts(a []int) []int {
	if len(a) == 0 {
		return a
	}
	out := []int{a[0]}
	for i := 1; i < len(a); i++ {
		if a[i] != a[i-1] {
			out = append(out, a[i])
		}
	}
	return out
}

func thresholdsEqual(n int) []int {
	if n < 1 {
		n = 1
	}
	// n layers => n-1 thresholds
	if n == 1 {
		return nil
	}
	step := 256.0 / float64(n)
	var th []int
	for i := 1; i < n; i++ {
		t := int(math.Round(step*float64(i))) - 1
		if t < 0 {
			t = 0
		}
		if t > 255 {
			t = 255
		}
		th = append(th, t)
	}
	th = uniqueInts(th)
	return th
}

func thresholdsPercentile(g *image.Gray, n int) []int {
	if n < 1 {
		n = 1
	}
	if n == 1 {
		return nil
	}
	// Build histogram
	var hist [256]int
	b := g.Bounds()
	for y := b.Min.Y; y < b.Max.Y; y++ {
		i := g.PixOffset(b.Min.X, y)
		row := g.Pix[i : i+(b.Dx())]
		for _, v := range row {
			hist[int(v)]++
		}
	}
	total := b.Dx() * b.Dy()
	var th []int
	for i := 1; i < n; i++ {
		target := int(math.Round(float64(total) * float64(i) / float64(n)))
		cum := 0
		t := 0
		for t = 0; t < 256; t++ {
			cum += hist[t]
			if cum >= target {
				break
			}
		}
		if t > 255 {
			t = 255
		}
		th = append(th, t)
	}
	th = uniqueInts(th)
	return th
}

type Range struct{ Min, Max int }

func buildRanges(th []int) []Range {
	if len(th) == 0 {
		return []Range{{Min: 0, Max: 255}}
	}
	var ranges []Range
	prev := 0
	for _, t := range th {
		ranges = append(ranges, Range{Min: prev, Max: t})
		prev = t + 1
	}
	if prev <= 255 {
		ranges = append(ranges, Range{Min: prev, Max: 255})
	}
	return ranges
}

func maskForRange(gray *image.Gray, minV, maxV int) *image.Gray {
	b := gray.Bounds()
	out := image.NewGray(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			v := int(gray.GrayAt(x, y).Y)
			if v >= minV && v <= maxV {
				out.SetGray(x, y, color.Gray{Y: 255})
			} else {
				out.SetGray(x, y, color.Gray{Y: 0})
			}
		}
	}
	return out
}

// Dither: keep only pixels within band; map them to intensity 0..1 within band then FS dither to binary.
// This gives a halftone-like texture in transition areas.
func ditherBandToBinary(gray *image.Gray, minV, maxV int) *image.Gray {
	b := gray.Bounds()
	w, h := b.Dx(), b.Dy()

	// working buffer float64 0..1
	buf := make([]float64, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := int(gray.GrayAt(b.Min.X+x, b.Min.Y+y).Y)
			if v < minV || v > maxV {
				buf[y*w+x] = 0.0
			} else {
				// map band range to [0..1] darker->1 stronger
				// since 0=black, we want darker => more "on"
				// within band: minV (dark) => 1, maxV (light) => 0
				t := float64(v-minV) / float64(maxV-minV+1)
				buf[y*w+x] = 1.0 - t
			}
		}
	}

	out := image.NewGray(b)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			old := buf[y*w+x]
			var newV float64
			if old >= 0.5 {
				newV = 1.0
			} else {
				newV = 0.0
			}
			err := old - newV
			buf[y*w+x] = newV

			// distribute error
			if x+1 < w {
				buf[y*w+(x+1)] += err * 7.0 / 16.0
			}
			if y+1 < h {
				if x > 0 {
					buf[(y+1)*w+(x-1)] += err * 3.0 / 16.0
				}
				buf[(y+1)*w+x] += err * 5.0 / 16.0
				if x+1 < w {
					buf[(y+1)*w+(x+1)] += err * 1.0 / 16.0
				}
			}
			// clamp neighbors can be omitted; FS tolerates small drift

			if newV >= 0.5 {
				out.SetGray(b.Min.X+x, b.Min.Y+y, color.Gray{Y: 255})
			} else {
				out.SetGray(b.Min.X+x, b.Min.Y+y, color.Gray{Y: 0})
			}
		}
	}
	return out
}

// -------------------- SVG contour extraction (marching squares) --------------------

type Point struct{ X, Y float64 }
type Polyline []Point

// marchingSquares returns polylines in pixel coordinates for white regions in mask (Y==255).
func marchingSquares(mask *image.Gray) []Polyline {
	b := mask.Bounds()
	w, h := b.Dx(), b.Dy()

	// Helper: sample cell corner as 0/1
	get := func(x, y int) int {
		if x < 0 || y < 0 || x >= w || y >= h {
			return 0
		}
		v := mask.GrayAt(b.Min.X+x, b.Min.Y+y).Y
		if v >= 128 {
			return 1
		}
		return 0
	}

	// Edge midpoint positions within a cell at (x,y) spanning [x,x+1]x[y,y+1]
	edgePt := func(edge int, x, y int) Point {
		switch edge {
		case 0: // top
			return Point{X: float64(x) + 0.5, Y: float64(y)}
		case 1: // right
			return Point{X: float64(x+1), Y: float64(y) + 0.5}
		case 2: // bottom
			return Point{X: float64(x) + 0.5, Y: float64(y+1)}
		case 3: // left
			return Point{X: float64(x), Y: float64(y) + 0.5}
		default:
			return Point{X: float64(x), Y: float64(y)}
		}
	}

	// Lookup table: for each case (0..15) return list of segments as pairs of edges.
	// This is a standard MS table for isolines at 0.5. We treat "inside" as 1 (white).
	table := map[int][][]int{
		0:  {},
		1:  {{3, 0}},
		2:  {{0, 1}},
		3:  {{3, 1}},
		4:  {{1, 2}},
		5:  {{3, 2}, {0, 1}}, // ambiguous; split
		6:  {{0, 2}},
		7:  {{3, 2}},
		8:  {{2, 3}},
		9:  {{0, 2}},
		10: {{0, 3}, {1, 2}}, // ambiguous; split
		11: {{1, 2}},
		12: {{1, 3}},
		13: {{0, 1}},
		14: {{3, 0}},
		15: {},
	}

	type Seg struct{ A, B Point }
	var segs []Seg

	for y := 0; y < h-1; y++ {
		for x := 0; x < w-1; x++ {
			// corners: tl,tr,br,bl
			tl := get(x, y)
			tr := get(x+1, y)
			br := get(x+1, y+1)
			bl := get(x, y+1)
			c := tl*1 + tr*2 + br*4 + bl*8
			for _, s := range table[c] {
				a := edgePt(s[0], x, y)
				bp := edgePt(s[1], x, y)
				segs = append(segs, Seg{A: a, B: bp})
			}
		}
	}

	// Stitch segments into polylines by endpoint matching (with tolerance)
	const eps = 1e-6
	key := func(p Point) string {
		// quantize to fixed grid to avoid float issues
		qx := int(math.Round(p.X * 1000))
		qy := int(math.Round(p.Y * 1000))
		return fmt.Sprintf("%d,%d", qx, qy)
	}

	adj := make(map[string][]Point) // endpoint -> connected points
	for _, s := range segs {
		ka := key(s.A)
		kb := key(s.B)
		adj[ka] = append(adj[ka], s.B)
		adj[kb] = append(adj[kb], s.A)
	}

	visited := make(map[string]map[string]bool)
	mark := func(a, b string) {
		if visited[a] == nil {
			visited[a] = map[string]bool{}
		}
		visited[a][b] = true
	}
	isMarked := func(a, b string) bool {
		if visited[a] == nil {
			return false
		}
		return visited[a][b]
	}

	var polys []Polyline
	for aKey, neigh := range adj {
		for _, nb := range neigh {
			bKey := key(nb)
			if isMarked(aKey, bKey) {
				continue
			}
			// start walking
			var poly Polyline
			// reconstruct starting point from key by parsing (approx)
			curKey := aKey
			prevKey := ""
			for {
				// decode curKey to point (approx)
				parts := strings.Split(curKey, ",")
				qx, _ := strconv.Atoi(parts[0])
				qy, _ := strconv.Atoi(parts[1])
				curPt := Point{X: float64(qx) / 1000.0, Y: float64(qy) / 1000.0}
				if len(poly) == 0 || dist(poly[len(poly)-1], curPt) > eps {
					poly = append(poly, curPt)
				}

				// choose next neighbor not coming from prevKey and not visited
				nextKey := ""
				for _, cand := range adj[curKey] {
					ck := key(cand)
					if ck == prevKey {
						continue
					}
					if isMarked(curKey, ck) {
						continue
					}
					nextKey = ck
					break
				}
				if nextKey == "" {
					break
				}
				mark(curKey, nextKey)
				mark(nextKey, curKey)
				prevKey, curKey = curKey, nextKey

				// closed loop stop condition
				if curKey == aKey {
					break
				}
			}
			if len(poly) >= 2 {
				polys = append(polys, poly)
			}
		}
	}

	return polys
}

func dist(a, b Point) float64 {
	dx := a.X - b.X
	dy := a.Y - b.Y
	return math.Sqrt(dx*dx + dy*dy)
}

func saveSVG(path string, polys []Polyline, w, h int, pxmm float64) error {
	// Simple SVG with polylines; coordinates in mm.
	var buf bytes.Buffer
	widthMM := float64(w) * pxmm
	heightMM := float64(h) * pxmm

	buf.WriteString(`<?xml version="1.0" encoding="UTF-8"?>` + "\n")
	buf.WriteString(fmt.Sprintf(`<svg xmlns="http://www.w3.org/2000/svg" width="%.3fmm" height="%.3fmm" viewBox="0 0 %.3f %.3f">`+"\n",
		widthMM, heightMM, widthMM, heightMM))

	// White background optional (comment out if unwanted)
	// buf.WriteString(`<rect x="0" y="0" width="100%" height="100%" fill="white"/>` + "\n")

	// Stroke only; fill none. You can change in editor if you want fill.
	buf.WriteString(`<g fill="none" stroke="black" stroke-width="0.1">` + "\n")
	for _, pl := range polys {
		if len(pl) < 2 {
			continue
		}
		buf.WriteString(`<polyline points="`)
		for i, p := range pl {
			x := p.X * pxmm
			y := p.Y * pxmm
			if i > 0 {
				buf.WriteByte(' ')
			}
			buf.WriteString(fmt.Sprintf("%.3f,%.3f", x, y))
		}
		buf.WriteString(`" />` + "\n")
	}
	buf.WriteString(`</g>` + "\n")
	buf.WriteString(`</svg>` + "\n")

	return ioutil.WriteFile(path, buf.Bytes(), 0644)
}

// -------------------- CSV --------------------

func writeCSV(path string, rows [][]string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	defer w.Flush()
	for _, r := range rows {
		if err := w.Write(r); err != nil {
			return err
		}
	}
	return w.Error()
}

// -------------------- Helpers --------------------

func copyToTemp(r io.Reader) (*os.File, error) {
	f, err := ioutil.TempFile("", "img-")
	if err != nil {
		return nil, err
	}
	if _, err := io.Copy(f, r); err != nil {
		f.Close()
		return nil, err
	}
	if _, err := f.Seek(0, 0); err != nil {
		f.Close()
		return nil, err
	}
	return f, nil
}
