package detect

import (
	"math"
	"strings"
)

// Decode128x15 interprets a 128*128*15 face-detection map.
// Official npu_server string order: confidance, y1, x1, y2, x2, landmarky, landmarkx...
// Channel-major (C,H,W) is tried first; values in [0,1] are treated as
// normalized box corners. This is best-effort until a matching RKNN is packed.
func Decode128x15(feat []float32, imgW, imgH int, scoreTh, iouTh float32) []Face {
	const g = 128
	need := 15 * g * g
	if len(feat) < need || imgW <= 0 || imgH <= 0 {
		return nil
	}
	if scoreTh <= 0 {
		scoreTh = 0.5
	}
	if iouTh <= 0 {
		iouTh = 0.5
	}
	var cands []Face
	plane := g * g
	for y := 0; y < g; y++ {
		for x := 0; x < g; x++ {
			i := y*g + x
			conf := feat[i]
			if conf < scoreTh {
				continue
			}
			y1 := feat[plane+i]
			x1 := feat[2*plane+i]
			y2 := feat[3*plane+i]
			x2 := feat[4*plane+i]
			if looksNormalized(x1, y1, x2, y2) {
				x1 *= float32(imgW)
				x2 *= float32(imgW)
				y1 *= float32(imgH)
				y2 *= float32(imgH)
			}
			if x2 <= x1 || y2 <= y1 {
				continue
			}
			lmX := make([]float32, 5)
			lmY := make([]float32, 5)
			for k := 0; k < 5; k++ {
				lmY[k] = feat[(5+2*k)*plane+i]
				lmX[k] = feat[(6+2*k)*plane+i]
				if lmX[k] >= 0 && lmX[k] <= 1 && lmY[k] >= 0 && lmY[k] <= 1 {
					lmX[k] *= float32(imgW)
					lmY[k] *= float32(imgH)
				}
			}
			cands = append(cands, Face{
				X1: x1, Y1: y1, X2: x2, Y2: y2,
				Score: conf, LX: lmX, LY: lmY,
			})
		}
	}
	return nms(cands, iouTh)
}

type Face struct {
	X1, Y1, X2, Y2 float32
	Score          float32
	LX, LY         []float32
}

func looksNormalized(x1, y1, x2, y2 float32) bool {
	return x1 >= 0 && y1 >= 0 && x2 <= 1.05 && y2 <= 1.05 && x2 > x1 && y2 > y1
}

func nms(in []Face, iouTh float32) []Face {
	if len(in) == 0 {
		return nil
	}
	// insertion sort by score desc
	for i := 1; i < len(in); i++ {
		j := i
		for j > 0 && in[j].Score > in[j-1].Score {
			in[j], in[j-1] = in[j-1], in[j]
			j--
		}
	}
	keep := make([]Face, 0, len(in))
	suppressed := make([]bool, len(in))
	for i := range in {
		if suppressed[i] {
			continue
		}
		keep = append(keep, in[i])
		for j := i + 1; j < len(in); j++ {
			if !suppressed[j] && iou(in[i], in[j]) > iouTh {
				suppressed[j] = true
			}
		}
	}
	return keep
}

func iou(a, b Face) float32 {
	ix1 := max32(a.X1, b.X1)
	iy1 := max32(a.Y1, b.Y1)
	ix2 := min32(a.X2, b.X2)
	iy2 := min32(a.Y2, b.Y2)
	iw := max32(0, ix2-ix1)
	ih := max32(0, iy2-iy1)
	inter := iw * ih
	if inter == 0 {
		return 0
	}
	ua := (a.X2-a.X1)*(a.Y2-a.Y1) + (b.X2-b.X1)*(b.Y2-b.Y1) - inter
	if ua <= 0 {
		return 0
	}
	return inter / ua
}

func max32(a, b float32) float32 {
	return float32(math.Max(float64(a), float64(b)))
}
func min32(a, b float32) float32 {
	return float32(math.Min(float64(a), float64(b)))
}

// Official npu_rtd1619b 112x112 alignment template.
var refLandmarks = [5][2]float32{
	{38.2946, 51.6963},
	{73.5318, 51.5014},
	{56.0252, 71.7366},
	{41.5493, 92.3655},
	{70.7299, 92.2041},
}

func FakeLandmarks(x1, y1, x2, y2 float32) (lx, ly []float32) {
	w := x2 - x1
	h := y2 - y1
	lx = make([]float32, 5)
	ly = make([]float32, 5)
	for i, p := range refLandmarks {
		lx[i] = x1 + p[0]/112*w
		ly[i] = y1 + p[1]/112*h
	}
	return
}

func DecodeAuto(outs [][]float32, netW, netH int, scoreTh, iouTh float32) []Face {
	if looksRetina(outs, netW, netH) {
		return DecodeRetinaFace(outs, netW, netH, scoreTh, iouTh)
	}
	if looksUltra(outs) {
		return DecodeUltraFace(outs, netW, netH, scoreTh, iouTh)
	}
	if len(outs) == 1 && len(outs[0]) >= 15*128*128 {
		return Decode128x15(outs[0], netW, netH, scoreTh, iouTh)
	}
	if len(outs) == 2 {
		return DecodeUltraFace(outs, netW, netH, scoreTh, iouTh)
	}
	if len(outs) >= 3 {
		return DecodeRetinaFace(outs, netW, netH, scoreTh, iouTh)
	}
	return nil
}

func DecodeNamed(name string, outs [][]float32, netW, netH int, scoreTh, iouTh float32) []Face {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "retinaface", "retina":
		return DecodeRetinaFace(outs, netW, netH, scoreTh, iouTh)
	case "ultraface", "ultra":
		return DecodeUltraFace(outs, netW, netH, scoreTh, iouTh)
	case "map128x15", "128x15", "official":
		if len(outs) == 0 {
			return nil
		}
		return Decode128x15(outs[0], netW, netH, scoreTh, iouTh)
	default:
		return DecodeAuto(outs, netW, netH, scoreTh, iouTh)
	}
}
