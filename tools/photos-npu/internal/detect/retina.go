package detect

import "math"

var retinaMinSizes = [][]int{{16, 32}, {64, 128}, {256, 512}}
var retinaSteps = []int{8, 16, 32}

func PriorBox(imgH, imgW int) [][4]float32 {
	if imgH <= 0 || imgW <= 0 {
		return nil
	}
	var anchors [][4]float32
	for k, step := range retinaSteps {
		fmH := (imgH + step - 1) / step
		fmW := (imgW + step - 1) / step
		for i := 0; i < fmH; i++ {
			for j := 0; j < fmW; j++ {
				for _, ms := range retinaMinSizes[k] {
					skx := float32(ms) / float32(imgW)
					sky := float32(ms) / float32(imgH)
					cx := (float32(j) + 0.5) * float32(step) / float32(imgW)
					cy := (float32(i) + 0.5) * float32(step) / float32(imgH)
					anchors = append(anchors, [4]float32{cx, cy, skx, sky})
				}
			}
		}
	}
	return anchors
}

func looksRetina(outs [][]float32, netW, netH int) bool {
	if len(outs) < 3 || netW <= 0 || netH <= 0 {
		return false
	}
	n := len(PriorBox(netH, netW))
	if n == 0 {
		return false
	}
	got := map[int]bool{}
	for _, o := range outs {
		got[len(o)] = true
	}
	return got[n*4] && got[n*2] && got[n*10]
}

func splitRetina(outs [][]float32, nAnchor int) (loc, conf, landm []float32) {
	for _, o := range outs {
		switch len(o) {
		case nAnchor * 10:
			landm = o
		case nAnchor * 4:
			loc = o
		case nAnchor * 2:
			conf = o
		}
	}
	if loc == nil || conf == nil {
		// fall back to model output order loc, conf, landms
		if len(outs) >= 3 {
			if loc == nil {
				loc = outs[0]
			}
			if conf == nil {
				conf = outs[1]
			}
			if landm == nil {
				landm = outs[2]
			}
		}
	}
	return
}

// DecodeRetinaFace decodes airockchip / biubug6 RetinaFace outputs
// (loc, conf, landms) for a square or rectangular 320/640 net.
// Returned boxes are in network pixel space.
func DecodeRetinaFace(outs [][]float32, netW, netH int, scoreTh, iouTh float32) []Face {
	if netW <= 0 || netH <= 0 {
		return nil
	}
	priors := PriorBox(netH, netW)
	n := len(priors)
	if n == 0 {
		return nil
	}
	loc, conf, landm := splitRetina(outs, n)
	if len(loc) < n*4 || len(conf) < n {
		return nil
	}
	if scoreTh <= 0 {
		scoreTh = 0.5
	}
	if iouTh <= 0 {
		iouTh = 0.5
	}
	confStride := 1
	if len(conf) >= n*2 {
		confStride = 2
	}
	var cands []Face
	for i := 0; i < n; i++ {
		score := conf[i]
		if confStride == 2 {
			score = conf[i*2+1]
		}
		if score < scoreTh {
			continue
		}
		lx := loc[i*4+0]
		ly := loc[i*4+1]
		lw := loc[i*4+2]
		lh := loc[i*4+3]
		cx := priors[i][0] + lx*0.1*priors[i][2]
		cy := priors[i][1] + ly*0.1*priors[i][3]
		bw := priors[i][2] * float32(math.Exp(float64(lw*0.2)))
		bh := priors[i][3] * float32(math.Exp(float64(lh*0.2)))
		x1 := (cx - bw/2) * float32(netW)
		y1 := (cy - bh/2) * float32(netH)
		x2 := (cx + bw/2) * float32(netW)
		y2 := (cy + bh/2) * float32(netH)
		if x2 <= x1 || y2 <= y1 {
			continue
		}
		lmX := make([]float32, 5)
		lmY := make([]float32, 5)
		if len(landm) >= (i+1)*10 {
			for k := 0; k < 5; k++ {
				px := priors[i][0] + landm[i*10+2*k]*0.1*priors[i][2]
				py := priors[i][1] + landm[i*10+2*k+1]*0.1*priors[i][3]
				lmX[k] = px * float32(netW)
				lmY[k] = py * float32(netH)
			}
		} else {
			lmX, lmY = FakeLandmarks(x1, y1, x2, y2)
		}
		cands = append(cands, Face{
			X1: x1, Y1: y1, X2: x2, Y2: y2,
			Score: score, LX: lmX, LY: lmY,
		})
	}
	return nms(cands, iouTh)
}
