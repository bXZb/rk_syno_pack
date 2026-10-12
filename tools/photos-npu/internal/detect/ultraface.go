package detect

func looksUltra(outs [][]float32) bool {
	if len(outs) < 2 {
		return false
	}
	a, b := len(outs[0]), len(outs[1])
	if a == 0 || b == 0 {
		return false
	}
	// boxes N*4 and scores N*2 (or N)
	if a%4 == 0 && (b == a/4 || b == a/2) {
		return true
	}
	if b%4 == 0 && (a == b/4 || a == b/2) {
		return true
	}
	return false
}

func splitUltra(outs [][]float32) (boxes, scores []float32) {
	if len(outs) < 2 {
		return nil, nil
	}
	a, b := outs[0], outs[1]
	if len(a)%4 == 0 && (len(b) == len(a)/4 || len(b) == len(a)/2) {
		return a, b
	}
	if len(b)%4 == 0 && (len(a) == len(b)/4 || len(a) == len(b)/2) {
		return b, a
	}
	// prefer the tensor whose length is divisible by 4 as boxes
	if len(a)%4 == 0 && len(b)%2 == 0 {
		return a, b
	}
	return b, a
}

// DecodeUltraFace decodes ONNX Model Zoo / Linzaer UltraFace outputs
// (boxes [N,4] in x1y1x2y2, scores [N,2]). Boxes may be normalized 0-1
// or already in network pixels. Returned boxes are in network pixel space.
func DecodeUltraFace(outs [][]float32, netW, netH int, scoreTh, iouTh float32) []Face {
	boxes, scores := splitUltra(outs)
	if len(boxes) < 4 || netW <= 0 || netH <= 0 {
		return nil
	}
	n := len(boxes) / 4
	if n == 0 || len(scores) < n {
		return nil
	}
	if scoreTh <= 0 {
		scoreTh = 0.5
	}
	if iouTh <= 0 {
		iouTh = 0.5
	}
	stride := 1
	if len(scores) >= n*2 {
		stride = 2
	}
	var cands []Face
	for i := 0; i < n; i++ {
		score := scores[i]
		if stride == 2 {
			score = scores[i*2+1]
		}
		if score < scoreTh {
			continue
		}
		x1 := boxes[i*4+0]
		y1 := boxes[i*4+1]
		x2 := boxes[i*4+2]
		y2 := boxes[i*4+3]
		if looksNormalized(x1, y1, x2, y2) {
			x1 *= float32(netW)
			x2 *= float32(netW)
			y1 *= float32(netH)
			y2 *= float32(netH)
		}
		if x2 <= x1 || y2 <= y1 {
			continue
		}
		lx, ly := FakeLandmarks(x1, y1, x2, y2)
		cands = append(cands, Face{
			X1: x1, Y1: y1, X2: x2, Y2: y2,
			Score: score, LX: lx, LY: ly,
		})
	}
	return nms(cands, iouTh)
}
