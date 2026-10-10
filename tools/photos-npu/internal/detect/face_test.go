package detect

import "testing"

func TestNMSKeepsHighest(t *testing.T) {
	faces := []Face{
		{X1: 0, Y1: 0, X2: 10, Y2: 10, Score: 0.4},
		{X1: 1, Y1: 1, X2: 11, Y2: 11, Score: 0.9},
		{X1: 50, Y1: 50, X2: 60, Y2: 60, Score: 0.7},
	}
	got := nms(faces, 0.3)
	if len(got) != 2 {
		t.Fatalf("len=%d", len(got))
	}
	if got[0].Score != 0.9 {
		t.Fatalf("first=%v", got[0].Score)
	}
}

func TestDecodeTooShort(t *testing.T) {
	if got := Decode128x15([]float32{1, 2, 3}, 512, 512, 0.5, 0.5); got != nil {
		t.Fatalf("got %v", got)
	}
}

func TestPriorBox320(t *testing.T) {
	p := PriorBox(320, 320)
	if len(p) != 4200 {
		t.Fatalf("anchors=%d want 4200", len(p))
	}
}

func TestDecodeUltraFaceNormalized(t *testing.T) {
	// one high-score box in normalized coords + one low-score
	boxes := []float32{
		0.1, 0.2, 0.4, 0.6,
		0.0, 0.0, 0.1, 0.1,
	}
	scores := []float32{
		0.1, 0.9,
		0.8, 0.05,
	}
	got := DecodeUltraFace([][]float32{boxes, scores}, 320, 240, 0.5, 0.5)
	if len(got) != 1 {
		t.Fatalf("faces=%d", len(got))
	}
	if got[0].X1 < 30 || got[0].X1 > 34 {
		t.Fatalf("x1=%v", got[0].X1)
	}
	if len(got[0].LX) != 5 {
		t.Fatalf("landmarks=%d", len(got[0].LX))
	}
}

func TestDecodeRetinaZeroLocCenterish(t *testing.T) {
	n := len(PriorBox(320, 320))
	loc := make([]float32, n*4)
	conf := make([]float32, n*2)
	landm := make([]float32, n*10)
	// force the middle-ish prior above threshold
	conf[n+1] = 0.99 // i=0 uses conf[1] when stride=2; set many
	for i := 0; i < n; i++ {
		conf[i*2+1] = 0.01
	}
	conf[100*2+1] = 0.95
	got := DecodeRetinaFace([][]float32{loc, conf, landm}, 320, 320, 0.5, 0.5)
	if len(got) != 1 {
		t.Fatalf("faces=%d", len(got))
	}
	if got[0].X2 <= got[0].X1 || got[0].Y2 <= got[0].Y1 {
		t.Fatalf("bad box %+v", got[0])
	}
}

func TestDecodeNamedAutoUltra(t *testing.T) {
	boxes := []float32{0.2, 0.2, 0.8, 0.8}
	scores := []float32{0.05, 0.8}
	got := DecodeNamed("auto", [][]float32{boxes, scores}, 100, 100, 0.5, 0.5)
	if len(got) != 1 {
		t.Fatalf("faces=%d", len(got))
	}
}
