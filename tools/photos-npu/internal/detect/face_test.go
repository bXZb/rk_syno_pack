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
