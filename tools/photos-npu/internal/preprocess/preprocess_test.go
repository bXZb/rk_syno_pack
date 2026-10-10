package preprocess

import (
	"bytes"
	"image"
	"image/color"
	"image/jpeg"
	"testing"
)

func TestNCHWFloat32ShapeAndMean(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 8, 8))
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			img.Set(x, y, color.RGBA{R: 127, G: 127, B: 127, A: 255})
		}
	}
	out := NCHWFloat32(img, Config{Width: 4, Height: 4, Mean: 127.5, Scale: 0.0078125})
	if len(out) != 3*4*4 {
		t.Fatalf("len=%d", len(out))
	}
	for i, v := range out {
		if v < -0.02 || v > 0.02 {
			t.Fatalf("out[%d]=%v, expected near 0", i, v)
		}
	}
}

func TestPrepareStretchAndMap(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 200, 100))
	res := Prepare(img, Config{Width: 100, Height: 100, Mean: 0, Scale: 1, Stretch: true})
	if res.ScaleX != 0.5 || res.ScaleY != 1 {
		t.Fatalf("scale=%v,%v", res.ScaleX, res.ScaleY)
	}
	x1, y1, x2, y2 := res.MapBox(50, 50, 100, 100)
	if x1 < 99 || x1 > 101 || y1 < 49 || y1 > 51 || x2 < 199 || x2 > 201 {
		t.Fatalf("mapped %v %v %v %v", x1, y1, x2, y2)
	}
}

func TestPrepareReorderBGR(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 2, 2))
	for y := 0; y < 2; y++ {
		for x := 0; x < 2; x++ {
			img.Set(x, y, color.RGBA{R: 200, G: 10, B: 30, A: 255})
		}
	}
	out := NCHWFloat32(img, Config{Width: 2, Height: 2, Mean: 0, Scale: 1, Reorder: true})
	// plane 0 should be B=30, plane 2 should be R=200
	if out[0] < 29 || out[0] > 31 {
		t.Fatalf("B plane %v", out[0])
	}
	if out[2*4] < 199 || out[2*4] > 201 {
		t.Fatalf("R plane %v", out[2*4])
	}
}

func TestDecodeJPEG(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 16, 16))
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 80}); err != nil {
		t.Fatal(err)
	}
	got, err := DecodeImage(buf.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if got.Bounds().Dx() != 16 {
		t.Fatalf("w=%d", got.Bounds().Dx())
	}
}
