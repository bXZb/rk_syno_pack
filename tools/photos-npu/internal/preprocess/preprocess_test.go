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
