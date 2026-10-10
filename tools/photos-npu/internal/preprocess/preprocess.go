package preprocess

import (
	"bytes"
	"fmt"
	"image"
	"image/draw"
	_ "image/jpeg"
	_ "image/png"
	"math"
)

// Config matches npu_model_conf.json network blocks.
type Config struct {
	Width  int
	Height int
	Mean   float32
	Scale  float32
}

func DecodeImage(buf []byte) (image.Image, error) {
	img, _, err := image.Decode(bytes.NewReader(buf))
	if err != nil {
		return nil, fmt.Errorf("decode image: %w", err)
	}
	return img, nil
}

// NCHWFloat32 letterboxes the image into a WxH RGB NCHW float32 tensor:
// out[c*H*W + y*W + x] = (pixel[c] - mean) * scale
func NCHWFloat32(img image.Image, cfg Config) []float32 {
	w, h := cfg.Width, cfg.Height
	dst := image.NewRGBA(image.Rect(0, 0, w, h))
	draw.Draw(dst, dst.Bounds(), image.Black, image.Point{}, draw.Src)

	sb := img.Bounds()
	sw, sh := sb.Dx(), sb.Dy()
	if sw <= 0 || sh <= 0 {
		return make([]float32, 3*w*h)
	}
	scale := math.Min(float64(w)/float64(sw), float64(h)/float64(sh))
	nw := int(math.Round(float64(sw) * scale))
	nh := int(math.Round(float64(sh) * scale))
	if nw < 1 {
		nw = 1
	}
	if nh < 1 {
		nh = 1
	}
	ox := (w - nw) / 2
	oy := (h - nh) / 2
	resized := resizeNearest(img, nw, nh)
	draw.Draw(dst, image.Rect(ox, oy, ox+nw, oy+nh), resized, image.Point{}, draw.Src)

	out := make([]float32, 3*w*h)
	mean, sc := cfg.Mean, cfg.Scale
	plane := w * h
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := dst.RGBAAt(x, y).RGBA()
			i := y*w + x
			out[i] = (float32(r>>8) - mean) * sc
			out[plane+i] = (float32(g>>8) - mean) * sc
			out[2*plane+i] = (float32(b>>8) - mean) * sc
		}
	}
	return out
}

func resizeNearest(src image.Image, nw, nh int) *image.RGBA {
	sb := src.Bounds()
	sw, sh := sb.Dx(), sb.Dy()
	dst := image.NewRGBA(image.Rect(0, 0, nw, nh))
	for y := 0; y < nh; y++ {
		sy := sb.Min.Y + y*sh/nh
		for x := 0; x < nw; x++ {
			sx := sb.Min.X + x*sw/nw
			dst.Set(x, y, src.At(sx, sy))
		}
	}
	return dst
}
